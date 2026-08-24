package openairesponses

import (
	"encoding/base64"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"mime"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/openai/openai-go/v3/packages/param"
	openairesponses "github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const (
	providerDataVersion = 1
)

// Codec translates llm-go's neutral conversation representation to and from
// the OpenAI Responses protocol.
type Codec struct {
	Provider             string
	ProviderDataAPI      string
	MaxProviderDataBytes int
}

// RequestOptions captures endpoint differences within the Responses protocol.
type RequestOptions struct {
	Subscription     bool
	ReasoningSummary bool
}

type providerDataEnvelope struct {
	API     string           `json:"api"`
	Version int              `json:"version"`
	Model   string           `json:"model"`
	Output  []jsontext.Value `json:"output"`
}

// Request converts a validated neutral request into Responses API parameters.
func (c Codec) Request(op, model string, request llm.Request, options RequestOptions) (openairesponses.ResponseNewParams, error) {
	instructions := make([]string, 0, 1)
	input := make(openairesponses.ResponseInputParam, 0, len(request.Messages))
	for i, message := range request.Messages {
		text := message.Text()
		switch message.Role {
		case llm.RoleSystem:
			if text != "" {
				instructions = append(instructions, text)
			}
		case llm.RoleUser:
			content, hasImage, err := c.userContentToWire(op, i, message.Content)
			if err != nil {
				return openairesponses.ResponseNewParams{}, err
			}
			if hasImage {
				input = append(input, openairesponses.ResponseInputItemParamOfMessage(content, openairesponses.EasyInputMessageRoleUser))
			} else {
				input = append(input, openairesponses.ResponseInputItemParamOfMessage(text, openairesponses.EasyInputMessageRoleUser))
			}
		case llm.RoleAssistant:
			items, replayed, err := c.providerItems(message.ProviderData, model)
			if err != nil {
				return openairesponses.ResponseNewParams{}, c.requestError(op, "messages[%d] provider data: %v", i, err)
			}
			if replayed {
				input = append(input, items...)
				continue
			}
			if text != "" {
				input = append(input, openairesponses.ResponseInputItemParamOfMessage(text, openairesponses.EasyInputMessageRoleAssistant))
			}
			for _, call := range message.ToolCalls {
				input = append(input, openairesponses.ResponseInputItemParamOfFunctionCall(string(call.Arguments), call.ID, call.Name))
			}
		case llm.RoleTool:
			for _, result := range message.ToolResults {
				input = append(input, openairesponses.ResponseInputItemParamOfFunctionCallOutput(result.CallID, toolResultText(result)))
			}
		}
	}
	if len(instructions) == 0 {
		instructions = append(instructions, "You are a helpful assistant.")
	}

	tools := make([]openairesponses.ToolUnionParam, 0, len(request.Tools))
	for i, tool := range request.Tools {
		if jsontext.Value(tool.InputSchema).Kind() != jsontext.KindBeginObject {
			return openairesponses.ResponseNewParams{}, c.unsupported(op, fmt.Sprintf("tool %d uses a boolean JSON Schema, which OpenAI function tools do not support", i))
		}
		var schema map[string]any
		if err := jsonv2.Unmarshal(tool.InputSchema, &schema); err != nil {
			return openairesponses.ResponseNewParams{}, c.requestError(op, "decode tool %d input schema: %v", i, err)
		}
		wireTool := openairesponses.ToolParamOfFunction(tool.Name, schema, tool.Strict)
		wireTool.OfFunction.Description = param.NewOpt(tool.Description)
		tools = append(tools, wireTool)
	}

	params := openairesponses.ResponseNewParams{
		Model:             shared.ResponsesModel(model),
		Store:             param.NewOpt(false),
		Instructions:      param.NewOpt(strings.Join(instructions, "\n\n")),
		Input:             openairesponses.ResponseNewParamsInputUnion{OfInputItemList: input},
		ParallelToolCalls: param.NewOpt(true),
		Tools:             tools,
	}
	if options.ReasoningSummary {
		params.Include = []openairesponses.ResponseIncludable{openairesponses.ResponseIncludableReasoningEncryptedContent}
		params.Reasoning = shared.ReasoningParam{Summary: shared.ReasoningSummaryAuto}
		params.Text.Verbosity = openairesponses.ResponseTextConfigVerbosityLow
	}
	if request.Temperature != nil {
		params.Temperature = param.NewOpt(*request.Temperature)
	}
	if !options.Subscription {
		if request.MaxOutputTokens != 0 {
			params.MaxOutputTokens = param.NewOpt(int64(request.MaxOutputTokens))
		}
		if request.TopP != nil {
			params.TopP = param.NewOpt(*request.TopP)
		}
	}
	return params, nil
}

func (c Codec) userContentToWire(op string, messageIndex int, parts []llm.Part) (openairesponses.ResponseInputMessageContentListParam, bool, error) {
	content := make(openairesponses.ResponseInputMessageContentListParam, 0, len(parts))
	hasImage := false
	for partIndex, part := range parts {
		switch part.Kind {
		case llm.PartText:
			content = append(content, openairesponses.ResponseInputContentParamOfInputText(part.Text))
		case llm.PartImage:
			mediaType, _, err := mime.ParseMediaType(part.MediaType)
			mediaType = strings.ToLower(mediaType)
			if err != nil {
				return nil, false, c.requestError(op, "messages[%d] content[%d] image media type %q is invalid", messageIndex, partIndex, part.MediaType)
			}
			switch mediaType {
			case "image/png", "image/jpeg", "image/webp", "image/gif":
			default:
				return nil, false, c.requestError(op, "messages[%d] content[%d] image media type %q is not supported", messageIndex, partIndex, part.MediaType)
			}
			image := openairesponses.ResponseInputContentParamOfInputImage(openairesponses.ResponseInputImageDetailAuto)
			image.OfInputImage.ImageURL = param.NewOpt("data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data))
			content = append(content, image)
			hasImage = true
		default:
			return nil, false, c.unsupported(op, fmt.Sprintf("messages[%d] content[%d] kind %d is not implemented", messageIndex, partIndex, part.Kind))
		}
	}
	return content, hasImage, nil
}

func (c Codec) providerItems(data json.RawMessage, model string) ([]openairesponses.ResponseInputItemUnionParam, bool, error) {
	if len(data) == 0 {
		return nil, false, nil
	}
	if len(data) > c.MaxProviderDataBytes {
		return nil, false, fmt.Errorf("exceeds %d bytes", c.MaxProviderDataBytes)
	}
	if jsontext.Value(data).Kind() != jsontext.KindBeginObject {
		return nil, false, nil
	}
	var header map[string]jsontext.Value
	if err := jsonv2.Unmarshal(data, &header); err != nil {
		return nil, false, fmt.Errorf("decode: %w", err)
	}
	var api string
	if rawAPI, exists := header["api"]; exists {
		if err := jsonv2.Unmarshal(rawAPI, &api); err != nil {
			return nil, false, nil
		}
	}
	if api != c.ProviderDataAPI {
		return nil, false, nil
	}

	var envelope providerDataEnvelope
	if err := jsonv2.Unmarshal(data, &envelope); err != nil {
		return nil, false, fmt.Errorf("decode: %w", err)
	}
	if envelope.Version != providerDataVersion {
		return nil, false, fmt.Errorf("unsupported version %d", envelope.Version)
	}
	if strings.TrimSpace(envelope.Model) == "" {
		return nil, false, errors.New("model must not be empty")
	}
	if envelope.Model != model {
		return nil, false, nil
	}
	if len(envelope.Output) == 0 {
		return nil, false, errors.New("must contain at least one output item")
	}
	items := make([]openairesponses.ResponseInputItemUnionParam, len(envelope.Output))
	for i, value := range envelope.Output {
		if value.Kind() != jsontext.KindBeginObject {
			return nil, false, fmt.Errorf("item %d must be an object", i)
		}
		raw := append(json.RawMessage(nil), value...)
		items[i] = param.Override[openairesponses.ResponseInputItemUnionParam](raw)
	}
	return items, true, nil
}

func (c Codec) requestError(op, format string, args ...any) *llm.Error {
	return &llm.Error{Kind: llm.KindInvalidRequest, Op: op, Provider: c.Provider, Err: fmt.Errorf(format, args...)}
}

func (c Codec) unsupported(op, message string) *llm.Error {
	return &llm.Error{Kind: llm.KindUnsupported, Op: op, Provider: c.Provider, Err: errors.New(message)}
}

func toolResultText(result llm.ToolResult) string {
	var text strings.Builder
	for _, part := range result.Content {
		text.WriteString(part.Text)
	}
	if result.IsError {
		return "Error: " + text.String()
	}
	return text.String()
}
