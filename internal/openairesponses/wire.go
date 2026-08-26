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
	Subscription        bool
	ReasoningSummary    bool
	EncryptedReasoning  bool
	JSONObjectOutput    bool
	NamedToolChoice     bool
	StrictTools         bool
	ExplicitPromptCache bool
	InputAudio          *InputAudioOptions
}

// InputAudioOptions enables the private Responses input_audio content variant
// used by endpoints that explicitly support it. Public Responses callers leave
// this nil.
type InputAudioOptions struct {
	MaxDecodedBytes int
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
			content, hasStructuredContent, err := c.userContentToWire(op, i, message.Content, options.InputAudio)
			if err != nil {
				return openairesponses.ResponseNewParams{}, err
			}
			if hasStructuredContent {
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
			for resultIndex, result := range message.ToolResults {
				items, hasImage, err := c.toolResultToWire(op, i, resultIndex, result)
				if err != nil {
					return openairesponses.ResponseNewParams{}, err
				}
				if hasImage {
					input = append(input, openairesponses.ResponseInputItemParamOfFunctionCallOutput(result.CallID, items))
				} else {
					input = append(input, openairesponses.ResponseInputItemParamOfFunctionCallOutput(result.CallID, toolResultText(result)))
				}
			}
		}
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
		wireTool := openairesponses.ToolParamOfFunction(tool.Name, schema, tool.StrictEnabled(options.StrictTools))
		wireTool.OfFunction.Description = param.NewOpt(tool.Description)
		tools = append(tools, wireTool)
	}

	params := openairesponses.ResponseNewParams{
		Model:             shared.ResponsesModel(model),
		Store:             param.NewOpt(false),
		Input:             openairesponses.ResponseNewParamsInputUnion{OfInputItemList: input},
		ParallelToolCalls: param.NewOpt(true),
		Tools:             tools,
	}
	if len(instructions) != 0 {
		params.Instructions = param.NewOpt(strings.Join(instructions, "\n\n"))
	}
	if options.EncryptedReasoning && request.ReasoningEffort != llm.ReasoningEffortNone {
		params.Include = []openairesponses.ResponseIncludable{openairesponses.ResponseIncludableReasoningEncryptedContent}
	}
	if options.ReasoningSummary {
		if request.ReasoningEffort != llm.ReasoningEffortNone {
			params.Reasoning.Summary = shared.ReasoningSummaryAuto
		}
		params.Text.Verbosity = openairesponses.ResponseTextConfigVerbosityLow
	}
	if request.ResponseFormat == llm.ResponseFormatJSON {
		if !options.JSONObjectOutput {
			return openairesponses.ResponseNewParams{}, c.unsupported(op, "JSON response format is not supported by this Responses endpoint")
		}
		format := shared.NewResponseFormatJSONObjectParam()
		params.Text.Format.OfJSONObject = &format
	}
	if err := c.applyToolChoice(op, &params, request.ToolChoice, options.NamedToolChoice); err != nil {
		return openairesponses.ResponseNewParams{}, err
	}
	if request.CacheRetention == llm.CacheRetentionNone && options.ExplicitPromptCache {
		params.PromptCacheOptions.Mode = "explicit"
	}
	if request.CacheRetention != llm.CacheRetentionNone {
		key := request.CacheKey
		if key == "" {
			key = request.SessionID
		}
		if key != "" {
			params.PromptCacheKey = param.NewOpt(key)
		}
	}
	switch request.CacheRetention {
	case llm.CacheRetentionLong:
		params.PromptCacheRetention = openairesponses.ResponseNewParamsPromptCacheRetention24h
	case llm.CacheRetentionShort:
		params.PromptCacheRetention = openairesponses.ResponseNewParamsPromptCacheRetentionInMemory
	}
	if request.ReasoningEffort != llm.ReasoningEffortDefault {
		effort := request.ReasoningEffort
		if options.Subscription && effort == llm.ReasoningEffortMinimal {
			effort = llm.ReasoningEffortLow
		}
		params.Reasoning.Effort = shared.ReasoningEffort(effort)
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

func (c Codec) userContentToWire(op string, messageIndex int, parts []llm.Part, audioOptions *InputAudioOptions) (openairesponses.ResponseInputMessageContentListParam, bool, error) {
	content := make(openairesponses.ResponseInputMessageContentListParam, 0, len(parts))
	hasStructuredContent := false
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
			hasStructuredContent = true
		case llm.PartAudio:
			if audioOptions == nil {
				return nil, false, c.unsupported(op, fmt.Sprintf("messages[%d] content[%d] audio is not supported", messageIndex, partIndex))
			}
			mediaType, err := canonicalAudioMediaType(part.MediaType)
			if err != nil {
				return nil, false, c.requestError(op, "messages[%d] content[%d] audio media type %q is invalid: %v", messageIndex, partIndex, part.MediaType, err)
			}
			if mediaType == "" {
				return nil, false, c.requestError(op, "messages[%d] content[%d] audio media type %q is not supported", messageIndex, partIndex, part.MediaType)
			}
			if audioOptions.MaxDecodedBytes > 0 && len(part.Data) > audioOptions.MaxDecodedBytes {
				return nil, false, c.requestError(op, "messages[%d] content[%d] audio input exceeds %d decoded bytes", messageIndex, partIndex, audioOptions.MaxDecodedBytes)
			}
			raw, err := json.Marshal(struct {
				Type     string `json:"type"`
				AudioURL string `json:"audio_url"`
			}{
				Type:     "input_audio",
				AudioURL: "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data),
			})
			if err != nil {
				return nil, false, c.requestError(op, "messages[%d] content[%d] encode audio: %v", messageIndex, partIndex, err)
			}
			content = append(content, param.Override[openairesponses.ResponseInputContentUnionParam](json.RawMessage(raw)))
			hasStructuredContent = true
		default:
			return nil, false, c.unsupported(op, fmt.Sprintf("messages[%d] content[%d] kind %d is not implemented", messageIndex, partIndex, part.Kind))
		}
	}
	return content, hasStructuredContent, nil
}

func canonicalAudioMediaType(value string) (string, error) {
	mediaType, _, err := mime.ParseMediaType(value)
	if err != nil {
		return "", err
	}
	switch strings.ToLower(mediaType) {
	case "audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave":
		return "audio/wav", nil
	case "audio/mpeg", "audio/mp3":
		return "audio/mpeg", nil
	case "audio/mp4", "audio/m4a", "audio/x-m4a":
		return "audio/mp4", nil
	case "audio/webm":
		return "audio/webm", nil
	case "audio/ogg":
		return "audio/ogg", nil
	default:
		return "", nil
	}
}

func (c Codec) applyToolChoice(op string, params *openairesponses.ResponseNewParams, choice llm.ToolChoice, named bool) error {
	switch choice.Mode {
	case llm.ToolChoiceAuto:
		return nil
	case llm.ToolChoiceNone, llm.ToolChoiceRequired:
		params.ToolChoice.OfToolChoiceMode = param.NewOpt(openairesponses.ToolChoiceOptions(choice.Mode))
		return nil
	case llm.ToolChoiceNamed:
		if !named {
			return c.unsupported(op, "named tool choice is not supported by this Responses endpoint")
		}
		params.ToolChoice.OfFunctionTool = &openairesponses.ToolChoiceFunctionParam{Name: choice.Name}
		return nil
	default:
		return c.requestError(op, "invalid tool choice %q", choice.Mode)
	}
}

func (c Codec) toolResultToWire(op string, messageIndex, resultIndex int, result llm.ToolResult) (openairesponses.ResponseFunctionCallOutputItemListParam, bool, error) {
	items := make(openairesponses.ResponseFunctionCallOutputItemListParam, 0, len(result.Content)+1)
	if result.IsError {
		items = append(items, openairesponses.ResponseFunctionCallOutputItemParamOfInputText("Error: "))
	}
	hasImage := false
	for partIndex, part := range result.Content {
		switch part.Kind {
		case llm.PartText:
			items = append(items, openairesponses.ResponseFunctionCallOutputItemParamOfInputText(part.Text))
		case llm.PartImage:
			mediaType, _, err := mime.ParseMediaType(part.MediaType)
			mediaType = strings.ToLower(mediaType)
			if err != nil {
				return nil, false, c.requestError(op, "messages[%d] tool result[%d] content[%d] image media type %q is invalid", messageIndex, resultIndex, partIndex, part.MediaType)
			}
			switch mediaType {
			case "image/png", "image/jpeg", "image/webp", "image/gif":
			default:
				return nil, false, c.requestError(op, "messages[%d] tool result[%d] content[%d] image media type %q is not supported", messageIndex, resultIndex, partIndex, part.MediaType)
			}
			image := openairesponses.ResponseInputImageContentParam{ImageURL: param.NewOpt("data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data))}
			items = append(items, openairesponses.ResponseFunctionCallOutputItemUnionParam{OfInputImage: &image})
			hasImage = true
		default:
			return nil, false, c.unsupported(op, fmt.Sprintf("messages[%d] tool result[%d] content[%d] kind %d is not supported", messageIndex, resultIndex, partIndex, part.Kind))
		}
	}
	return items, hasImage, nil
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
