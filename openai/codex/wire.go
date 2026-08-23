package codex

import (
	"encoding/base64"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/openai/openai-go/v3/packages/param"
	openairesponses "github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

const (
	accountClaim        = "https://api.openai.com/auth"
	providerDataAPI     = "openai-codex-responses"
	providerDataVersion = 1
)

type providerDataEnvelope struct {
	API     string           `json:"api"`
	Version int              `json:"version"`
	Model   string           `json:"model"`
	Output  []jsontext.Value `json:"output"`
}

// AccountIDFromToken extracts the ChatGPT account identifier from an OpenAI ID
// or access JWT without verifying its signature. The server verifies the token
// when it is used; this helper only reads the routing claim.
func AccountIDFromToken(token string) (string, error) {
	parts := strings.Split(strings.TrimSpace(token), ".")
	if len(parts) != 3 {
		return "", errors.New("access token is not a JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("decode access token: %w", err)
	}
	var claims map[string]jsontext.Value
	if err := jsonv2.Unmarshal(payload, &claims); err != nil {
		return "", fmt.Errorf("decode access token claims: %w", err)
	}
	rawAuth, exists := claims[accountClaim]
	if !exists {
		return "", errors.New("access token is missing OpenAI auth claims")
	}
	var auth struct {
		AccountID string `json:"chatgpt_account_id"`
	}
	if err := jsonv2.Unmarshal(rawAuth, &auth); err != nil {
		return "", fmt.Errorf("decode OpenAI auth claims: %w", err)
	}
	accountID := strings.TrimSpace(auth.AccountID)
	if accountID == "" {
		return "", errors.New("access token is missing chatgpt_account_id")
	}
	return accountID, nil
}

func requestToWire(op, model string, request llm.Request) (openairesponses.ResponseNewParams, error) {
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
			input = append(input, openairesponses.ResponseInputItemParamOfMessage(text, openairesponses.EasyInputMessageRoleUser))
		case llm.RoleAssistant:
			items, replayed, err := providerItems(message.ProviderData, model)
			if err != nil {
				return openairesponses.ResponseNewParams{}, requestError(op, "messages[%d] provider data: %v", i, err)
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
			return openairesponses.ResponseNewParams{}, unsupported(op, fmt.Sprintf("tool %d uses a boolean JSON Schema, which Codex function tools do not support", i))
		}
		var schema map[string]any
		if err := jsonv2.Unmarshal(tool.InputSchema, &schema); err != nil {
			return openairesponses.ResponseNewParams{}, requestError(op, "decode tool %d input schema: %v", i, err)
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
		Include:           []openairesponses.ResponseIncludable{openairesponses.ResponseIncludableReasoningEncryptedContent},
		ParallelToolCalls: param.NewOpt(true),
		Text: openairesponses.ResponseTextConfigParam{
			Verbosity: openairesponses.ResponseTextConfigVerbosityLow,
		},
		Tools: tools,
	}
	if request.Temperature != nil {
		params.Temperature = param.NewOpt(*request.Temperature)
	}
	return params, nil
}

func providerItems(data json.RawMessage, model string) ([]openairesponses.ResponseInputItemUnionParam, bool, error) {
	if len(data) == 0 {
		return nil, false, nil
	}
	if len(data) > maxProviderDataBytes {
		return nil, false, fmt.Errorf("exceeds %d bytes", maxProviderDataBytes)
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
	if api != providerDataAPI {
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
