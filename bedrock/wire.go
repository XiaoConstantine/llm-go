package bedrock

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

const messageDataVersion = 1

type messageData struct {
	Provider  string          `json:"provider"`
	Version   int             `json:"version"`
	Model     string          `json:"model"`
	Reasoning []reasoningData `json:"reasoning"`
	Order     []orderData     `json:"order,omitempty"`
}

type orderData struct {
	Kind  string `json:"kind"`
	Index int    `json:"index"`
}

type reasoningData struct {
	Index     int    `json:"index"`
	Text      string `json:"text"`
	Signature string `json:"signature,omitempty"`
}

func buildInput(op, model string, reasoning bool, request llm.Request) (*bedrockruntime.ConverseStreamInput, error) {
	return buildInputFor(op, model, "", reasoning, request)
}

func buildInputFor(op, model, region string, reasoning bool, request llm.Request) (*bedrockruntime.ConverseStreamInput, error) {
	if request.MaxOutputTokens > math.MaxInt32 || request.ReasoningBudgetTokens > math.MaxInt32 {
		return nil, requestError(op, "token limits must not exceed %d", math.MaxInt32)
	}
	temperature, err := float32Value(op, "temperature", request.Temperature)
	if err != nil {
		return nil, err
	}
	topP, err := float32Value(op, "top-p", request.TopP)
	if err != nil {
		return nil, err
	}
	input := &bedrockruntime.ConverseStreamInput{ModelId: aws.String(model),
		InferenceConfig: &types.InferenceConfiguration{Temperature: temperature, TopP: topP, StopSequences: append([]string(nil), request.Stop...)}}
	if request.MaxOutputTokens != 0 {
		input.InferenceConfig.MaxTokens = aws.Int32(int32(request.MaxOutputTokens))
	}
	input.System, input.Messages, err = messagesToSDK(op, model, request)
	if err != nil {
		return nil, err
	}
	input.ToolConfig, err = toolsToSDK(op, request)
	if err != nil {
		return nil, err
	}
	if reasoning && request.ReasoningEffort != llm.ReasoningEffortDefault && request.ReasoningEffort != llm.ReasoningEffortNone || request.ReasoningBudgetTokens != 0 {
		fields, reasoningErr := reasoningFields(op, model, region, request)
		if reasoningErr != nil {
			return nil, reasoningErr
		}
		if fields != nil {
			input.AdditionalModelRequestFields = document.NewLazyDocument(fields)
		}
	}
	return input, nil
}

func float32Value(op, name string, value *float64) (*float32, error) {
	if value == nil {
		return nil, nil
	}
	converted := float32(*value)
	if math.IsInf(float64(converted), 0) {
		return nil, requestError(op, "%s exceeds Bedrock's numeric range", name)
	}
	return &converted, nil
}

func messagesToSDK(op, model string, request llm.Request) ([]types.SystemContentBlock, []types.Message, error) {
	var system []types.SystemContentBlock
	messages := make([]types.Message, 0, len(request.Messages))
	encoder := newMessageEncoder()
	retention := resolveCacheRetention(request.CacheRetention)
	cache := retention != llm.CacheRetentionNone && supportsPromptCaching(model)
	previousTool := false
	for messageIndex, message := range request.Messages {
		if message.Role == llm.RoleSystem {
			for _, part := range message.Content {
				if part.Kind != llm.PartText {
					return nil, nil, unsupported(op, "Bedrock system prompts support only text")
				}
				system = append(system, &types.SystemContentBlockMemberText{Value: part.Text})
			}
			continue
		}
		converted, err := messageToSDK(op, model, message, encoder)
		if err != nil {
			return nil, nil, requestError(op, "messages[%d]: %v", messageIndex, err)
		}
		if len(converted.Content) != 0 {
			if message.Role == llm.RoleTool && previousTool {
				messages[len(messages)-1].Content = append(messages[len(messages)-1].Content, converted.Content...)
			} else {
				messages = append(messages, converted)
			}
		}
		previousTool = message.Role == llm.RoleTool
	}
	if cache {
		point := cachePoint(retention)
		if len(system) != 0 {
			system = append(system, &types.SystemContentBlockMemberCachePoint{Value: point})
		}
		if len(messages) != 0 && messages[len(messages)-1].Role == types.ConversationRoleUser {
			messages[len(messages)-1].Content = append(messages[len(messages)-1].Content, &types.ContentBlockMemberCachePoint{Value: point})
		}
	}
	return system, messages, nil
}

func cachePoint(retention llm.CacheRetention) types.CachePointBlock {
	point := types.CachePointBlock{Type: types.CachePointTypeDefault}
	if retention == llm.CacheRetentionLong {
		point.Ttl = types.CacheTTLOneHour
	}
	return point
}

type messageEncoder struct {
	normalized map[string]string
	owners     map[string]string
	pending    map[string][]string
	counter    int
}

func newMessageEncoder() *messageEncoder {
	return &messageEncoder{normalized: make(map[string]string), owners: make(map[string]string), pending: make(map[string][]string)}
}

func messageToSDK(op, model string, message llm.Message, encoder *messageEncoder) (types.Message, error) {
	switch message.Role {
	case llm.RoleUser:
		content := make([]types.ContentBlock, len(message.Content))
		for index, part := range message.Content {
			block, err := contentBlock(part)
			if err != nil {
				return types.Message{}, err
			}
			content[index] = block
		}
		return types.Message{Role: types.ConversationRoleUser, Content: content}, nil
	case llm.RoleAssistant:
		data, ok := parseMessageData(message.ProviderData, model)
		if ok && len(data.Order) != 0 {
			return assistantFromOrder(model, message, data, encoder)
		}
		content := make([]types.ContentBlock, 0, len(message.Content)+len(message.ToolCalls))
		for _, reasoning := range data.Reasoning {
			if !ok || strings.TrimSpace(reasoning.Text) == "" {
				continue
			}
			if isClaude(model) && reasoning.Signature == "" {
				content = append(content, &types.ContentBlockMemberText{Value: reasoning.Text})
				continue
			}
			block := types.ReasoningTextBlock{Text: aws.String(reasoning.Text)}
			if isClaude(model) {
				block.Signature = aws.String(reasoning.Signature)
			}
			content = append(content, &types.ContentBlockMemberReasoningContent{Value: &types.ReasoningContentBlockMemberReasoningText{Value: block}})
		}
		for _, part := range message.Content {
			block, err := contentBlock(part)
			if err != nil {
				return types.Message{}, err
			}
			content = append(content, block)
		}
		for index, call := range message.ToolCalls {
			var arguments any
			if err := jsonv2.Unmarshal(call.Arguments, &arguments); err != nil {
				return types.Message{}, fmt.Errorf("tool call %q arguments: %w", call.Name, err)
			}
			rawID := call.ID
			if rawID == "" {
				encoder.counter++
				rawID = fmt.Sprintf("toolcall:%d:%d:%s:%s", encoder.counter, index, call.Name, call.Arguments)
			}
			id := encoder.normalize(rawID)
			encoder.pending[call.Name] = append(encoder.pending[call.Name], id)
			content = append(content, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				ToolUseId: aws.String(id), Name: aws.String(call.Name), Input: document.NewLazyDocument(arguments)}})
		}
		return types.Message{Role: types.ConversationRoleAssistant, Content: content}, nil
	case llm.RoleTool:
		content := make([]types.ContentBlock, len(message.ToolResults))
		for index, result := range message.ToolResults {
			id, err := encoder.resultID(result)
			if err != nil {
				return types.Message{}, err
			}
			parts := make([]types.ToolResultContentBlock, len(result.Content))
			for partIndex, part := range result.Content {
				switch part.Kind {
				case llm.PartText:
					parts[partIndex] = &types.ToolResultContentBlockMemberText{Value: part.Text}
				case llm.PartImage:
					image, err := imageBlock(part)
					if err != nil {
						return types.Message{}, err
					}
					parts[partIndex] = &types.ToolResultContentBlockMemberImage{Value: image}
				default:
					return types.Message{}, fmt.Errorf("tool result content kind %d is unsupported", part.Kind)
				}
			}
			status := types.ToolResultStatusSuccess
			if result.IsError {
				status = types.ToolResultStatusError
			}
			content[index] = &types.ContentBlockMemberToolResult{Value: types.ToolResultBlock{
				ToolUseId: aws.String(id), Content: parts, Status: status}}
		}
		return types.Message{Role: types.ConversationRoleUser, Content: content}, nil
	default:
		return types.Message{}, fmt.Errorf("role %q is unsupported", message.Role)
	}
}

func assistantFromOrder(model string, message llm.Message, data messageData, encoder *messageEncoder) (types.Message, error) {
	content := make([]types.ContentBlock, 0, len(data.Order))
	usedContent := make([]bool, len(message.Content))
	usedCalls := make([]bool, len(message.ToolCalls))
	usedReasoning := make([]bool, len(data.Reasoning))
	for orderIndex, entry := range data.Order {
		switch entry.Kind {
		case "content":
			if entry.Index < 0 || entry.Index >= len(message.Content) || usedContent[entry.Index] {
				return types.Message{}, fmt.Errorf("provider order %d has invalid content index %d", orderIndex, entry.Index)
			}
			block, err := contentBlock(message.Content[entry.Index])
			if err != nil {
				return types.Message{}, err
			}
			content = append(content, block)
			usedContent[entry.Index] = true
		case "tool_call":
			if entry.Index < 0 || entry.Index >= len(message.ToolCalls) || usedCalls[entry.Index] {
				return types.Message{}, fmt.Errorf("provider order %d has invalid tool call index %d", orderIndex, entry.Index)
			}
			call := message.ToolCalls[entry.Index]
			var arguments any
			if err := jsonv2.Unmarshal(call.Arguments, &arguments); err != nil {
				return types.Message{}, fmt.Errorf("tool call %q arguments: %w", call.Name, err)
			}
			rawID := call.ID
			if rawID == "" {
				encoder.counter++
				rawID = fmt.Sprintf("toolcall:%d:%s:%s", encoder.counter, call.Name, call.Arguments)
			}
			id := encoder.normalize(rawID)
			encoder.pending[call.Name] = append(encoder.pending[call.Name], id)
			content = append(content, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				ToolUseId: aws.String(id), Name: aws.String(call.Name), Input: document.NewLazyDocument(arguments)}})
			usedCalls[entry.Index] = true
		case "reasoning":
			if entry.Index < 0 || entry.Index >= len(data.Reasoning) || usedReasoning[entry.Index] {
				return types.Message{}, fmt.Errorf("provider order %d has invalid reasoning index %d", orderIndex, entry.Index)
			}
			reasoning := data.Reasoning[entry.Index]
			if isClaude(model) && reasoning.Signature == "" {
				content = append(content, &types.ContentBlockMemberText{Value: reasoning.Text})
				usedReasoning[entry.Index] = true
				continue
			}
			block := types.ReasoningTextBlock{Text: aws.String(reasoning.Text)}
			if isClaude(model) {
				block.Signature = aws.String(reasoning.Signature)
			}
			content = append(content, &types.ContentBlockMemberReasoningContent{Value: &types.ReasoningContentBlockMemberReasoningText{Value: block}})
			usedReasoning[entry.Index] = true
		default:
			return types.Message{}, fmt.Errorf("provider order %d has unknown kind %q", orderIndex, entry.Kind)
		}
	}
	for index, used := range usedContent {
		if !used {
			return types.Message{}, fmt.Errorf("provider order omits content index %d", index)
		}
	}
	for index, used := range usedCalls {
		if !used {
			return types.Message{}, fmt.Errorf("provider order omits tool call index %d", index)
		}
	}
	for index, used := range usedReasoning {
		if !used {
			return types.Message{}, fmt.Errorf("provider order omits reasoning index %d", index)
		}
	}
	return types.Message{Role: types.ConversationRoleAssistant, Content: content}, nil
}

func contentBlock(part llm.Part) (types.ContentBlock, error) {
	switch part.Kind {
	case llm.PartText:
		return &types.ContentBlockMemberText{Value: part.Text}, nil
	case llm.PartImage:
		image, err := imageBlock(part)
		if err != nil {
			return nil, err
		}
		return &types.ContentBlockMemberImage{Value: image}, nil
	default:
		return nil, fmt.Errorf("content kind %d is unsupported", part.Kind)
	}
}

func imageBlock(part llm.Part) (types.ImageBlock, error) {
	var format types.ImageFormat
	switch strings.ToLower(part.MediaType) {
	case "image/jpeg", "image/jpg":
		format = types.ImageFormatJpeg
	case "image/png":
		format = types.ImageFormatPng
	case "image/gif":
		format = types.ImageFormatGif
	case "image/webp":
		format = types.ImageFormatWebp
	default:
		return types.ImageBlock{}, fmt.Errorf("unsupported image media type %q", part.MediaType)
	}
	return types.ImageBlock{Format: format, Source: &types.ImageSourceMemberBytes{Value: append([]byte(nil), part.Data...)}}, nil
}

func toolsToSDK(op string, request llm.Request) (*types.ToolConfiguration, error) {
	if len(request.Tools) == 0 || request.ToolChoice.Mode == llm.ToolChoiceNone {
		return nil, nil
	}
	tools := make([]types.Tool, len(request.Tools))
	for index, tool := range request.Tools {
		if tool.RequiresStrict() {
			return nil, unsupported(op, fmt.Sprintf("tool %d requires strict schemas, which Bedrock portable mode does not guarantee", index))
		}
		var schema any
		if err := jsonv2.Unmarshal(tool.InputSchema, &schema); err != nil {
			return nil, requestError(op, "decode tools[%d] schema: %w", index, err)
		}
		tools[index] = &types.ToolMemberToolSpec{Value: types.ToolSpecification{Name: aws.String(tool.Name),
			Description: aws.String(tool.Description), InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(schema)}}}
	}
	config := &types.ToolConfiguration{Tools: tools}
	switch request.ToolChoice.Mode {
	case llm.ToolChoiceAuto:
		config.ToolChoice = &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}
	case llm.ToolChoiceRequired:
		config.ToolChoice = &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}
	case llm.ToolChoiceNamed:
		config.ToolChoice = &types.ToolChoiceMemberTool{Value: types.SpecificToolChoice{Name: aws.String(request.ToolChoice.Name)}}
	}
	return config, nil
}

func normalizeToolID(id string) string {
	var builder strings.Builder
	for _, character := range id {
		if character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || character == '_' || character == '-' {
			builder.WriteRune(character)
		} else {
			builder.WriteByte('_')
		}
		if builder.Len() == 64 {
			break
		}
	}
	return builder.String()
}

func (encoder *messageEncoder) normalize(id string) string {
	if existing := encoder.normalized[id]; existing != "" {
		return existing
	}
	base := normalizeToolID(id)
	if base != "" {
		if owner := encoder.owners[base]; owner == "" || owner == id {
			encoder.normalized[id], encoder.owners[base] = base, id
			return base
		}
	}
	for attempt := 0; ; attempt++ {
		digest := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", id, attempt)))
		candidate := "call_" + hex.EncodeToString(digest[:])[:32]
		if owner := encoder.owners[candidate]; owner == "" || owner == id {
			encoder.normalized[id], encoder.owners[candidate] = candidate, id
			return candidate
		}
	}
}

func (encoder *messageEncoder) resultID(result llm.ToolResult) (string, error) {
	if result.CallID != "" {
		id := encoder.normalize(result.CallID)
		for name, queue := range encoder.pending {
			for index, candidate := range queue {
				if candidate == id {
					encoder.pending[name] = append(queue[:index], queue[index+1:]...)
					return id, nil
				}
			}
		}
		return id, nil
	}
	queue := encoder.pending[result.Name]
	if len(queue) == 0 {
		return "", fmt.Errorf("tool result has no pending call named %q", result.Name)
	}
	id := queue[0]
	encoder.pending[result.Name] = queue[1:]
	return id, nil
}

func reasoningFields(op, model, region string, request llm.Request) (map[string]any, error) {
	if !isClaude(model) {
		return nil, nil
	}
	if adaptiveThinking(model) {
		if request.ReasoningBudgetTokens != 0 {
			return nil, requestError(op, "explicit reasoning budgets are not supported with adaptive thinking")
		}
		thinking := map[string]any{"type": "adaptive"}
		if !isGovCloud(model, region) {
			thinking["display"] = "summarized"
		}
		return map[string]any{"thinking": thinking,
			"output_config": map[string]any{"effort": reasoningEffort(model, request.ReasoningEffort)}}, nil
	}
	budget := request.ReasoningBudgetTokens
	if budget == 0 {
		switch request.ReasoningEffort {
		case llm.ReasoningEffortMinimal:
			budget = 1024
		case llm.ReasoningEffortLow:
			budget = 2048
		case llm.ReasoningEffortMedium:
			budget = 8192
		default:
			budget = 16384
		}
		if request.MaxOutputTokens != 0 {
			budget = min(budget, request.MaxOutputTokens-1024)
		}
	}
	if budget < 1024 {
		return nil, requestError(op, "reasoning budget must be at least 1024 tokens")
	}
	if request.MaxOutputTokens != 0 && budget >= request.MaxOutputTokens {
		return nil, requestError(op, "reasoning budget must be smaller than max output tokens")
	}
	thinking := map[string]any{"type": "enabled", "budget_tokens": budget}
	if !isGovCloud(model, region) {
		thinking["display"] = "summarized"
	}
	return map[string]any{"thinking": thinking,
		"anthropic_beta": []string{"interleaved-thinking-2025-05-14"}}, nil
}

func isGovCloud(model, region string) bool {
	model = strings.ToLower(model)
	return strings.HasPrefix(strings.ToLower(region), "us-gov-") || strings.HasPrefix(model, "us-gov.") || strings.HasPrefix(model, "arn:aws-us-gov:")
}

func reasoningEffort(model string, effort llm.ReasoningEffort) string {
	switch effort {
	case llm.ReasoningEffortMinimal, llm.ReasoningEffortLow:
		return "low"
	case llm.ReasoningEffortMedium:
		return "medium"
	case llm.ReasoningEffortXHigh, llm.ReasoningEffortMax:
		value := strings.NewReplacer("_", "-", ".", "-", ":", "-").Replace(strings.ToLower(model))
		if strings.Contains(value, "opus-4-7") {
			return "xhigh"
		}
		if strings.Contains(value, "opus-4-6") {
			return "max"
		}
		return "high"
	default:
		return "high"
	}
}

func isClaude(model string) bool { return strings.Contains(strings.ToLower(model), "claude") }
func adaptiveThinking(model string) bool {
	value := strings.NewReplacer("_", "-", ".", "-", ":", "-").Replace(strings.ToLower(model))
	return strings.Contains(value, "opus-4-6") || strings.Contains(value, "opus-4-7") || strings.Contains(value, "sonnet-4-6")
}
func supportsPromptCaching(model string) bool {
	value := strings.NewReplacer("_", "-", ".", "-", ":", "-").Replace(strings.ToLower(model))
	if !strings.Contains(value, "claude") && os.Getenv("AWS_BEDROCK_FORCE_CACHE") == "1" {
		return true
	}
	return strings.Contains(value, "claude") && (strings.Contains(value, "-4-") || strings.Contains(value, "claude-3-7-sonnet") || strings.Contains(value, "claude-3-5-haiku"))
}

func resolveCacheRetention(retention llm.CacheRetention) llm.CacheRetention {
	if retention == llm.CacheRetentionDefault {
		return llm.CacheRetentionNone
	}
	return retention
}

func parseMessageData(raw json.RawMessage, model string) (messageData, bool) {
	if len(raw) == 0 || jsontext.Value(raw).Kind() != jsontext.KindBeginObject {
		return messageData{}, false
	}
	var data messageData
	if jsonv2.Unmarshal(raw, &data) != nil || data.Provider != defaultProvider || data.Version != messageDataVersion || data.Model != model {
		return messageData{}, false
	}
	return data, true
}

func marshalMessageData(model string, reasoning []reasoningData, order []orderData) (json.RawMessage, error) {
	if len(reasoning) == 0 {
		if len(order) == 0 {
			return nil, nil
		}
	}
	return jsonv2.Marshal(messageData{Provider: defaultProvider, Version: messageDataVersion, Model: model, Reasoning: reasoning, Order: order})
}
