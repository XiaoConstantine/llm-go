package anthropic

import (
	"encoding/base64"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"fmt"
	"mime"
	"slices"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
)

const maxMessages = 100_000

type messageRequest struct {
	Model         string           `json:"model"`
	MaxTokens     int              `json:"max_tokens"`
	Messages      []inputMessage   `json:"messages"`
	System        []contentBlock   `json:"system,omitempty"`
	Temperature   *float64         `json:"temperature,omitempty"`
	TopP          *float64         `json:"top_p,omitempty"`
	StopSequences []string         `json:"stop_sequences,omitempty"`
	Tools         []toolDefinition `json:"tools,omitempty"`
	ToolChoice    *toolChoice      `json:"tool_choice,omitempty"`
	Thinking      *thinkingConfig  `json:"thinking,omitempty"`
	OutputConfig  *outputConfig    `json:"output_config,omitempty"`
	Stream        bool             `json:"stream,omitzero"`
}

type inputMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type cacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type contentBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *cacheControl `json:"cache_control,omitempty"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type thinkingConfig struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitzero"`
	Display      string `json:"display,omitempty"`
}

type outputConfig struct {
	Effort llm.ReasoningEffort `json:"effort,omitempty"`
}

type toolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type inputContentBlock struct {
	Type         string          `json:"type"`
	Text         *string         `json:"text,omitzero"`
	Source       *imageSource    `json:"source,omitempty"`
	ID           string          `json:"id,omitempty"`
	Name         string          `json:"name,omitempty"`
	Input        json.RawMessage `json:"input,omitzero"`
	ToolUseID    string          `json:"tool_use_id,omitempty"`
	Content      any             `json:"content,omitempty"`
	IsError      bool            `json:"is_error,omitzero"`
	Thinking     *string         `json:"thinking,omitzero"`
	Signature    *string         `json:"signature,omitzero"`
	Data         string          `json:"data,omitempty"`
	CacheControl *cacheControl   `json:"cache_control,omitempty"`
}

type toolDefinition struct {
	Name                string          `json:"name"`
	Description         string          `json:"description,omitempty"`
	InputSchema         json.RawMessage `json:"input_schema"`
	Strict              bool            `json:"strict,omitzero"`
	EagerInputStreaming bool            `json:"eager_input_streaming,omitzero"`
	CacheControl        *cacheControl   `json:"cache_control,omitempty"`
}

type messageResponse struct {
	ID         string           `json:"id"`
	Type       string           `json:"type"`
	Role       string           `json:"role"`
	Content    *[]responseBlock `json:"content"`
	Model      string           `json:"model"`
	StopReason *string          `json:"stop_reason"`
	Usage      *responseUsage   `json:"usage"`
}

type responseBlock struct {
	Type      string           `json:"type"`
	Text      *string          `json:"text"`
	ID        *string          `json:"id"`
	Name      *string          `json:"name"`
	Input     *json.RawMessage `json:"input"`
	Thinking  *string          `json:"thinking"`
	Signature *string          `json:"signature"`
	Data      *string          `json:"data"`
}

type responseUsage struct {
	InputTokens              *int                    `json:"input_tokens"`
	OutputTokens             *int                    `json:"output_tokens"`
	CacheCreationInputTokens int                     `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int                     `json:"cache_read_input_tokens"`
	CacheCreation            *cacheCreationBreakdown `json:"cache_creation"`
	OutputTokensDetails      *outputTokenDetails     `json:"output_tokens_details"`
}

type cacheCreationBreakdown struct {
	Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
}

type outputTokenDetails struct {
	ThinkingTokens *int `json:"thinking_tokens"`
}

type errorEnvelope struct {
	Error     *APIError `json:"error"`
	RequestID string    `json:"request_id"`
}

const anthropicMessageDataVersion = 1

type anthropicMessageData struct {
	Provider string              `json:"provider"`
	Version  int                 `json:"version"`
	Model    string              `json:"model"`
	Thinking []anthropicThinking `json:"thinking"`
}

type anthropicThinking struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

func checkRequest(op string, request llm.Request) error {
	if request.PresencePenalty != nil || request.FrequencyPenalty != nil {
		return unsupported(op, "presence and frequency penalties are not supported")
	}
	if request.Temperature != nil && *request.Temperature > 1 {
		return requestError(op, "temperature must not exceed 1")
	}

	declared := make(map[string]struct{}, len(request.Tools))
	for index, tool := range request.Tools {
		if !validToolName(tool.Name) {
			return requestError(op, "tools[%d].name %q must contain 1-64 letters, digits, underscores, or dashes", index, tool.Name)
		}
		if jsontext.Value(tool.InputSchema).Kind() != jsontext.KindBeginObject {
			return requestError(op, "tools[%d].input schema must be a JSON object", index)
		}
		var schema struct {
			Type string `json:"type"`
		}
		if err := jsonv2.Unmarshal(tool.InputSchema, &schema); err != nil || schema.Type != "object" {
			return requestError(op, "tools[%d].input schema must declare top-level type %q", index, "object")
		}
		declared[tool.Name] = struct{}{}
	}

	conversationMessages := 0
	seenCallIDs := make(map[string]struct{})
	for index, message := range request.Messages {
		if message.Role == llm.RoleSystem {
			if conversationMessages != 0 {
				return requestError(op, "messages[%d]: system messages must precede conversation messages", index)
			}
		} else {
			conversationMessages++
			if conversationMessages > maxMessages {
				return requestError(op, "messages contains %d conversation entries, maximum is %d", conversationMessages, maxMessages)
			}
		}
		for callIndex, call := range message.ToolCalls {
			if !validToolName(call.Name) {
				return requestError(op, "messages[%d].tool calls[%d].name %q must contain 1-64 letters, digits, underscores, or dashes", index, callIndex, call.Name)
			}
			if jsontext.Value(call.Arguments).Kind() != jsontext.KindBeginObject {
				return requestError(op, "messages[%d].tool calls[%d].arguments must be a JSON object", index, callIndex)
			}
			if _, exists := declared[call.Name]; !exists {
				return requestError(op, "messages[%d].tool calls[%d] names undeclared tool %q", index, callIndex, call.Name)
			}
			if call.ID != "" {
				if _, exists := seenCallIDs[call.ID]; exists {
					return requestError(op, "messages[%d].tool calls[%d] repeats ID %q", index, callIndex, call.ID)
				}
				seenCallIDs[call.ID] = struct{}{}
			}
		}
	}
	if conversationMessages == 0 {
		return requestError(op, "messages must contain a user or assistant message")
	}
	return checkToolHistory(op, request.Messages)
}

func requestToWire(op, model string, defaultMaxOutputTokens int, request llm.Request) (messageRequest, map[string]struct{}, error) {
	return requestToWireWithCompatibility(op, model, defaultMaxOutputTokens, request, llm.AnthropicCompatibility{})
}

func requestToWireWithCompatibility(op, model string, defaultMaxOutputTokens int, request llm.Request, compatibility llm.AnthropicCompatibility) (messageRequest, map[string]struct{}, error) {
	return requestToWireWithIdentity(op, model, defaultMaxOutputTokens, request, compatibility, false)
}

func requestToWireWithIdentity(op, model string, defaultMaxOutputTokens int, request llm.Request, compatibility llm.AnthropicCompatibility, subscriptionOAuth bool) (messageRequest, map[string]struct{}, error) {
	maxTokens := request.MaxOutputTokens
	if maxTokens == 0 {
		maxTokens = defaultMaxOutputTokens
	}

	wrequest := messageRequest{
		Model:         model,
		MaxTokens:     maxTokens,
		Temperature:   request.Temperature,
		TopP:          request.TopP,
		StopSequences: append([]string(nil), request.Stop...),
	}
	cache := anthropicCacheControl(request.CacheRetention)
	if subscriptionOAuth {
		wrequest.System = append(wrequest.System, contentBlock{
			Type: "text", Text: "You are Claude Code, Anthropic's official CLI for Claude.", CacheControl: cache,
		})
	}
	wrequest.Tools = make([]toolDefinition, len(request.Tools))
	for index, tool := range request.Tools {
		wrequest.Tools[index] = toolDefinition{
			Name:                tool.Name,
			Description:         tool.Description,
			InputSchema:         append(json.RawMessage(nil), tool.InputSchema...),
			Strict:              tool.Strict,
			EagerInputStreaming: compatibility.EagerToolInputStreaming != llm.CompatibilityDisabled,
		}
		if cache != nil && index == len(request.Tools)-1 && compatibility.CacheControlOnTools != llm.CompatibilityDisabled {
			wrequest.Tools[index].CacheControl = cache
		}
	}
	applyAnthropicToolChoice(&wrequest, request.ToolChoice)
	if err := applyAnthropicThinking(op, &wrequest, request, compatibility); err != nil {
		return messageRequest{}, nil, err
	}
	encoder := newMessageEncoder(op, request.Messages, model, compatibility.EmptyThinkingSignature == llm.CompatibilityEnabled)
	for _, message := range request.Messages {
		if message.Role == llm.RoleSystem {
			for _, part := range message.Content {
				wrequest.System = append(wrequest.System, contentBlock{Type: "text", Text: part.Text})
			}
			continue
		}
		converted, err := encoder.toWire(message)
		if err != nil {
			return messageRequest{}, nil, err
		}
		wrequest.Messages = append(wrequest.Messages, converted)
	}
	if cache != nil {
		if len(wrequest.System) != 0 {
			wrequest.System[len(wrequest.System)-1].CacheControl = cache
		}
		for index := len(wrequest.Messages) - 1; index >= 0; index-- {
			if wrequest.Messages[index].Role != string(llm.RoleUser) {
				continue
			}
			blocks, ok := wrequest.Messages[index].Content.([]inputContentBlock)
			if !ok {
				text, _ := wrequest.Messages[index].Content.(string)
				blocks = []inputContentBlock{{Type: "text", Text: &text}}
			}
			if len(blocks) != 0 {
				blocks[len(blocks)-1].CacheControl = cache
				wrequest.Messages[index].Content = blocks
			}
			break
		}
	}
	return wrequest, encoder.usedIDs, nil
}

func anthropicCacheControl(retention llm.CacheRetention) *cacheControl {
	switch retention {
	case llm.CacheRetentionShort:
		return &cacheControl{Type: "ephemeral"}
	case llm.CacheRetentionLong:
		return &cacheControl{Type: "ephemeral", TTL: "1h"}
	default:
		return nil
	}
}

func applyAnthropicToolChoice(request *messageRequest, choice llm.ToolChoice) {
	switch choice.Mode {
	case llm.ToolChoiceNone:
		request.ToolChoice = &toolChoice{Type: "none"}
	case llm.ToolChoiceRequired:
		request.ToolChoice = &toolChoice{Type: "any"}
	case llm.ToolChoiceNamed:
		request.ToolChoice = &toolChoice{Type: "tool", Name: choice.Name}
	}
}

func applyAnthropicThinking(op string, wire *messageRequest, request llm.Request, compatibility llm.AnthropicCompatibility) error {
	if request.ReasoningEffort == llm.ReasoningEffortDefault && request.ReasoningBudgetTokens == 0 {
		return nil
	}
	if request.ReasoningEffort == llm.ReasoningEffortNone {
		wire.Thinking = &thinkingConfig{Type: "disabled"}
		return nil
	}
	if compatibility.AdaptiveThinking == llm.CompatibilityEnabled {
		if request.ReasoningBudgetTokens != 0 {
			return requestError(op, "explicit reasoning budgets are not supported with adaptive thinking")
		}
		wire.Thinking = &thinkingConfig{Type: "adaptive", Display: "summarized"}
		effort := request.ReasoningEffort
		if effort == llm.ReasoningEffortMinimal {
			effort = llm.ReasoningEffortLow
		}
		if effort != llm.ReasoningEffortDefault {
			wire.OutputConfig = &outputConfig{Effort: effort}
		}
		return nil
	}
	budget := request.ReasoningBudgetTokens
	explicitBudget := budget != 0
	if !explicitBudget {
		switch request.ReasoningEffort {
		case llm.ReasoningEffortMinimal:
			budget = 1024
		case llm.ReasoningEffortLow:
			budget = 2048
		case llm.ReasoningEffortMedium:
			budget = 8192
		case llm.ReasoningEffortHigh, llm.ReasoningEffortXHigh, llm.ReasoningEffortMax:
			budget = 16384
		default:
			budget = 1024
		}
		// Anthropic's max_tokens ceiling includes thinking. Preserve at least
		// 1024 answer tokens while fitting legacy effort budgets under the cap.
		if available := wire.MaxTokens - 1024; budget > available {
			budget = available
		}
	}
	if budget < 1024 {
		return requestError(op, "reasoning budget must be at least 1024 tokens")
	}
	if budget >= wire.MaxTokens {
		return requestError(op, "reasoning budget must be less than max output tokens")
	}
	wire.Thinking = &thinkingConfig{Type: "enabled", BudgetTokens: budget}
	return nil
}

func validToolName(name string) bool {
	if len(name) == 0 || len(name) > 64 {
		return false
	}
	for index := 0; index < len(name); index++ {
		character := name[index]
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			character == '_' || character == '-' {
			continue
		}
		return false
	}
	return true
}

func checkToolHistory(op string, messages []llm.Message) error {
	pending := 0
	for index, message := range messages {
		if pending != 0 && message.Role != llm.RoleTool {
			return requestError(op, "messages[%d] must contain tool results for %d pending calls", index, pending)
		}
		switch message.Role {
		case llm.RoleAssistant:
			pending = len(message.ToolCalls)
		case llm.RoleTool:
			pending -= len(message.ToolResults)
		}
	}
	if pending != 0 {
		return requestError(op, "messages end with %d pending tool calls", pending)
	}
	return nil
}

type messageEncoder struct {
	op                          string
	model                       string
	allowEmptyThinkingSignature bool
	usedIDs                     map[string]struct{}
	pendingByName               map[string][]string
	nextID                      int
}

func newMessageEncoder(op string, messages []llm.Message, model string, allowEmptyThinkingSignature bool) *messageEncoder {
	usedIDs := make(map[string]struct{})
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			if call.ID != "" {
				usedIDs[call.ID] = struct{}{}
			}
		}
	}
	return &messageEncoder{
		op:                          op,
		model:                       model,
		allowEmptyThinkingSignature: allowEmptyThinkingSignature,
		usedIDs:                     usedIDs,
		pendingByName:               make(map[string][]string),
	}
}

func (encoder *messageEncoder) toWire(message llm.Message) (inputMessage, error) {
	if message.Role == llm.RoleTool {
		blocks := make([]inputContentBlock, 0, len(message.ToolResults))
		for index, result := range message.ToolResults {
			callID, err := encoder.resultID(result)
			if err != nil {
				return inputMessage{}, err
			}
			content, err := anthropicParts(encoder.op, result.Content)
			if err != nil {
				return inputMessage{}, fmt.Errorf("tool result %d: %w", index, err)
			}
			var value any
			if allTextParts(result.Content) {
				value = optionalTextContent(result.Content)
			} else if len(content) != 0 {
				value = content
			}
			blocks = append(blocks, inputContentBlock{Type: "tool_result", ToolUseID: callID, Content: value, IsError: result.IsError})
		}
		return inputMessage{Role: string(llm.RoleUser), Content: blocks}, nil
	}

	blocks, err := anthropicParts(encoder.op, message.Content)
	if err != nil {
		return inputMessage{}, err
	}
	replayedThinking := false
	if message.Role == llm.RoleAssistant {
		replay, recognized, err := parseAnthropicMessageData(message.ProviderData, encoder.model)
		if err != nil {
			return inputMessage{}, requestError(encoder.op, "assistant provider data: %v", err)
		}
		if recognized {
			replayed, err := thinkingBlocksToWire(replay.Thinking, encoder.allowEmptyThinkingSignature)
			if err != nil {
				return inputMessage{}, requestError(encoder.op, "assistant provider data: %v", err)
			}
			replayedThinking = true
			blocks = append(replayed, blocks...)
		}
	}
	if len(message.ToolCalls) == 0 {
		if allTextParts(message.Content) && !replayedThinking {
			return inputMessage{Role: string(message.Role), Content: message.Text()}, nil
		}
		if len(blocks) == 1 && blocks[0].Type == "text" && blocks[0].Text != nil {
			return inputMessage{Role: string(message.Role), Content: *blocks[0].Text}, nil
		}
		return inputMessage{Role: string(message.Role), Content: blocks}, nil
	}
	blocks = slices.Grow(blocks, len(message.ToolCalls))
	for _, call := range message.ToolCalls {
		callID := call.ID
		if callID == "" {
			callID = encoder.newID()
			encoder.pendingByName[call.Name] = append(encoder.pendingByName[call.Name], callID)
		}
		blocks = append(blocks, inputContentBlock{
			Type:  "tool_use",
			ID:    callID,
			Name:  call.Name,
			Input: append(json.RawMessage(nil), call.Arguments...),
		})
	}
	return inputMessage{Role: string(llm.RoleAssistant), Content: blocks}, nil
}

func (encoder *messageEncoder) newID() string {
	for {
		encoder.nextID++
		id := fmt.Sprintf("toolu_llm_go_%d", encoder.nextID)
		if _, exists := encoder.usedIDs[id]; exists {
			continue
		}
		encoder.usedIDs[id] = struct{}{}
		return id
	}
}

func (encoder *messageEncoder) resultID(result llm.ToolResult) (string, error) {
	if result.CallID != "" {
		return result.CallID, nil
	}
	queue := encoder.pendingByName[result.Name]
	if len(queue) == 0 {
		return "", requestError(encoder.op, "tool result %q has no pending tool call", result.Name)
	}
	callID := queue[0]
	if len(queue) == 1 {
		delete(encoder.pendingByName, result.Name)
	} else {
		encoder.pendingByName[result.Name] = queue[1:]
	}
	return callID, nil
}

func anthropicParts(op string, parts []llm.Part) ([]inputContentBlock, error) {
	blocks := make([]inputContentBlock, 0, len(parts))
	for index, part := range parts {
		switch part.Kind {
		case llm.PartText:
			text := part.Text
			blocks = append(blocks, inputContentBlock{Type: "text", Text: &text})
		case llm.PartImage:
			mediaType, _, err := mime.ParseMediaType(part.MediaType)
			mediaType = strings.ToLower(mediaType)
			if err != nil {
				return nil, requestError(op, "content[%d] image media type %q is invalid", index, part.MediaType)
			}
			switch mediaType {
			case "image/jpeg", "image/png", "image/gif", "image/webp":
			default:
				return nil, requestError(op, "content[%d] image media type %q is not supported", index, part.MediaType)
			}
			blocks = append(blocks, inputContentBlock{Type: "image", Source: &imageSource{Type: "base64", MediaType: mediaType, Data: base64.StdEncoding.EncodeToString(part.Data)}})
		default:
			return nil, unsupported(op, fmt.Sprintf("content[%d] kind %d is not supported", index, part.Kind))
		}
	}
	return blocks, nil
}

func parseAnthropicMessageData(raw json.RawMessage, model string) (anthropicMessageData, bool, error) {
	if len(raw) == 0 || jsontext.Value(raw).Kind() != jsontext.KindBeginObject {
		return anthropicMessageData{}, false, nil
	}
	var data anthropicMessageData
	if err := jsonv2.Unmarshal(raw, &data); err != nil {
		return anthropicMessageData{}, false, err
	}
	if data.Provider != "anthropic" {
		return anthropicMessageData{}, false, nil
	}
	if data.Version != anthropicMessageDataVersion {
		return anthropicMessageData{}, false, fmt.Errorf("unsupported version %d", data.Version)
	}
	if data.Model != model {
		return anthropicMessageData{}, false, nil
	}
	return data, true, nil
}

func thinkingBlocksToWire(thinking []anthropicThinking, allowEmptySignature bool) ([]inputContentBlock, error) {
	blocks := make([]inputContentBlock, 0, len(thinking))
	for index, item := range thinking {
		switch item.Type {
		case "redacted_thinking":
			if item.Data == "" {
				return nil, fmt.Errorf("redacted thinking block %d has no data", index)
			}
			blocks = append(blocks, inputContentBlock{Type: item.Type, Data: item.Data})
		case "thinking":
			if item.Signature == "" && !allowEmptySignature {
				return nil, fmt.Errorf("thinking block %d has no signature", index)
			}
			thinking := item.Thinking
			signature := item.Signature
			blocks = append(blocks, inputContentBlock{Type: item.Type, Thinking: &thinking, Signature: &signature})
		default:
			return nil, fmt.Errorf("thinking block %d has unsupported type %q", index, item.Type)
		}
	}
	return blocks, nil
}

func marshalAnthropicMessageData(model string, thinking []anthropicThinking) (json.RawMessage, error) {
	if len(thinking) == 0 {
		return nil, nil
	}
	return jsonv2.Marshal(anthropicMessageData{Provider: "anthropic", Version: anthropicMessageDataVersion, Model: model, Thinking: thinking})
}

func allTextParts(parts []llm.Part) bool {
	for _, part := range parts {
		if part.Kind != llm.PartText {
			return false
		}
	}
	return true
}

func optionalTextContent(parts []llm.Part) *string {
	if len(parts) == 0 {
		return nil
	}
	var builder strings.Builder
	for _, part := range parts {
		builder.WriteString(part.Text)
	}
	text := builder.String()
	return &text
}

func responseFromWire(configuredModel string, request llm.Request, priorToolIDs map[string]struct{}, response messageResponse, allowEmptyThinkingSignature ...bool) (*llm.Response, error) {
	if response.ID == "" {
		return nil, malformedResponse("response has no ID")
	}
	if response.Type != "message" {
		return nil, malformedResponse("response has type %q, want message", response.Type)
	}
	if response.Role != string(llm.RoleAssistant) {
		return nil, malformedResponse("response has role %q, want assistant", response.Role)
	}
	if response.Model == "" {
		return nil, malformedResponse("response has no model")
	}
	if response.StopReason == nil {
		return nil, malformedResponse("response has no stop reason")
	}
	finishReason, err := finishReason("generate", *response.StopReason)
	if err != nil {
		return nil, err
	}
	if response.Content == nil {
		return nil, malformedResponse("response has no content")
	}

	declared := make(map[string]struct{}, len(request.Tools))
	for _, tool := range request.Tools {
		declared[tool.Name] = struct{}{}
	}
	content := make([]llm.Part, 0, len(*response.Content))
	toolCalls := make([]llm.ToolCall, 0)
	thinking := make([]anthropicThinking, 0)
	var reasoning strings.Builder
	seenIDs := make(map[string]struct{}, len(priorToolIDs))
	for id := range priorToolIDs {
		seenIDs[id] = struct{}{}
	}
	for index, block := range *response.Content {
		switch block.Type {
		case "text":
			if block.Text == nil {
				return nil, malformedResponse("response content %d has no text", index)
			}
			content = append(content, llm.Part{Kind: llm.PartText, Text: *block.Text})
		case "tool_use":
			call, err := toolCallFromWire(index, block, declared, seenIDs)
			if err != nil {
				return nil, err
			}
			toolCalls = append(toolCalls, call)
		case "thinking":
			if block.Thinking == nil || block.Signature == nil {
				return nil, malformedResponse("response thinking block %d is missing thinking or signature", index)
			}
			allowEmpty := len(allowEmptyThinkingSignature) != 0 && allowEmptyThinkingSignature[0]
			if *block.Signature == "" && !allowEmpty {
				return nil, malformedResponse("response thinking block %d has an empty signature", index)
			}
			thinking = append(thinking, anthropicThinking{Type: "thinking", Thinking: *block.Thinking, Signature: *block.Signature})
			reasoning.WriteString(*block.Thinking)
		case "redacted_thinking":
			if block.Data == nil || *block.Data == "" {
				return nil, malformedResponse("response redacted thinking block %d is missing data", index)
			}
			thinking = append(thinking, anthropicThinking{Type: "redacted_thinking", Data: *block.Data})
			if reasoning.Len() != 0 {
				reasoning.WriteString("\n\n")
			}
			reasoning.WriteString("[Reasoning redacted]")
		default:
			return nil, malformedResponse("response content %d has unsupported type %q", index, block.Type)
		}
	}
	if finishReason == llm.FinishReasonToolCall {
		if len(toolCalls) == 0 {
			return nil, malformedResponse("response stop reason tool_use has no tool calls")
		}
	} else if len(toolCalls) != 0 {
		if finishReason == llm.FinishReasonLength {
			return nil, unsupported("generate", "truncated tool use is not representable")
		}
		return nil, malformedResponse("response stop reason %q is inconsistent with tool use", *response.StopReason)
	}
	usage, err := usageFromWire(response.Usage)
	if err != nil {
		return nil, err
	}
	providerData, err := marshalAnthropicMessageData(configuredModel, thinking)
	if err != nil {
		return nil, malformedResponse("encode thinking replay: %v", err)
	}
	return &llm.Response{
		ID:               response.ID,
		Model:            response.Model,
		Message:          llm.Message{Role: llm.RoleAssistant, Content: content, ToolCalls: toolCalls, ProviderData: providerData},
		ReasoningSummary: reasoning.String(),
		FinishReason:     finishReason,
		Usage:            usage,
	}, nil
}

func toolCallFromWire(index int, block responseBlock, declared, seenIDs map[string]struct{}) (llm.ToolCall, error) {
	if block.ID == nil || *block.ID == "" {
		return llm.ToolCall{}, malformedResponse("response tool use %d has no ID", index)
	}
	if _, exists := seenIDs[*block.ID]; exists {
		return llm.ToolCall{}, malformedResponse("response tool use %d repeats ID %q", index, *block.ID)
	}
	seenIDs[*block.ID] = struct{}{}
	if block.Name == nil || !validToolName(*block.Name) {
		return llm.ToolCall{}, malformedResponse("response tool use %d has invalid name", index)
	}
	if _, exists := declared[*block.Name]; !exists {
		return llm.ToolCall{}, malformedResponse("response tool use %d names undeclared tool %q", index, *block.Name)
	}
	if block.Input == nil {
		return llm.ToolCall{}, malformedResponse("response tool use %d input must be a JSON object", index)
	}
	input := jsontext.Value(*block.Input)
	if !input.IsValid() || input.Kind() != jsontext.KindBeginObject {
		return llm.ToolCall{}, malformedResponse("response tool use %d input must be a strict JSON object", index)
	}
	return llm.ToolCall{
		ID:        *block.ID,
		Name:      *block.Name,
		Arguments: append(json.RawMessage(nil), (*block.Input)...),
	}, nil
}

func finishReason(op, reason string) (llm.FinishReason, error) {
	switch reason {
	case "end_turn", "stop_sequence":
		return llm.FinishReasonStop, nil
	case "max_tokens", "model_context_window_exceeded":
		return llm.FinishReasonLength, nil
	case "refusal":
		return llm.FinishReasonContentFilter, nil
	case "tool_use":
		return llm.FinishReasonToolCall, nil
	case "pause_turn":
		return "", unsupported(op, fmt.Sprintf("response stop reason %q is not implemented", reason))
	default:
		if op == "generate" {
			return "", malformedResponse("response has unsupported stop reason %q", reason)
		}
		return "", malformedStream("response has unsupported stop reason %q", reason)
	}
}

func usageFromWire(usage *responseUsage) (*llm.Usage, error) {
	if usage == nil || usage.InputTokens == nil || usage.OutputTokens == nil {
		return nil, malformedResponse("response has incomplete usage")
	}
	cacheWrite1h := 0
	if usage.CacheCreation != nil {
		cacheWrite1h = usage.CacheCreation.Ephemeral1hInputTokens
	}
	reasoning := 0
	if usage.OutputTokensDetails != nil && usage.OutputTokensDetails.ThinkingTokens != nil {
		reasoning = *usage.OutputTokensDetails.ThinkingTokens
	}
	values := []int{
		*usage.InputTokens,
		usage.CacheCreationInputTokens,
		usage.CacheReadInputTokens,
		cacheWrite1h,
		*usage.OutputTokens,
		reasoning,
	}
	for _, value := range values {
		if value < 0 {
			return nil, malformedResponse("response has negative token usage")
		}
	}
	if cacheWrite1h > usage.CacheCreationInputTokens {
		return nil, malformedResponse("response one-hour cache-write tokens exceed cache-write tokens")
	}
	if reasoning > *usage.OutputTokens {
		return nil, malformedResponse("response thinking tokens exceed output tokens")
	}
	inputUsage, ok := addInts(values[0], values[1], values[2])
	if !ok {
		return nil, malformedResponse("response token usage overflows int")
	}
	totalTokens, ok := addInts(inputUsage, values[4])
	if !ok {
		return nil, malformedResponse("response token usage overflows int")
	}
	return &llm.Usage{
		InputTokens:        values[0],
		OutputTokens:       values[4],
		CacheReadTokens:    values[2],
		CacheWriteTokens:   values[1],
		CacheWrite1hTokens: values[3],
		ReasoningTokens:    values[5],
		TotalTokens:        totalTokens,
	}, nil
}

func addInts(values ...int) (int, bool) {
	const maxInt = int(^uint(0) >> 1)
	total := 0
	for _, value := range values {
		if value > maxInt-total {
			return 0, false
		}
		total += value
	}
	return total, true
}
