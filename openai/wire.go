package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime"
	"strings"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
)

type chatRequest struct {
	Model               string              `json:"model"`
	Messages            []chatMessage       `json:"messages"`
	Tools               []chatTool          `json:"tools,omitempty"`
	Stream              bool                `json:"stream,omitzero"`
	StreamOptions       *streamOptions      `json:"stream_options,omitempty"`
	Temperature         *float64            `json:"temperature,omitempty"`
	MaxCompletionTokens *int                `json:"max_completion_tokens,omitempty"`
	MaxTokens           *int                `json:"max_tokens,omitempty"`
	ResponseFormat      *responseFormat     `json:"response_format,omitempty"`
	ReasoningEffort     llm.ReasoningEffort `json:"reasoning_effort,omitempty"`
	TopP                *float64            `json:"top_p,omitempty"`
	FrequencyPenalty    *float64            `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64            `json:"presence_penalty,omitempty"`
	Stop                []string            `json:"stop,omitempty"`
}

type chatMessage struct {
	Role             string             `json:"role"`
	Content          any                `json:"content"`
	Refusal          *string            `json:"refusal,omitzero"`
	ReasoningContent *string            `json:"reasoning_content,omitzero"`
	Reasoning        *string            `json:"reasoning,omitzero"`
	ReasoningDetails *[]json.RawMessage `json:"reasoning_details,omitzero"`
	ToolCallID       string             `json:"tool_call_id,omitempty"`
	ToolCalls        []chatToolCall     `json:"tool_calls,omitempty"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     *string   `json:"text,omitzero"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type chatTool struct {
	Type     string             `json:"type"`
	Function functionDefinition `json:"function"`
}

type functionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict,omitzero"`
}

type chatToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function functionCall `json:"function"`
}

type functionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type responseFormat struct {
	Type string `json:"type"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type chatResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Choices []chatChoice   `json:"choices"`
	Usage   *responseUsage `json:"usage,omitempty"`
}

type chatChoice struct {
	Index        *int             `json:"index"`
	Message      *responseMessage `json:"message"`
	FinishReason *string          `json:"finish_reason"`
}

type responseMessage struct {
	Role             string             `json:"role"`
	Content          *string            `json:"content"`
	Refusal          *string            `json:"refusal"`
	ReasoningContent *string            `json:"reasoning_content"`
	Reasoning        *string            `json:"reasoning"`
	ReasoningDetails *[]json.RawMessage `json:"reasoning_details"`
	ToolCalls        []chatToolCall     `json:"tool_calls,omitempty"`
	FunctionCall     *functionCall      `json:"function_call"`
}

type responseUsage struct {
	PromptTokens            *int                    `json:"prompt_tokens"`
	CompletionTokens        *int                    `json:"completion_tokens"`
	TotalTokens             *int                    `json:"total_tokens"`
	PromptTokensDetails     *promptTokenDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *completionTokenDetails `json:"completion_tokens_details"`
	PromptCacheHitTokens    *int                    `json:"prompt_cache_hit_tokens"`
	CachedTokens            *int                    `json:"cached_tokens"`
}

type promptTokenDetails struct {
	CachedTokens     *int `json:"cached_tokens"`
	CacheWriteTokens *int `json:"cache_write_tokens"`
}

type completionTokenDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

type errorEnvelope struct {
	Error APIError `json:"error"`
}

const messageDataVersion = 1

type messageDataEnvelope struct {
	Provider string           `json:"provider"`
	Version  int              `json:"version"`
	Data     *wireMessageData `json:"data"`
}

// MessageData is OpenAI-specific state stored in llm.Message.ProviderData.
// Callers may inspect it with ParseMessageData or pass the message back to the
// same provider unchanged.
type MessageData struct {
	Refusal *string `json:"refusal,omitzero"`
}

// ReasoningData is opaque reasoning state returned by an OpenAI-compatible
// provider. Exactly one of ReasoningContent and Reasoning is set, preserving
// the provider's original text field. ReasoningDetails contains independently
// owned raw JSON blocks in provider order.
type ReasoningData struct {
	ReasoningContent    *string
	Reasoning           *string
	ReasoningDetails    []json.RawMessage
	HasReasoningDetails bool
}

type wireMessageData struct {
	Refusal          *string            `json:"refusal,omitzero"`
	ReasoningContent *string            `json:"reasoning_content,omitzero"`
	Reasoning        *string            `json:"reasoning,omitzero"`
	ReasoningDetails *[]json.RawMessage `json:"reasoning_details,omitzero"`
}

// ParseMessageData decodes OpenAI-specific state from message. It returns a
// zero MessageData when message has no recognized OpenAI provider data.
func ParseMessageData(message llm.Message) (MessageData, error) {
	data, _, err := parseMessageData(message.ProviderData)
	return MessageData{Refusal: cloneString(data.Refusal)}, err
}

// ParseReasoningData decodes provider reasoning state from message. It returns
// zero data when message has no recognized OpenAI-compatible provider data.
func ParseReasoningData(message llm.Message) (ReasoningData, error) {
	data, _, err := parseMessageData(message.ProviderData)
	if err != nil {
		return ReasoningData{}, err
	}
	return ReasoningData{
		ReasoningContent:    cloneString(data.ReasoningContent),
		Reasoning:           cloneString(data.Reasoning),
		ReasoningDetails:    cloneRawMessagesPointer(data.ReasoningDetails),
		HasReasoningDetails: data.ReasoningDetails != nil,
	}, nil
}

func parseMessageData(raw json.RawMessage) (wireMessageData, bool, error) {
	if len(raw) == 0 {
		return wireMessageData{}, false, nil
	}
	value := jsontext.Value(raw)
	if !value.IsValid() {
		return wireMessageData{}, false, fmt.Errorf("decode OpenAI message data: invalid JSON")
	}
	if value.Kind() != jsontext.KindBeginObject {
		return wireMessageData{}, false, nil
	}

	var identity struct {
		Provider string `json:"provider"`
	}
	if err := jsonv2.Unmarshal(raw, &identity); err != nil || identity.Provider != "openai" {
		return wireMessageData{}, false, nil
	}
	var header struct {
		Version int `json:"version"`
	}
	if err := jsonv2.Unmarshal(raw, &header); err != nil {
		return wireMessageData{}, false, fmt.Errorf("decode OpenAI message data header: %w", err)
	}
	if header.Version != messageDataVersion {
		return wireMessageData{}, false, nil
	}

	var envelope messageDataEnvelope
	if err := jsonv2.Unmarshal(raw, &envelope); err != nil {
		return wireMessageData{}, false, fmt.Errorf("decode OpenAI message data: %w", err)
	}
	if envelope.Data == nil {
		return wireMessageData{}, false, fmt.Errorf("decode OpenAI message data: missing data")
	}
	return *envelope.Data, true, nil
}

func marshalMessageData(data wireMessageData) (json.RawMessage, error) {
	data.ReasoningDetails = cloneRawMessagePointer(data.ReasoningDetails)
	raw, err := jsonv2.Marshal(&messageDataEnvelope{
		Provider: "openai",
		Version:  messageDataVersion,
		Data:     &data,
	})
	if err != nil {
		return nil, err
	}
	return json.RawMessage(raw), nil
}

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	clone := *value
	return &clone
}

func cloneRawMessagesPointer(values *[]json.RawMessage) []json.RawMessage {
	if values == nil {
		return nil
	}
	return cloneRawMessages(*values)
}

func cloneRawMessagePointer(values *[]json.RawMessage) *[]json.RawMessage {
	if values == nil {
		return nil
	}
	clone := cloneRawMessages(*values)
	return &clone
}

func cloneRawMessages(values []json.RawMessage) []json.RawMessage {
	if values == nil {
		return nil
	}
	clone := make([]json.RawMessage, len(values))
	for index := range values {
		clone[index] = append(json.RawMessage(nil), values[index]...)
	}
	return clone
}

func newChatRequest(model string, request llm.Request) (chatRequest, error) {
	return newChatRequestFor("generate", model, request, MaxTokensFieldCompletion)
}

func newChatRequestFor(op, model string, request llm.Request, maxTokensField MaxTokensField) (chatRequest, error) {
	encoder := newMessageEncoder(op, request.Messages)
	messages := make([]chatMessage, 0, len(request.Messages))
	for _, message := range request.Messages {
		converted, err := encoder.toWire(message)
		if err != nil {
			return chatRequest{}, err
		}
		messages = append(messages, converted...)
	}

	tools := make([]chatTool, len(request.Tools))
	for i, tool := range request.Tools {
		tools[i] = chatTool{
			Type: "function",
			Function: functionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  append(json.RawMessage(nil), tool.InputSchema...),
				Strict:      tool.Strict,
			},
		}
	}

	wrequest := chatRequest{
		Model:            model,
		Messages:         messages,
		Tools:            tools,
		Temperature:      request.Temperature,
		TopP:             request.TopP,
		FrequencyPenalty: request.FrequencyPenalty,
		PresencePenalty:  request.PresencePenalty,
		Stop:             append([]string(nil), request.Stop...),
		ReasoningEffort:  request.ReasoningEffort,
	}
	if request.MaxOutputTokens != 0 {
		maxTokens := request.MaxOutputTokens
		if maxTokensField == MaxTokensFieldLegacy {
			wrequest.MaxTokens = &maxTokens
		} else {
			wrequest.MaxCompletionTokens = &maxTokens
		}
	}
	if request.ResponseFormat == llm.ResponseFormatJSON {
		wrequest.ResponseFormat = &responseFormat{Type: "json_object"}
	}
	return wrequest, nil
}

type messageEncoder struct {
	op            string
	usedIDs       map[string]struct{}
	pendingByName map[string][]string
	nextID        int
}

func newMessageEncoder(op string, messages []llm.Message) *messageEncoder {
	usedIDs := make(map[string]struct{})
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			if call.ID != "" {
				usedIDs[call.ID] = struct{}{}
			}
		}
	}
	return &messageEncoder{
		op:            op,
		usedIDs:       usedIDs,
		pendingByName: make(map[string][]string),
	}
}

func (e *messageEncoder) toWire(message llm.Message) ([]chatMessage, error) {
	if message.Role == llm.RoleTool {
		messages := make([]chatMessage, len(message.ToolResults))
		for i, result := range message.ToolResults {
			callID, err := e.resultID(result)
			if err != nil {
				return nil, err
			}
			messages[i] = chatMessage{
				Role:       string(llm.RoleTool),
				Content:    toolResultText(result),
				ToolCallID: callID,
			}
		}
		return messages, nil
	}

	content, err := contentToWire(e.op, message.Content)
	if err != nil {
		return nil, err
	}
	wmessage := chatMessage{
		Role:    string(message.Role),
		Content: content,
	}
	if message.Role == llm.RoleAssistant {
		data, recognized, err := parseMessageData(message.ProviderData)
		if err != nil {
			return nil, requestError(e.op, "decode assistant provider data: %w", err)
		}
		if recognized {
			if data.Refusal != nil {
				refusal := *data.Refusal
				wmessage.Refusal = &refusal
				if len(message.Content) == 0 {
					wmessage.Content = nil
				}
			}
			wmessage.ReasoningContent = cloneString(data.ReasoningContent)
			wmessage.Reasoning = cloneString(data.Reasoning)
			wmessage.ReasoningDetails = cloneRawMessagePointer(data.ReasoningDetails)
		}
	}
	if len(message.ToolCalls) != 0 {
		wmessage.ToolCalls = make([]chatToolCall, len(message.ToolCalls))
		for i, call := range message.ToolCalls {
			callID := call.ID
			if callID == "" {
				callID = e.newID()
				e.pendingByName[call.Name] = append(e.pendingByName[call.Name], callID)
			}
			wmessage.ToolCalls[i] = chatToolCall{
				ID:   callID,
				Type: "function",
				Function: functionCall{
					Name:      call.Name,
					Arguments: string(call.Arguments),
				},
			}
		}
	}
	return []chatMessage{wmessage}, nil
}

func (e *messageEncoder) newID() string {
	for {
		e.nextID++
		id := fmt.Sprintf("call_llm_go_%d", e.nextID)
		if _, exists := e.usedIDs[id]; exists {
			continue
		}
		e.usedIDs[id] = struct{}{}
		return id
	}
}

func (e *messageEncoder) resultID(result llm.ToolResult) (string, error) {
	if result.CallID != "" {
		return result.CallID, nil
	}
	queue := e.pendingByName[result.Name]
	if len(queue) == 0 {
		return "", requestError(e.op, "tool result %q has no pending tool call", result.Name)
	}
	callID := queue[0]
	if len(queue) == 1 {
		delete(e.pendingByName, result.Name)
	} else {
		e.pendingByName[result.Name] = queue[1:]
	}
	return callID, nil
}

func contentToWire(op string, parts []llm.Part) (any, error) {
	hasBinary := false
	for _, part := range parts {
		if part.Kind != llm.PartText {
			hasBinary = true
			break
		}
	}
	if !hasBinary {
		return textContent(parts), nil
	}

	content := make([]contentPart, len(parts))
	for i, part := range parts {
		switch part.Kind {
		case llm.PartText:
			text := part.Text
			content[i] = contentPart{Type: "text", Text: &text}
		case llm.PartImage:
			mediaType, _, err := mime.ParseMediaType(part.MediaType)
			if err != nil || !strings.HasPrefix(strings.ToLower(mediaType), "image/") {
				return nil, requestError(op, "image media type %q is invalid", part.MediaType)
			}
			content[i] = contentPart{
				Type: "image_url",
				ImageURL: &imageURL{URL: "data:" + mediaType + ";base64," +
					base64.StdEncoding.EncodeToString(part.Data)},
			}
		default:
			return nil, unsupported(op, "content kind is not supported")
		}
	}
	return content, nil
}

func textContent(parts []llm.Part) string {
	var builder strings.Builder
	for _, part := range parts {
		if part.Kind == llm.PartText {
			builder.WriteString(part.Text)
		}
	}
	return builder.String()
}

func toolResultText(result llm.ToolResult) string {
	text := textContent(result.Content)
	if result.IsError {
		return "Error: " + text
	}
	return text
}

func responseFromWire(configuredModel string, request llm.Request, response chatResponse) (*llm.Response, error) {
	if len(response.Choices) == 0 {
		return nil, malformedResponse("response contains no choices")
	}
	if len(response.Choices) != 1 {
		return nil, malformedResponse("response contains %d choices, want exactly one", len(response.Choices))
	}
	choice := response.Choices[0]
	if choice.Index == nil {
		return nil, malformedResponse("response choice has no index")
	}
	if *choice.Index != 0 {
		return nil, malformedResponse("response choice has index %d, want 0", *choice.Index)
	}
	if choice.Message == nil {
		return nil, malformedResponse("response choice has no message")
	}
	if choice.Message.Role != string(llm.RoleAssistant) {
		return nil, malformedResponse("response message has role %q, want assistant", choice.Message.Role)
	}
	if choice.FinishReason == nil {
		return nil, malformedResponse("response choice has no finish reason")
	}
	finish, err := finishReason(*choice.FinishReason)
	if err != nil {
		return nil, err
	}

	modernCalls := len(choice.Message.ToolCalls) != 0
	legacyCall := choice.Message.FunctionCall != nil
	if modernCalls && legacyCall {
		return nil, malformedResponse("response contains both tool_calls and function_call")
	}
	switch *choice.FinishReason {
	case "tool_calls":
		if !modernCalls {
			return nil, malformedResponse("finish reason tool_calls has no tool calls")
		}
	case "function_call":
		if !legacyCall {
			return nil, malformedResponse("finish reason function_call has no function call")
		}
	default:
		if modernCalls || legacyCall {
			return nil, malformedResponse("finish reason %q is inconsistent with a tool call", *choice.FinishReason)
		}
	}

	message := llm.Message{Role: llm.RoleAssistant}
	if choice.Message.Content != nil {
		message.Content = []llm.Part{{Text: *choice.Message.Content}}
	}
	reasoning, err := reasoningFromWire(choice.Message.ReasoningContent, choice.Message.Reasoning)
	if err != nil {
		return nil, malformedResponse("response message %w", err)
	}
	if choice.Message.Refusal != nil || reasoning != nil || choice.Message.ReasoningDetails != nil {
		data, err := marshalMessageData(wireMessageData{
			Refusal:          choice.Message.Refusal,
			ReasoningContent: choice.Message.ReasoningContent,
			Reasoning:        choice.Message.Reasoning,
			ReasoningDetails: choice.Message.ReasoningDetails,
		})
		if err != nil {
			return nil, malformedResponse("encode provider message state: %w", err)
		}
		message.ProviderData = data
	}
	if modernCalls {
		message.ToolCalls = make([]llm.ToolCall, len(choice.Message.ToolCalls))
		ids := make(map[string]struct{}, len(choice.Message.ToolCalls))
		for i, call := range choice.Message.ToolCalls {
			if call.Type != "function" {
				return nil, malformedResponse("tool call %d has unsupported type %q", i, call.Type)
			}
			if call.ID == "" {
				return nil, malformedResponse("tool call %d has no ID", i)
			}
			if _, exists := ids[call.ID]; exists {
				return nil, malformedResponse("tool call %d repeats ID %q", i, call.ID)
			}
			ids[call.ID] = struct{}{}
			converted, err := functionCallFromWire(call.ID, call.Function, fmt.Sprintf("tool call %d", i))
			if err != nil {
				return nil, err
			}
			message.ToolCalls[i] = converted
		}
	} else if legacyCall {
		converted, err := functionCallFromWire("", *choice.Message.FunctionCall, "function call")
		if err != nil {
			return nil, err
		}
		message.ToolCalls = []llm.ToolCall{converted}
	}
	if len(message.ToolCalls) != 0 {
		declared := declaredToolNames(request.Tools)
		for i, call := range message.ToolCalls {
			if _, ok := declared[call.Name]; !ok {
				return nil, malformedResponse("tool call %d names undeclared function %q", i, call.Name)
			}
		}
	}

	if request.ResponseFormat == llm.ResponseFormatJSON &&
		finish == llm.FinishReasonStop && choice.Message.Refusal == nil {
		if choice.Message.Content == nil {
			return nil, malformedResponse("JSON-mode response contains no JSON content")
		}
		if !jsontext.Value(*choice.Message.Content).IsValid() {
			return nil, malformedResponse("JSON-mode response content is not strict JSON")
		}
	}

	usage, err := usageFromWire(response.Usage)
	if err != nil {
		return nil, malformedResponse("response contains %w", err)
	}

	model := response.Model
	if model == "" {
		model = configuredModel
	}
	return &llm.Response{
		ID:           response.ID,
		Model:        model,
		Message:      message,
		FinishReason: finish,
		Usage:        usage,
	}, nil
}

func reasoningFromWire(reasoningContent, reasoning *string) (*string, error) {
	if reasoningContent != nil && reasoning != nil {
		return nil, fmt.Errorf("contains both reasoning_content and reasoning")
	}
	if reasoningContent != nil {
		value := *reasoningContent
		return &value, nil
	}
	if reasoning != nil {
		value := *reasoning
		return &value, nil
	}
	return nil, nil
}

func functionCallFromWire(id string, call functionCall, label string) (llm.ToolCall, error) {
	converted, err := functionCallValue(id, call, label)
	if err != nil {
		return llm.ToolCall{}, malformedResponse("%w", err)
	}
	return converted, nil
}

func functionCallValue(id string, call functionCall, label string) (llm.ToolCall, error) {
	if !validFunctionName(call.Name) {
		return llm.ToolCall{}, fmt.Errorf("%s has invalid function name %q", label, call.Name)
	}
	arguments := json.RawMessage(call.Arguments)
	if !jsontext.Value(arguments).IsValid() {
		return llm.ToolCall{}, fmt.Errorf("%s arguments are not strict JSON", label)
	}
	return llm.ToolCall{
		ID:        id,
		Name:      call.Name,
		Arguments: append(json.RawMessage(nil), arguments...),
	}, nil
}

func declaredToolNames(tools []llm.Tool) map[string]struct{} {
	if len(tools) == 0 {
		return nil
	}
	names := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		names[tool.Name] = struct{}{}
	}
	return names
}

func finishReason(reason string) (llm.FinishReason, error) {
	if reason == "insufficient_system_resource" {
		return "", providerInterruption("generate", reason)
	}
	finish, ok := finishReasonValue(reason)
	if !ok {
		return "", malformedResponse("response has unknown finish reason %q", reason)
	}
	return finish, nil
}

func finishReasonValue(reason string) (llm.FinishReason, bool) {
	switch reason {
	case "stop":
		return llm.FinishReasonStop, true
	case "length":
		return llm.FinishReasonLength, true
	case "tool_calls", "function_call":
		return llm.FinishReasonToolCall, true
	case "content_filter":
		return llm.FinishReasonContentFilter, true
	default:
		return "", false
	}
}

func providerInterruption(op, reason string) error {
	return &llm.Error{
		Kind:     llm.KindProvider,
		Op:       op,
		Provider: "openai",
		Err:      fmt.Errorf("provider interrupted generation: %s", reason),
	}
}

func usageFromWire(usage *responseUsage) (*llm.Usage, error) {
	if usage == nil {
		return nil, nil
	}
	if usage.PromptTokens == nil || usage.CompletionTokens == nil || usage.TotalTokens == nil {
		return nil, fmt.Errorf("incomplete token usage")
	}
	if *usage.PromptTokens < 0 || *usage.CompletionTokens < 0 || *usage.TotalTokens < 0 {
		return nil, fmt.Errorf("negative token usage")
	}
	if *usage.CompletionTokens > *usage.TotalTokens ||
		*usage.PromptTokens != *usage.TotalTokens-*usage.CompletionTokens {
		return nil, fmt.Errorf("inconsistent token usage")
	}
	cacheRead := 0
	cacheWrite := 0
	hasCacheRead := false
	if usage.PromptTokensDetails != nil {
		if usage.PromptTokensDetails.CachedTokens != nil {
			cacheRead = *usage.PromptTokensDetails.CachedTokens
			hasCacheRead = true
		}
		if usage.PromptTokensDetails.CacheWriteTokens != nil {
			cacheWrite = *usage.PromptTokensDetails.CacheWriteTokens
		}
	}
	if !hasCacheRead && usage.PromptCacheHitTokens != nil {
		cacheRead = *usage.PromptCacheHitTokens
		hasCacheRead = true
	}
	if !hasCacheRead && usage.CachedTokens != nil {
		cacheRead = *usage.CachedTokens
	}
	reasoning := 0
	if usage.CompletionTokensDetails != nil && usage.CompletionTokensDetails.ReasoningTokens != nil {
		reasoning = *usage.CompletionTokensDetails.ReasoningTokens
	}
	if cacheRead < 0 || cacheWrite < 0 || reasoning < 0 || cacheRead > *usage.PromptTokens ||
		cacheWrite > *usage.PromptTokens-cacheRead || reasoning > *usage.CompletionTokens {
		return nil, fmt.Errorf("inconsistent token usage details")
	}
	return &llm.Usage{
		InputTokens:      *usage.PromptTokens - cacheRead - cacheWrite,
		OutputTokens:     *usage.CompletionTokens,
		CacheReadTokens:  cacheRead,
		CacheWriteTokens: cacheWrite,
		ReasoningTokens:  reasoning,
		TotalTokens:      *usage.TotalTokens,
	}, nil
}
