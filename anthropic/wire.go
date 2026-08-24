package anthropic

import (
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"fmt"
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
	Stream        bool             `json:"stream,omitzero"`
}

type inputMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type inputContentBlock struct {
	Type      string          `json:"type"`
	Text      *string         `json:"text,omitzero"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   *string         `json:"content,omitzero"`
	IsError   bool            `json:"is_error,omitzero"`
}

type toolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
	Strict      bool            `json:"strict,omitzero"`
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
	Type  string           `json:"type"`
	Text  *string          `json:"text"`
	ID    *string          `json:"id"`
	Name  *string          `json:"name"`
	Input *json.RawMessage `json:"input"`
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

func checkRequest(op string, request llm.Request) error {
	if request.ReasoningEffort != llm.ReasoningEffortDefault {
		return unsupported(op, "reasoning effort is not implemented")
	}
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
	wrequest.Tools = make([]toolDefinition, len(request.Tools))
	for index, tool := range request.Tools {
		wrequest.Tools[index] = toolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: append(json.RawMessage(nil), tool.InputSchema...),
			Strict:      tool.Strict,
		}
	}
	encoder := newMessageEncoder(op, request.Messages)
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
	return wrequest, encoder.usedIDs, nil
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

func (encoder *messageEncoder) toWire(message llm.Message) (inputMessage, error) {
	if message.Role == llm.RoleTool {
		blocks := make([]inputContentBlock, len(message.ToolResults))
		for index, result := range message.ToolResults {
			callID, err := encoder.resultID(result)
			if err != nil {
				return inputMessage{}, err
			}
			blocks[index] = inputContentBlock{
				Type:      "tool_result",
				ToolUseID: callID,
				Content:   optionalTextContent(result.Content),
				IsError:   result.IsError,
			}
		}
		return inputMessage{Role: string(llm.RoleUser), Content: blocks}, nil
	}

	if len(message.ToolCalls) == 0 {
		return inputMessage{Role: string(message.Role), Content: message.Text()}, nil
	}
	blocks := make([]inputContentBlock, 0, len(message.Content)+len(message.ToolCalls))
	for _, part := range message.Content {
		text := part.Text
		blocks = append(blocks, inputContentBlock{Type: "text", Text: &text})
	}
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

func responseFromWire(request llm.Request, priorToolIDs map[string]struct{}, response messageResponse) (*llm.Response, error) {
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
	return &llm.Response{
		ID:           response.ID,
		Model:        response.Model,
		Message:      llm.Message{Role: llm.RoleAssistant, Content: content, ToolCalls: toolCalls},
		FinishReason: finishReason,
		Usage:        usage,
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
