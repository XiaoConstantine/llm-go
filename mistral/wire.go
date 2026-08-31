package mistral

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
)

const messageDataVersion = 1

type chatRequest struct {
	Model           string              `json:"model"`
	Messages        []chatMessage       `json:"messages"`
	Tools           []chatTool          `json:"tools,omitempty"`
	Stream          bool                `json:"stream"`
	Temperature     *float64            `json:"temperature,omitempty"`
	MaxTokens       *int                `json:"max_tokens,omitempty"`
	ToolChoice      any                 `json:"tool_choice,omitempty"`
	PromptMode      string              `json:"prompt_mode,omitempty"`
	ReasoningEffort llm.ReasoningEffort `json:"reasoning_effort,omitempty"`
}

type chatMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content,omitempty"`
	ToolCalls  []chatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	Name       string         `json:"name,omitempty"`
}

type contentPart struct {
	Type     string         `json:"type"`
	Text     string         `json:"text,omitempty"`
	ImageURL string         `json:"image_url,omitempty"`
	Thinking []thinkingPart `json:"thinking,omitempty"`
}

type thinkingPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type chatTool struct {
	Type     string             `json:"type"`
	Function functionDefinition `json:"function"`
}

type functionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict"`
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

type mistralMessageData struct {
	Provider string   `json:"provider"`
	Version  int      `json:"version"`
	Model    string   `json:"model"`
	Thinking []string `json:"thinking"`
}

type requestEncoder struct {
	op          string
	model       string
	vision      bool
	normalizeID *toolCallIDNormalizer
	pending     map[string][]string
}

func buildRequest(op, model string, reasoning, vision bool, request llm.Request) ([]byte, map[string]struct{}, error) {
	encoder := requestEncoder{op: op, model: model, vision: vision,
		normalizeID: newToolCallIDNormalizer(), pending: make(map[string][]string)}
	messages := make([]chatMessage, 0, len(request.Messages))
	priorIDs := make(map[string]struct{})
	for index, message := range request.Messages {
		converted, err := encoder.message(message, priorIDs)
		if err != nil {
			return nil, nil, requestError(op, "messages[%d]: %v", index, err)
		}
		messages = append(messages, converted...)
	}
	wire := chatRequest{Model: model, Messages: messages, Stream: true, Temperature: request.Temperature}
	if request.MaxOutputTokens != 0 {
		value := request.MaxOutputTokens
		wire.MaxTokens = &value
	}
	if len(request.Tools) != 0 {
		wire.Tools = make([]chatTool, len(request.Tools))
		for index, tool := range request.Tools {
			if tool.RequiresStrict() {
				return nil, nil, unsupported(op, fmt.Sprintf("tool %d requires strict schemas, which Mistral Conversations does not support", index))
			}
			wire.Tools[index] = chatTool{Type: "function", Function: functionDefinition{Name: tool.Name,
				Description: tool.Description, Parameters: append(json.RawMessage(nil), tool.InputSchema...), Strict: false}}
		}
	}
	switch request.ToolChoice.Mode {
	case llm.ToolChoiceAuto:
	case llm.ToolChoiceNone:
		wire.ToolChoice = "none"
	case llm.ToolChoiceRequired:
		wire.ToolChoice = "required"
	case llm.ToolChoiceNamed:
		wire.ToolChoice = map[string]any{"type": "function", "function": map[string]string{"name": request.ToolChoice.Name}}
	}
	if reasoning && request.ReasoningEffort != llm.ReasoningEffortDefault && request.ReasoningEffort != llm.ReasoningEffortNone {
		if usesReasoningEffort(model) {
			wire.ReasoningEffort = llm.ReasoningEffortHigh
		} else {
			wire.PromptMode = "reasoning"
		}
	}
	encoded, err := jsonv2.Marshal(wire)
	if err != nil {
		return nil, nil, requestError(op, "encode request: %v", err)
	}
	return encoded, priorIDs, nil
}

func (e *requestEncoder) message(message llm.Message, priorIDs map[string]struct{}) ([]chatMessage, error) {
	switch message.Role {
	case llm.RoleSystem, llm.RoleUser:
		content, err := e.messageParts(message.Content)
		if err != nil {
			return nil, err
		}
		return []chatMessage{{Role: string(message.Role), Content: content}}, nil
	case llm.RoleAssistant:
		return e.assistant(message, priorIDs)
	case llm.RoleTool:
		output := make([]chatMessage, 0, len(message.ToolResults))
		for index, result := range message.ToolResults {
			callID, err := e.resultID(result)
			if err != nil {
				return nil, fmt.Errorf("tool results[%d]: %w", index, err)
			}
			content, err := e.toolResultParts(result)
			if err != nil {
				return nil, fmt.Errorf("tool results[%d]: %w", index, err)
			}
			output = append(output, chatMessage{Role: string(llm.RoleTool), ToolCallID: callID, Name: result.Name, Content: content})
		}
		return output, nil
	default:
		return nil, fmt.Errorf("role %q is unsupported", message.Role)
	}
}

func (e *requestEncoder) assistant(message llm.Message, priorIDs map[string]struct{}) ([]chatMessage, error) {
	content := make([]contentPart, 0, len(message.Content)+1)
	data, ok, err := parseMessageData(message.ProviderData, e.model)
	if err != nil {
		return nil, fmt.Errorf("provider data: %w", err)
	}
	if ok {
		for _, thinking := range data.Thinking {
			content = append(content, contentPart{Type: "thinking", Thinking: []thinkingPart{{Type: "text", Text: thinking}}})
		}
	}
	parts, err := e.parts(message.Content, false)
	if err != nil {
		return nil, err
	}
	content = append(content, parts...)
	wire := chatMessage{Role: string(llm.RoleAssistant)}
	if len(content) != 0 {
		wire.Content = content
	}
	if len(message.ToolCalls) != 0 {
		wire.ToolCalls = make([]chatToolCall, len(message.ToolCalls))
		for index, call := range message.ToolCalls {
			id := call.ID
			if id == "" {
				id = fmt.Sprintf("toolcall:%d:%s:%s", index, call.Name, call.Arguments)
			}
			id = e.normalizeID.normalize(id)
			if _, duplicate := priorIDs[id]; duplicate {
				return nil, fmt.Errorf("tool calls[%d] normalizes to repeated ID %q", index, id)
			}
			priorIDs[id] = struct{}{}
			e.pending[call.Name] = append(e.pending[call.Name], id)
			wire.ToolCalls[index] = chatToolCall{ID: id, Type: "function",
				Function: functionCall{Name: call.Name, Arguments: string(call.Arguments)}}
		}
	}
	return []chatMessage{wire}, nil
}

func (e *requestEncoder) resultID(result llm.ToolResult) (string, error) {
	if result.CallID != "" {
		id := e.normalizeID.normalize(result.CallID)
		e.removePending(result.Name, id)
		return id, nil
	}
	queue := e.pending[result.Name]
	if len(queue) == 0 {
		return "", fmt.Errorf("has no pending tool call named %q", result.Name)
	}
	id := queue[0]
	if len(queue) == 1 {
		delete(e.pending, result.Name)
	} else {
		e.pending[result.Name] = queue[1:]
	}
	return id, nil
}

func (e *requestEncoder) removePending(name, id string) {
	remove := func(key string) bool {
		queue := e.pending[key]
		for index, candidate := range queue {
			if candidate != id {
				continue
			}
			e.pending[key] = append(queue[:index], queue[index+1:]...)
			if len(e.pending[key]) == 0 {
				delete(e.pending, key)
			}
			return true
		}
		return false
	}
	if name != "" && remove(name) {
		return
	}
	for key := range e.pending {
		if remove(key) {
			return
		}
	}
}

func (e *requestEncoder) messageParts(parts []llm.Part) (any, error) {
	converted, err := e.parts(parts, false)
	if err != nil {
		return nil, err
	}
	if len(converted) == 1 && converted[0].Type == "text" {
		return converted[0].Text, nil
	}
	return converted, nil
}

func (e *requestEncoder) parts(parts []llm.Part, toolResult bool) ([]contentPart, error) {
	converted := make([]contentPart, 0, len(parts))
	for index, part := range parts {
		switch part.Kind {
		case llm.PartText:
			converted = append(converted, contentPart{Type: "text", Text: part.Text})
		case llm.PartImage:
			if !e.vision {
				if !toolResult {
					converted = append(converted, contentPart{Type: "text", Text: "(image omitted: model does not support images)"})
				}
				continue
			}
			converted = append(converted, contentPart{Type: "image_url",
				ImageURL: "data:" + part.MediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data)})
		default:
			return nil, fmt.Errorf("content[%d] kind %d is not supported", index, part.Kind)
		}
	}
	return converted, nil
}

func (e *requestEncoder) toolResultParts(result llm.ToolResult) (any, error) {
	converted, err := e.parts(result.Content, true)
	if err != nil {
		return nil, err
	}
	texts := make([]string, 0, len(converted))
	images := make([]contentPart, 0, len(converted))
	for _, part := range converted {
		switch part.Type {
		case "text":
			texts = append(texts, part.Text)
		case "image_url":
			images = append(images, part)
		}
	}
	value := strings.TrimSpace(strings.Join(texts, "\n"))
	hasImage := len(images) != 0
	if !e.vision {
		for _, part := range result.Content {
			hasImage = hasImage || part.Kind == llm.PartImage
		}
	}
	if value == "" {
		switch {
		case hasImage && e.vision:
			value = "(see attached image)"
		case hasImage:
			value = "(image omitted: model does not support images)"
		default:
			value = "(no tool output)"
		}
	} else if hasImage && !e.vision {
		value += "\n[tool image omitted: model does not support images]"
	}
	if result.IsError {
		value = "[tool error] " + value
	}
	return append([]contentPart{{Type: "text", Text: value}}, images...), nil
}

func usesReasoningEffort(model string) bool {
	switch model {
	case "mistral-small-2603", "mistral-small-latest", "mistral-medium-3.5":
		return true
	default:
		return false
	}
}

type toolCallIDNormalizer struct {
	forward map[string]string
	reverse map[string]string
}

func newToolCallIDNormalizer() *toolCallIDNormalizer {
	return &toolCallIDNormalizer{forward: make(map[string]string), reverse: make(map[string]string)}
}

func (n *toolCallIDNormalizer) normalize(id string) string {
	if existing := n.forward[id]; existing != "" {
		return existing
	}
	var alphanumeric strings.Builder
	for index := range len(id) {
		character := id[index]
		if character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' || character >= '0' && character <= '9' {
			alphanumeric.WriteByte(character)
		}
	}
	if candidate := alphanumeric.String(); len(candidate) == 9 {
		if owner := n.reverse[candidate]; owner == "" || owner == id {
			n.forward[id], n.reverse[candidate] = candidate, id
			return candidate
		}
	}
	seed := alphanumeric.String()
	if seed == "" {
		seed = id
	}
	for attempt := 0; ; attempt++ {
		value := seed
		if attempt != 0 {
			value = fmt.Sprintf("%s:%d", seed, attempt)
		}
		digest := sha256.Sum256([]byte(value))
		candidate := hex.EncodeToString(digest[:])[:9]
		if owner := n.reverse[candidate]; owner == "" || owner == id {
			n.forward[id], n.reverse[candidate] = candidate, id
			return candidate
		}
	}
}

func marshalMessageData(model string, thinking []string) (json.RawMessage, error) {
	if len(thinking) == 0 {
		return nil, nil
	}
	return jsonv2.Marshal(mistralMessageData{Provider: defaultProvider, Version: messageDataVersion, Model: model, Thinking: thinking})
}

func parseMessageData(raw json.RawMessage, model string) (mistralMessageData, bool, error) {
	if len(raw) == 0 || jsontext.Value(raw).Kind() != jsontext.KindBeginObject {
		return mistralMessageData{}, false, nil
	}
	var data mistralMessageData
	if err := jsonv2.Unmarshal(raw, &data); err != nil {
		return mistralMessageData{}, false, err
	}
	if data.Provider != defaultProvider {
		return mistralMessageData{}, false, nil
	}
	if data.Version != messageDataVersion {
		return mistralMessageData{}, false, fmt.Errorf("unsupported version %d", data.Version)
	}
	if data.Model != model {
		return mistralMessageData{}, false, nil
	}
	return data, true, nil
}
