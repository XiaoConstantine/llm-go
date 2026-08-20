package llm

import (
	"encoding/json"
	"strings"
)

// Role identifies a message participant. The constants below are the complete
// set of valid roles; the zero value is invalid.
type Role string

const (
	// RoleSystem carries instructions that apply to the conversation.
	RoleSystem Role = "system"
	// RoleUser carries input from the end user.
	RoleUser Role = "user"
	// RoleAssistant carries model output.
	RoleAssistant Role = "assistant"
	// RoleTool carries results produced by tools.
	RoleTool Role = "tool"
)

// PartKind identifies the representation of a message part. The zero value is
// PartText.
type PartKind uint8

const (
	// PartText contains UTF-8 text in Part.Text.
	PartText PartKind = iota
	// PartImage contains encoded image bytes in Part.Data.
	PartImage
	// PartAudio contains encoded audio bytes in Part.Data.
	PartAudio
)

// Part is one piece of message content. Text parts use Text and leave Data and
// MediaType empty. Image and audio parts use Data and MediaType and leave Text
// empty. Other combinations are invalid.
type Part struct {
	Kind      PartKind
	Text      string
	Data      []byte
	MediaType string
}

// Message is one turn in a model conversation. ToolCalls are valid only on an
// assistant message and may accompany Content. ToolResults are valid only on a
// tool message; its Content must be empty because each result carries its own
// content. Providers translate this canonical form to their wire format.
//
// ProviderData is opaque state returned by a provider. Callers may pass it back
// unchanged on a later turn. A provider must ignore data it does not recognize.
type Message struct {
	Role         Role
	Content      []Part
	ToolCalls    []ToolCall
	ToolResults  []ToolResult
	ProviderData json.RawMessage
}

// Text returns the concatenation of the message's text parts. For a tool
// message, it concatenates the text parts in each result in result order.
func (m Message) Text() string {
	var b strings.Builder
	for _, part := range m.Content {
		if part.Kind == PartText {
			b.WriteString(part.Text)
		}
	}
	for _, result := range m.ToolResults {
		for _, part := range result.Content {
			if part.Kind == PartText {
				b.WriteString(part.Text)
			}
		}
	}
	return b.String()
}

// Tool describes a function that a model may call. InputSchema is a JSON
// Schema describing the function arguments. Strict requires generated
// arguments to conform to InputSchema. An implementation that cannot enforce
// strict schemas must return an error instead of silently weakening the request.
type Tool struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Strict      bool
}

// ToolCall is a tool invocation requested by a model. Arguments contains one
// JSON value, normally an object. ID may be empty when a provider supplies only
// a function name. The order of ID-less calls is significant.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolResult is the output of one tool invocation. CallID must match the call
// when its ID is available. Otherwise, Name must match the call and results are
// paired with preceding unmatched ID-less calls of that name in order. When
// both CallID and Name are set, both must identify the same call.
type ToolResult struct {
	CallID  string
	Name    string
	Content []Part
	IsError bool
}

// ResponseFormat identifies the requested response representation. The zero
// value is ResponseFormatText.
type ResponseFormat uint8

const (
	// ResponseFormatText requests ordinary model output.
	ResponseFormatText ResponseFormat = iota
	// ResponseFormatJSON requests a valid JSON value.
	ResponseFormatJSON
)

// Request describes one generation operation. Messages must not be empty.
// MaxOutputTokens set to zero and nil sampling or penalty fields ask the
// provider to use its default. An empty Tools or Stop list means none.
// Implementations must treat Request as read-only.
type Request struct {
	Messages         []Message
	Tools            []Tool
	ResponseFormat   ResponseFormat
	MaxOutputTokens  int
	Temperature      *float64
	TopP             *float64
	PresencePenalty  *float64
	FrequencyPenalty *float64
	Stop             []string
}

// FinishReason describes why generation stopped.
type FinishReason string

const (
	// FinishReasonStop indicates a natural or requested stop sequence.
	FinishReasonStop FinishReason = "stop"
	// FinishReasonLength indicates that a token limit was reached.
	FinishReasonLength FinishReason = "length"
	// FinishReasonToolCall indicates that the model requested a tool call.
	FinishReasonToolCall FinishReason = "tool_call"
	// FinishReasonContentFilter indicates that content was withheld by a safety
	// system.
	FinishReasonContentFilter FinishReason = "content_filter"
)

// Usage reports token counts when a provider supplies them.
type Usage struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

// Response is the result of a generation operation.
type Response struct {
	ID           string
	Model        string
	Message      Message
	FinishReason FinishReason
	Usage        *Usage
}

// Text returns the concatenation of the response message's text parts.
func (r Response) Text() string {
	return r.Message.Text()
}

// Capability identifies an operation supported by a model.
type Capability string

const (
	// CapabilityGeneration indicates text or multimodal generation support.
	CapabilityGeneration Capability = "generation"
	// CapabilityStreaming indicates incremental generation support.
	CapabilityStreaming Capability = "streaming"
	// CapabilityTools indicates tool-calling support.
	CapabilityTools Capability = "tools"
	// CapabilityJSON indicates structured JSON output support.
	CapabilityJSON Capability = "json"
	// CapabilityVision indicates image input support.
	CapabilityVision Capability = "vision"
	// CapabilityAudio indicates audio input support.
	CapabilityAudio Capability = "audio"
)

// ModelInfo describes a configured model.
type ModelInfo struct {
	Provider     string
	Model        string
	Capabilities []Capability
}
