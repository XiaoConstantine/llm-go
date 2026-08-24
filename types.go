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

// ReasoningEffort controls how much internal reasoning a capable model uses.
// The zero value leaves the provider's default unchanged. Providers may map an
// effort to the nearest level supported by the configured model.
type ReasoningEffort string

const (
	ReasoningEffortDefault ReasoningEffort = ""
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortMax     ReasoningEffort = "max"
)

// Request describes one generation operation. Messages must not be empty.
// MaxOutputTokens set to zero and nil sampling or penalty fields ask the
// provider to use its default. An empty Tools or Stop list means none.
// Implementations must treat Request as read-only.
type Request struct {
	Messages         []Message
	Tools            []Tool
	ResponseFormat   ResponseFormat
	ReasoningEffort  ReasoningEffort
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

// Usage reports token counts when a provider supplies them. InputTokens excludes
// cache reads and writes; TotalTokens is the sum of input, output, cache-read,
// and cache-write tokens. ReasoningTokens is a subset of OutputTokens.
// CacheWrite1hTokens is a subset of CacheWriteTokens. Cost is nil when no
// pricing metadata was configured.
type Usage struct {
	InputTokens        int
	OutputTokens       int
	CacheReadTokens    int
	CacheWriteTokens   int
	CacheWrite1hTokens int
	ReasoningTokens    int
	TotalTokens        int
	Cost               *UsageCost
}

// UsageCost reports cost in US dollars for each token category.
type UsageCost struct {
	Input      float64
	Output     float64
	CacheRead  float64
	CacheWrite float64
	Total      float64
}

// Response is the result of a generation operation.
type Response struct {
	ID      string
	Model   string
	Message Message
	// ReasoningSummary is provider-generated explanatory text about the model's
	// reasoning. It is empty when the provider omits a summary and is separate
	// from the assistant message returned by Text.
	ReasoningSummary string
	FinishReason     FinishReason
	Usage            *Usage
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

// ModelCost contains prices in US dollars per million tokens. Tiers apply the
// highest threshold exceeded by total input usage to the entire request.
type ModelCost struct {
	Input      float64
	Output     float64
	CacheRead  float64
	CacheWrite float64
	Tiers      []ModelCostTier
}

// ModelCostTier overrides all rates when total input usage is greater than
// InputTokensAbove.
type ModelCostTier struct {
	InputTokensAbove int
	Input            float64
	Output           float64
	CacheRead        float64
	CacheWrite       float64
}

// ModelInfo describes a configured model. Cost is nil when pricing is unknown.
type ModelInfo struct {
	Provider     string
	Model        string
	Capabilities []Capability
	Cost         *ModelCost
}

// API identifies a provider wire protocol. Unknown nonempty values are valid so
// applications can describe protocols implemented outside this module.
type API string

const (
	APIAnthropicMessages     API = "anthropic-messages"
	APIOpenAIChatCompletions API = "openai-chat-completions"
	APIOpenAIResponses       API = "openai-responses"
	APIOpenAICodexResponses  API = "openai-codex-responses"
	APIGeminiGenerateContent API = "gemini-generate-content"
)

// Model describes one catalog entry. Name is a human-readable display name and
// defaults conceptually to ID when empty. ContextWindow and MaxOutputTokens are
// zero when unknown.
type Model struct {
	Provider        string
	ID              string
	Name            string
	API             API
	Capabilities    []Capability
	ContextWindow   int
	MaxOutputTokens int
	Cost            *ModelCost
}

// Info returns the provider-neutral configuration used to construct a
// Generator. The returned capability slice is owned by the caller.
func (m Model) Info() ModelInfo {
	return ModelInfo{
		Provider:     m.Provider,
		Model:        m.ID,
		Capabilities: append([]Capability(nil), m.Capabilities...),
		Cost:         cloneModelCost(m.Cost),
	}
}
