package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"
)

// Generator produces model responses. A Generator is configured for the model
// reported by Info and may be used concurrently by multiple goroutines.
// Implementations must first call Request.Validate and return its error unchanged
// or wrapped so errors.As preserves its Error fields. They must next return an
// error matching ctx.Err if the context is already done, then apply capability
// checks, then model-specific checks, all before provider I/O. Implementations
// must continue to honor cancellation until Generate returns or a stream commits
// its terminal state. They return a nil response when Generate returns an error
// and a nil Stream when Stream returns an error. Before returning a Stream error,
// an implementation must release all storage borrowed from the request. Info
// must include CapabilityGeneration. If it does not include
// CapabilityStreaming, Stream returns an error classified KindUnsupported.
// Callers should give operations a context deadline appropriate to their
// workload when an unbounded request is not acceptable.
type Generator interface {
	Info() ModelInfo
	Generate(ctx context.Context, request Request) (*Response, error)
	Stream(ctx context.Context, request Request) (Stream, error)
}

// BackgroundStatus describes the lifecycle of a durable provider-side
// generation. The zero value is invalid.
type BackgroundStatus string

const (
	BackgroundQueued     BackgroundStatus = "queued"
	BackgroundInProgress BackgroundStatus = "in_progress"
	BackgroundCompleted  BackgroundStatus = "completed"
	BackgroundIncomplete BackgroundStatus = "incomplete"
	BackgroundCancelled  BackgroundStatus = "cancelled"
	BackgroundFailed     BackgroundStatus = "failed"
)

// BackgroundHandle contains the durable identity and output-validation contract
// for a provider-side generation. Tools and their schemas are retained so a
// fetched response receives the same validation as Generate. Handles returned by
// implementations are caller-owned and may be persisted. Callers must not modify
// a handle concurrently with FetchBackground or CancelBackground.
type BackgroundHandle struct {
	Provider       string
	Model          string
	API            API
	ID             string
	Tools          []Tool
	ResponseFormat ResponseFormat
}

// Validate reports whether h has complete identity and valid output-validation
// metadata.
func (h BackgroundHandle) Validate() error {
	if err := h.validate(); err != nil {
		return &Error{Kind: KindInvalidRequest, Op: "validate_background_handle", Provider: h.Provider, Err: err}
	}
	return nil
}

func (h BackgroundHandle) validate() error {
	for _, field := range []struct{ name, value string }{
		{"provider", h.Provider}, {"model", h.Model}, {"API", string(h.API)}, {"ID", h.ID},
	} {
		if field.value == "" {
			return fmt.Errorf("%s must not be empty", field.name)
		}
		if !utf8.ValidString(field.value) {
			return fmt.Errorf("%s must be valid UTF-8", field.name)
		}
		if field.value != strings.TrimSpace(field.value) {
			return fmt.Errorf("%s must not contain surrounding whitespace", field.name)
		}
	}
	if err := validateTools(h.Tools); err != nil {
		return err
	}
	return validateResponseFormat(h.ResponseFormat)
}

// BackgroundResult is one observed state of a background generation. Response
// is non-nil only when Status is BackgroundCompleted or BackgroundIncomplete
// and output conversion succeeds. Handle and all nested response storage are
// caller-owned. Callers must retain a non-nil result even when the accompanying
// error is non-nil, because StartBackground may have created a durable job.
type BackgroundResult struct {
	Handle   BackgroundHandle
	Status   BackgroundStatus
	Response *Response
}

// BackgroundGenerator starts, fetches, and cancels durable provider-side
// generations. A handle passed to FetchBackground or CancelBackground must
// match the provider, model, and API reported by Info. Implementations validate
// requests and handles before I/O and are safe for concurrent use. A failed
// provider generation returns a result with BackgroundFailed together with its
// provider error. Malformed output or cancellation that races with an observed
// provider response may also return its handle and status with an error;
// transport errors before a response is observed return a nil result.
type BackgroundGenerator interface {
	Generator
	StartBackground(context.Context, Request) (*BackgroundResult, error)
	FetchBackground(context.Context, BackgroundHandle) (*BackgroundResult, error)
	CancelBackground(context.Context, BackgroundHandle) (*BackgroundResult, error)
}

// StreamEventKind identifies one semantic event in a generation stream.
type StreamEventKind string

const (
	StreamEventStart          StreamEventKind = "start"
	StreamEventTextStart      StreamEventKind = "text_start"
	StreamEventTextDelta      StreamEventKind = "text_delta"
	StreamEventTextEnd        StreamEventKind = "text_end"
	StreamEventReasoningStart StreamEventKind = "reasoning_start"
	StreamEventReasoningDelta StreamEventKind = "reasoning_delta"
	StreamEventReasoningEnd   StreamEventKind = "reasoning_end"
	StreamEventToolCallStart  StreamEventKind = "tool_call_start"
	StreamEventToolCallDelta  StreamEventKind = "tool_call_delta"
	StreamEventToolCallEnd    StreamEventKind = "tool_call_end"
	StreamEventDone           StreamEventKind = "done"
)

// StreamEvent describes a semantic transition represented by a Chunk. Index
// identifies a provider output item and Subindex identifies a part within it;
// both are stable for events belonging to that item. Providers without nested
// output parts leave Subindex zero. Delta is newly received text, reasoning text, or raw tool
// argument fragment; a tool argument delta need not be valid JSON by itself.
// Content is the completed text for an end event when the provider makes it
// available. ToolCallID and ToolName contain the identity known at a tool-call
// start or delta. ToolCall is set on a successfully completed ToolCallEnd and
// contains complete, valid arguments; it is nil when an incomplete response
// interrupts a started call. FinishReason is set only for Done.
//
// Streams report failures through Recv rather than as events. Event storage is
// owned by the caller after Recv returns.
type StreamEvent struct {
	Kind         StreamEventKind
	Index        int
	Subindex     int
	Delta        string
	Content      string
	ToolCallID   string
	ToolName     string
	ToolCall     *ToolCall
	FinishReason FinishReason
}

// Chunk contains new output from a generation stream. To assemble a Response,
// append Content, ToolCalls, and ReasoningSummary in chunk order, retain any
// nonempty ID, Model, and FinishReason, use the single nonempty ProviderData,
// and use the last non-nil Usage. Events describe the same output at a finer
// granularity and must not be appended to the response. The assembled message
// has RoleAssistant.
//
// Text parts may be fragments; other parts and ToolCalls must be complete
// values. Events may expose incomplete tool argument fragments, but ToolCalls
// never do. Nonempty ID and Model values must not change between chunks. A
// nonzero FinishReason must not change between chunks. ProviderData must be a
// complete value and appear in at most one chunk. Usage, when present, is
// cumulative for the request.
type Chunk struct {
	ID               string
	Model            string
	Content          []Part
	ToolCalls        []ToolCall
	ReasoningSummary string
	ProviderData     json.RawMessage
	FinishReason     FinishReason
	Usage            *Usage
	Events           []StreamEvent
}

// Stream is a sequence of generation chunks. Callers must close every Stream,
// including after Recv returns an error. Only one goroutine may call Recv at a
// time. Close is safe to call concurrently with any Stream method.
//
// A stream commits an ordered sequence of chunks followed by exactly one
// terminal state: io.EOF or a non-EOF error. Recv delivers every committed chunk
// exactly once, in order, before the terminal state; Close does not discard
// committed chunks. Natural completion and Close commit io.EOF, while provider
// and context failures commit their error. The first terminal transition wins.
// However, if the context passed to Generator.Stream is already done when Close
// is called, its error precedes Close. If cancellation and Close occur
// concurrently, either may win.
type Stream interface {
	// Recv returns the next chunk. It returns io.EOF on clean completion. When
	// it returns an error, the chunk is zero and later calls return the same
	// error. Context cancellation and deadline errors remain discoverable with
	// errors.Is.
	Recv() (Chunk, error)

	// Close stops the stream and releases its resources. It is idempotent and
	// must unblock a pending Recv. It commits io.EOF unless another terminal
	// state won first or the context was already done. Close does not return
	// until the implementation has released all storage borrowed from the
	// request. Repeated calls return the same error; a Close error does not
	// replace the stream's terminal state.
	Close() error
}
