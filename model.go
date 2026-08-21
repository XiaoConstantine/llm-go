package llm

import (
	"context"
	"encoding/json"
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

// Chunk contains new output from a generation stream. To assemble a Response,
// append Content and ToolCalls in chunk order, retain any nonempty ID, Model,
// and FinishReason, use the single nonempty ProviderData, and use the last
// non-nil Usage. The assembled message has RoleAssistant.
//
// Text parts may be fragments; other parts and ToolCalls must be complete
// values. Nonempty ID and Model values must not change between chunks. A
// nonzero FinishReason must not change between chunks. ProviderData must be a
// complete value and appear in at most one chunk. Usage, when present, is
// cumulative for the request.
type Chunk struct {
	ID           string
	Model        string
	Content      []Part
	ToolCalls    []ToolCall
	ProviderData json.RawMessage
	FinishReason FinishReason
	Usage        *Usage
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
