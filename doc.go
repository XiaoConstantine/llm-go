// Package llm defines provider-neutral contracts for working with large
// language models in Go. It is independent of agent and prompt-programming
// frameworks. Provider implementations may use provider SDKs internally, but
// those SDK types are not part of this package's requests, responses, streams,
// errors, or interfaces.
//
// Unless documented otherwise, an implementation borrows all storage reachable
// from a request, including slices, pointers, and JSON values, until an operation
// returns. For a successful streaming operation, the borrow ends when the
// stream is closed; if establishing the stream fails, it ends when Stream
// returns. Callers must not mutate borrowed storage concurrently. All storage
// reachable from values returned by an implementation belongs to the caller and
// is not mutated by the implementation after delivery.
package llm
