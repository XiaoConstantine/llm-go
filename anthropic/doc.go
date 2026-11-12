// Package anthropic implements Anthropic's Messages API using llm's neutral
// request, response, tool, and stream contracts. It supports generation,
// image input and image tool results, signed/redacted thinking replay, prompt
// cache controls, non-streaming client tool use, and typed text/thinking/tool
// streaming. Anthropic Messages has no audio input content block, so
// CapabilityAudio is unsupported. JSON response mode is not implemented.
//
// Temporary overloaded_error failures carry an explicit retryable hint, also
// when received as SSE error events without an HTTP failure status. Retries
// remain opt-in through llm.WithRetry and stop once a stream delivers a chunk.
package anthropic
