// Package anthropic implements Anthropic's Messages API using llm's neutral
// request, response, tool, and stream contracts. It supports generation,
// image input and image tool results, signed/redacted thinking replay, prompt
// cache controls, non-streaming client tool use, and typed text/thinking/tool
// streaming. Anthropic Messages has no audio input content block, so
// CapabilityAudio is unsupported. JSON response mode is not implemented.
// File parts in user messages become document blocks: PDF as base64 and
// plain text as UTF-8 text, with Part.Filename as the title. Other document
// media types are unsupported.
//
// Temporary overloaded_error failures carry an explicit retryable hint, also
// when received as SSE error events without an HTTP failure status. Retries
// remain opt-in through llm.WithRetry and stop once a stream delivers a chunk.
//
// Streams report the input and cache usage from message_start on the first
// chunk that carries output, so a stream that fails or is canceled after
// producing output still reports what the request consumed. The final chunk
// carries the complete cumulative usage. The early usage is never a chunk of
// its own, so it does not end retry eligibility.
package anthropic
