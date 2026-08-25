// Package anthropic implements Anthropic's Messages API using llm's neutral
// request, response, tool, and stream contracts. It supports generation,
// image input and image tool results, signed/redacted thinking replay, prompt
// cache controls, non-streaming client tool use, and typed text/thinking/tool
// streaming. Anthropic Messages has no audio input content block, so
// CapabilityAudio is unsupported. JSON response mode is not implemented.
package anthropic
