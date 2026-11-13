// Package openai implements OpenAI-compatible Chat Completions using llm's
// neutral contracts. It supports generation, streaming, function tools, JSON
// mode, image input, WAV or MP3 audio input, and PDF document input in user
// messages. For OpenAI's native Responses API, use the openai/responses
// subpackage. Audio output is not implemented.
// Missing, null, or zero usage totals are derived from prompt and completion
// counts. A nonzero total that disagrees with those categories leaves Usage nil
// without discarding response content. Negative prompt, completion, or total
// counts remain malformed responses. Category subdivisions are checked only
// when the total agrees or is derived. Streaming accepts opaque Azure prompt
// annotations, including events with empty choices and no usage.
package openai
