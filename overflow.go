package llm

import (
	"errors"
	"regexp"
)

var contextOverflowPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)prompt is too long`),
	regexp.MustCompile(`(?i)request_too_large`),
	regexp.MustCompile(`(?i)input is too long for requested model`),
	regexp.MustCompile(`(?i)exceeds the context window`),
	regexp.MustCompile(`(?i)input token count.*exceeds the maximum`),
	regexp.MustCompile(`(?i)maximum prompt length is \d+`),
	regexp.MustCompile(`(?i)reduce the length of the messages`),
	regexp.MustCompile(`(?i)maximum context length is \d+ tokens`),
	regexp.MustCompile(`(?i)exceeds the limit of \d+`),
	regexp.MustCompile(`(?i)exceeds the available context size`),
	regexp.MustCompile(`(?i)greater than the context length`),
	regexp.MustCompile(`(?i)context window exceeds limit`),
	regexp.MustCompile(`(?i)exceeded model token limit`),
	regexp.MustCompile(`(?i)too large for model with \d+ maximum context length`),
	regexp.MustCompile(`(?i)model_context_window_exceeded`),
	regexp.MustCompile(`(?i)prompt too long; exceeded (max )?context length`),
	regexp.MustCompile(`(?i)context[_ ]length[_ ]exceeded`),
	regexp.MustCompile(`(?i)too many tokens`),
	regexp.MustCompile(`(?i)token limit exceeded`),
	regexp.MustCompile(`(?i)^4(00|13)\s*(status code)?\s*\(no body\)`),
}

var nonContextOverflowPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^(throttling error|service unavailable):`),
	regexp.MustCompile(`(?i)rate limit`),
	regexp.MustCompile(`(?i)too many requests`),
}

// IsContextOverflow reports whether a failed or successful generation indicates
// that its input exceeded the model context window. Built-in generators classify
// explicit provider failures as KindContextLimit. The message-pattern fallback
// supports custom generators that retain an unclassified provider error.
//
// A positive contextWindow additionally detects providers that silently accept
// oversized input and providers that truncate input until no output tokens
// remain. response may be nil when err is non-nil.
func IsContextOverflow(response *Response, err error, contextWindow int) bool {
	if err != nil {
		var modelErr *Error
		if errors.As(err, &modelErr) && modelErr != nil {
			switch modelErr.Kind {
			case KindContextLimit:
				return true
			case KindRateLimit:
				return false
			}
		}
		message := err.Error()
		for _, pattern := range nonContextOverflowPatterns {
			if pattern.MatchString(message) {
				return false
			}
		}
		for _, pattern := range contextOverflowPatterns {
			if pattern.MatchString(message) {
				return true
			}
		}
	}
	if response == nil || response.Usage == nil || contextWindow <= 0 {
		return false
	}
	inputTokens := response.Usage.InputTokens + response.Usage.CacheReadTokens
	if response.FinishReason == FinishReasonStop && inputTokens > contextWindow {
		return true
	}
	return response.FinishReason == FinishReasonLength && response.Usage.OutputTokens == 0 &&
		inputTokens >= contextWindow-contextWindow/100
}
