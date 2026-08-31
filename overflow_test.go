package llm

import (
	"errors"
	"testing"
)

func TestIsContextOverflowClassifiedErrors(t *testing.T) {
	if !IsContextOverflow(nil, &Error{Kind: KindContextLimit, Err: errors.New("provider rejected input")}, 0) {
		t.Fatal("classified context-limit error was not detected")
	}
	if IsContextOverflow(nil, &Error{Kind: KindRateLimit, Err: errors.New("too many tokens, try later")}, 0) {
		t.Fatal("classified rate-limit error was detected as overflow")
	}
	var typedNil *Error
	for _, err := range []error{typedNil, errors.Join(typedNil, errors.New("ordinary failure"))} {
		if IsContextOverflow(nil, err, 0) {
			t.Fatalf("typed-nil error %v was detected as overflow", err)
		}
	}
}

func TestIsContextOverflowProviderMessages(t *testing.T) {
	for _, message := range []string{
		"prompt is too long: 210000 tokens > 200000 maximum",
		"The input token count (200) exceeds the maximum number of tokens allowed (100)",
		"Prompt contains 100 tokens too large for model with 50 maximum context length",
		"413 status code (no body)",
	} {
		if !IsContextOverflow(nil, errors.New(message), 0) {
			t.Errorf("message %q was not detected", message)
		}
	}
	for _, message := range []string{
		"Throttling error: Too many tokens, please wait",
		"rate limit: too many tokens",
		"too many requests: token limit exceeded",
	} {
		if IsContextOverflow(nil, errors.New(message), 0) {
			t.Errorf("message %q was detected as overflow", message)
		}
	}
}

func TestIsContextOverflowUsageSignals(t *testing.T) {
	tests := []struct {
		name     string
		response *Response
		window   int
		want     bool
	}{
		{name: "silent overflow", response: usageResponse(FinishReasonStop, 101, 0, 0), window: 100, want: true},
		{name: "cache read included", response: usageResponse(FinishReasonStop, 90, 0, 11), window: 100, want: true},
		{name: "exact stop fits", response: usageResponse(FinishReasonStop, 100, 0, 0), window: 100},
		{name: "length fills 99 percent", response: usageResponse(FinishReasonLength, 990, 0, 0), window: 1000, want: true},
		{name: "length produced output", response: usageResponse(FinishReasonLength, 1000, 1, 0), window: 1000},
		{name: "unknown context", response: usageResponse(FinishReasonStop, 101, 0, 0), window: 0},
		{name: "nil response", window: 100},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := IsContextOverflow(test.response, nil, test.window); got != test.want {
				t.Fatalf("IsContextOverflow() = %v, want %v", got, test.want)
			}
		})
	}
}

func usageResponse(reason FinishReason, input, output, cacheRead int) *Response {
	return &Response{FinishReason: reason, Usage: &Usage{InputTokens: input, OutputTokens: output, CacheReadTokens: cacheRead}}
}
