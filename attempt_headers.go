package llm

import (
	"context"
	"net/http"
)

type attemptHeadersKey struct{}

func withAttemptHeaders(ctx context.Context, headers http.Header) context.Context {
	if len(headers) == 0 {
		return ctx
	}
	return context.WithValue(ctx, attemptHeadersKey{}, headers.Clone())
}

// AttemptHeaders returns an owned copy of headers supplied by the current
// WithRetry AttemptHook. It returns nil outside an attempt carrying headers.
// Custom Generator implementations may read these values for their validated
// transport request; mutating the result never changes retry or sibling state.
func AttemptHeaders(ctx context.Context) http.Header {
	headers, _ := ctx.Value(attemptHeadersKey{}).(http.Header)
	return headers.Clone()
}
