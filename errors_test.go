package llm

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestError(t *testing.T) {
	cause := context.DeadlineExceeded
	err := &Error{
		Kind:       KindRateLimit,
		Op:         "generate",
		Provider:   "example",
		HTTPStatus: 429,
		RetryAfter: 2 * time.Second,
		Err:        cause,
	}

	if got, want := err.Error(), "example: generate: rate limit (HTTP 429): context deadline exceeded"; got != want {
		t.Fatalf("Error() = %q, want %q", got, want)
	}
	if !errors.Is(err, cause) {
		t.Fatalf("errors.Is(%v, %v) = false, want true", err, cause)
	}

	var modelErr *Error
	if !errors.As(err, &modelErr) {
		t.Fatalf("errors.As(%v, *Error) = false, want true", err)
	}
	if modelErr.RetryAfter != 2*time.Second {
		t.Fatalf("RetryAfter = %v, want %v", modelErr.RetryAfter, 2*time.Second)
	}
}

func TestZeroError(t *testing.T) {
	if got, want := (&Error{}).Error(), "unknown"; got != want {
		t.Fatalf("Error() = %q, want %q", got, want)
	}
}

func TestErrorKindString(t *testing.T) {
	tests := []struct {
		kind ErrorKind
		want string
	}{
		{KindUnknown, "unknown"},
		{KindInvalidRequest, "invalid request"},
		{KindAuthentication, "authentication"},
		{KindPermission, "permission"},
		{KindRateLimit, "rate limit"},
		{KindContextLimit, "context limit"},
		{KindUnsupported, "unsupported"},
		{KindTransport, "transport"},
		{KindProvider, "provider"},
		{KindMalformedResponse, "malformed response"},
		{ErrorKind(255), "error kind 255"},
	}

	for _, test := range tests {
		if got := test.kind.String(); got != test.want {
			t.Errorf("ErrorKind(%d).String() = %q, want %q", test.kind, got, test.want)
		}
	}
}
