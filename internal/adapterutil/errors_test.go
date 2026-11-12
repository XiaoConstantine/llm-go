package adapterutil

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestRetryAfter(t *testing.T) {
	for _, test := range []struct {
		value string
		want  time.Duration
	}{
		{" 7 ", 7 * time.Second},
		{"9223372036", 9223372036 * time.Second},
		{"9223372037", time.Duration(math.MaxInt64)},
		{"9223372036854775807", time.Duration(math.MaxInt64)},
		{"99999999999999999999", time.Duration(math.MaxInt64)},
		{"+99999999999999999999", time.Duration(math.MaxInt64)},
		{"-99999999999999999999", 0},
		{"99999999999999999999x", 0},
		{"0", 0}, {"-1", 0}, {"bad", 0}, {"7m", 0},
		{"Wed, 01 Jan 2020 00:00:00 GMT", 0},
	} {
		t.Run(test.value, func(t *testing.T) {
			if got := RetryAfter(test.value); got != test.want {
				t.Fatalf("RetryAfter(%q) = %v, want %v", test.value, got, test.want)
			}
		})
	}
	t.Run("trimmed HTTP date", func(t *testing.T) {
		when := time.Now().UTC().Add(time.Hour).Truncate(time.Second)
		before := time.Now()
		got := RetryAfter(" \t" + when.Format(http.TimeFormat) + " \t")
		after := time.Now()
		if got < when.Sub(after) || got > when.Sub(before) {
			t.Fatalf("RetryAfter(date) = %v, want between %v and %v", got, when.Sub(after), when.Sub(before))
		}
	})
}

func TestContextError(t *testing.T) {
	if err := ContextError(context.Background()); err != nil {
		t.Fatalf("live context = %v", err)
	}
	for _, cause := range []error{nil, errors.New("stop"), fmt.Errorf("stop: %w", context.Canceled)} {
		ctx, cancel := context.WithCancelCause(context.Background())
		cancel(cause)
		err := ContextError(ctx)
		if !errors.Is(err, context.Canceled) || cause != nil && !errors.Is(err, cause) {
			t.Fatalf("cancellation = %v, want canceled and %v", err, cause)
		}
		switch {
		case cause == nil:
			if err != context.Canceled {
				t.Fatalf("plain cancellation must be returned as is: %#v", err)
			}
		case errors.Is(cause, context.Canceled):
			if err != cause {
				t.Fatalf("wrapping cause must be returned as is: %#v", err)
			}
		default:
			if err == cause || err == context.Canceled {
				t.Fatalf("distinct cause and cancellation must be joined: %#v", err)
			}
		}
	}
	cause := errors.New("deadline cause")
	ctx, cancel := context.WithDeadlineCause(context.Background(), time.Time{}, cause)
	defer cancel()
	if err := ContextError(ctx); !errors.Is(err, context.DeadlineExceeded) || !errors.Is(err, cause) {
		t.Fatalf("deadline = %v", err)
	}
}

func TestRelabelProviderError(t *testing.T) {
	var typedNil *llm.Error
	for _, err := range []error{nil, typedNil} {
		if got := RelabelProviderError(err, "old", "new"); got != err {
			t.Fatalf("nil error = %#v, want %#v", got, err)
		}
	}
	cause := errors.New("provider cause")
	original := &llm.Error{Kind: llm.KindRateLimit, Provider: "old", Err: cause}
	other := errors.New("cleanup cause")
	err := RelabelProviderError(errors.Join(original, other), "old", "new")
	var got *llm.Error
	if !errors.As(err, &got) || got == original || got.Provider != "new" || got.Kind != original.Kind ||
		!errors.Is(err, cause) || !errors.Is(err, other) || original.Provider != "old" {
		t.Fatalf("relabeled = %#v, original = %#v, error = %v", got, original, err)
	}
	for _, err := range []error{other, fmt.Errorf("wrapper: %w", original), &llm.Error{Provider: "another"}} {
		if got := RelabelProviderError(err, "old", "new"); got != err {
			t.Fatalf("unowned error changed: %#v -> %#v", err, got)
		}
	}
	if got := RelabelProviderError(original, "old", "old"); got != original {
		t.Fatalf("same-provider error changed: %#v", got)
	}
}
