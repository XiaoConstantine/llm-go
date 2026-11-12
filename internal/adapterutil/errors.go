// Package adapterutil contains provider-neutral adapter error mechanics.
// Protocol-specific classifications belong in the adapters, not here.
package adapterutil

import (
	"context"
	"errors"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
)

// ContextError preserves both cancellation identity and a distinct cause,
// including a cause which itself wraps ctx.Err().
func ContextError(ctx context.Context) error {
	err := ctx.Err()
	if err == nil {
		return nil
	}
	cause := context.Cause(ctx)
	if cause == nil {
		return err
	}
	if errors.Is(cause, err) {
		return cause
	}
	return errors.Join(err, cause)
}

// RelabelProviderError copies adapter-owned errors at the client boundary.
// It preserves joined causes and does not unwrap arbitrary error wrappers.
func RelabelProviderError(err error, from, to string) error {
	if err == nil || from == to {
		return err
	}
	relabeled, _ := relabelProviderError(err, from, to)
	return relabeled
}

func relabelProviderError(err error, from, to string) (error, bool) {
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		relabeled := make([]error, len(causes))
		changed := false
		for i, cause := range causes {
			var childChanged bool
			relabeled[i], childChanged = relabelProviderError(cause, from, to)
			changed = changed || childChanged
		}
		if changed {
			return errors.Join(relabeled...), true
		}
		return err, false
	}
	providerErr, ok := err.(*llm.Error)
	if !ok || providerErr == nil || providerErr.Provider != from {
		return err, false
	}
	clone := *providerErr
	clone.Provider = to
	return &clone, true
}

// RetryableHint parses an explicit boolean hint. The adapter decides which
// provider headers, if any, have that meaning.
func RetryableHint(value string) *bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "true":
		value := true
		return &value
	case "false":
		value := false
		return &value
	default:
		return nil
	}
}

// RetryAfter parses an HTTP Retry-After delay or date, saturating durations
// including positive integers too large for int64. Invalid or elapsed delays
// are ignored.
func RetryAfter(value string) time.Duration {
	value = strings.TrimSpace(value)
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil || errors.Is(err, strconv.ErrRange) {
		if seconds <= 0 {
			return 0
		}
		if err != nil {
			// ParseInt may report range overflow before inspecting a suffix.
			for _, digit := range strings.TrimPrefix(value, "+") {
				if digit < '0' || digit > '9' {
					return 0
				}
			}
		}
		if seconds > math.MaxInt64/int64(time.Second) {
			return time.Duration(math.MaxInt64)
		}
		return time.Duration(seconds) * time.Second
	}
	if when, err := http.ParseTime(value); err == nil {
		return max(0, time.Until(when))
	}
	return 0
}
