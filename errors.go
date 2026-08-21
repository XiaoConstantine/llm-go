package llm

import (
	"fmt"
	"strings"
	"time"
)

// ErrorKind classifies an error independently of a provider's wire format. The
// constants below are the complete set of valid kinds. Implementations must use
// the most specific applicable kind. The zero value is KindUnknown.
type ErrorKind uint8

const (
	// KindUnknown identifies an error that has no more specific classification.
	KindUnknown ErrorKind = iota
	// KindInvalidRequest identifies an invalid request without a more specific
	// classification, whether detected locally or rejected by a provider.
	KindInvalidRequest
	// KindAuthentication identifies missing or invalid credentials.
	KindAuthentication
	// KindPermission identifies a request forbidden for valid credentials.
	KindPermission
	// KindRateLimit identifies provider throttling.
	KindRateLimit
	// KindContextLimit identifies a request that does not fit within a model's
	// context limit.
	KindContextLimit
	// KindUnsupported identifies a capability the implementation cannot honor.
	KindUnsupported
	// KindTransport identifies a failure communicating with a provider.
	KindTransport
	// KindProvider identifies another provider-side failure.
	KindProvider
	// KindMalformedResponse identifies a response that cannot be decoded or
	// violates the provider protocol.
	KindMalformedResponse
)

// String returns a stable name for k.
func (k ErrorKind) String() string {
	switch k {
	case KindUnknown:
		return "unknown"
	case KindInvalidRequest:
		return "invalid request"
	case KindAuthentication:
		return "authentication"
	case KindPermission:
		return "permission"
	case KindRateLimit:
		return "rate limit"
	case KindContextLimit:
		return "context limit"
	case KindUnsupported:
		return "unsupported"
	case KindTransport:
		return "transport"
	case KindProvider:
		return "provider"
	case KindMalformedResponse:
		return "malformed response"
	default:
		return fmt.Sprintf("error kind %d", k)
	}
}

// Error describes a provider-neutral model error. Provider and Op identify its
// source when known. HTTPStatus is zero when unavailable; provider-native codes
// remain available through Err. RetryAfter is positive when a provider advised
// a delay. Err retains the underlying cause, including context cancellation and
// deadline errors.
type Error struct {
	Kind       ErrorKind
	Op         string
	Provider   string
	HTTPStatus int
	RetryAfter time.Duration
	Err        error
}

// Error returns a concise description of e.
func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}

	var parts []string
	if e.Provider != "" {
		parts = append(parts, e.Provider)
	}
	if e.Op != "" {
		parts = append(parts, e.Op)
	}
	parts = append(parts, e.Kind.String())

	message := strings.Join(parts, ": ")
	if e.HTTPStatus != 0 {
		message += fmt.Sprintf(" (HTTP %d)", e.HTTPStatus)
	}
	if e.Err != nil {
		message += ": " + e.Err.Error()
	}
	return message
}

// Unwrap returns the underlying cause.
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}
