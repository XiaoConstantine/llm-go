// Package openaicommon contains rules shared by OpenAI protocol adapters.
// These are not provider-neutral defaults.
package openaicommon

import (
	"net/http"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
)

// ClassifyStatus gives HTTP authentication, permission, and throttling signals
// precedence over context-limit text in an error body.
func ClassifyStatus(status int, errorType, code, message string) llm.ErrorKind {
	switch status {
	case http.StatusUnauthorized:
		return llm.KindAuthentication
	case http.StatusForbidden:
		return llm.KindPermission
	case http.StatusTooManyRequests:
		return llm.KindRateLimit
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		if IsContextLimit(errorType, code, message) {
			return llm.KindContextLimit
		}
		return llm.KindInvalidRequest
	case http.StatusNotFound, http.StatusMethodNotAllowed:
		return llm.KindInvalidRequest
	default:
		return llm.KindProvider
	}
}

// ClassifyError classifies a Chat Completions in-band error without an HTTP status.
// Unknown codes must not acquire semantics merely by containing a known word.
func ClassifyError(errorType, code, message string) llm.ErrorKind {
	if IsContextLimit(errorType, code, message) {
		return llm.KindContextLimit
	}
	switch strings.ToLower(strings.TrimSpace(code)) {
	case "invalid_api_key", "authentication_error":
		return llm.KindAuthentication
	case "insufficient_permissions":
		return llm.KindPermission
	case "rate_limit_exceeded", "insufficient_quota":
		return llm.KindRateLimit
	}
	switch strings.ToLower(strings.TrimSpace(errorType)) {
	case "authentication_error":
		return llm.KindAuthentication
	case "permission_error":
		return llm.KindPermission
	case "rate_limit_error":
		return llm.KindRateLimit
	case "invalid_request_error":
		return llm.KindInvalidRequest
	}
	return llm.KindProvider
}

// ClassifyResponsesError preserves Responses' legacy substring aliases across
// the combined type and code. Context limit takes precedence, followed by
// authentication, permission, rate limit, and invalid request, regardless of
// which field supplies the signal. Chat uses its own classifier.
func ClassifyResponsesError(errorType, code, message string) llm.ErrorKind {
	if IsContextLimit(errorType, code, message) {
		return llm.KindContextLimit
	}
	detail := strings.ToLower(errorType + " " + code)
	switch {
	case strings.Contains(detail, "auth"), strings.Contains(detail, "invalid_api_key"):
		return llm.KindAuthentication
	case strings.Contains(detail, "permission"), strings.Contains(detail, "forbidden"):
		return llm.KindPermission
	case strings.Contains(detail, "rate_limit"), strings.Contains(detail, "usage_limit"), strings.Contains(detail, "quota"):
		return llm.KindRateLimit
	case strings.Contains(detail, "invalid_request"):
		return llm.KindInvalidRequest
	default:
		return llm.KindProvider
	}
}

// IsContextLimit recognizes the context-window failures used by these APIs.
func IsContextLimit(errorType, code, message string) bool {
	detail := strings.ToLower(errorType + " " + code + " " + message)
	return strings.Contains(detail, "context_length_exceeded") ||
		strings.Contains(detail, "maximum context length") ||
		strings.Contains(detail, "context window")
}
