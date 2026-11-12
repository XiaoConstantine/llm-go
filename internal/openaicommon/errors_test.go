package openaicommon

import (
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestClassifyResponsesError(t *testing.T) {
	for _, test := range []struct {
		value string
		kind  llm.ErrorKind
	}{
		{"authentication_error", llm.KindAuthentication}, {"invalid_api_key", llm.KindAuthentication},
		{"unauthorized", llm.KindAuthentication}, {"permission_denied", llm.KindPermission},
		{"permission_error", llm.KindPermission}, {"insufficient_permissions", llm.KindPermission},
		{"forbidden", llm.KindPermission}, {"rate_limit_error", llm.KindRateLimit},
		{"rate_limit_reached", llm.KindRateLimit}, {"rate_limit_exceeded", llm.KindRateLimit},
		{"usage_limit_reached", llm.KindRateLimit}, {"insufficient_quota", llm.KindRateLimit},
		{"quota_exceeded", llm.KindRateLimit}, {"invalid_request", llm.KindInvalidRequest},
		{"invalid_request_error", llm.KindInvalidRequest}, {"UNAUTHORIZED", llm.KindAuthentication},
		// Retain legacy substring compatibility; absence from the docs does not
		// prove an alias cannot arrive through an OpenAI-compatible endpoint.
		{"server_authorization_cache_failed", llm.KindAuthentication},
		{"rate_limit_probe_failed", llm.KindRateLimit}, {"future_error", llm.KindProvider},
	} {
		t.Run(test.value, func(t *testing.T) {
			for _, fields := range [][2]string{{test.value, ""}, {"", test.value}} {
				if got := ClassifyResponsesError(fields[0], fields[1], "failed"); got != test.kind {
					t.Errorf("type=%q code=%q: got %v, want %v", fields[0], fields[1], got, test.kind)
				}
			}
		})
	}
	for _, test := range []struct {
		errorType, code, message string
		want                     llm.ErrorKind
	}{
		{"authentication_error", "insufficient_quota", "", llm.KindAuthentication},
		{"permission_denied", "unauthorized", "", llm.KindAuthentication},
		{"rate_limit_error", "invalid_api_key", "", llm.KindAuthentication},
		{"invalid_request_error", "usage_limit_reached", "", llm.KindRateLimit},
		{"invalid_request_error", "rate_limit_exceeded", "", llm.KindRateLimit},
		{"invalid_request_error", "insufficient_quota", "", llm.KindRateLimit},
		{"invalid_request_error", "invalid_api_key", "", llm.KindAuthentication},
		{"future_error", "usage_limit_reached", "", llm.KindRateLimit},
		{"authentication_error", "", "maximum context length", llm.KindContextLimit},
	} {
		t.Run(test.errorType+"/"+test.code+"/"+test.message, func(t *testing.T) {
			if got := ClassifyResponsesError(test.errorType, test.code, test.message); got != test.want {
				t.Errorf("classification = %v, want %v", got, test.want)
			}
		})
	}
}

func TestClassifyChatErrorKeepsCodePrecedence(t *testing.T) {
	for _, test := range []struct {
		errorType, code string
		want            llm.ErrorKind
	}{
		{"authentication_error", "insufficient_quota", llm.KindRateLimit},
		{"", "authentication_error", llm.KindAuthentication},
		{"", "server_authorization_cache_failed", llm.KindProvider},
		{"", "rate_limit_probe_failed", llm.KindProvider},
	} {
		if got := ClassifyError(test.errorType, test.code, ""); got != test.want {
			t.Errorf("type=%q code=%q: got %v, want %v", test.errorType, test.code, got, test.want)
		}
	}
}
