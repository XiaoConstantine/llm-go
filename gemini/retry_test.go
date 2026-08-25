package gemini

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestAttemptHookCannotOverrideGeminiAPIKey(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", APIKey: "real-key", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	retrying, err := llm.WithRetry(client, llm.RetryPolicy{MaxAttempts: 1, Hook: func(context.Context, llm.Attempt) (http.Header, error) {
		return http.Header{"X-Goog-Api-Key": {"replacement-key"}}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	response, err := retrying.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if response != nil || err == nil || !strings.Contains(err.Error(), "reserved") {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
	if calls.Load() != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls.Load())
	}
}
