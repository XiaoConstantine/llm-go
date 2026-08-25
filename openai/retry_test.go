package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestRetryAttemptHeadersReachTransport(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		attempt := calls.Add(1)
		if got := request.Header.Get("X-Attempt"); got != strconv.Itoa(int(attempt)) {
			t.Errorf("X-Attempt = %q, want %d", got, attempt)
		}
		writer.Header().Set("Content-Type", "application/json")
		if attempt == 1 {
			writer.WriteHeader(http.StatusServiceUnavailable)
			_, _ = io.WriteString(writer, `{"error":{"message":"retry"}}`)
			return
		}
		_, _ = io.WriteString(writer, `{"id":"id","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	retrying, err := llm.WithRetry(client, llm.RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Nanosecond, MaxBackoff: time.Nanosecond,
		Hook: func(_ context.Context, attempt llm.Attempt) (http.Header, error) {
			return http.Header{"X-Attempt": {strconv.Itoa(attempt.Number)}}, nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	response, err := retrying.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil || response.Text() != "ok" || calls.Load() != 2 {
		t.Fatalf("Generate() = %#v, %v; calls %d", response, err, calls.Load())
	}
}
