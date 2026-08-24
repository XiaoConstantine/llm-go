package responses

import (
	"context"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestGenerateUsesResponsesProtocol(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			t.Errorf("path = %q, want /v1/responses", request.URL.Path)
		}
		if got := request.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("Authorization = %q", got)
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
			return
		}
		var payload map[string]any
		if err := jsonv2.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode body: %v", err)
			return
		}
		if payload["model"] != "gpt-5.6-test" || payload["stream"] != true || payload["store"] != false {
			t.Errorf("model/stream/store = %#v/%#v/%#v", payload["model"], payload["stream"], payload["store"])
		}
		if payload["max_output_tokens"] != float64(321) || payload["top_p"] != 0.75 {
			t.Errorf("max_output_tokens/top_p = %#v/%#v", payload["max_output_tokens"], payload["top_p"])
		}
		reasoning, _ := payload["reasoning"].(map[string]any)
		if reasoning["summary"] != "auto" || reasoning["effort"] != "high" {
			t.Errorf("reasoning = %#v, want summary auto and effort high", payload["reasoning"])
		}
		tools, _ := payload["tools"].([]any)
		if len(tools) != 1 || tools[0].(map[string]any)["name"] != "read" {
			t.Errorf("tools = %#v", tools)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.reasoning_summary_text.delta","delta":"Checking."}`,
			`{"type":"response.output_text.delta","delta":"hello"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`,
			`{"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.6-test","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
		)
	}))
	defer server.Close()

	client, err := New(Config{
		Model:        "gpt-5.6-test",
		APIKey:       "secret",
		BaseURL:      server.URL + "/v1",
		HTTPClient:   server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityTools},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	topP := 0.75
	response, err := client.Generate(context.Background(), llm.Request{
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
		Tools:           []llm.Tool{{Name: "read", InputSchema: jsontext.Value(`{"type":"object"}`), Strict: true}},
		ReasoningEffort: llm.ReasoningEffortHigh,
		MaxOutputTokens: 321,
		TopP:            &topP,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.ID != "resp_1" || response.Text() != "hello" || response.ReasoningSummary != "Checking." || response.FinishReason != llm.FinishReasonStop {
		t.Fatalf("Generate() = %#v", response)
	}
	if response.Usage == nil || *response.Usage != (llm.Usage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5}) {
		t.Fatalf("Usage = %#v", response.Usage)
	}
}

func TestNewRequiresAPIKey(t *testing.T) {
	client, err := New(Config{Model: "gpt-5.6-test"})
	if client != nil || err == nil {
		t.Fatalf("New() = %#v, %v", client, err)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindInvalidRequest || modelErr.Op != "configure" {
		t.Fatalf("New() error = %#v", err)
	}
}

func TestSupportsReasoning(t *testing.T) {
	for _, test := range []struct {
		model string
		want  bool
	}{
		{model: "gpt-5.6-luna", want: true},
		{model: "o4-mini", want: true},
		{model: "gpt-4o", want: false},
	} {
		if got := supportsReasoning(test.model); got != test.want {
			t.Errorf("supportsReasoning(%q) = %t, want %t", test.model, got, test.want)
		}
	}
}

func TestGenerateClassifiesAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"type":"invalid_request_error","code":"bad_schema","message":"bad schema"}}`)
	}))
	defer server.Close()
	client, err := New(Config{Model: "gpt-5.6-test", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindInvalidRequest || modelErr.HTTPStatus != http.StatusBadRequest {
		t.Fatalf("Generate() error = %#v", err)
	}
}

func writeSSE(t *testing.T, writer io.Writer, events ...string) {
	t.Helper()
	for _, event := range events {
		if _, err := fmt.Fprintf(writer, "data: %s\n\n", event); err != nil {
			t.Fatalf("write SSE: %v", err)
		}
	}
}
