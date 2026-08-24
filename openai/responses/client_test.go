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
	"strings"
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
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
			`{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"item_id":"rs_1","delta":"Checking."}`,
			`{"type":"response.reasoning_summary_text.done","output_index":0,"summary_index":0,"item_id":"rs_1","text":"Checking."}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}`,
			`{"type":"response.output_text.delta","output_index":1,"content_index":0,"item_id":"msg_1","delta":"hello"}`,
			`{"type":"response.output_text.done","output_index":1,"content_index":0,"item_id":"msg_1","text":"hello"}`,
			`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`,
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

func TestGeneratePreservesDoneOnlyText(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","content":[]}}`,
			`{"type":"response.output_text.done","output_index":0,"content_index":0,"item_id":"msg_1","text":"done only"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"done only","annotations":[]}]}}`,
			`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[]}}`,
		)
	}))
	defer server.Close()
	client, err := New(Config{Model: "gpt-test", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.Text() != "done only" {
		t.Fatalf("Generate().Text() = %q", response.Text())
	}
}

func TestStreamEmitsPartialToolArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":""}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{\"path\":\""}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"README.md\"}"}`,
			`{"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_1","name":"read","arguments":"{\"path\":\"README.md\"}"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{\"path\":\"README.md\"}","status":"completed"}}`,
			`{"type":"response.completed","response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[]}}`,
		)
	}))
	defer server.Close()
	client, err := New(Config{
		Model: "gpt-test", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "read"}}}},
		Tools:    []llm.Tool{{Name: "read", InputSchema: jsontext.Value(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer stream.Close()
	var chunks []llm.Chunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv() error = %v", err)
		}
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 5 {
		t.Fatalf("len(chunks) = %d, want 5: %#v", len(chunks), chunks)
	}
	if len(chunks[0].Events) != 2 || chunks[0].Events[0].Kind != llm.StreamEventStart || chunks[0].Events[1].Kind != llm.StreamEventToolCallStart {
		t.Fatalf("start chunk = %#v", chunks[0])
	}
	if chunks[1].Events[0].Kind != llm.StreamEventToolCallDelta || chunks[2].Events[0].Kind != llm.StreamEventToolCallDelta ||
		chunks[1].Events[0].Delta+chunks[2].Events[0].Delta != `{"path":"README.md"}` {
		t.Fatalf("argument delta chunks = %#v, %#v", chunks[1], chunks[2])
	}
	if chunks[3].Events[0].Kind != llm.StreamEventToolCallEnd || chunks[3].Events[0].ToolCall == nil {
		t.Fatalf("tool end chunk = %#v", chunks[3])
	}
	final := chunks[4]
	if final.FinishReason != llm.FinishReasonToolCall || len(final.ToolCalls) != 1 || string(final.ToolCalls[0].Arguments) != `{"path":"README.md"}` ||
		len(final.Events) != 1 || final.Events[0].Kind != llm.StreamEventDone {
		t.Fatalf("final chunk = %#v", final)
	}
}

func TestStreamEndsInterruptedToolCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":""}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{"}`,
			`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`,
		)
	}))
	defer server.Close()
	client, err := New(Config{Model: "gpt-test", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "read", InputSchema: jsontext.Value(`{"type":"object"}`)}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer stream.Close()
	var chunks []llm.Chunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv() error = %v", err)
		}
		chunks = append(chunks, chunk)
	}
	final := chunks[len(chunks)-1]
	if final.FinishReason != llm.FinishReasonLength || len(final.Events) != 2 || final.Events[0].Kind != llm.StreamEventToolCallEnd || final.Events[0].ToolCall != nil || final.Events[1].Kind != llm.StreamEventDone {
		t.Fatalf("final chunk = %#v", final)
	}
}

func TestStreamRejectsInconsistentPartialToolArguments(t *testing.T) {
	for _, test := range []struct {
		name   string
		events []string
		want   string
	}{
		{
			name: "completed text differs",
			events: []string{
				`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","content":[]}}`,
				`{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg_1","delta":"first"}`,
				`{"type":"response.output_text.done","output_index":0,"content_index":0,"item_id":"msg_1","text":"different"}`,
			},
			want: "does not match",
		},
		{
			name: "wrong item ID",
			events: []string{
				`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":""}}`,
				`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"other","delta":"{}"}`,
			},
			want: "does not match",
		},
		{
			name: "completed call differs",
			events: []string{
				`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":""}}`,
				`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{}"}`,
				`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{\"changed\":true}"}}`,
			},
			want: "do not match",
		},
		{
			name: "done differs from deltas",
			events: []string{
				`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":""}}`,
				`{"type":"response.function_call_arguments.delta","output_index":0,"item_id":"fc_1","delta":"{}"}`,
				`{"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_1","name":"read","arguments":"{\"changed\":true}"}`,
			},
			want: "do not match",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				writeSSE(t, w, test.events...)
			}))
			defer server.Close()
			client, err := New(Config{Model: "gpt-test", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "read", InputSchema: jsontext.Value(`{"type":"object"}`)}}})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			for {
				_, err = stream.Recv()
				if err != nil {
					break
				}
			}
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Recv() error = %v, want %q", err, test.want)
			}
			_ = stream.Close()
		})
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

func TestStreamRejectsDuplicateToolCallIDs(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":""}}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_1","name":"read","arguments":""}}`,
		)
	}))
	defer server.Close()

	client, err := New(Config{
		Model: "gpt-test", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
		Tools:    []llm.Tool{{Name: "read", InputSchema: jsontext.Value(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer stream.Close()
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("first Recv() error = %v", err)
	}
	_, err = stream.Recv()
	var responseErr *llm.Error
	if !errors.As(err, &responseErr) || responseErr.Kind != llm.KindMalformedResponse || !strings.Contains(err.Error(), "call_1") {
		t.Fatalf("second Recv() error = %#v, want duplicate-call malformed response", err)
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
