package responses

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestJSONModeAllowsToolCallCompletionGenerateAndStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":""}}`,
			`{"type":"response.function_call_arguments.done","output_index":0,"item_id":"fc_1","name":"read","arguments":"{}"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{}"}}`,
			`{"type":"response.completed","response":{"id":"resp","model":"model","status":"completed","output":[]}}`,
		)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityJSON}})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "read", InputSchema: []byte(`{"type":"object"}`)}}, ResponseFormat: llm.ResponseFormatJSON}
	response, err := client.Generate(context.Background(), request)
	if err != nil || response.FinishReason != llm.FinishReasonToolCall || len(response.Message.ToolCalls) != 1 {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
	stream, err := client.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	var terminal llm.Chunk
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if chunk.FinishReason != "" {
			terminal = chunk
		}
	}
	if terminal.FinishReason != llm.FinishReasonToolCall || len(terminal.ToolCalls) != 1 {
		t.Fatalf("terminal chunk = %#v", terminal)
	}
}
