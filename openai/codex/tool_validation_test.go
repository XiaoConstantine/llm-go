package codex

import (
	json "encoding/json/v2"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestSchemaInvalidToolArgumentsPreserveCodexResponseAndReplay(t *testing.T) {
	wireRequests := make(chan map[string]any, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.UnmarshalRead(r.Body, &payload); err != nil {
			t.Error(err)
			return
		}
		wireRequests <- payload
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
			`{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"history_retrieve","arguments":"{\"limit\":14000}"}}`,
			`{"type":"response.completed","response":{"id":"resp_1","model":"model","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[]}],"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14}}}`,
		)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", AccessToken: "token", AccountID: "account", BaseURL: server.URL, HTTPClient: server.Client(), Transport: TransportSSE, Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}})
	if err != nil {
		t.Fatal(err)
	}
	request := textRequest("retrieve history")
	request.Tools = []llm.Tool{{Name: "history_retrieve", InputSchema: []byte(`{"type":"object","properties":{"limit":{"type":"integer","maximum":6144}},"required":["limit"],"additionalProperties":false}`)}}
	var firstWire map[string]any
	for _, streaming := range []bool{false, true} {
		var response *llm.Response
		var err error
		if streaming {
			stream, streamErr := client.Stream(t.Context(), request)
			if streamErr != nil {
				t.Fatal(streamErr)
			}
			response, err = llm.CollectStructural(stream)
		} else {
			response, err = client.Generate(t.Context(), request)
		}
		if err != nil {
			t.Fatalf("streaming=%t: %v", streaming, err)
		}
		wire := <-wireRequests
		if firstWire == nil {
			firstWire = wire
		}
		if !reflect.DeepEqual(wire, firstWire) {
			t.Fatalf("streaming=%t changed provider request: %#v; want %#v", streaming, wire, firstWire)
		}
		if response.ID != "resp_1" || response.FinishReason != llm.FinishReasonToolCall || response.Usage == nil || response.Usage.TotalTokens != 14 || len(response.Message.ToolCalls) != 1 || response.Message.ToolCalls[0].ID != "call_1" || string(response.Message.ToolCalls[0].Arguments) != `{"limit":14000}` {
			t.Fatalf("incomplete response: %#v", response)
		}
		if err := llm.ValidateToolCalls(request.Tools, response.Message.ToolCalls); err == nil {
			t.Fatal("explicit validation accepted oversized limit")
		}
		// Schema-invalid arguments remain replayable alongside corrective feedback.
		replay := request
		replay.Messages = []llm.Message{response.Message, {Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call_1", Name: "history_retrieve", IsError: true, Content: []llm.Part{{Text: "limit must not exceed 6144"}}}}}}
		if err := replay.Validate(); err != nil {
			t.Fatal(err)
		}
		encoded, err := requestToWire("generate", "model", replay)
		if err != nil {
			t.Fatal(err)
		}
		data, err := json.Marshal(encoded)
		if err != nil {
			t.Fatal(err)
		}
		var body struct {
			Input []map[string]any `json:"input"`
		}
		if err := json.Unmarshal(data, &body); err != nil {
			t.Fatal(err)
		}
		wantCall := map[string]any{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "history_retrieve", "arguments": `{"limit":14000}`}
		if len(body.Input) != 3 || body.Input[0]["type"] != "reasoning" || body.Input[0]["encrypted_content"] != "opaque" || !reflect.DeepEqual(body.Input[1], wantCall) || body.Input[2]["type"] != "function_call_output" {
			t.Fatalf("replay input = %s", data)
		}
	}
}
