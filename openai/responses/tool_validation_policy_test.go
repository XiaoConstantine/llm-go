package responses

import (
	json "encoding/json/v2"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestBackgroundLeavesArgumentValidationToExecutionOwner(t *testing.T) {
	for _, immediate := range []bool{false, true} {
		t.Run(fmt.Sprintf("immediate=%t", immediate), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if !immediate && r.Method == http.MethodPost && r.URL.Path == "/responses" {
					_, _ = io.WriteString(w, `{"id":"resp_bg","object":"response","created_at":1,"model":"model","status":"queued","output":[]}`)
					return
				}
				// Cancel may race with completion and return completed output too.
				_, _ = io.WriteString(w, `{"id":"resp_bg","object":"response","created_at":1,"model":"model","status":"completed","output":[{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"lookup","arguments":"{\"limit\":14000}"}]}`)
			}))
			defer server.Close()
			client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityTools}})
			if err != nil {
				t.Fatal(err)
			}
			request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "lookup"}}}}, Tools: []llm.Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","properties":{"limit":{"type":"integer","maximum":6144}}}`)}}}
			check := func(result *llm.BackgroundResult, err error) {
				t.Helper()
				if err != nil || result == nil || result.Status != llm.BackgroundCompleted || result.Response == nil || len(result.Response.Message.ToolCalls) != 1 {
					t.Fatalf("result = %#v, %v", result, err)
				}
				call := result.Response.Message.ToolCalls[0]
				if string(call.Arguments) != `{"limit":14000}` {
					t.Fatalf("call = %+v", call)
				}
				if err := llm.ValidateToolCalls(result.Handle.Tools, result.Response.Message.ToolCalls); err == nil {
					t.Fatal("explicit validation accepted oversized limit")
				}
			}
			started, err := client.StartBackground(t.Context(), request)
			if immediate {
				check(started, err)
				return
			}
			if err != nil || started == nil || started.Status != llm.BackgroundQueued {
				t.Fatalf("start = %#v, %v", started, err)
			}
			encoded, err := json.Marshal(started.Handle)
			if err != nil {
				t.Fatal(err)
			}
			var handle llm.BackgroundHandle
			if err := json.Unmarshal(encoded, &handle); err != nil {
				t.Fatal(err)
			}
			check(client.FetchBackground(t.Context(), handle))
			check(client.CancelBackground(t.Context(), handle))
		})
	}
}
