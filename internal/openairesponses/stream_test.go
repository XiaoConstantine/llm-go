package openairesponses_test

import (
	"context"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/openairesponses"
	"github.com/XiaoConstantine/llm-go/openai/azure"
	"github.com/XiaoConstantine/llm-go/openai/codex"
	"github.com/XiaoConstantine/llm-go/openai/responses"
	sdkresponses "github.com/openai/openai-go/v3/responses"
)

func TestOutputItemStatusAcrossAdapters(t *testing.T) {
	caps := []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}
	factories := map[string]func(*httptest.Server) (llm.Generator, error){
		"responses": func(s *httptest.Server) (llm.Generator, error) {
			return responses.New(responses.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		},
		"azure": func(s *httptest.Server) (llm.Generator, error) {
			return azure.New(azure.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		},
		"codex": func(s *httptest.Server) (llm.Generator, error) {
			return codex.New(codex.Config{Model: "model", AccessToken: "token", AccountID: "account", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		},
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "read"}}}},
		Tools: []llm.Tool{{Name: "read", InputSchema: jsontext.Value(`{"type":"object"}`)}}}
	for _, test := range []struct {
		name, kind, status, arguments, terminal string
		wantError, withoutAdded                 bool
	}{
		{"truncated call", "function_call", "incomplete", `{"q":`, "incomplete", false, false},
		{"nonterminal call", "function_call", "in_progress", `{"q":`, "incomplete", false, false},
		{"nonexecutable valid arguments", "function_call", "incomplete", `{}`, "incomplete", false, false},
		{"truncated message", "message", "incomplete", "", "incomplete", false, false},
		{"completed malformed call", "function_call", "completed", `{"q":`, "completed", true, false},
		{"incomplete item in completed response", "function_call", "incomplete", `{}`, "completed", true, false},
		{"unknown item status", "function_call", "unknown", `{}`, "incomplete", true, false},
		{"incomplete call without added", "function_call", "incomplete", `{"q":`, "incomplete", false, true},
		{"hosted in progress", "web_search_call", "in_progress", "", "completed", false, false},
		{"hosted incomplete", "web_search_call", "incomplete", "", "completed", false, true},
		{"hosted completed", "web_search_call", "completed", "", "completed", false, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			message := `{"type":"message","id":"msg_0","role":"assistant","status":"completed","content":[{"type":"output_text","text":"answer","annotations":[]}]}`
			item := fmt.Sprintf(`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","status":%q,"arguments":%q}`, test.status, test.arguments)
			added := `{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","status":"in_progress","arguments":""}`
			delta := fmt.Sprintf(`{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc_1","delta":%q}`, test.arguments)
			done := fmt.Sprintf(`{"type":"response.function_call_arguments.done","output_index":1,"item_id":"fc_1","arguments":%q}`, test.arguments)
			wantText := "answer"
			wantFinish := llm.FinishReasonLength
			if test.terminal == "completed" {
				wantFinish = llm.FinishReasonStop
			}
			wantReplay := test.kind == "web_search_call" && test.status == "completed"
			switch test.kind {
			case "message":
				item = fmt.Sprintf(`{"type":"message","id":"msg_1","role":"assistant","status":%q,"content":[{"type":"output_text","text":"partial","annotations":[]}]}`, test.status)
				added = `{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}`
				delta = `{"type":"response.output_text.delta","output_index":1,"content_index":0,"item_id":"msg_1","delta":"partial"}`
				done = `{"type":"response.output_text.done","output_index":1,"content_index":0,"item_id":"msg_1","text":"partial"}`
				wantText += "partial"
			case "web_search_call":
				item = fmt.Sprintf(`{"type":"web_search_call","status":%q}`, test.status)
				added = `{"type":"web_search_call","status":"in_progress"}`
			}
			terminal := fmt.Sprintf(`{"id":"resp","status":%q,"output":[%s,%s],"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}`, test.terminal, message, item)
			check := func(t *testing.T, response *llm.Response, err error) {
				t.Helper()
				if test.wantError {
					var modelErr *llm.Error
					if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse {
						t.Fatalf("error = %v, want malformed response", err)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if response.Text() != wantText || response.FinishReason != wantFinish || len(response.Message.ToolCalls) != 0 || (len(response.Message.ProviderData) != 0) != wantReplay || response.Usage == nil || response.Usage.TotalTokens != 7 {
					t.Fatalf("response = %#v", response)
				}
			}
			t.Run("full response", func(t *testing.T) {
				var wire sdkresponses.Response
				if err := wire.UnmarshalJSON([]byte(terminal)); err != nil {
					t.Fatal(err)
				}
				codec := openairesponses.Codec{Provider: "openai", ProviderDataAPI: "openai-responses", MaxProviderDataBytes: 1 << 20}
				response, err := codec.FullResponse("generate", "model", request.Tools, llm.ResponseFormatText, wire)
				check(t, response, err)
			})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				events := []string{
					`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_0","role":"assistant","status":"in_progress","content":[]}}`,
					`{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg_0","delta":"answer"}`,
					`{"type":"response.output_item.done","output_index":0,"item":` + message + `}`,
				}
				if !test.withoutAdded {
					events = append(events, `{"type":"response.output_item.added","output_index":1,"item":`+added+`}`)
					if test.kind != "web_search_call" {
						events = append(events, delta, done)
					}
				}
				events = append(events,
					`{"type":"response.output_item.done","output_index":1,"item":`+item+`}`,
					`{"type":"response.`+test.terminal+`","response":`+terminal+`}`)
				for _, event := range events {
					_, _ = fmt.Fprintf(w, "data: %s\n\n", event)
				}
			}))
			defer server.Close()
			for name, factory := range factories {
				for _, mode := range []string{"generate", "stream"} {
					t.Run(name+"/"+mode, func(t *testing.T) {
						client, err := factory(server)
						if err != nil {
							t.Fatal(err)
						}
						if mode == "generate" {
							response, err := client.Generate(context.Background(), request)
							check(t, response, err)
							return
						}
						stream, err := client.Stream(context.Background(), request)
						if err != nil {
							t.Fatal(err)
						}
						defer func() { _ = stream.Close() }()
						response := &llm.Response{}
						toolStarts, toolEnds := 0, 0
						for {
							chunk, recvErr := stream.Recv()
							if recvErr != nil {
								if recvErr != io.EOF {
									err = recvErr
								}
								break
							}
							openairesponses.MergeChunk(response, chunk)
							for _, event := range chunk.Events {
								if event.Kind == llm.StreamEventToolCallStart {
									toolStarts++
								}
								if event.Kind == llm.StreamEventToolCallEnd {
									toolEnds++
									if toolStarts != 1 {
										t.Fatalf("tool end without exactly one start: starts=%d", toolStarts)
									}
									if !test.wantError && event.ToolCall != nil {
										t.Fatal("incomplete tool end contains executable call")
									}
								}
							}
						}
						check(t, response, err)
						if !test.wantError && test.kind == "function_call" && toolEnds != 1 {
							t.Fatalf("tool ends = %d, want 1", toolEnds)
						}
					})
				}
			}
		})
	}
}
