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
	"sync/atomic"
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
			`{"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.6-test","status":"completed","output":[],"usage":{"input_tokens":2,"input_tokens_details":{"cached_tokens":1,"cache_write_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":2},"total_tokens":5}}}`,
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
	if response.Usage == nil || *response.Usage != (llm.Usage{InputTokens: 1, OutputTokens: 3, CacheReadTokens: 1, ReasoningTokens: 2, TotalTokens: 5}) {
		t.Fatalf("Usage = %#v", response.Usage)
	}
}

func TestBackgroundLifecycleUsesStoredNonStreamingResponses(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		switch {
		case request.Method == http.MethodPost && request.URL.Path == "/responses":
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
			if payload["background"] != true || payload["store"] != true || payload["stream"] == true {
				t.Errorf("background/store/stream = %#v/%#v/%#v; body = %s", payload["background"], payload["store"], payload["stream"], body)
			}
			_, _ = io.WriteString(writer, `{"id":"resp_bg","object":"response","created_at":1,"model":"model","status":"queued","output":[]}`)
		case request.Method == http.MethodGet && request.URL.Path == "/responses/resp_bg":
			_, _ = io.WriteString(writer, `{"id":"resp_bg","object":"response","created_at":1,"model":"served-model","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]},{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"read","arguments":"{\"path\":\"file\"}"}],"usage":{"input_tokens":2,"input_tokens_details":{"cached_tokens":1},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":3}}`)
		case request.Method == http.MethodPost && request.URL.Path == "/responses/resp_bg/cancel":
			_, _ = io.WriteString(writer, `{"id":"resp_bg","object":"response","created_at":1,"model":"model","status":"cancelled","output":[]}`)
		default:
			t.Errorf("unexpected request %s %s", request.Method, request.URL.Path)
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()

	client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityTools}})
	if err != nil {
		t.Fatal(err)
	}
	backgroundRequest := llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "work"}}}},
		Tools:    []llm.Tool{{Name: "read", InputSchema: []byte(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}},
	}
	started, err := client.StartBackground(context.Background(), backgroundRequest)
	if err != nil || started.Status != llm.BackgroundQueued || started.Response != nil || started.Handle.ID != "resp_bg" {
		t.Fatalf("StartBackground() = (%#v, %v)", started, err)
	}
	backgroundRequest.Tools[0].InputSchema[0] = '['
	if started.Handle.Tools[0].InputSchema[0] != '{' {
		t.Fatal("StartBackground() handle aliases request tool schema")
	}
	fetched, err := client.FetchBackground(context.Background(), started.Handle)
	if err != nil || fetched.Status != llm.BackgroundCompleted || fetched.Response == nil || fetched.Response.Text() != "done" ||
		fetched.Response.Model != "served-model" || fetched.Response.FinishReason != llm.FinishReasonToolCall ||
		len(fetched.Response.Message.ToolCalls) != 1 || fetched.Response.Message.ToolCalls[0].Name != "read" ||
		fetched.Response.Usage == nil || fetched.Response.Usage.CacheReadTokens != 1 {
		t.Fatalf("FetchBackground() = (%#v, %v)", fetched, err)
	}
	if len(fetched.Response.Message.ProviderData) == 0 {
		t.Fatal("FetchBackground() omitted replayable provider data")
	}
	cancelled, err := client.CancelBackground(context.Background(), started.Handle)
	if err != nil || cancelled.Status != llm.BackgroundCancelled || cancelled.Response != nil {
		t.Fatalf("CancelBackground() = (%#v, %v)", cancelled, err)
	}

	wrong := started.Handle
	wrong.Model = "other"
	if result, err := client.FetchBackground(context.Background(), wrong); result != nil || err == nil {
		t.Fatalf("FetchBackground(wrong handle) = (%#v, %v)", result, err)
	}
	if calls.Load() != 3 {
		t.Fatalf("HTTP calls = %d, want 3", calls.Load())
	}
}

func TestBackgroundFailedResponseReturnsStateAndProviderError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_failed","object":"response","created_at":1,"model":"model","status":"failed","output":[],"error":{"code":"server_error","message":"job failed"}}`)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.StartBackground(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if result == nil || result.Status != llm.BackgroundFailed || result.Handle.ID != "resp_failed" {
		t.Fatalf("StartBackground() result = %#v", result)
	}
	_ = requireResponseError(t, err, llm.KindProvider, "start_background", "job failed")
}

func TestBackgroundRejectsNonStrictJSONBeforeSDKDecoding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_duplicate","object":"response","created_at":1,"model":"model","status":"queued","status":"completed","output":[]}`)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	result, err := client.StartBackground(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if result != nil {
		t.Fatalf("StartBackground() result = %#v, want nil", result)
	}
	_ = requireResponseError(t, err, llm.KindMalformedResponse, "start_background", "strict JSON")
}

func TestBackgroundReturnsObservedHandleWithOutputOrCancellationError(t *testing.T) {
	t.Run("malformed output", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"id":"resp_bad_output","object":"response","created_at":1,"model":"model","status":"completed","output":[{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"read","arguments":"{\"path\":1}"}]}`)
		}))
		defer server.Close()
		client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
			Capabilities: []llm.Capability{llm.CapabilityTools}})
		if err != nil {
			t.Fatal(err)
		}
		result, err := client.StartBackground(context.Background(), llm.Request{
			Messages: []llm.Message{{Role: llm.RoleUser}},
			Tools:    []llm.Tool{{Name: "read", InputSchema: []byte(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}},
		})
		if result == nil || result.Handle.ID != "resp_bad_output" || result.Status != llm.BackgroundCompleted || result.Response != nil {
			t.Fatalf("StartBackground() result = %#v", result)
		}
		_ = requireResponseError(t, err, llm.KindMalformedResponse, "start_background", "validate tool call")
	})

	t.Run("missing output item ID", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"id":"resp_missing_item_id","object":"response","created_at":1,"model":"model","status":"completed","output":[{"type":"function_call","status":"completed","call_id":"call_1","name":"read","arguments":"{}"}]}`)
		}))
		defer server.Close()
		client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
			Capabilities: []llm.Capability{llm.CapabilityTools}})
		if err != nil {
			t.Fatal(err)
		}
		result, err := client.StartBackground(context.Background(), llm.Request{
			Messages: []llm.Message{{Role: llm.RoleUser}},
			Tools:    []llm.Tool{{Name: "read", InputSchema: []byte(`{"type":"object"}`)}},
		})
		if result == nil || result.Handle.ID != "resp_missing_item_id" || result.Status != llm.BackgroundCompleted || result.Response != nil {
			t.Fatalf("StartBackground() result = %#v", result)
		}
		_ = requireResponseError(t, err, llm.KindMalformedResponse, "start_background", "has no ID")
	})

	t.Run("body close error after complete response", func(t *testing.T) {
		body := `{"id":"resp_close_error","object":"response","created_at":1,"model":"model","status":"queued","output":[]}`
		client, err := New(Config{Model: "model", APIKey: "key", BaseURL: "https://example.com/v1",
			HTTPClient: &http.Client{Transport: backgroundRoundTripFunc(func(request *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Request: request,
					Header: http.Header{"Content-Type": []string{"application/json"}},
					Body:   &errorOnCloseBody{Reader: strings.NewReader(body), err: errors.New("close failed")}}, nil
			})}})
		if err != nil {
			t.Fatal(err)
		}
		result, err := client.StartBackground(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
		if err != nil || result == nil || result.Handle.ID != "resp_close_error" || result.Status != llm.BackgroundQueued {
			t.Fatalf("StartBackground() = (%#v, %v)", result, err)
		}
	})

	t.Run("cancellation after response", func(t *testing.T) {
		cause := errors.New("caller stopped waiting")
		ctx, cancel := context.WithCancelCause(context.Background())
		body := `{"id":"resp_cancel_race","object":"response","created_at":1,"model":"model","status":"queued","output":[]}`
		client, err := New(Config{Model: "model", APIKey: "key", BaseURL: "https://example.com/v1",
			HTTPClient: &http.Client{Transport: backgroundRoundTripFunc(func(request *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Request: request,
					Header: http.Header{"Content-Type": []string{"application/json"}},
					Body:   &cancelOnCloseBody{Reader: strings.NewReader(body), cancel: func() { cancel(cause) }}}, nil
			})}})
		if err != nil {
			t.Fatal(err)
		}
		result, err := client.StartBackground(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
		if result == nil || result.Handle.ID != "resp_cancel_race" || result.Status != llm.BackgroundQueued {
			t.Fatalf("StartBackground() = (%#v, %v)", result, err)
		}
		if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
			t.Fatalf("StartBackground() error = %v, want cancellation and cause", err)
		}
	})
}

func TestJSONCapabilityConfigurationAndInfoCopy(t *testing.T) {
	client, err := New(Config{
		Model: "json-model", APIKey: "key",
		Capabilities: []llm.Capability{llm.CapabilityJSON, llm.CapabilityJSON},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	first := client.Info()
	second := client.Info()
	want := []llm.Capability{llm.CapabilityGeneration, llm.CapabilityJSON}
	if fmt.Sprint(first.Capabilities) != fmt.Sprint(want) {
		t.Fatalf("Info().Capabilities = %v, want %v", first.Capabilities, want)
	}
	first.Capabilities[0] = llm.CapabilityAudio
	if fmt.Sprint(second.Capabilities) != fmt.Sprint(want) {
		t.Fatalf("Info returned aliased capabilities: %v", second.Capabilities)
	}
}

func TestGenerateJSONModeUsesJSONObjectWireFormat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
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
		text, ok := payload["text"].(map[string]any)
		if !ok || len(text) != 1 {
			t.Errorf("text = %#v, want exact format object", payload["text"])
		} else if format, ok := text["format"].(map[string]any); !ok || len(format) != 1 || format["type"] != "json_object" {
			t.Errorf("text.format = %#v, want {type:json_object}", text["format"])
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w, responseTextEvents(`{"value":1}`, `{"type":"response.completed","response":{"id":"resp_json","status":"completed","output":[]}}`)...)
	}))
	defer server.Close()
	client, err := New(Config{
		Model: "json-model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityJSON},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), llm.Request{
		Messages:       []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "JSON please"}}}},
		ResponseFormat: llm.ResponseFormatJSON,
	})
	if err != nil || response == nil || response.Text() != `{"value":1}` || response.FinishReason != llm.FinishReasonStop {
		t.Fatalf("Generate() = (%#v, %v)", response, err)
	}
}

func TestJSONModeRequiresCapabilityBeforeIO(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()
	client, err := New(Config{
		Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, ResponseFormat: llm.ResponseFormatJSON}
	response, err := client.Generate(context.Background(), request)
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	_ = requireResponseError(t, err, llm.KindUnsupported, "generate", "JSON capability")
	stream, err := client.Stream(context.Background(), request)
	if stream != nil {
		t.Fatalf("Stream() = %#v, want nil", stream)
	}
	_ = requireResponseError(t, err, llm.KindUnsupported, "stream", "JSON capability")
	if calls.Load() != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls.Load())
	}
}

func TestGenerateJSONModeRejectsMalformedCompletedOutput(t *testing.T) {
	for _, test := range []struct {
		name string
		text string
	}{
		{name: "malformed", text: "{"},
		{name: "trailing", text: `{} trailing`},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				writeSSE(t, w, responseTextEvents(test.text, `{"type":"response.completed","response":{"status":"completed","output":[]}}`)...)
			}))
			defer server.Close()
			client, err := New(Config{
				Model: "json-model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
				Capabilities: []llm.Capability{llm.CapabilityJSON},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			response, err := client.Generate(context.Background(), llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}}, ResponseFormat: llm.ResponseFormatJSON,
			})
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			_ = requireResponseError(t, err, llm.KindMalformedResponse, "generate", "not strict JSON")
		})
	}
}

func TestGenerateJSONModeSkipsValidationForRefusalIncompleteAndError(t *testing.T) {
	t.Run("refusal", func(t *testing.T) {
		events := []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","content":[]}}`,
			`{"type":"response.refusal.delta","output_index":0,"content_index":0,"item_id":"msg_1","delta":"cannot"}`,
			`{"type":"response.refusal.done","output_index":0,"content_index":0,"item_id":"msg_1","refusal":"cannot"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"refusal","refusal":"cannot"}]}}`,
			`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","id":"msg_1","status":"completed","content":[{"type":"refusal","refusal":"cannot"}]}]}}`,
		}
		response, err := generateJSONFromEvents(t, events)
		if err != nil || response == nil || response.FinishReason != llm.FinishReasonStop {
			t.Fatalf("Generate(refusal) = (%#v, %v)", response, err)
		}
	})

	t.Run("incomplete", func(t *testing.T) {
		events := responseTextEvents("not JSON", `{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`)
		response, err := generateJSONFromEvents(t, events)
		if err != nil || response == nil || response.FinishReason != llm.FinishReasonLength {
			t.Fatalf("Generate(incomplete) = (%#v, %v)", response, err)
		}
	})

	t.Run("error", func(t *testing.T) {
		response, err := generateJSONFromEvents(t, []string{`{"type":"response.failed","response":{"error":{"type":"server_error","message":"failed"}}}`})
		if response != nil {
			t.Fatalf("Generate(error) response = %#v, want nil", response)
		}
		_ = requireResponseError(t, err, llm.KindProvider, "generate", "failed")
	})
}

func TestGenerateJSONModeRefusalCannotExemptMalformedText(t *testing.T) {
	for _, test := range []struct {
		name   string
		events func() []string
	}{
		{
			name: "fabricated terminal refusal",
			events: func() []string {
				return responseTextEvents("{", `{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","id":"unrelated","status":"completed","content":[{"type":"refusal","refusal":"cannot"}]}]}}`)
			},
		},
		{
			name: "completed item refusal",
			events: func() []string {
				events := responseTextEvents("{", `{"type":"response.completed","response":{"status":"completed","output":[]}}`)
				events[3] = `{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"refusal","refusal":"cannot"}]}}`
				return events
			},
		},
		{
			name: "invalid refusal content",
			events: func() []string {
				events := responseTextEvents("", `{"type":"response.completed","response":{"status":"completed","output":[]}}`)
				events[3] = `{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"refusal"}]}}`
				return events
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			response, err := generateJSONFromEvents(t, test.events())
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			_ = requireResponseError(t, err, llm.KindMalformedResponse, "generate", "not strict JSON")
		})
	}
}

func TestStreamJSONModeTerminalAndStickyBehavior(t *testing.T) {
	for _, test := range []struct {
		name                 string
		text                 string
		malformed            bool
		strayRefusal         bool
		terminalRefusal      bool
		completedItemRefusal bool
		legitimateRefusal    bool
	}{
		{name: "valid", text: `{"value":1}`},
		{name: "legitimate refusal only", legitimateRefusal: true},
		{name: "malformed", text: "{", malformed: true},
		{name: "trailing", text: `{} trailing`, malformed: true},
		{name: "stray refusal cannot bypass validation", text: "{", malformed: true, strayRefusal: true},
		{name: "terminal refusal cannot exempt malformed text", text: "{", malformed: true, terminalRefusal: true},
		{name: "completed item refusal cannot exempt malformed text", text: "{", malformed: true, completedItemRefusal: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				var events []string
				if test.legitimateRefusal {
					events = []string{
						`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","content":[]}}`,
						`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"refusal","refusal":"cannot"}]}}`,
						`{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","id":"msg_1","status":"completed","content":[{"type":"refusal","refusal":"cannot"}]}]}}`,
					}
				} else {
					events = responseTextEvents(test.text, `{"type":"response.completed","response":{"status":"completed","output":[]}}`)
				}
				if test.strayRefusal {
					events = append([]string{
						events[0],
						`{"type":"response.refusal.delta","output_index":0,"content_index":0,"item_id":"msg_1","delta":"cannot"}`,
					}, events[1:]...)
				}
				if test.terminalRefusal {
					events[len(events)-1] = `{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","id":"unrelated","status":"completed","content":[{"type":"refusal","refusal":"cannot"}]}]}}`
				}
				if test.completedItemRefusal {
					events[len(events)-2] = `{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"refusal","refusal":"cannot"}]}}`
				}
				writeSSE(t, w, events...)
			}))
			defer server.Close()
			client, err := New(Config{
				Model: "json-model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
				Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityJSON},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			stream, err := client.Stream(context.Background(), llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}}, ResponseFormat: llm.ResponseFormatJSON,
			})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			var successfulTerminal bool
			for {
				chunk, recvErr := stream.Recv()
				if recvErr != nil {
					err = recvErr
					break
				}
				successfulTerminal = successfulTerminal || chunk.FinishReason != ""
				for _, event := range chunk.Events {
					successfulTerminal = successfulTerminal || event.Kind == llm.StreamEventDone
				}
			}
			if test.malformed {
				_ = requireResponseError(t, err, llm.KindMalformedResponse, "stream", "not strict JSON")
				if successfulTerminal {
					t.Fatal("Stream emitted a successful terminal chunk for malformed JSON")
				}
				_, sticky := stream.Recv()
				if sticky != err {
					t.Fatalf("sticky Recv() error = %v, want identical %v", sticky, err)
				}
			} else {
				if !errors.Is(err, io.EOF) || !successfulTerminal {
					t.Fatalf("Stream terminal = %v, successful terminal chunk = %t", err, successfulTerminal)
				}
				_, sticky := stream.Recv()
				if !errors.Is(sticky, io.EOF) {
					t.Fatalf("sticky Recv() error = %v, want EOF", sticky)
				}
			}
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		})
	}
}

func TestCompatibleResponsesStreamsRawReasoningAndRequestsEncryptedState(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
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
		include, _ := payload["include"].([]any)
		if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
			t.Errorf("include = %#v", payload["include"])
		}
		if reasoning, _ := payload["reasoning"].(map[string]any); reasoning["summary"] != nil {
			t.Errorf("reasoning = %#v, want no summary request", reasoning)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","content":[]}}`,
			`{"type":"response.reasoning_text.delta","output_index":0,"content_index":0,"item_id":"rs_1","delta":"raw thought"}`,
			`{"type":"response.reasoning_text.done","output_index":0,"content_index":0,"item_id":"rs_1","text":"raw thought"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","content":[{"type":"reasoning_text","text":"raw thought"}]}}`,
			`{"type":"response.completed","response":{"id":"resp_1","model":"grok-test","status":"completed","output":[{"type":"reasoning","id":"rs_1","content":[{"type":"reasoning_text","text":"raw thought"}],"encrypted_content":"secret"}]}}`,
		)
	}))
	defer server.Close()
	client, err := NewWithOptions(Config{
		Model: "grok-test", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
	}, Options{EncryptedReasoning: true})
	if err != nil {
		t.Fatalf("NewWithOptions() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer func() { _ = stream.Close() }()
	var events []llm.StreamEvent
	var summary string
	var providerData jsontext.Value
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv() error = %v", err)
		}
		events = append(events, chunk.Events...)
		summary += chunk.ReasoningSummary
		if len(chunk.ProviderData) != 0 {
			providerData = append(jsontext.Value(nil), chunk.ProviderData...)
		}
	}
	if summary != "" {
		t.Fatalf("ReasoningSummary = %q, want raw reasoning kept separate", summary)
	}
	wantKinds := []llm.StreamEventKind{llm.StreamEventStart, llm.StreamEventReasoningStart, llm.StreamEventReasoningDelta, llm.StreamEventReasoningEnd, llm.StreamEventDone}
	if len(events) != len(wantKinds) {
		t.Fatalf("events = %#v", events)
	}
	for index, kind := range wantKinds {
		if events[index].Kind != kind {
			t.Fatalf("event %d = %#v, want %q", index, events[index], kind)
		}
	}
	if events[2].Delta != "raw thought" || !strings.Contains(string(providerData), `"encrypted_content":"secret"`) {
		t.Fatalf("raw reasoning events/provider data = (%#v, %s)", events, providerData)
	}
}

func TestCompatibilityDisablesAutomaticEncryptedReasoning(t *testing.T) {
	payloads := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
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
		payloads <- payload
		writer.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, writer, `{"type":"response.completed","response":{"id":"response","model":"gpt-5-test","status":"completed","output":[]}}`)
	}))
	defer server.Close()
	client, err := NewWithCompatibility(Config{Model: "gpt-5-test", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()},
		&llm.OpenAIResponsesCompatibility{EncryptedReasoning: llm.CompatibilityDisabled})
	if err != nil {
		t.Fatalf("NewWithOptions() error = %v", err)
	}
	if _, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, ReasoningEffort: llm.ReasoningEffortHigh}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	payload := <-payloads
	if _, exists := payload["include"]; exists {
		t.Fatalf("payload contains include: %#v", payload)
	}
}

func TestClientEnforcesModelCompatibilityBeforeIO(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()
	client, err := NewWithCompatibility(Config{
		Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityTools},
	}, &llm.OpenAIResponsesCompatibility{StrictTools: llm.CompatibilityDisabled})
	if err != nil {
		t.Fatalf("NewWithOptions() error = %v", err)
	}
	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
		Tools:    []llm.Tool{{Name: "tool", InputSchema: jsontext.Value(`{"type":"object"}`), Strict: true}},
	})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported || modelErr.Op != "generate" {
		t.Fatalf("Generate() error = %#v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("provider calls = %d, want zero", calls.Load())
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
	defer func() { _ = stream.Close() }()
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
	defer func() { _ = stream.Close() }()
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

func TestAudioUnsupportedBeforeIO(t *testing.T) {
	configured, err := New(Config{Model: "model", APIKey: "key", Capabilities: []llm.Capability{llm.CapabilityAudio}})
	if configured != nil {
		t.Fatalf("New(audio) client = %#v, want nil", configured)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindInvalidRequest || modelErr.Op != "configure" {
		t.Fatalf("New(audio) error = %#v", err)
	}

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()
	client, err := New(Config{
		Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{
		Kind: llm.PartAudio, Data: []byte("audio"), MediaType: "audio/wav",
	}}}}}
	response, err := client.Generate(context.Background(), request)
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported || modelErr.Op != "generate" ||
		!strings.Contains(err.Error(), "does not support audio") {
		t.Fatalf("Generate() error = %#v", err)
	}
	stream, err := client.Stream(context.Background(), request)
	if stream != nil {
		t.Fatalf("Stream() = %#v, want nil", stream)
	}
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported || modelErr.Op != "stream" ||
		!strings.Contains(err.Error(), "does not support audio") {
		t.Fatalf("Stream() error = %#v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls.Load())
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
	defer func() { _ = stream.Close() }()
	if _, err := stream.Recv(); err != nil {
		t.Fatalf("first Recv() error = %v", err)
	}
	_, err = stream.Recv()
	var responseErr *llm.Error
	if !errors.As(err, &responseErr) || responseErr.Kind != llm.KindMalformedResponse || !strings.Contains(err.Error(), "call_1") {
		t.Fatalf("second Recv() error = %#v, want duplicate-call malformed response", err)
	}
}

func responseTextEvents(text, terminal string) []string {
	encoded, err := jsonv2.Marshal(text)
	if err != nil {
		panic(err)
	}
	quoted := string(encoded)
	return []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","content":[]}}`,
		fmt.Sprintf(`{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg_1","delta":%s}`, quoted),
		fmt.Sprintf(`{"type":"response.output_text.done","output_index":0,"content_index":0,"item_id":"msg_1","text":%s}`, quoted),
		fmt.Sprintf(`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":%s,"annotations":[]}]}}`, quoted),
		terminal,
	}
}

func generateJSONFromEvents(t *testing.T, events []string) (*llm.Response, error) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w, events...)
	}))
	defer server.Close()
	client, err := New(Config{
		Model: "json-model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityJSON},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}}, ResponseFormat: llm.ResponseFormatJSON,
	})
}

func requireResponseError(t *testing.T, err error, kind llm.ErrorKind, op, contains string) *llm.Error {
	t.Helper()
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != kind || modelErr.Op != op || !strings.Contains(err.Error(), contains) {
		t.Fatalf("error = %#v, want kind %q op %q containing %q", err, kind, op, contains)
	}
	return modelErr
}

func writeSSE(t *testing.T, writer io.Writer, events ...string) {
	t.Helper()
	for _, event := range events {
		if _, err := fmt.Fprintf(writer, "data: %s\n\n", event); err != nil {
			t.Fatalf("write SSE: %v", err)
		}
	}
}

type backgroundRoundTripFunc func(*http.Request) (*http.Response, error)

func (function backgroundRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type cancelOnCloseBody struct {
	io.Reader
	cancel func()
}

func (body *cancelOnCloseBody) Close() error {
	body.cancel()
	return nil
}

type errorOnCloseBody struct {
	io.Reader
	err error
}

func (body *errorOnCloseBody) Close() error { return body.err }
