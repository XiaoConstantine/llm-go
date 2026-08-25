package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestStreamTranslatesTextCompletion(t *testing.T) {
	type requestRecord struct {
		path       string
		forceQuery bool
		header     http.Header
		body       []byte
	}
	records := make(chan requestRecord, 1)
	server := httptest.NewTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			panic(err)
		}
		records <- requestRecord{
			path:       request.URL.Path,
			forceQuery: request.URL.ForceQuery,
			header:     request.Header.Clone(),
			body:       body,
		}
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = io.WriteString(w, strings.Join([]string{
			": heartbeat\n\n",
			"data: {\"id\":\"chat-1\",\"model\":\"served-model\",\n",
			"data: \"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
			"data: {\"id\":\"chat-1\",\"model\":\"served-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
			"data: {\"id\":\"chat-1\",\"model\":\"served-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
			"data: {\"id\":\"chat-1\",\"model\":\"served-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
			"data: {\"id\":\"chat-1\",\"model\":\"served-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":1},\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n",
			"data: [DONE]\n\n",
		}, ""))
	}))

	client, err := New(Config{
		Model:        "request-model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		APIKey:       "ignored-key",
		BaseURL:      server.URL + "/proxy/v1?",
		HTTPClient:   server.Client(),
		Headers: http.Header{
			"authorization": {"Basic stream-owned"},
			"x-trace":       {"stream-trace"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if !errors.Is(terminal, io.EOF) {
		t.Fatalf("Recv() terminal = %v, want io.EOF", terminal)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	var text strings.Builder
	var finish llm.FinishReason
	var usage *llm.Usage
	for i, chunk := range chunks {
		if chunk.ID != "chat-1" || chunk.Model != "served-model" {
			t.Fatalf("chunk %d metadata = (%q, %q)", i, chunk.ID, chunk.Model)
		}
		for _, part := range chunk.Content {
			text.WriteString(part.Text)
		}
		if chunk.FinishReason != "" {
			finish = chunk.FinishReason
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if text.String() != "hello" || finish != llm.FinishReasonStop {
		t.Fatalf("assembled output = (%q, %q), want (hello, stop)", text.String(), finish)
	}
	if usage == nil || *usage != (llm.Usage{InputTokens: 1, OutputTokens: 1, CacheReadTokens: 1, ReasoningTokens: 1, TotalTokens: 3}) {
		t.Fatalf("assembled usage = %#v", usage)
	}
	assertStreamEventKinds(t, chunks, [][]llm.StreamEventKind{
		{llm.StreamEventStart},
		{llm.StreamEventTextStart, llm.StreamEventTextDelta},
		{llm.StreamEventTextDelta},
		{llm.StreamEventTextEnd, llm.StreamEventDone},
		nil,
	})
	if chunks[1].Events[1].Delta+chunks[2].Events[0].Delta != "hello" || chunks[3].Events[0].Content != "hello" {
		t.Fatalf("text events = %#v", chunks)
	}

	record := <-records
	if record.path != "/proxy/v1/chat/completions" || !record.forceQuery {
		t.Fatalf("request endpoint = (%q, ForceQuery=%v)", record.path, record.forceQuery)
	}
	if record.header.Get("Accept") != "text/event-stream" || record.header.Get("Authorization") != "Basic stream-owned" ||
		record.header.Get("X-Trace") != "stream-trace" {
		t.Fatalf("request headers = %#v", record.header)
	}
	if record.header.Get("User-Agent") != "OpenAI/Go 3.52.0" || record.header.Get("X-Stainless-Retry-Count") != "0" {
		t.Fatalf("SDK headers = %#v", record.header)
	}
	var payload map[string]any
	if err := jsonv2.Unmarshal(record.body, &payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if payload["stream"] != true {
		t.Fatalf("stream = %#v", payload["stream"])
	}
	options, ok := payload["stream_options"].(map[string]any)
	if !ok || options["include_usage"] != true {
		t.Fatalf("stream_options = %#v", payload["stream_options"])
	}
}

func TestStreamTranslatesCompatibleReasoning(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, strings.Join([]string{
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"reasoning_content":"ing"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n",
			"data: [DONE]\n\n",
		}, ""))
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityStreaming}, BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if terminal != io.EOF {
		t.Fatalf("Recv() terminal = %v, want EOF", terminal)
	}
	assertStreamEventKinds(t, chunks, [][]llm.StreamEventKind{
		{llm.StreamEventStart},
		{llm.StreamEventReasoningStart, llm.StreamEventReasoningDelta},
		{llm.StreamEventReasoningDelta},
		{llm.StreamEventReasoningEnd, llm.StreamEventTextStart, llm.StreamEventTextDelta},
		{llm.StreamEventTextEnd, llm.StreamEventDone},
	})
	var reasoningEvents, text strings.Builder
	var providerData []byte
	for _, chunk := range chunks {
		for _, event := range chunk.Events {
			if event.Kind == llm.StreamEventReasoningDelta {
				reasoningEvents.WriteString(event.Delta)
			}
		}
		for _, part := range chunk.Content {
			text.WriteString(part.Text)
		}
		if len(chunk.ProviderData) != 0 {
			providerData = append([]byte(nil), chunk.ProviderData...)
		}
	}
	if reasoningEvents.String() != "thinking" || text.String() != "answer" {
		t.Fatalf("assembled output = reasoning events %q, text %q", reasoningEvents.String(), text.String())
	}
	data, err := ParseReasoningData(llm.Message{ProviderData: providerData})
	if err != nil || data.ReasoningContent == nil || *data.ReasoningContent != "thinking" {
		t.Fatalf("ParseReasoningData() = (%#v, %v)", data, err)
	}
}

func TestStreamClosesReasoningBeforeToolCallsAndPreservesDetails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, strings.Join([]string{
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"reasoning":"thinking","reasoning_details":[{"type":"reasoning.encrypted","data":"secret"}]},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"tool","arguments":"{}"}}],"reasoning_details":[{"type":"reasoning.text","text":"thinking"}]},"finish_reason":null}]}` + "\n\n",
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n",
			"data: [DONE]\n\n",
		}, ""))
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
		Tools:    []llm.Tool{{Name: "tool", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if terminal != io.EOF {
		t.Fatalf("Recv() terminal = %v, want EOF", terminal)
	}
	assertStreamEventKinds(t, chunks, [][]llm.StreamEventKind{
		{llm.StreamEventStart},
		{llm.StreamEventReasoningStart, llm.StreamEventReasoningDelta},
		{llm.StreamEventReasoningEnd, llm.StreamEventToolCallStart, llm.StreamEventToolCallDelta},
		{llm.StreamEventToolCallEnd, llm.StreamEventDone},
	})
	last := chunks[len(chunks)-1]
	data, err := ParseReasoningData(llm.Message{ProviderData: last.ProviderData})
	if err != nil || data.Reasoning == nil || *data.Reasoning != "thinking" || len(data.ReasoningDetails) != 2 {
		t.Fatalf("ParseReasoningData() = (%#v, %v)", data, err)
	}
	if len(last.ToolCalls) != 1 || last.ToolCalls[0].ID != "call" || last.ToolCalls[0].Name != "tool" || string(last.ToolCalls[0].Arguments) != `{}` {
		t.Fatalf("ToolCalls = %#v", last.ToolCalls)
	}
}

func TestStreamPreservesEmptyReasoningDetails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer,
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"reasoning_details":[]},"finish_reason":null}]}`+"\n\n"+
				`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n"+
				"data: [DONE]\n\n")
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityStreaming}, BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if terminal != io.EOF {
		t.Fatalf("Recv() terminal = %v, want EOF", terminal)
	}
	last := chunks[len(chunks)-1]
	data, err := ParseReasoningData(llm.Message{ProviderData: last.ProviderData})
	if err != nil || !data.HasReasoningDetails || len(data.ReasoningDetails) != 0 {
		t.Fatalf("ParseReasoningData() = (%#v, %v)", data, err)
	}
	wire, err := newChatRequest("model", llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, {Role: llm.RoleAssistant, ProviderData: last.ProviderData}, {Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("newChatRequest() error = %v", err)
	}
	encoded, err := jsonv2.Marshal(wire)
	if err != nil || !strings.Contains(string(encoded), `"reasoning_details":[]`) {
		t.Fatalf("encoded continuation = (%s, %v)", encoded, err)
	}
}

func TestStreamRejectsReasoningAfterToolOutputStarts(t *testing.T) {
	body := `data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"tool","arguments":""}}]},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"reasoning":"late"},"finish_reason":null}]}` + "\n\n"
	client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, http.StatusOK, "text/event-stream", body)
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
		Tools:    []llm.Tool{{Name: "tool", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if len(chunks) != 1 {
		t.Fatalf("chunks = %#v, want one tool chunk", chunks)
	}
	requireModelError(t, terminal, llm.KindMalformedResponse, "stream")
	if !strings.Contains(terminal.Error(), "after text or tool output started") {
		t.Fatalf("Recv() terminal = %v", terminal)
	}
	_ = stream.Close()
}

func TestStreamSDKIgnoresAmbientConfiguration(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "ambient-key")
	t.Setenv("OPENAI_ADMIN_KEY", "ambient-admin-key")
	t.Setenv("OPENAI_ORG_ID", "ambient-org")
	t.Setenv("OPENAI_PROJECT_ID", "ambient-project")
	t.Setenv("OPENAI_BASE_URL", "http://ambient.invalid/")
	t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Ambient: leaked")

	requests := make(chan *http.Request, 1)
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		APIKey:       "ignored-key",
		BaseURL:      "http://configured.example/proxy/v1?",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests <- request.Clone(request.Context())
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body: io.NopCloser(strings.NewReader(
					"data: {\"id\":\"chat\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
						"data: [DONE]\n\n",
				)),
			}, nil
		})},
		Headers: http.Header{
			"authorization": {"Basic header-owned"},
			"x-trace":       {"trace"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_, terminal := receiveAll(stream)
	if !errors.Is(terminal, io.EOF) {
		t.Fatalf("Recv() terminal = %v", terminal)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	request := <-requests
	if request.URL.Host != "configured.example" || request.URL.Path != "/proxy/v1/chat/completions" ||
		request.URL.RawQuery != "" || !request.URL.ForceQuery {
		t.Fatalf("request URL = %#v", request.URL)
	}
	if request.Header.Get("Accept") != "text/event-stream" || request.Header.Get("Authorization") != "Basic header-owned" ||
		request.Header.Get("X-Trace") != "trace" || request.Header.Get("OpenAI-Organization") != "" ||
		request.Header.Get("OpenAI-Project") != "" || request.Header.Get("X-Ambient") != "" {
		t.Fatalf("request headers = %#v", request.Header)
	}
}

func TestStreamSDKMakesOneAttemptAndBoundsErrorBody(t *testing.T) {
	var calls atomic.Int64
	var closed atomic.Bool
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return &http.Response{
				StatusCode: http.StatusInternalServerError,
				Status:     "500 Internal Server Error",
				Header:     make(http.Header),
				Body: &observedCloseBody{
					Reader: strings.NewReader(strings.Repeat("x", maxErrorBodyBytes+1)),
					closed: &closed,
				},
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_, terminal := receiveAll(stream)
	modelErr := requireModelError(t, terminal, llm.KindProvider, "stream")
	var apiErr *APIError
	if modelErr.HTTPStatus != http.StatusInternalServerError || !errors.As(terminal, &apiErr) ||
		!strings.Contains(apiErr.Message, "response body exceeds") {
		t.Fatalf("Recv() error = %#v, API error = %#v", modelErr, apiErr)
	}
	if calls.Load() != 1 {
		t.Fatalf("HTTP calls = %d, want one", calls.Load())
	}
	if !closed.Load() {
		t.Fatal("error response body was not closed")
	}
	_ = stream.Close()
}

func TestStreamSDKCancellationAfterResponseClosesBody(t *testing.T) {
	cause := errors.New("cancel after response")
	ctx, cancel := context.WithCancelCause(context.Background())
	body := newBlockingStreamBody("")
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			cancel(cause)
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	if _, err := stream.Recv(); !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Recv() error = %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("canceled response body was not closed")
	}
}

func TestStreamSDKRedirectCancellation(t *testing.T) {
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		http.Redirect(writer, request, "/redirected", http.StatusTemporaryRedirect)
	}))
	cause := errors.New("cancel during redirect")
	redirectErr := errors.New("reject redirect")
	ctx, cancel := context.WithCancelCause(context.Background())
	httpClient := server.Client()
	httpClient.CheckRedirect = func(*http.Request, []*http.Request) error {
		cancel(cause)
		return redirectErr
	}
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      server.URL,
		HTTPClient:   httpClient,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_, terminal := receiveAll(stream)
	if !errors.Is(terminal, context.Canceled) || !errors.Is(terminal, cause) || errors.Is(terminal, redirectErr) {
		t.Fatalf("Recv() terminal = %v", terminal)
	}
	var modelErr *llm.Error
	if errors.As(terminal, &modelErr) {
		t.Fatalf("Recv() error contains model error: %#v", modelErr)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestStreamAcceptsStandardSSELineEndings(t *testing.T) {
	finish := `{"id":"chat","model":"served-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	tests := []struct {
		name      string
		separator string
	}{
		{name: "LF", separator: "\n"},
		{name: "CRLF", separator: "\r\n"},
		{name: "CR", separator: "\r"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := "\xef\xbb\xbfdata: " + finish + test.separator + test.separator +
				"data:" + test.separator + test.separator +
				"data: [DONE]" + test.separator + test.separator
			client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming}, http.StatusOK, "text/event-stream", body)
			stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			chunks, terminal := receiveAll(stream)
			if !errors.Is(terminal, io.EOF) {
				t.Fatalf("Recv() terminal = %v, want io.EOF", terminal)
			}
			_ = stream.Close()
			if len(chunks) != 1 || chunks[0].FinishReason != llm.FinishReasonStop || chunks[0].Model != "served-model" {
				t.Fatalf("chunks = %#v", chunks)
			}
		})
	}
}

func TestStreamPreservesRefusalData(t *testing.T) {
	tests := []struct {
		name      string
		fragments []string
		want      string
	}{
		{name: "fragmented", fragments: []string{"cannot", " comply"}, want: "cannot comply"},
		{name: "present empty", fragments: []string{""}, want: ""},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var body strings.Builder
			for _, fragment := range test.fragments {
				body.WriteString("data: {\"id\":\"chat-1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{\"refusal\":")
				encoded, err := jsonv2.Marshal(fragment)
				if err != nil {
					t.Fatal(err)
				}
				body.Write(encoded)
				body.WriteString("},\"finish_reason\":null}]}\n\n")
			}
			body.WriteString("data: {\"id\":\"chat-1\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
			body.WriteString("data: [DONE]\n\n")

			client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming}, http.StatusOK, "text/event-stream", body.String())
			stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			chunks, terminal := receiveAll(stream)
			if !errors.Is(terminal, io.EOF) {
				t.Fatalf("Recv() terminal = %v", terminal)
			}
			_ = stream.Close()

			var providerData []byte
			for _, chunk := range chunks {
				if len(chunk.ProviderData) != 0 {
					providerData = chunk.ProviderData
				}
			}
			data, err := ParseMessageData(llm.Message{ProviderData: providerData})
			if err != nil {
				t.Fatalf("ParseMessageData() error = %v", err)
			}
			if data.Refusal == nil || *data.Refusal != test.want {
				t.Fatalf("refusal = %#v, want %q", data.Refusal, test.want)
			}
		})
	}
}

func TestStreamAllowsPartialJSONForIncompleteOutcomes(t *testing.T) {
	tests := []struct {
		name   string
		finish string
		want   llm.FinishReason
	}{
		{name: "length", finish: "length", want: llm.FinishReasonLength},
		{name: "content filter", finish: "content_filter", want: llm.FinishReasonContentFilter},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := "data: {\"id\":\"chat\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chat\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" + test.finish + "\"}]}\n\n" +
				"data: [DONE]\n\n"
			client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityJSON}, http.StatusOK, "text/event-stream", body)
			stream, err := client.Stream(context.Background(), llm.Request{
				Messages:       []llm.Message{{Role: llm.RoleUser}},
				ResponseFormat: llm.ResponseFormatJSON,
			})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			chunks, terminal := receiveAll(stream)
			if !errors.Is(terminal, io.EOF) {
				t.Fatalf("Recv() terminal = %v", terminal)
			}
			_ = stream.Close()

			var text strings.Builder
			var finish llm.FinishReason
			for _, chunk := range chunks {
				for _, part := range chunk.Content {
					text.WriteString(part.Text)
				}
				if chunk.FinishReason != "" {
					finish = chunk.FinishReason
				}
			}
			if text.String() != "{" || finish != test.want {
				t.Fatalf("assembled output = (%q, %q), want ({, %q)", text.String(), finish, test.want)
			}
		})
	}
}

func TestStreamAssemblesModernToolCalls(t *testing.T) {
	body := strings.Join([]string{
		`data: {"id":"chat-1","model":"served-model","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":1,"id":"call_","type":"fun","function":{"name":"s","arguments":"{\"a\":"}},{"index":0,"id":"call_","type":"fun","function":{"name":"look","arguments":"{\"q\":\""}}]},"finish_reason":null}]}`,
		`data: {"id":"chat-1","model":"served-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"one","type":"ction","function":{"name":"up","arguments":"go\"}"}},{"index":1,"id":"two","type":"ction","function":{"name":"um","arguments":"1}"}}]},"finish_reason":null}]}`,
		`data: {"id":"chat-1","model":"served-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		"data: [DONE]",
	}, "\n\n") + "\n\n"
	client := newStaticStreamClient(t,
		[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
		http.StatusOK, "text/event-stream", body)
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
		Tools: []llm.Tool{
			{Name: "lookup", InputSchema: []byte(`{}`)},
			{Name: "sum", InputSchema: []byte(`{}`)},
		},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if !errors.Is(terminal, io.EOF) {
		t.Fatalf("Recv() terminal = %v, want io.EOF", terminal)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("len(chunks) = %d, want 3", len(chunks))
	}
	final := chunks[len(chunks)-1]
	if final.ID != "chat-1" || final.Model != "served-model" || final.FinishReason != llm.FinishReasonToolCall {
		t.Fatalf("final chunk = %#v", final)
	}
	wantEventKinds := [][]llm.StreamEventKind{
		{llm.StreamEventStart, llm.StreamEventToolCallStart, llm.StreamEventToolCallDelta, llm.StreamEventToolCallStart, llm.StreamEventToolCallDelta},
		{llm.StreamEventToolCallDelta, llm.StreamEventToolCallDelta},
		{llm.StreamEventToolCallEnd, llm.StreamEventToolCallEnd, llm.StreamEventDone},
	}
	assertStreamEventKinds(t, chunks, wantEventKinds)
	if got := chunks[0].Events[2].Delta + chunks[1].Events[1].Delta; got != `{"a":1}` {
		t.Fatalf("second tool argument deltas = %q", got)
	}
	want := []llm.ToolCall{
		{ID: "call_one", Name: "lookup", Arguments: []byte(`{"q":"go"}`)},
		{ID: "call_two", Name: "sum", Arguments: []byte(`{"a":1}`)},
	}
	if got := final.ToolCalls; len(got) != len(want) || got[0].ID != want[0].ID ||
		got[0].Name != want[0].Name || string(got[0].Arguments) != string(want[0].Arguments) ||
		got[1].ID != want[1].ID || got[1].Name != want[1].Name ||
		string(got[1].Arguments) != string(want[1].Arguments) {
		t.Fatalf("ToolCalls = %#v, want %#v", got, want)
	}
}

func TestStreamAssemblesLegacyFunctionCall(t *testing.T) {
	body := strings.Join([]string{
		`data: {"id":"chat-1","model":"served-model","choices":[{"index":0,"delta":{"function_call":{"name":"look","arguments":"{\"q\":\""}},"finish_reason":null}]}`,
		`data: {"id":"chat-1","model":"served-model","choices":[{"index":0,"delta":{"function_call":{"name":"up","arguments":"go\"}"}},"finish_reason":null}]}`,
		`data: {"id":"chat-1","model":"served-model","choices":[{"index":0,"delta":{},"finish_reason":"function_call"}]}`,
		"data: [DONE]",
	}, "\n\n") + "\n\n"
	client := newStaticStreamClient(t,
		[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
		http.StatusOK, "text/event-stream", body)
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
		Tools:    []llm.Tool{{Name: "lookup", InputSchema: []byte(`{}`)}},
	})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if !errors.Is(terminal, io.EOF) {
		t.Fatalf("Recv() terminal = %v, want io.EOF", terminal)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if len(chunks) != 3 || chunks[2].FinishReason != llm.FinishReasonToolCall || len(chunks[2].ToolCalls) != 1 {
		t.Fatalf("chunks = %#v", chunks)
	}
	assertStreamEventKinds(t, chunks, [][]llm.StreamEventKind{
		{llm.StreamEventStart, llm.StreamEventToolCallStart, llm.StreamEventToolCallDelta},
		{llm.StreamEventToolCallDelta},
		{llm.StreamEventToolCallEnd, llm.StreamEventDone},
	})
	if got := chunks[0].Events[2].Delta + chunks[1].Events[0].Delta; got != `{"q":"go"}` {
		t.Fatalf("tool argument deltas = %q", got)
	}
	call := chunks[2].ToolCalls[0]
	if call.ID != "" || call.Name != "lookup" || string(call.Arguments) != `{"q":"go"}` {
		t.Fatalf("ToolCall = %#v", call)
	}
}

func TestStreamRejectsMalformedToolCalls(t *testing.T) {
	modernFinish := `data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`
	legacyFinish := `data: {"choices":[{"index":0,"delta":{},"finish_reason":"function_call"}]}`
	tests := []struct {
		name   string
		events []string
		want   string
	}{
		{
			name: "missing index",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call"}]},"finish_reason":null}]}`,
			},
			want: "no index",
		},
		{
			name: "index outside limit",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1024}]},"finish_reason":null}]}`,
			},
			want: "outside",
		},
		{
			name: "sparse indices",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}`,
				modernFinish,
			},
			want: "no call at index 0",
		},
		{
			name: "missing ID",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}`,
				modernFinish,
			},
			want: "no ID",
		},
		{
			name: "duplicate ID",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"same","type":"function","function":{"name":"lookup","arguments":"{}"}},{"index":1,"id":"same","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}`,
				modernFinish,
			},
			want: "repeats ID",
		},
		{
			name: "unsupported type",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","type":"other","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":null}]}`,
				modernFinish,
			},
			want: "unsupported type",
		},
		{
			name: "invalid function name",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"bad name","arguments":"{}"}}]},"finish_reason":null}]}`,
				modernFinish,
			},
			want: "invalid function name",
		},
		{
			name: "invalid arguments",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"lookup","arguments":"{"}}]},"finish_reason":null}]}`,
				modernFinish,
			},
			want: "arguments are not strict JSON",
		},
		{
			name: "undeclared function",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"other","arguments":"{}"}}]},"finish_reason":null}]}`,
				modernFinish,
			},
			want: "undeclared function",
		},
		{
			name: "both formats in one event",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0}],"function_call":{}},"finish_reason":null}]}`,
			},
			want: "both tool_calls and function_call",
		},
		{
			name: "formats mixed across events",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0}]},"finish_reason":null}]}`,
				`data: {"choices":[{"index":0,"delta":{"function_call":{}},"finish_reason":null}]}`,
			},
			want: "mixes tool_calls and function_call",
		},
		{
			name: "modern call with legacy finish",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0}]},"finish_reason":null}]}`,
				legacyFinish,
			},
			want: "inconsistent with streamed tool_calls",
		},
		{
			name: "legacy call with modern finish",
			events: []string{
				`data: {"choices":[{"index":0,"delta":{"function_call":{}},"finish_reason":null}]}`,
				modernFinish,
			},
			want: "inconsistent with streamed function_call",
		},
		{
			name:   "modern finish without calls",
			events: []string{modernFinish},
			want:   "has no tool calls",
		},
		{
			name:   "legacy finish without call",
			events: []string{legacyFinish},
			want:   "has no function call",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			body := strings.Join(test.events, "\n\n") + "\n\n"
			client := newStaticStreamClient(t,
				[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
				http.StatusOK, "text/event-stream", body)
			stream, err := client.Stream(context.Background(), llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
				Tools:    []llm.Tool{{Name: "lookup", InputSchema: []byte(`{}`)}},
			})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, llm.KindMalformedResponse, "stream")
			if !strings.Contains(terminal.Error(), test.want) {
				t.Fatalf("Recv() error = %q, want substring %q", terminal, test.want)
			}
			_ = stream.Close()
		})
	}
}

func TestStreamUsesStreamOperationForPreflightErrors(t *testing.T) {
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityVision},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	request := llm.Request{Messages: []llm.Message{{
		Role:    llm.RoleUser,
		Content: []llm.Part{{Kind: llm.PartImage, Data: []byte("image"), MediaType: "not a media type"}},
	}}}
	stream, err := client.Stream(context.Background(), request)
	if stream != nil {
		t.Fatalf("Stream() = %#v, want nil", stream)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "stream")
}

func TestStreamClassifiesHTTPError(t *testing.T) {
	client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming}, http.StatusUnauthorized, "application/json", `{"error":{"message":"bad key"}}`)
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_, terminal := receiveAll(stream)
	modelErr := requireModelError(t, terminal, llm.KindAuthentication, "stream")
	if modelErr.HTTPStatus != http.StatusUnauthorized {
		t.Fatalf("HTTPStatus = %d", modelErr.HTTPStatus)
	}
	_ = stream.Close()
}

func TestStreamClassifiesProviderInterruption(t *testing.T) {
	body := `data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{},"finish_reason":"insufficient_system_resource"}]}` + "\n\n"
	client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming}, http.StatusOK, "text/event-stream", body)
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if len(chunks) != 0 {
		t.Fatalf("chunks = %#v, want none", chunks)
	}
	requireModelError(t, terminal, llm.KindProvider, "stream")
	_ = stream.Close()
}

func TestStreamClassifiesProviderEventError(t *testing.T) {
	tests := []struct {
		name string
		body string
		kind llm.ErrorKind
		code string
	}{
		{
			name: "authentication type",
			body: `{"error":{"message":"bad key","type":"authentication_error"}}`,
			kind: llm.KindAuthentication,
		},
		{
			name: "authentication code",
			body: `{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`,
			kind: llm.KindAuthentication,
			code: "invalid_api_key",
		},
		{
			name: "permission",
			body: `{"error":{"message":"forbidden","type":"permission_error"}}`,
			kind: llm.KindPermission,
		},
		{
			name: "rate limit",
			body: `{"error":{"message":"slow down","type":"rate_limit_error"}}`,
			kind: llm.KindRateLimit,
		},
		{
			name: "invalid request",
			body: `{"error":{"message":"bad input","type":"invalid_request_error","code":"invalid_value"}}`,
			kind: llm.KindInvalidRequest,
			code: "invalid_value",
		},
		{
			name: "context limit",
			body: `{"error":{"message":"too long","type":"invalid_request_error","code":"context_length_exceeded"}}`,
			kind: llm.KindContextLimit,
			code: "context_length_exceeded",
		},
		{
			name: "provider",
			body: `{"error":{"message":"generation failed","type":"server_error","code":"stream_failed"}}`,
			kind: llm.KindProvider,
			code: "stream_failed",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming}, http.StatusOK, "text/event-stream",
				"data: "+test.body+"\n\n")
			stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, test.kind, "stream")
			var apiErr *APIError
			if !errors.As(terminal, &apiErr) || apiErr.Code != test.code {
				t.Fatalf("Recv() error = %v, API error = %#v", terminal, apiErr)
			}
			_ = stream.Close()
		})
	}
}

func TestStreamPreservesTransportError(t *testing.T) {
	sentinel := errors.New("transport sentinel")
	validBody := "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	tests := []struct {
		name      string
		transport http.RoundTripper
	}{
		{
			name: "send",
			transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, sentinel
			}),
		},
		{
			name: "read",
			transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": {"text/event-stream"}},
					Body:       errorReadCloser{err: sentinel},
				}, nil
			}),
		},
		{
			name: "error response read",
			transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusInternalServerError,
					Header:     make(http.Header),
					Body:       errorReadCloser{err: sentinel},
				}, nil
			}),
		},
		{
			name: "close",
			transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": {"text/event-stream"}},
					Body:       &closeErrorBody{Reader: strings.NewReader(validBody), err: sentinel},
				}, nil
			}),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(Config{
				Model:        "model",
				Capabilities: []llm.Capability{llm.CapabilityStreaming},
				BaseURL:      "http://example.com/v1",
				HTTPClient:   &http.Client{Transport: test.transport},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, llm.KindTransport, "stream")
			if !errors.Is(terminal, sentinel) {
				t.Fatalf("Recv() error = %v, want sentinel", terminal)
			}
			_ = stream.Close()
		})
	}
}

func TestStreamKeepsFallbackModelStableWhenMetadataArrivesLate(t *testing.T) {
	body := "data: {\"id\":\"chat\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: {\"id\":\"chat\",\"model\":\"late-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n" +
		"data: [DONE]\n\n"
	client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming}, http.StatusOK, "text/event-stream", body)
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if !errors.Is(terminal, io.EOF) {
		t.Fatalf("Recv() terminal = %v, want io.EOF", terminal)
	}
	_ = stream.Close()
	if len(chunks) != 2 || chunks[0].Model != "model" || chunks[1].Model != "model" {
		t.Fatalf("chunk models = %#v, want stable configured fallback", chunks)
	}
}

func TestStreamBoundsBufferedOutput(t *testing.T) {
	tests := []struct {
		name   string
		format llm.ResponseFormat
		delta  string
	}{
		{name: "JSON content", format: llm.ResponseFormatJSON, delta: `{"content":"xx"}`},
		{name: "refusal", format: llm.ResponseFormatText, delta: `{"refusal":"xx"}`},
		{name: "tool arguments", format: llm.ResponseFormatText, delta: `{"tool_calls":[{"index":0,"function":{"arguments":"xx"}}]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decoder := newStreamDecoder("model", test.format, nil)
			decoder.bufferedBytes = maxResponseBodyBytes - 1
			_, err := decoder.consume(`{"choices":[{"index":0,"delta":`+test.delta+`,"finish_reason":null}]}`,
				func(llm.Chunk) bool { return true })
			requireModelError(t, err, llm.KindMalformedResponse, "stream")
			if !strings.Contains(err.Error(), "buffered stream output") {
				t.Fatalf("consume() error = %v", err)
			}
		})
	}
}

func TestStreamRejectsMalformedProtocol(t *testing.T) {
	validFinish := `{"id":"chat","model":"model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	tests := []struct {
		name        string
		contentType string
		body        string
		caps        []llm.Capability
		request     llm.Request
		want        string
	}{
		{name: "content type", contentType: "application/json", body: `{}`, want: "content type"},
		{name: "malformed JSON", contentType: "text/event-stream", body: "data: {\n\n", want: "decode event"},
		{name: "missing choices", contentType: "text/event-stream", body: "data: {\"id\":\"chat\"}\n\n", want: "no choices field"},
		{name: "empty event", contentType: "text/event-stream", body: "data: {\"choices\":[]}\n\n", want: "neither a choice nor usage"},
		{name: "multiple choices", contentType: "text/event-stream", body: "data: {\"choices\":[{},{}]}\n\n", want: "want exactly one"},
		{name: "missing index", contentType: "text/event-stream", body: "data: {\"choices\":[{\"delta\":{},\"finish_reason\":null}]}\n\n", want: "no index"},
		{name: "missing delta", contentType: "text/event-stream", body: "data: {\"choices\":[{\"index\":0,\"finish_reason\":null}]}\n\n", want: "no delta"},
		{name: "missing finish reason", contentType: "text/event-stream", body: "data: {\"choices\":[{\"index\":0,\"delta\":{}}]}\n\n", want: "no finish reason"},
		{name: "tool delta", contentType: "text/event-stream", body: "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{}]},\"finish_reason\":null}]}\n\n", want: "streamed tool call"},
		{name: "done before finish", contentType: "text/event-stream", body: "data: [DONE]\n\n", want: "before a finish reason"},
		{name: "missing done", contentType: "text/event-stream", body: "data: " + validFinish + "\n\n", want: "without [DONE]"},
		{name: "unterminated done", contentType: "text/event-stream", body: "data: " + validFinish + "\n\ndata: [DONE]", want: "without [DONE]"},
		{name: "discard incomplete EOF event", contentType: "text/event-stream", body: "data: " + validFinish + "\n\ndata: {", want: "without [DONE]"},
		{name: "altered done marker", contentType: "text/event-stream", body: "data: " + validFinish + "\n\ndata: [DONE] \n\n", want: "decode event"},
		{
			name:        "changed ID",
			contentType: "text/event-stream",
			body: "data: {\"id\":\"one\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"two\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
			want: "ID changed",
		},
		{
			name:        "choice after finish",
			contentType: "text/event-stream",
			body: "data: " + validFinish + "\n\n" +
				"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"late\"},\"finish_reason\":null}]}\n\n",
			want: "after the finish reason",
		},
		{
			name:        "incomplete usage",
			contentType: "text/event-stream",
			body: "data: " + validFinish + "\n\n" +
				"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1}}\n\n",
			want: "incomplete token usage",
		},
		{
			name:        "invalid completed JSON",
			contentType: "text/event-stream",
			caps:        []llm.Capability{llm.CapabilityStreaming, llm.CapabilityJSON},
			request: llm.Request{
				Messages:       []llm.Message{{Role: llm.RoleUser}},
				ResponseFormat: llm.ResponseFormatJSON,
			},
			body: "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\"},\"finish_reason\":null}]}\n\n" +
				"data: " + validFinish + "\n\n",
			want: "not strict JSON",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			caps := test.caps
			if caps == nil {
				caps = []llm.Capability{llm.CapabilityStreaming}
			}
			request := test.request
			if len(request.Messages) == 0 {
				request.Messages = []llm.Message{{Role: llm.RoleUser}}
			}
			client := newStaticStreamClient(t, caps, http.StatusOK, test.contentType, test.body)
			stream, err := client.Stream(context.Background(), request)
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, llm.KindMalformedResponse, "stream")
			if !strings.Contains(terminal.Error(), test.want) {
				t.Fatalf("Recv() error = %q, want substring %q", terminal, test.want)
			}
			_ = stream.Close()
		})
	}
}

func TestStreamPreservesMalformedEventCause(t *testing.T) {
	client := newStaticStreamClient(t, []llm.Capability{llm.CapabilityStreaming}, http.StatusOK, "text/event-stream", "data: {\n\n")
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_, terminal := receiveAll(stream)
	requireModelError(t, terminal, llm.KindMalformedResponse, "stream")
	var syntaxErr *jsontext.SyntacticError
	if !errors.As(terminal, &syntaxErr) {
		t.Fatalf("errors.As(%v, *jsontext.SyntacticError) = false", terminal)
	}
	_ = stream.Close()
}

func TestEventStreamRejectsSizeLimits(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "line",
			body: "data: " + strings.Repeat("x", maxStreamEventBytes+1) + "\n\n",
			want: "line exceeds",
		},
		{
			name: "event",
			body: strings.Repeat(": keepalive\n", maxStreamEventBytes/len(": keepalive\n")+1),
			want: "event exceeds",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := readEventStream(context.Background(), strings.NewReader(test.body),
				newStreamDecoder("model", llm.ResponseFormatText, nil), func(llm.Chunk) bool { return true })
			requireModelError(t, err, llm.KindMalformedResponse, "stream")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("readEventStream() error = %v", err)
			}
		})
	}
}

func TestEventStreamAcceptsBOMAtSizeBoundary(t *testing.T) {
	finish := `data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	for _, separator := range []string{"\r", "\r\n"} {
		name := "CR"
		if separator == "\r\n" {
			name = "CRLF"
		}
		t.Run(name, func(t *testing.T) {
			comment := ":" + strings.Repeat("x", maxStreamEventBytes-3)
			body := streamUTF8BOM + comment + separator + separator +
				finish + separator + separator + "data: [DONE]" + separator + separator
			var chunks []llm.Chunk
			err := readEventStream(context.Background(), strings.NewReader(body),
				newStreamDecoder("model", llm.ResponseFormatText, nil), func(chunk llm.Chunk) bool {
					chunks = append(chunks, chunk)
					return true
				})
			if err != nil {
				t.Fatalf("readEventStream() error = %v", err)
			}
			if len(chunks) != 1 || chunks[0].FinishReason != llm.FinishReasonStop {
				t.Fatalf("chunks = %#v", chunks)
			}
		})
	}
}

func TestStreamCloseClosesBlockedResponseBody(t *testing.T) {
	body := newBlockingStreamBody("data: {\"id\":\"chat\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"committed\"},\"finish_reason\":null}]}\n\n")
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunk, err := stream.Recv()
	if err != nil || chunk.Content[0].Text != "committed" {
		t.Fatalf("Recv() = (%#v, %v)", chunk, err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("Close() returned before response body closed")
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("Recv() after Close error = %v, want io.EOF", err)
	}
}

func TestStreamClosePreservesBodyCloseError(t *testing.T) {
	closeErr := errors.New("close failed")
	body := newBlockingStreamBody("")
	body.closeErr = closeErr
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	<-body.reading
	for range 2 {
		err := stream.Close()
		requireModelError(t, err, llm.KindTransport, "stream")
		if !errors.Is(err, closeErr) {
			t.Fatalf("Close() error = %v, want %v", err, closeErr)
		}
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("Recv() after Close error = %v, want io.EOF", err)
	}
}

func TestStreamClosePreservesInFlightResponseCloseError(t *testing.T) {
	closeErr := errors.New("close failed")
	body := newBlockingStreamBody("")
	body.closeErr = closeErr
	requestStarted := make(chan struct{})
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			close(requestStarted)
			<-request.Context().Done()
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	<-requestStarted
	for range 2 {
		err := stream.Close()
		requireModelError(t, err, llm.KindTransport, "stream")
		if !errors.Is(err, closeErr) {
			t.Fatalf("Close() error = %v, want %v", err, closeErr)
		}
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("Close() returned before the in-flight response body closed")
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("Recv() after Close error = %v, want io.EOF", err)
	}
}

func TestStreamClosePreservesStalledErrorResponseCloseError(t *testing.T) {
	closeErr := errors.New("close failed")
	body := newBlockingStreamBody(`{"error":{"message":"partial"}}`)
	body.closeErr = closeErr
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusInternalServerError,
				Header:     http.Header{"Content-Type": {"application/json"}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	<-body.reading
	for range 2 {
		err := stream.Close()
		requireModelError(t, err, llm.KindTransport, "stream")
		if !errors.Is(err, closeErr) {
			t.Fatalf("Close() error = %v, want %v", err, closeErr)
		}
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("Recv() after Close error = %v, want io.EOF", err)
	}
}

func TestStreamPreservesEarlyResponseCloseErrors(t *testing.T) {
	closeErr := errors.New("close failed")
	tests := []struct {
		name        string
		status      int
		contentType string
		body        string
		kind        llm.ErrorKind
	}{
		{
			name:        "HTTP error",
			status:      http.StatusUnauthorized,
			contentType: "application/json",
			body:        `{"error":{"message":"bad key"}}`,
			kind:        llm.KindAuthentication,
		},
		{
			name:        "invalid content type",
			status:      http.StatusOK,
			contentType: "application/json",
			body:        `{}`,
			kind:        llm.KindMalformedResponse,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(Config{
				Model:        "model",
				Capabilities: []llm.Capability{llm.CapabilityStreaming},
				BaseURL:      "http://example.com/v1",
				HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: test.status,
						Header:     http.Header{"Content-Type": {test.contentType}},
						Body:       &closeErrorBody{Reader: strings.NewReader(test.body), err: closeErr},
					}, nil
				})},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, test.kind, "stream")
			if !errors.Is(terminal, closeErr) {
				t.Fatalf("Recv() error = %v, want close error", terminal)
			}
			_ = stream.Close()
		})
	}
}

func TestStreamPreservesCloseErrorWithSSEFailure(t *testing.T) {
	closeErr := errors.New("close failed")
	tests := []struct {
		name string
		body string
		kind llm.ErrorKind
	}{
		{
			name: "malformed event",
			body: "data: {\"choices\":\n\n",
			kind: llm.KindMalformedResponse,
		},
		{
			name: "provider error",
			body: "data: {\"error\":{\"message\":\"failed\"}}\n\n",
			kind: llm.KindProvider,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(Config{
				Model:        "model",
				Capabilities: []llm.Capability{llm.CapabilityStreaming},
				BaseURL:      "http://example.com/v1",
				HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     http.Header{"Content-Type": {"text/event-stream"}},
						Body:       &closeErrorBody{Reader: strings.NewReader(test.body), err: closeErr},
					}, nil
				})},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, test.kind, "stream")
			if !errors.Is(terminal, closeErr) {
				t.Fatalf("Recv() error = %v, want close error", terminal)
			}
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		})
	}
}

func TestStreamCancellationClosesBodyAndPreservesCause(t *testing.T) {
	closeErr := errors.New("close failed")
	body := newBlockingStreamBody("")
	body.closeErr = closeErr
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	cause := errors.New("cancel stream")
	ctx, cancel := context.WithCancelCause(context.Background())
	stream, err := client.Stream(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	<-body.reading
	cancel(cause)
	_, terminal := stream.Recv()
	if !errors.Is(terminal, context.Canceled) || !errors.Is(terminal, cause) || errors.Is(terminal, closeErr) {
		t.Fatalf("Recv() error = %v, want cancellation and custom cause without close error", terminal)
	}
	closeDiagnostic := stream.Close()
	requireModelError(t, closeDiagnostic, llm.KindTransport, "stream")
	if !errors.Is(closeDiagnostic, closeErr) || errors.Is(closeDiagnostic, context.Canceled) || errors.Is(closeDiagnostic, cause) {
		t.Fatalf("Close() error = %v, want only response-body close diagnostic", closeDiagnostic)
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("response body remained open after cancellation")
	}
}

func TestProduceStreamCancellationPreservesBodyCloseError(t *testing.T) {
	closeErr := errors.New("close failed")
	body := newBlockingStreamBody("")
	body.closeErr = closeErr
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	cause := errors.New("cancel stream")
	ctx, cancel := context.WithCancelCause(context.Background())
	result := make(chan error, 1)
	go func() {
		result <- client.produceStream(ctx, llm.ResponseFormatText, nil, []byte(`{}`), func(llm.Chunk) bool { return true })
	}()
	<-body.reading
	cancel(cause)
	err = <-result
	requireModelError(t, err, llm.KindTransport, "stream")
	if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || !errors.Is(err, closeErr) {
		t.Fatalf("produceStream() error = %v, want cancellation, cause, and close error", err)
	}
}

func TestStreamOpeningPathsPreserveCancellationAndCloseError(t *testing.T) {
	for _, name := range []string{"opening response", "HTTP error response", "invalid content type"} {
		t.Run(name, func(t *testing.T) {
			closeErr := errors.New("close failed")
			cause := errors.New("cancel stream")
			ctx, cancel := context.WithCancelCause(context.Background())
			defer cancel(nil)
			var ready <-chan struct{}
			var transport roundTripFunc
			switch name {
			case "opening response":
				started := make(chan struct{})
				ready = started
				transport = func(request *http.Request) (*http.Response, error) {
					close(started)
					<-request.Context().Done()
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     http.Header{"Content-Type": {"text/event-stream"}},
						Body:       &closeErrorBody{Reader: strings.NewReader(""), err: closeErr},
					}, nil
				}
			case "HTTP error response":
				body := newBlockingStreamBody(`{"error":{"message":"partial"}}`)
				body.closeErr = closeErr
				ready = body.reading
				transport = func(*http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusUnauthorized,
						Header:     http.Header{"Content-Type": {"application/json"}},
						Body:       body,
					}, nil
				}
			case "invalid content type":
				transport = func(*http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     http.Header{"Content-Type": {"application/json"}},
						Body: &cancelingCloseBody{Reader: strings.NewReader(`{}`), cancel: func() {
							cancel(cause)
						}, err: closeErr},
					}, nil
				}
			}
			client, err := New(Config{
				Model: "model", Capabilities: []llm.Capability{llm.CapabilityStreaming}, BaseURL: "http://example.com/v1",
				HTTPClient: &http.Client{Transport: transport},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			stream, err := client.Stream(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			if ready != nil {
				<-ready
				cancel(cause)
			}
			_, terminal := stream.Recv()
			if !errors.Is(terminal, context.Canceled) || !errors.Is(terminal, cause) || errors.Is(terminal, closeErr) {
				t.Fatalf("Recv() error = %v, want cancellation and custom cause without close error", terminal)
			}
			closeDiagnostic := stream.Close()
			requireModelError(t, closeDiagnostic, llm.KindTransport, "stream")
			if !errors.Is(closeDiagnostic, closeErr) || errors.Is(closeDiagnostic, context.Canceled) || errors.Is(closeDiagnostic, cause) {
				t.Fatalf("Close() error = %v, want only response-body close diagnostic", closeDiagnostic)
			}
		})
	}
}

func TestStreamCancellationReachesHTTPServer(t *testing.T) {
	requestStarted := make(chan struct{})
	requestCanceled := make(chan struct{})
	handlerRelease := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("response writer does not support flushing")
		}
		_, _ = io.WriteString(w, "data: {\"id\":\"chat\",\"model\":\"model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ready\"},\"finish_reason\":null}]}\n\n")
		flusher.Flush()
		close(requestStarted)
		select {
		case <-request.Context().Done():
			close(requestCanceled)
		case <-handlerRelease:
		}
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(handlerRelease) })

	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      server.URL,
		HTTPClient:   server.Client(),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.Stream(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunk, err := stream.Recv()
	if err != nil || chunk.Content[0].Text != "ready" {
		t.Fatalf("Recv() = (%#v, %v)", chunk, err)
	}
	<-requestStarted
	cancel()
	if _, err := stream.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv() error = %v, want context.Canceled", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	select {
	case <-requestCanceled:
	case <-time.After(5 * time.Second):
		t.Fatal("stream cancellation did not reach HTTP server")
	}
}

func assertStreamEventKinds(t *testing.T, chunks []llm.Chunk, want [][]llm.StreamEventKind) {
	t.Helper()
	if len(chunks) != len(want) {
		t.Fatalf("event chunk count = %d, want %d", len(chunks), len(want))
	}
	for chunkIndex, kinds := range want {
		if len(chunks[chunkIndex].Events) != len(kinds) {
			t.Fatalf("chunk %d event count = %d, want %d: %#v", chunkIndex, len(chunks[chunkIndex].Events), len(kinds), chunks[chunkIndex].Events)
		}
		for eventIndex, kind := range kinds {
			if got := chunks[chunkIndex].Events[eventIndex].Kind; got != kind {
				t.Fatalf("chunk %d event %d kind = %q, want %q", chunkIndex, eventIndex, got, kind)
			}
		}
	}
}

func receiveAll(stream llm.Stream) ([]llm.Chunk, error) {
	var chunks []llm.Chunk
	for {
		chunk, err := stream.Recv()
		if err != nil {
			return chunks, err
		}
		chunks = append(chunks, chunk)
	}
}

func newStaticStreamClient(t *testing.T, capabilities []llm.Capability, status int, contentType, body string) *Client {
	t.Helper()
	client, err := New(Config{
		Model:        "model",
		Capabilities: capabilities,
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticStreamResponse{
			status:      status,
			contentType: contentType,
			body:        body,
		}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client
}

type staticStreamResponse struct {
	status      int
	contentType string
	body        string
}

func (response staticStreamResponse) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: response.status,
		Status:     http.StatusText(response.status),
		Header:     http.Header{"Content-Type": {response.contentType}},
		Body:       io.NopCloser(strings.NewReader(response.body)),
	}, nil
}

type errorReadCloser struct {
	err error
}

func (body errorReadCloser) Read([]byte) (int, error) {
	return 0, body.err
}

func (errorReadCloser) Close() error {
	return nil
}

type closeErrorBody struct {
	io.Reader
	err error
}

func (body *closeErrorBody) Close() error {
	return body.err
}

type cancelingCloseBody struct {
	io.Reader
	cancel func()
	err    error
	once   sync.Once
}

func (body *cancelingCloseBody) Close() error {
	body.once.Do(body.cancel)
	return body.err
}

type blockingStreamBody struct {
	first    []byte
	closed   chan struct{}
	reading  chan struct{}
	closeErr error
	readOnce sync.Once
	once     sync.Once
}

func newBlockingStreamBody(first string) *blockingStreamBody {
	return &blockingStreamBody{
		first:   []byte(first),
		closed:  make(chan struct{}),
		reading: make(chan struct{}),
	}
}

func (body *blockingStreamBody) Read(buffer []byte) (int, error) {
	if len(body.first) != 0 {
		n := copy(buffer, body.first)
		body.first = body.first[n:]
		return n, nil
	}
	body.readOnce.Do(func() { close(body.reading) })
	<-body.closed
	return 0, errors.New("body closed")
}

func (body *blockingStreamBody) Close() error {
	body.once.Do(func() { close(body.closed) })
	return body.closeErr
}
