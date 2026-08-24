package anthropic

import (
	"context"
	"errors"
	"fmt"
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

func TestStreamTranslatesTextResponse(t *testing.T) {
	type requestRecord struct {
		path   string
		header http.Header
		body   []byte
	}
	records := make(chan requestRecord, 1)
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			panic(err)
		}
		records <- requestRecord{path: request.URL.Path, header: request.Header.Clone(), body: body}
		writer.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		_, _ = io.WriteString(writer, strings.Join([]string{
			": heartbeat\n\n",
			"event: message_start\n",
			"data: {\"type\":\"message_start\",\n",
			"data: \"message\":{\"id\":\"msg_stream\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-served\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":2,\"cache_creation\":{\"ephemeral_1h_input_tokens\":1},\"cache_read_input_tokens\":3,\"output_tokens\":1}}}\n\n",
			"event: ping\n",
			"data: {\"type\":\"ping\"}\n\n",
			"event: future_metadata\n",
			"data: {\"type\":\"future_metadata\",\"value\":true}\n\n",
			"event: content_block_start\n",
			"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
			"event: content_block_delta\n",
			"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
			"event: content_block_delta\n",
			"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
			"event: content_block_stop\n",
			"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
			"event: content_block_start\n",
			"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"!\"}}\n\n",
			"event: content_block_stop\n",
			"data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
			"event: message_delta\n",
			"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5,\"output_tokens_details\":{\"thinking_tokens\":2}}}\n\n",
			"event: message_stop\n",
			"data: {\"type\":\"message_stop\"}\n\n",
		}, ""))
	}))

	client, err := New(Config{
		Model:        "claude-request",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      server.URL + "/proxy/v1/",
		HTTPClient:   server.Client(),
		Headers:      http.Header{"X-Trace": {"stream-trace"}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	info := client.Info()
	if len(info.Capabilities) != 2 || info.Capabilities[0] != llm.CapabilityGeneration ||
		info.Capabilities[1] != llm.CapabilityStreaming {
		t.Fatalf("Info() = %#v", info)
	}
	stream, err := client.Stream(context.Background(), textRequest("hello"))
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
	for index, chunk := range chunks {
		if chunk.ID != "msg_stream" || chunk.Model != "claude-served" {
			t.Fatalf("chunk %d metadata = (%q, %q)", index, chunk.ID, chunk.Model)
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
	if text.String() != "hello!" || finish != llm.FinishReasonStop {
		t.Fatalf("assembled response = (%q, %q)", text.String(), finish)
	}
	if usage == nil || *usage != (llm.Usage{InputTokens: 10, OutputTokens: 5, CacheReadTokens: 3, CacheWriteTokens: 2, CacheWrite1hTokens: 1, ReasoningTokens: 2, TotalTokens: 20}) {
		t.Fatalf("assembled usage = %#v", usage)
	}

	record := <-records
	if record.path != "/proxy/v1/messages" {
		t.Fatalf("request path = %q", record.path)
	}
	if record.header.Get("Accept") != "text/event-stream" || record.header.Get("Anthropic-Version") != defaultAPIVersion ||
		record.header.Get("X-Trace") != "stream-trace" {
		t.Fatalf("request headers = %#v", record.header)
	}
	if record.header.Get("User-Agent") != "Anthropic/Go 1.66.0" || record.header.Get("X-Stainless-Retry-Count") != "0" {
		t.Fatalf("SDK headers = %#v", record.header)
	}
	var payload map[string]any
	if err := jsonv2.Unmarshal(record.body, &payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if payload["stream"] != true || payload["model"] != "claude-request" {
		t.Fatalf("request payload = %#v", payload)
	}
}

func TestStreamSDKConfigurationAndSingleAttempt(t *testing.T) {
	t.Run("environment prefix empty query and headers", func(t *testing.T) {
		t.Setenv("ANTHROPIC_API_KEY", "ambient-key")
		t.Setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
		t.Setenv("ANTHROPIC_BASE_URL", "http://ambient.invalid/")
		t.Setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Ambient: leaked")
		t.Setenv("ANTHROPIC_PROFILE", "ambient-profile-must-not-load")

		requests := make(chan *http.Request, 1)
		client, err := New(Config{
			Model:        "model",
			Capabilities: []llm.Capability{llm.CapabilityStreaming},
			APIKey:       "configured-key",
			BaseURL:      "http://configured.example/proxy/v1?",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				requests <- request.Clone(request.Context())
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     "200 OK",
					Header:     http.Header{"Content-Type": {"text/event-stream"}},
					Body: io.NopCloser(strings.NewReader(
						streamEvent("message_start", validStreamStart) +
							streamEvent("message_delta", streamFinishEvent("end_turn", 2)) +
							streamEvent("message_stop", `{"type":"message_stop"}`),
					)),
				}, nil
			})},
			Headers: http.Header{
				"x-api-key":       {"header-key"},
				"x-stream-header": {"owned"},
			},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		stream, err := client.Stream(context.Background(), textRequest("hello"))
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
		if request.URL.Host != "configured.example" || request.URL.Path != "/proxy/v1/messages" ||
			request.URL.RawQuery != "" || !request.URL.ForceQuery {
			t.Fatalf("request URL = %#v", request.URL)
		}
		if values := request.Header["x-stream-header"]; len(values) != 1 || values[0] != "owned" {
			t.Fatalf("x-stream-header = %#v", values)
		}
		if values := request.Header["x-api-key"]; len(values) != 1 || values[0] != "header-key" {
			t.Fatalf("x-api-key = %#v", values)
		}
		authVariants := 0
		for key := range request.Header {
			if strings.EqualFold(key, "X-Api-Key") {
				authVariants++
			}
		}
		if request.Header.Get("Accept") != "text/event-stream" || authVariants != 1 ||
			request.Header.Get("Authorization") != "" || request.Header.Get("X-Ambient") != "" {
			t.Fatalf("request headers = %#v", request.Header)
		}
	})

	t.Run("one attempt and bounded error body", func(t *testing.T) {
		var calls atomic.Int64
		var closed atomic.Bool
		body := &errorBody{data: strings.Repeat("x", int(maxErrorBodyBytes)+1), onClose: func() { closed.Store(true) }}
		client, err := New(Config{
			Model:        "model",
			Capabilities: []llm.Capability{llm.CapabilityStreaming},
			BaseURL:      "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				calls.Add(1)
				return &http.Response{
					StatusCode: http.StatusInternalServerError,
					Status:     "500 Internal Server Error",
					Header:     http.Header{"Content-Type": {"application/json"}},
					Body:       body,
				}, nil
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		stream, err := client.Stream(context.Background(), textRequest("hello"))
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
	})
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
	stream, err := client.Stream(ctx, textRequest("hello"))
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

func TestStreamRejectsOversizedRequestWithoutIO(t *testing.T) {
	var calls atomic.Int64
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, errors.New("unexpected request")
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), textRequest(strings.Repeat("x", maxRequestBodyBytes)))
	if stream != nil {
		t.Fatalf("Stream() = %#v", stream)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "stream")
	if calls.Load() != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls.Load())
	}
}

func TestStreamPreflightAndNoIO(t *testing.T) {
	var calls atomic.Int64
	httpClient := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, errors.New("unexpected request")
	})}
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient:   httpClient,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	penalty := 0.1
	temperature := 1.1
	image := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Kind: llm.PartImage, Data: []byte{1}, MediaType: "image/png"}}}}}
	for _, test := range []struct {
		name    string
		request llm.Request
		kind    llm.ErrorKind
		want    string
	}{
		{name: "JSON", request: llm.Request{Messages: textRequest("hello").Messages, ResponseFormat: llm.ResponseFormatJSON}, kind: llm.KindUnsupported, want: "JSON"},
		{name: "image", request: image, kind: llm.KindUnsupported, want: "binary"},
		{name: "penalty", request: llm.Request{Messages: textRequest("hello").Messages, PresencePenalty: &penalty}, kind: llm.KindUnsupported, want: "penalties"},
		{name: "temperature", request: llm.Request{Messages: textRequest("hello").Messages, Temperature: &temperature}, kind: llm.KindInvalidRequest, want: "must not exceed 1"},
		{name: "tool capability", request: toolRequest(), kind: llm.KindUnsupported, want: "tool capability"},
	} {
		t.Run(test.name, func(t *testing.T) {
			stream, err := client.Stream(context.Background(), test.request)
			if stream != nil {
				t.Fatalf("Stream() = %#v", stream)
			}
			requireModelError(t, err, test.kind, "stream")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Stream() error = %v, want %q", err, test.want)
			}
		})
	}

	if calls.Load() != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls.Load())
	}
}

func TestStreamAcceptsStandardLineEndingsAndFinalEOFEvent(t *testing.T) {
	for _, test := range []struct {
		name      string
		separator string
	}{
		{name: "LF", separator: "\n"},
		{name: "CRLF", separator: "\r\n"},
		{name: "CR", separator: "\r"},
	} {
		t.Run(test.name, func(t *testing.T) {
			separator := test.separator
			body := streamUTF8BOM + "event: message_start" + separator +
				"data: " + validStreamStart + separator + separator +
				"event: message_delta" + separator +
				"data: " + streamFinishEvent("end_turn", 2) + separator + separator +
				"event: message_stop" + separator +
				"data: {\"type\":\"message_stop\"}"
			client := newStaticStreamClient(t, http.StatusOK, "text/event-stream", body)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			chunks, terminal := receiveAll(stream)
			if !errors.Is(terminal, io.EOF) {
				t.Fatalf("Recv() terminal = %v", terminal)
			}
			_ = stream.Close()
			if len(chunks) != 1 || chunks[0].FinishReason != llm.FinishReasonStop {
				t.Fatalf("chunks = %#v", chunks)
			}
		})
	}
}

func TestStreamMapsFinishReasons(t *testing.T) {
	for reason, want := range map[string]llm.FinishReason{
		"end_turn":                      llm.FinishReasonStop,
		"stop_sequence":                 llm.FinishReasonStop,
		"max_tokens":                    llm.FinishReasonLength,
		"model_context_window_exceeded": llm.FinishReasonLength,
		"refusal":                       llm.FinishReasonContentFilter,
	} {
		t.Run(reason, func(t *testing.T) {
			body := streamEvent("message_start", validStreamStart) +
				streamEvent("message_delta", streamFinishEvent(reason, 2)) +
				streamEvent("message_stop", `{"type":"message_stop"}`)
			client := newStaticStreamClient(t, http.StatusOK, "text/event-stream", body)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			chunks, terminal := receiveAll(stream)
			if !errors.Is(terminal, io.EOF) || len(chunks) != 1 || chunks[0].FinishReason != want {
				t.Fatalf("stream result = (%#v, %v), want %q", chunks, terminal, want)
			}
			_ = stream.Close()
		})
	}
}

func TestStreamEmitsPartialToolArguments(t *testing.T) {
	body := streamEvent("message_start", validStreamStart) +
		streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_one","name":"lookup","input":{}}}`) +
		streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\""}}`) +
		streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"Paris\"}"}}`) +
		streamEvent("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		streamEvent("message_delta", streamFinishEvent("tool_use", 2)) +
		streamEvent("message_stop", `{"type":"message_stop"}`)
	client := newStaticStreamClientWithCapabilities(t, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, http.StatusOK, "text/event-stream", body)
	stream, err := client.Stream(context.Background(), toolRequest())
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunks, terminal := receiveAll(stream)
	if !errors.Is(terminal, io.EOF) {
		t.Fatalf("Recv() terminal = %v", terminal)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if len(chunks) != 5 {
		t.Fatalf("len(chunks) = %d, want 5: %#v", len(chunks), chunks)
	}
	wantKinds := [][]llm.StreamEventKind{
		{llm.StreamEventStart, llm.StreamEventToolCallStart},
		{llm.StreamEventToolCallDelta},
		{llm.StreamEventToolCallDelta},
		{llm.StreamEventToolCallEnd},
		{llm.StreamEventDone},
	}
	for index, want := range wantKinds {
		if len(chunks[index].Events) != len(want) {
			t.Fatalf("chunk %d events = %#v", index, chunks[index].Events)
		}
		for eventIndex, kind := range want {
			if chunks[index].Events[eventIndex].Kind != kind {
				t.Fatalf("chunk %d event %d = %q, want %q", index, eventIndex, chunks[index].Events[eventIndex].Kind, kind)
			}
		}
	}
	if got := chunks[1].Events[0].Delta + chunks[2].Events[0].Delta; got != `{"city":"Paris"}` {
		t.Fatalf("tool argument deltas = %q", got)
	}
	final := chunks[4]
	if final.FinishReason != llm.FinishReasonToolCall || len(final.ToolCalls) != 1 || string(final.ToolCalls[0].Arguments) != `{"city":"Paris"}` {
		t.Fatalf("final chunk = %#v", final)
	}
	endCall := chunks[3].Events[0].ToolCall
	if endCall == nil || endCall.ID != "toolu_one" || endCall.Name != "lookup" || string(endCall.Arguments) != `{"city":"Paris"}` {
		t.Fatalf("tool end event = %#v", chunks[3].Events[0])
	}
	endCall.Arguments[0] = '['
	if string(final.ToolCalls[0].Arguments) != `{"city":"Paris"}` {
		t.Fatalf("event arguments alias final tool call: %s", final.ToolCalls[0].Arguments)
	}
}

func TestStreamRejectsMalformedToolEvents(t *testing.T) {
	start := streamEvent("message_start", validStreamStart)
	for _, test := range []struct {
		name string
		body string
		want string
	}{
		{name: "missing ID", body: start + streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"lookup","input":{}}}`), want: "no ID"},
		{name: "undeclared name", body: start + streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"missing","input":{}}}`), want: "undeclared"},
		{name: "non-object input", body: start + streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":[]}}`), want: "JSON object"},
		{name: "malformed completed arguments", body: start +
			streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}}`) +
			streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{"}}`) +
			streamEvent("content_block_stop", `{"type":"content_block_stop","index":0}`), want: "not a strict JSON object"},
		{name: "duplicate ID", body: start +
			streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}}`) +
			streamEvent("content_block_stop", `{"type":"content_block_stop","index":0}`) +
			streamEvent("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}}`), want: "repeats ID"},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := newStaticStreamClientWithCapabilities(t, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, http.StatusOK, "text/event-stream", test.body)
			stream, err := client.Stream(context.Background(), toolRequest())
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, llm.KindMalformedResponse, "stream")
			if !strings.Contains(terminal.Error(), test.want) {
				t.Fatalf("Recv() error = %v, want %q", terminal, test.want)
			}
			_ = stream.Close()
		})
	}
}

func TestStreamRejectsMalformedProtocol(t *testing.T) {
	start := streamEvent("message_start", validStreamStart)
	finish := streamEvent("message_delta", streamFinishEvent("end_turn", 2))
	stop := streamEvent("message_stop", `{"type":"message_stop"}`)
	textStart := streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
	for _, test := range []struct {
		name string
		body string
		kind llm.ErrorKind
		want string
	}{
		{name: "invalid JSON", body: streamEvent("message_start", `{`), kind: llm.KindMalformedResponse, want: "decode"},
		{name: "duplicate member", body: streamEvent("message_start", `{"type":"message_start","type":"message_start"}`), kind: llm.KindMalformedResponse, want: "duplicate object member"},
		{name: "missing event name", body: "data: " + validStreamStart + "\n\n", kind: llm.KindMalformedResponse, want: "no name"},
		{name: "event type mismatch", body: streamEvent("ping", `{"type":"message_start"}`), kind: llm.KindMalformedResponse, want: "does not match"},
		{name: "repeated start", body: start + start, kind: llm.KindMalformedResponse, want: "repeated message_start"},
		{name: "content before start", body: textStart, kind: llm.KindMalformedResponse, want: "before message_start"},
		{name: "missing message", body: streamEvent("message_start", `{"type":"message_start"}`), kind: llm.KindMalformedResponse, want: "no message"},
		{name: "missing ID", body: streamEvent("message_start", strings.Replace(validStreamStart, `"id":"msg_stream",`, "", 1)), kind: llm.KindMalformedResponse, want: "no message ID"},
		{name: "wrong message type", body: streamEvent("message_start", strings.Replace(validStreamStart, `"type":"message"`, `"type":"other"`, 1)), kind: llm.KindMalformedResponse, want: "want message"},
		{name: "wrong role", body: streamEvent("message_start", strings.Replace(validStreamStart, `"role":"assistant"`, `"role":"user"`, 1)), kind: llm.KindMalformedResponse, want: "want assistant"},
		{name: "nonempty initial content", body: streamEvent("message_start", strings.Replace(validStreamStart, `"content":[]`, `"content":[{}]`, 1)), kind: llm.KindMalformedResponse, want: "empty array"},
		{name: "missing model", body: streamEvent("message_start", strings.Replace(validStreamStart, `"model":"claude-served",`, "", 1)), kind: llm.KindMalformedResponse, want: "no model"},
		{name: "non-null initial stop", body: streamEvent("message_start", strings.Replace(validStreamStart, `"stop_reason":null`, `"stop_reason":"end_turn"`, 1)), kind: llm.KindMalformedResponse, want: "must be null"},
		{name: "incomplete initial usage", body: streamEvent("message_start", strings.Replace(validStreamStart, `"input_tokens":1,`, "", 1)), kind: llm.KindMalformedResponse, want: "incomplete"},
		{name: "negative initial usage", body: streamEvent("message_start", strings.Replace(validStreamStart, `"input_tokens":1`, `"input_tokens":-1`, 1)), kind: llm.KindMalformedResponse, want: "negative"},
		{name: "missing block index", body: start + streamEvent("content_block_start", `{"type":"content_block_start","content_block":{"type":"text","text":""}}`), kind: llm.KindMalformedResponse, want: "index is missing"},
		{name: "block index gap", body: start + streamEvent("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`), kind: llm.KindMalformedResponse, want: "want 0"},
		{name: "missing block type", body: start + streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"text":""}}`), kind: llm.KindMalformedResponse, want: "no content block type"},
		{name: "unsupported block", body: start + streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`), kind: llm.KindUnsupported, want: "thinking"},
		{name: "delta without block", body: start + streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}`), kind: llm.KindMalformedResponse, want: "without an active block"},
		{name: "delta index mismatch", body: start + textStart + streamEvent("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"x"}}`), kind: llm.KindMalformedResponse, want: "want 0"},
		{name: "missing delta type", body: start + textStart + streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"text":"x"}}`), kind: llm.KindMalformedResponse, want: "no delta type"},
		{name: "missing delta text", body: start + textStart + streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta"}}`), kind: llm.KindMalformedResponse, want: "has no text"},
		{name: "unsupported delta", body: start + textStart + streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"x"}}`), kind: llm.KindUnsupported, want: "thinking_delta"},
		{name: "message delta during block", body: start + textStart + finish, kind: llm.KindMalformedResponse, want: "before content block"},
		{name: "content after message delta", body: start + streamEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":null},"usage":{"output_tokens":1}}`) + textStart, kind: llm.KindMalformedResponse, want: "after message_delta"},
		{name: "finish without delta usage", body: start + streamEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`), kind: llm.KindMalformedResponse, want: "without cumulative output usage"},
		{name: "finish with empty delta usage", body: start + streamEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{}}`), kind: llm.KindMalformedResponse, want: "without cumulative output usage"},
		{name: "usage decreases", body: start + streamEvent("message_delta", streamFinishEvent("end_turn", 0)), kind: llm.KindMalformedResponse, want: "decreased"},
		{name: "thinking exceeds output", body: start + streamEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2,"output_tokens_details":{"thinking_tokens":3}}}`), kind: llm.KindMalformedResponse, want: "thinking tokens exceed"},
		{name: "unknown stop reason", body: start + streamEvent("message_delta", streamFinishEvent("future", 2)), kind: llm.KindMalformedResponse, want: "unsupported stop reason"},
		{name: "pause turn", body: start + streamEvent("message_delta", streamFinishEvent("pause_turn", 2)), kind: llm.KindUnsupported, want: "pause_turn"},
		{name: "tool use without call", body: start + streamEvent("message_delta", streamFinishEvent("tool_use", 2)), kind: llm.KindMalformedResponse, want: "has no tool calls"},
		{name: "stop before finish", body: start + stop, kind: llm.KindMalformedResponse, want: "before a finish reason"},
		{name: "EOF before stop", body: start + finish, kind: llm.KindMalformedResponse, want: "ended before message_stop"},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := newStaticStreamClient(t, http.StatusOK, "text/event-stream", test.body)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, test.kind, "stream")
			if !strings.Contains(terminal.Error(), test.want) {
				t.Fatalf("Recv() error = %v, want %q", terminal, test.want)
			}
			_ = stream.Close()
		})
	}
}

func TestStreamPreservesMalformedEventCause(t *testing.T) {
	body := "event: message_start\ndata: {\n\n"
	client := newStaticStreamClient(t, http.StatusOK, "text/event-stream", body)
	stream, err := client.Stream(context.Background(), textRequest("hello"))
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

func TestStreamClassifiesTransportHTTPAndEventErrors(t *testing.T) {
	t.Run("transport", func(t *testing.T) {
		sentinel := errors.New("network down")
		client, err := New(Config{
			Model:        "model",
			Capabilities: []llm.Capability{llm.CapabilityStreaming},
			BaseURL:      "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, sentinel
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		stream, err := client.Stream(context.Background(), textRequest("hello"))
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		_, terminal := receiveAll(stream)
		requireModelError(t, terminal, llm.KindTransport, "stream")
		if !errors.Is(terminal, sentinel) {
			t.Fatalf("Recv() error = %v, want %v", terminal, sentinel)
		}
		_ = stream.Close()
	})

	t.Run("HTTP", func(t *testing.T) {
		body := `{"type":"error","error":{"type":"authentication_error","message":"bad key"},"request_id":"req_body"}`
		client := newStaticStreamClient(t, http.StatusUnauthorized, "application/json", body)
		stream, err := client.Stream(context.Background(), textRequest("hello"))
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		_, terminal := receiveAll(stream)
		modelErr := requireModelError(t, terminal, llm.KindAuthentication, "stream")
		if modelErr.HTTPStatus != http.StatusUnauthorized {
			t.Fatalf("HTTP status = %d", modelErr.HTTPStatus)
		}
		var apiErr *APIError
		if !errors.As(terminal, &apiErr) || apiErr.Message != "bad key" || apiErr.RequestID != "req_body" {
			t.Fatalf("API error = %#v", apiErr)
		}
		_ = stream.Close()
	})

	t.Run("event", func(t *testing.T) {
		body := streamEvent("error", `{"type":"error","error":{"type":"overloaded_error","message":"busy"},"request_id":"req_event"}`)
		client := newStaticStreamClient(t, http.StatusOK, "text/event-stream", body)
		stream, err := client.Stream(context.Background(), textRequest("hello"))
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		_, terminal := receiveAll(stream)
		requireModelError(t, terminal, llm.KindProvider, "stream")
		var apiErr *APIError
		if !errors.As(terminal, &apiErr) || apiErr.Type != "overloaded_error" || apiErr.RequestID != "req_event" {
			t.Fatalf("API error = %#v", apiErr)
		}
		_ = stream.Close()
	})

	t.Run("event request ID header fallback", func(t *testing.T) {
		body := streamEvent("error", `{"type":"error","error":{"type":"overloaded_error","message":"busy"}}`)
		client, err := New(Config{
			Model:        "model",
			Capabilities: []llm.Capability{llm.CapabilityStreaming},
			BaseURL:      "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     http.StatusText(http.StatusOK),
					Header: http.Header{
						"Content-Type": {"text/event-stream"},
						"Request-Id":   {"req_header"},
					},
					Body: io.NopCloser(strings.NewReader(body)),
				}, nil
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		stream, err := client.Stream(context.Background(), textRequest("hello"))
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		_, terminal := receiveAll(stream)
		requireModelError(t, terminal, llm.KindProvider, "stream")
		var apiErr *APIError
		if !errors.As(terminal, &apiErr) || apiErr.RequestID != "req_header" {
			t.Fatalf("API error = %#v", apiErr)
		}
		_ = stream.Close()
	})

	t.Run("content type", func(t *testing.T) {
		client := newStaticStreamClient(t, http.StatusOK, "application/json", `{}`)
		stream, err := client.Stream(context.Background(), textRequest("hello"))
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		_, terminal := receiveAll(stream)
		requireModelError(t, terminal, llm.KindMalformedResponse, "stream")
		_ = stream.Close()
	})
}

func TestStreamClassifiesEventErrorKinds(t *testing.T) {
	tests := []struct {
		name      string
		errorType string
		message   string
		kind      llm.ErrorKind
	}{
		{name: "authentication", errorType: "authentication_error", message: "bad key", kind: llm.KindAuthentication},
		{name: "permission", errorType: "permission_error", message: "forbidden", kind: llm.KindPermission},
		{name: "billing", errorType: "billing_error", message: "payment required", kind: llm.KindPermission},
		{name: "rate limit", errorType: "rate_limit_error", message: "slow down", kind: llm.KindRateLimit},
		{name: "invalid request", errorType: "invalid_request_error", message: "bad input", kind: llm.KindInvalidRequest},
		{name: "not found", errorType: "not_found_error", message: "unknown model", kind: llm.KindInvalidRequest},
		{name: "context limit", errorType: "invalid_request_error", message: "prompt is too long", kind: llm.KindContextLimit},
		{name: "overloaded", errorType: "overloaded_error", message: "busy", kind: llm.KindProvider},
		{name: "API", errorType: "api_error", message: "failed", kind: llm.KindProvider},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			data := fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":%q}}`, test.errorType, test.message)
			body := streamEvent("error", data)
			client := newStaticStreamClient(t, http.StatusOK, "text/event-stream", body)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, test.kind, "stream")
			var apiErr *APIError
			if !errors.As(terminal, &apiErr) || apiErr.Type != test.errorType {
				t.Fatalf("Recv() error = %v, API error = %#v", terminal, apiErr)
			}
			_ = stream.Close()
		})
	}
}

func TestEventStreamRejectsSizeLimits(t *testing.T) {
	for _, test := range []struct {
		name string
		body string
		want string
	}{
		{name: "line", body: "data: " + strings.Repeat("x", maxStreamEventBytes+1) + "\n\n", want: "line exceeds"},
		{name: "event", body: strings.Repeat(": keepalive\n", maxStreamEventBytes/len(": keepalive\n")+1), want: "event exceeds"},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := readEventStream(context.Background(), strings.NewReader(test.body), newStreamDecoder("", nil, nil), func(llm.Chunk) bool { return true })
			requireModelError(t, err, llm.KindMalformedResponse, "stream")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("readEventStream() error = %v", err)
			}
		})
	}
}

func TestStreamCloseAndCancellationReleaseBody(t *testing.T) {
	first := streamEvent("message_start", validStreamStart) +
		streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ready"}}`)

	t.Run("Close", func(t *testing.T) {
		body := newBlockingStreamBody(first)
		client := streamClientWithBody(t, body)
		stream, err := client.Stream(context.Background(), textRequest("hello"))
		if err != nil {
			t.Fatalf("Stream() error = %v", err)
		}
		chunk, err := stream.Recv()
		if err != nil || chunk.Content[0].Text != "ready" {
			t.Fatalf("Recv() = (%#v, %v)", chunk, err)
		}
		if err := stream.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		select {
		case <-body.closed:
		default:
			t.Fatal("Close returned before response body closed")
		}
		if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
			t.Fatalf("Recv() after Close = %v", err)
		}
	})

	t.Run("Close error", func(t *testing.T) {
		closeErr := errors.New("close failed")
		body := newBlockingStreamBody("")
		body.closeErr = closeErr
		client := streamClientWithBody(t, body)
		stream, err := client.Stream(context.Background(), textRequest("hello"))
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
	})

	t.Run("cancellation cause", func(t *testing.T) {
		closeErr := errors.New("close failed")
		body := newBlockingStreamBody("")
		body.closeErr = closeErr
		client := streamClientWithBody(t, body)
		cause := errors.New("cancel stream")
		ctx, cancel := context.WithCancelCause(context.Background())
		stream, err := client.Stream(ctx, textRequest("hello"))
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
			t.Fatal("cancellation left response body open")
		}
	})
}

func TestStreamPreservesReadAndCloseErrors(t *testing.T) {
	readErr := errors.New("read failed")
	closeErr := errors.New("close failed")
	for _, test := range []struct {
		name        string
		status      int
		contentType string
		body        *errorBody
		kind        llm.ErrorKind
		wantRead    bool
	}{
		{
			name:        "event read",
			status:      http.StatusOK,
			contentType: "text/event-stream",
			body:        &errorBody{readErr: readErr, closeErr: closeErr},
			kind:        llm.KindTransport,
			wantRead:    true,
		},
		{
			name:        "malformed event",
			status:      http.StatusOK,
			contentType: "text/event-stream",
			body:        &errorBody{data: streamEvent("message_start", `{`), closeErr: closeErr},
			kind:        llm.KindMalformedResponse,
		},
		{
			name:        "HTTP error",
			status:      http.StatusUnauthorized,
			contentType: "application/json",
			body:        &errorBody{data: `{"type":"error","error":{"message":"bad key"}}`, closeErr: closeErr},
			kind:        llm.KindAuthentication,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			client := streamClientWithResponse(t, test.status, test.contentType, test.body)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, test.kind, "stream")
			if !errors.Is(terminal, closeErr) {
				t.Fatalf("Recv() error = %v, want %v", terminal, closeErr)
			}
			if test.wantRead && !errors.Is(terminal, readErr) {
				t.Fatalf("Recv() error = %v, want %v", terminal, readErr)
			}
			_ = stream.Close()
		})
	}
}

func TestProduceStreamCancellationPreservesBodyCloseError(t *testing.T) {
	closeErr := errors.New("close failed")
	body := newBlockingStreamBody("")
	body.closeErr = closeErr
	client := streamClientWithBody(t, body)
	cause := errors.New("cancel stream")
	ctx, cancel := context.WithCancelCause(context.Background())
	result := make(chan error, 1)
	go func() {
		result <- client.produceStream(ctx, []byte(`{}`), nil, nil, func(llm.Chunk) bool { return true })
	}()
	<-body.reading
	cancel(cause)
	err := <-result
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
						Body:       &errorBody{closeErr: closeErr},
					}, nil
				}
			case "HTTP error response":
				body := newBlockingStreamBody(`{"type":"error","error":{"message":"partial"}}`)
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
			stream, err := client.Stream(ctx, textRequest("hello"))
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
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			panic("response writer does not support flushing")
		}
		_, _ = io.WriteString(writer, streamEvent("message_start", validStreamStart))
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
		BaseURL:      server.URL + "/v1",
		HTTPClient:   server.Client(),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.Stream(ctx, textRequest("hello"))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	<-requestStarted
	cancel()
	if _, err := stream.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv() error = %v", err)
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

const validStreamStart = `{"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","content":[],"model":"claude-served","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":1}}}`

func streamFinishEvent(reason string, outputTokens int) string {
	encoded, err := jsonv2.Marshal(reason)
	if err != nil {
		panic(err)
	}
	return `{"type":"message_delta","delta":{"stop_reason":` + string(encoded) + `},"usage":{"output_tokens":` +
		string(mustJSON(outputTokens)) + `}}`
}

func streamEvent(name, data string) string {
	return "event: " + name + "\ndata: " + data + "\n\n"
}

func mustJSON(value any) []byte {
	data, err := jsonv2.Marshal(value)
	if err != nil {
		panic(err)
	}
	return data
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

func newStaticStreamClient(t *testing.T, status int, contentType, body string) *Client {
	t.Helper()
	return newStaticStreamClientWithCapabilities(t, []llm.Capability{llm.CapabilityStreaming}, status, contentType, body)
}

func newStaticStreamClientWithCapabilities(t *testing.T, capabilities []llm.Capability, status int, contentType, body string) *Client {
	t.Helper()
	client, err := New(Config{
		Model:        "model",
		Capabilities: capabilities,
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: status,
				Status:     http.StatusText(status),
				Header:     http.Header{"Content-Type": {contentType}},
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client
}

func streamClientWithBody(t *testing.T, body io.ReadCloser) *Client {
	return streamClientWithResponse(t, http.StatusOK, "text/event-stream", body)
}

func streamClientWithResponse(t *testing.T, status int, contentType string, body io.ReadCloser) *Client {
	t.Helper()
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: status,
				Status:     http.StatusText(status),
				Header:     http.Header{"Content-Type": {contentType}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client
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
		count := copy(buffer, body.first)
		body.first = body.first[count:]
		return count, nil
	}
	body.readOnce.Do(func() { close(body.reading) })
	<-body.closed
	return 0, errors.New("body closed")
}

func (body *blockingStreamBody) Close() error {
	body.once.Do(func() { close(body.closed) })
	return body.closeErr
}
