package gemini

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestStreamTranslatesTextAndFinalMetadata(t *testing.T) {
	type requestRecord struct {
		path   string
		query  string
		header http.Header
		body   []byte
	}
	records := make(chan requestRecord, 1)
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		records <- requestRecord{
			path: request.URL.EscapedPath(), query: request.URL.RawQuery,
			header: request.Header.Clone(), body: body,
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, strings.Join([]string{
			`data: {"responseId":"response-1","modelVersion":"served-model","candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]} }],"usageMetadata":{"promptTokenCount":2,"cachedContentTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":3}}` + "\n\n",
			`data: {"responseId":"response-1","modelVersion":"served-model","candidates":[{"content":{"role":"model","parts":[{"text":"lo","thoughtSignature":"c2ln"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"cachedContentTokenCount":1,"candidatesTokenCount":2,"thoughtsTokenCount":1,"totalTokenCount":5}}` + "\n\n",
		}, ""))
	}))

	client, err := New(Config{
		Model:        "request-model",
		APIKey:       "secret",
		BaseURL:      server.URL,
		HTTPClient:   server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		Headers:      http.Header{"X-Route": {"owned"}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
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
	if len(chunks) != 3 {
		t.Fatalf("chunks = %#v, want two content chunks and one final chunk", chunks)
	}
	if chunkText(chunks[0]) != "hel" || chunkText(chunks[1]) != "lo" {
		t.Fatalf("content chunks = %#v", chunks[:2])
	}
	if len(chunks[0].Events) != 3 || chunks[0].Events[0].Kind != llm.StreamEventStart ||
		chunks[0].Events[1].Kind != llm.StreamEventTextStart || chunks[0].Events[2].Kind != llm.StreamEventTextDelta ||
		len(chunks[1].Events) != 1 || chunks[1].Events[0].Kind != llm.StreamEventTextDelta {
		t.Fatalf("content events = %#v, %#v", chunks[0].Events, chunks[1].Events)
	}
	for index, chunk := range chunks[:2] {
		if chunk.ID != "" || chunk.Model != "" || chunk.FinishReason != "" || chunk.Usage != nil || len(chunk.ProviderData) != 0 {
			t.Fatalf("content chunk %d contains premature metadata: %#v", index, chunk)
		}
	}
	final := chunks[2]
	if final.ID != "response-1" || final.Model != "served-model" || final.FinishReason != llm.FinishReasonStop ||
		len(final.Events) != 2 || final.Events[0].Kind != llm.StreamEventTextEnd || final.Events[1].Kind != llm.StreamEventDone {
		t.Fatalf("final chunk metadata = %#v", final)
	}
	if final.Usage == nil || *final.Usage != (llm.Usage{InputTokens: 1, OutputTokens: 3, CacheReadTokens: 1, ReasoningTokens: 1, TotalTokens: 5}) {
		t.Fatalf("final usage = %#v", final.Usage)
	}
	data, recognized, err := parseMessageData(final.ProviderData)
	if err != nil || !recognized || len(data.Parts) != 2 || string(data.Parts[1].ThoughtSignature) != "sig" {
		t.Fatalf("final provider data = (%#v, %v, %v)", data, recognized, err)
	}

	record := <-records
	if record.path != "/v1beta/models/request-model:streamGenerateContent" || record.query != "alt=sse" {
		t.Fatalf("stream endpoint = %q?%s", record.path, record.query)
	}
	if record.header.Get("X-Goog-Api-Key") != "secret" || record.header.Get("X-Route") != "owned" {
		t.Fatalf("stream headers = %#v", record.header)
	}
	var payload map[string]any
	if err := jsonv2.Unmarshal(record.body, &payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	contents, ok := payload["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("request contents = %#v", payload["contents"])
	}
}

func TestStreamTranslatesToolCallAndThoughtSignature(t *testing.T) {
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, `data: {"responseId":"tool-response","candidates":[{"content":{"role":"model","parts":[{"functionCall":{"id":"call-1","name":"weather","args":{"city":"Boston"}},"thoughtSignature":"c2lnbmF0dXJl"},{"text":"after"},{"inlineData":{"mimeType":"image/png","data":"AQ=="}}]},"finishReason":"STOP"}]}`+"\n\n")
	}))
	client := mustTestClient(t, server, llm.CapabilityStreaming, llm.CapabilityTools)
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: textRequest("weather?").Messages,
		Tools: []llm.Tool{{
			Name: "weather", InputSchema: []byte(`{"type":"object"}`),
		}},
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
	if len(chunks) != 2 || len(chunks[0].ToolCalls) != 1 {
		t.Fatalf("chunks = %#v", chunks)
	}
	if len(chunks[0].Events) != 5 || chunks[0].Events[0].Kind != llm.StreamEventStart || chunks[0].Events[1].Kind != llm.StreamEventToolCallStart || chunks[0].Events[2].Kind != llm.StreamEventToolCallEnd || chunks[0].Events[2].ToolCall == nil || chunks[0].Events[3].Kind != llm.StreamEventTextStart || chunks[0].Events[4].Kind != llm.StreamEventTextDelta || chunks[0].Events[4].Delta != "after" {
		t.Fatalf("tool call events = %#v", chunks[0].Events)
	}
	call := chunks[0].ToolCalls[0]
	if call.ID != "call-1" || call.Name != "weather" {
		t.Fatalf("tool call = %#v", call)
	}
	var arguments map[string]any
	if err := jsonv2.Unmarshal(call.Arguments, &arguments); err != nil || arguments["city"] != "Boston" {
		t.Fatalf("tool arguments = (%s, %v)", call.Arguments, err)
	}
	final := chunks[1]
	if final.ID != "tool-response" || final.FinishReason != llm.FinishReasonToolCall {
		t.Fatalf("final chunk = %#v", final)
	}
	data, recognized, err := parseMessageData(final.ProviderData)
	if err != nil || !recognized || len(data.Parts) != 3 || data.Parts[0].Kind != "tool_call" || data.Parts[1].Kind != "content" || data.Parts[2].Kind != "content" ||
		string(data.Parts[0].ThoughtSignature) != "signature" {
		t.Fatalf("provider data = (%#v, %v, %v)", data, recognized, err)
	}
}

func TestStreamRepresentsBlockedOutputWithoutContent(t *testing.T) {
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, `data: {"candidates":[{"finishReason":"SAFETY"}]}`+"\n\n")
	}))
	client := mustTestClient(t, server, llm.CapabilityStreaming)
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
	if len(chunks) != 1 || chunks[0].FinishReason != llm.FinishReasonContentFilter || chunks[0].Model != "gemini-model" {
		t.Fatalf("chunks = %#v", chunks)
	}
}

func TestStreamAcceptsFirstBlockedFeedbackAndTrailingMetadata(t *testing.T) {
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, `data: {"promptFeedback":{"blockReason":"SAFETY"}}`+"\n\n")
		_, _ = io.WriteString(writer, `data: {"responseId":"response-1","modelVersion":"served-model","usageMetadata":{"promptTokenCount":2,"totalTokenCount":2}}`+"\n\n")
	}))
	client := mustTestClient(t, server, llm.CapabilityStreaming)
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
	if len(chunks) != 1 || chunks[0].ID != "response-1" || chunks[0].Model != "served-model" ||
		chunks[0].FinishReason != llm.FinishReasonContentFilter || len(chunks[0].Content) != 0 ||
		!reflect.DeepEqual(chunks[0].Events, []llm.StreamEvent{
			{Kind: llm.StreamEventStart},
			{Kind: llm.StreamEventDone, FinishReason: llm.FinishReasonContentFilter},
		}) || chunks[0].Usage == nil || *chunks[0].Usage != (llm.Usage{InputTokens: 2, TotalTokens: 2}) {
		t.Fatalf("chunks = %#v", chunks)
	}
}

func TestStreamTreatsAbsentOrUnspecifiedPromptFeedbackAsNonBlocking(t *testing.T) {
	for _, test := range []struct {
		name     string
		feedback string
	}{
		{name: "absent"},
		{name: "unspecified", feedback: `"promptFeedback":{"blockReason":"BLOCKED_REASON_UNSPECIFIED"},`},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(writer, `data: {`+test.feedback+`"candidates":[{"content":{"role":"model","parts":[{"text":"visible"}]},"finishReason":"STOP"}]}`+"\n\n")
			}))
			client := mustTestClient(t, server, llm.CapabilityStreaming)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			chunks, terminal := receiveAll(stream)
			if !errors.Is(terminal, io.EOF) || len(chunks) != 2 || chunkText(chunks[0]) != "visible" || chunks[1].FinishReason != llm.FinishReasonStop {
				t.Fatalf("stream result = (%#v, %v)", chunks, terminal)
			}
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		})
	}
}

func TestStreamRejectsLateOrContradictoryPromptBlocking(t *testing.T) {
	tests := []struct {
		name       string
		events     []string
		wantChunks int
	}{
		{
			name: "candidate then blocked feedback",
			events: []string{
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"visible"}]}}]}`,
				`{"promptFeedback":{"blockReason":"SAFETY"}}`,
			},
			wantChunks: 1,
		},
		{
			name: "metadata then blocked feedback",
			events: []string{
				`{"responseId":"response-1"}`,
				`{"responseId":"response-1","promptFeedback":{"blockReason":"SAFETY"}}`,
			},
		},
		{
			name: "candidate after blocked feedback",
			events: []string{
				`{"promptFeedback":{"blockReason":"SAFETY"}}`,
				`{"candidates":[{"content":{"role":"model","parts":[{"text":"visible"}]},"finishReason":"STOP"}]}`,
			},
		},
		{
			name:   "candidate alongside blocked feedback",
			events: []string{`{"promptFeedback":{"blockReason":"SAFETY"},"candidates":[{"content":{"role":"model","parts":[{"text":"visible"}]}}]}`},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				for _, event := range test.events {
					_, _ = io.WriteString(writer, "data: "+event+"\n\n")
				}
			}))
			client := mustTestClient(t, server, llm.CapabilityStreaming)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			chunks, terminal := receiveAll(stream)
			requireModelError(t, terminal, llm.KindMalformedResponse, "stream", defaultProvider)
			if len(chunks) != test.wantChunks {
				t.Fatalf("len(chunks) = %d, want %d: %#v", len(chunks), test.wantChunks, chunks)
			}
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		})
	}
}

func TestStreamValidatesCompletedJSON(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		wantKind llm.ErrorKind
	}{
		{name: "strict JSON", text: `{"value":1}`},
		{name: "duplicate object name", text: `{"value":1,"value":2}`, wantKind: llm.KindMalformedResponse},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				first := test.text[:len(test.text)/2]
				second := test.text[len(test.text)/2:]
				_, _ = io.WriteString(writer, "data: "+streamTextResponse(first, "")+"\n\n")
				_, _ = io.WriteString(writer, "data: "+streamTextResponse(second, "STOP")+"\n\n")
			}))
			client := mustTestClient(t, server, llm.CapabilityStreaming, llm.CapabilityJSON)
			request := textRequest("JSON")
			request.ResponseFormat = llm.ResponseFormatJSON
			stream, err := client.Stream(context.Background(), request)
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			chunks, terminal := receiveAll(stream)
			if test.wantKind == 0 {
				if !errors.Is(terminal, io.EOF) || chunkText(chunks[0])+chunkText(chunks[1]) != test.text {
					t.Fatalf("stream result = (%#v, %v)", chunks, terminal)
				}
			} else {
				requireModelError(t, terminal, test.wantKind, "stream", defaultProvider)
			}
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		})
	}
}

func TestStreamClassifiesProviderAndMalformedEvents(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		kind   llm.ErrorKind
	}{
		{
			name: "HTTP authentication error", status: http.StatusUnauthorized,
			body: `{"error":{"code":401,"status":"UNAUTHENTICATED","message":"bad key"}}`,
			kind: llm.KindAuthentication,
		},
		{
			name: "malformed SSE", status: http.StatusOK,
			body: "data: {\n\n", kind: llm.KindMalformedResponse,
		},
		{
			name: "malformed provider shape", status: http.StatusOK,
			body: `data: {"candidates":{}}` + "\n\n", kind: llm.KindMalformedResponse,
		},
		{
			name: "in-stream rate limit", status: http.StatusOK,
			body: `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"slow down"}}` + "\n\n", kind: llm.KindRateLimit,
		},
		{
			name: "missing unblocked content", status: http.StatusOK,
			body: `data: {"candidates":[{"finishReason":"STOP"}]}` + "\n\n", kind: llm.KindMalformedResponse,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				if test.status == http.StatusOK {
					writer.Header().Set("Content-Type", "text/event-stream")
				} else {
					writer.Header().Set("Content-Type", "application/json")
				}
				writer.WriteHeader(test.status)
				_, _ = io.WriteString(writer, test.body)
			}))
			client := mustTestClient(t, server, llm.CapabilityStreaming)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Stream() error = %v", err)
			}
			_, terminal := receiveAll(stream)
			requireModelError(t, terminal, test.kind, "stream", defaultProvider)
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		})
	}
}

func TestStreamRelabelsProviderError(t *testing.T) {
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(writer, `{"error":{"code":401,"status":"UNAUTHENTICATED","message":"bad key"}}`)
	}))
	client, err := New(Config{
		Provider:     "google-gateway",
		Model:        "model",
		APIKey:       "key",
		BaseURL:      server.URL,
		HTTPClient:   server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), textRequest("hello"))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	_, terminal := receiveAll(stream)
	requireModelError(t, terminal, llm.KindAuthentication, "stream", "google-gateway")
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestProduceStreamStopsCleanlyWhenEmissionStops(t *testing.T) {
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"unused"}]},"finishReason":"STOP"}]}`+"\n\n")
	}))
	client := mustTestClient(t, server, llm.CapabilityStreaming)
	request := textRequest("hello")
	contents, config, err := requestToSDK("stream", request)
	if err != nil {
		t.Fatalf("requestToSDK() error = %v", err)
	}
	if err := client.produceStream(context.Background(), request, contents, config, func(llm.Chunk) bool { return false }); err != nil {
		t.Fatalf("produceStream() error = %v, want nil", err)
	}
}

func TestStreamClassifiesResponseReadError(t *testing.T) {
	readErr := errors.New("read failed")
	client, err := New(Config{
		Model:        "model",
		APIKey:       "key",
		BaseURL:      "http://provider.example",
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"text/event-stream"}},
				Body:       &readErrorBody{err: readErr},
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
	requireModelError(t, terminal, llm.KindTransport, "stream", defaultProvider)
	if !errors.Is(terminal, readErr) {
		t.Fatalf("Recv() error = %v, want read cause", terminal)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
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
		_, _ = io.WriteString(writer, `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ready"}]}}]}`+"\n\n")
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
		APIKey:       "key",
		BaseURL:      server.URL,
		HTTPClient:   server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	cause := errors.New("stop stream")
	ctx, cancel := context.WithCancelCause(context.Background())
	stream, err := client.Stream(ctx, textRequest("hello"))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	chunk, err := stream.Recv()
	if err != nil || chunkText(chunk) != "ready" {
		t.Fatalf("Recv() = (%#v, %v)", chunk, err)
	}
	<-requestStarted
	cancel(cause)
	if _, err := stream.Recv(); !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Recv() error = %v, want context cancellation and cause", err)
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

func chunkText(chunk llm.Chunk) string {
	var text strings.Builder
	for _, part := range chunk.Content {
		if part.Kind == llm.PartText {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

func streamTextResponse(text, finish string) string {
	finishField := ""
	if finish != "" {
		finishField = `,"finishReason":"` + finish + `"`
	}
	encoded, err := jsonv2.Marshal(text)
	if err != nil {
		panic(err)
	}
	return `{"candidates":[{"content":{"role":"model","parts":[{"text":` + string(encoded) + `}]}` + finishField + `}]}`
}

type readErrorBody struct {
	err error
}

func (body *readErrorBody) Read([]byte) (int, error) {
	return 0, body.err
}

func (*readErrorBody) Close() error {
	return nil
}
