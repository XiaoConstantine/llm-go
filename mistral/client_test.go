package mistral

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestGenerateReasoningAndReplay(t *testing.T) {
	requests := make(chan map[string]any, 2)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", request.URL.Path)
		}
		if got := request.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("Authorization = %q", got)
		}
		if got := request.Header.Get("X-Affinity"); got != "configured-affinity" {
			t.Errorf("X-Affinity = %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		requests <- body
		writer.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		writeSSE(writer,
			`{"id":"resp","model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"consider "}]},{"type":"text","text":"Hello"}]},"finish_reason":null}]}`,
			`{"id":"resp","model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`,
			"[DONE]")
	}))
	defer server.Close()

	client := mustClient(t, Config{Model: "mistral-large-latest", Reasoning: true, APIKey: "secret", BaseURL: server.URL + "/v1",
		HTTPClient: server.Client(), Headers: http.Header{"X-Affinity": {"configured-affinity"}}})
	request := llm.Request{SessionID: "session", ReasoningEffort: llm.ReasoningEffortLow,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}}
	response, err := client.Generate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if response.ID != "resp" || response.Model != "mistral-large-latest" || response.Text() != "Hello world" ||
		response.ReasoningSummary != "consider " || response.FinishReason != llm.FinishReasonStop {
		t.Fatalf("response = %#v", response)
	}
	if response.Usage == nil || response.Usage.InputTokens != 3 || response.Usage.OutputTokens != 4 || response.Usage.TotalTokens != 7 {
		t.Fatalf("usage = %#v", response.Usage)
	}
	first := <-requests
	if first["prompt_mode"] != "reasoning" || first["stream"] != true || first["model"] != "mistral-large-latest" {
		t.Fatalf("first request = %#v", first)
	}

	request.Messages = append(request.Messages, response.Message, llm.Message{Role: llm.RoleUser, Content: []llm.Part{{Text: "again"}}})
	if _, err := client.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	second := <-requests
	messages := second["messages"].([]any)
	assistant := messages[1].(map[string]any)
	content := assistant["content"].([]any)
	thinking := content[0].(map[string]any)
	if thinking["type"] != "thinking" {
		t.Fatalf("replayed assistant = %#v", assistant)
	}
}

func TestReasoningEffortModelUsesHigh(t *testing.T) {
	payload, _, err := buildRequest("generate", "mistral-small-latest", true, false, llm.Request{
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}},
		ReasoningEffort: llm.ReasoningEffortMinimal,
	})
	if err != nil {
		t.Fatal(err)
	}
	var request map[string]any
	if err := json.Unmarshal(payload, &request); err != nil {
		t.Fatal(err)
	}
	if request["reasoning_effort"] != "high" {
		t.Fatalf("request = %s", payload)
	}
	if _, present := request["prompt_mode"]; present {
		t.Fatalf("request = %s", payload)
	}
}

func TestStreamFragmentedToolCallWithoutRepeatedID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writeSSE(writer,
			`{"id":"resp","model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"abc123XYZ","type":"function","function":{"name":"read","arguments":"{\"path\":"}}]},"finish_reason":null}]}`,
			`{"id":"resp","model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"file\"}"}}]},"finish_reason":null}]}`,
			`{"id":"resp","model":"model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`,
			"[DONE]")
	}))
	defer server.Close()
	client := mustClient(t, Config{Model: "model", APIKey: "secret", BaseURL: server.URL,
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, HTTPClient: server.Client()})
	tools := []llm.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}}
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "read"}}}}, Tools: tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	response, err := llm.CollectStructural(stream)
	if err != nil {
		t.Fatal(err)
	}
	if response.FinishReason != llm.FinishReasonToolCall || len(response.Message.ToolCalls) != 1 {
		t.Fatalf("response = %#v", response)
	}
	call := response.Message.ToolCalls[0]
	if call.ID != "abc123XYZ" || call.Name != "read" || string(call.Arguments) != `{"path":"file"}` {
		t.Fatalf("call = %#v", call)
	}
}

func TestStreamDiscardedUsageDoesNotEmitEmptyChunks(t *testing.T) {
	for _, test := range []struct {
		name       string
		metadata   string
		usageFirst bool
		repeated   bool
	}{
		{name: "discarded usage"},
		{name: "response ID", metadata: `,"id":"resp"`},
		{name: "model", metadata: `,"model":"reported-model"`},
		{name: "start event", usageFirst: true},
		{name: "repeated usage", repeated: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			usage := `{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":99}` + test.metadata + `}`
			text := `{"choices":[{"index":0,"delta":{"content":"answer"}}]}`
			events := []string{text, usage}
			if test.usageFirst {
				events = []string{usage, text}
			}
			if test.repeated {
				events = append(events, usage)
			}
			events = append(events, `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, "[DONE]")
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				writeSSE(writer, events...)
			}))
			defer server.Close()
			client := mustClient(t, Config{Model: "model", APIKey: "secret", BaseURL: server.URL, HTTPClient: server.Client(),
				Capabilities: []llm.Capability{llm.CapabilityStreaming}})
			stream, err := client.Stream(context.Background(), textRequest())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = stream.Close() }()
			var content strings.Builder
			var finish llm.FinishReason
			starts, metadataChunks := 0, 0
			for {
				chunk, recvErr := stream.Recv()
				if recvErr != nil {
					if test.repeated {
						var modelErr *llm.Error
						if !errors.As(recvErr, &modelErr) || modelErr.Kind != llm.KindMalformedResponse || !strings.Contains(recvErr.Error(), "repeated usage") {
							t.Fatalf("Recv() = %v, want repeated usage error", recvErr)
						}
					} else if !errors.Is(recvErr, io.EOF) {
						t.Fatal(recvErr)
					}
					break
				}
				if chunk.ID == "" && chunk.Model == "" && len(chunk.Content) == 0 && len(chunk.ToolCalls) == 0 &&
					chunk.ReasoningSummary == "" && len(chunk.ProviderData) == 0 && chunk.FinishReason == "" && chunk.Usage == nil && len(chunk.Events) == 0 {
					t.Fatal("stream emitted an empty chunk")
				}
				if chunk.Usage != nil {
					t.Fatalf("usage = %#v, want nil", chunk.Usage)
				}
				if (chunk.ID == "resp" || chunk.Model == "reported-model") && len(chunk.Content) == 0 && len(chunk.Events) == 0 && chunk.FinishReason == "" {
					metadataChunks++
				}
				for _, part := range chunk.Content {
					content.WriteString(part.Text)
				}
				for _, event := range chunk.Events {
					if event.Kind == llm.StreamEventStart {
						starts++
					}
				}
				if chunk.FinishReason != "" {
					finish = chunk.FinishReason
				}
			}
			if content.String() != "answer" || starts != 1 || !test.repeated && finish != llm.FinishReasonStop {
				t.Fatalf("text = %q, starts = %d, finish = %q", content.String(), starts, finish)
			}
			if test.metadata != "" && metadataChunks != 1 {
				t.Fatalf("metadata-only chunks = %d, want 1", metadataChunks)
			}
		})
	}
}

func TestIDlessToolCallsAcrossTurns(t *testing.T) {
	for _, test := range []struct {
		mode       string
		providerID bool
	}{
		{"generate", false}, {"stream", false},
		{"generate", true}, {"stream", true},
	} {
		t.Run(fmt.Sprintf("%s/providerID=%t", test.mode, test.providerID), func(t *testing.T) {
			calls := `{"index":0,"function":{"name":"read","arguments":"{}"}}`
			wantCalls := 1
			if test.providerID {
				calls += `,{"index":1,"id":"toolcall0","function":{"name":"read","arguments":"{}"}}`
				wantCalls++
			}
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "text/event-stream")
				writeSSE(writer, `{"choices":[{"index":0,"delta":{"tool_calls":[`+calls+`]},"finish_reason":"tool_calls"}]}`, "[DONE]")
			}))
			defer server.Close()
			client := mustClient(t, Config{Model: "model", APIKey: "secret", BaseURL: server.URL, HTTPClient: server.Client(),
				Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}})
			request := textRequest()
			request.Tools = []llm.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}
			seen := make(map[string]bool)
			for turn := range 3 {
				// Only retain the generated ID in history; the explicit ID in the
				// same response also must not collide with the synthesized one.
				var response *llm.Response
				var err error
				if test.mode == "generate" {
					response, err = client.Generate(context.Background(), request)
				} else {
					stream, streamErr := client.Stream(context.Background(), request)
					if streamErr != nil {
						t.Fatal(streamErr)
					}
					response, err = llm.CollectStructural(stream)
				}
				if err != nil {
					t.Fatalf("turn %d: %v", turn, err)
				}
				if len(response.Message.ToolCalls) != wantCalls {
					t.Fatalf("calls = %#v", response.Message.ToolCalls)
				}
				call := response.Message.ToolCalls[0]
				if seen[call.ID] || test.providerID && call.ID == "toolcall0" || !regexp.MustCompile(`^[A-Za-z0-9]{9}$`).MatchString(call.ID) {
					t.Fatalf("turn %d: invalid or repeated ID %q", turn, call.ID)
				}
				seen[call.ID] = true
				request.Messages = append(request.Messages,
					llm.Message{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call}},
					llm.Message{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: call.ID, Name: call.Name, Content: []llm.Part{{Text: "ok"}}}}})
			}
		})
	}
}

func TestEmptyToolArguments(t *testing.T) {
	for _, test := range []struct {
		name, arguments string
		wantError       bool
	}{
		{"missing", "", false},
		{"empty", `,"arguments":""`, false},
		{"object", `,"arguments":"{}"`, false},
		{"truncated", `,"arguments":"{\"q\":"`, true},
	} {
		for _, mode := range []string{"generate", "stream"} {
			t.Run(test.name+"/"+mode, func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
					writer.Header().Set("Content-Type", "text/event-stream")
					writeSSE(writer, `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"abc123XYZ","function":{"name":"read"`+test.arguments+`}}]},"finish_reason":"tool_calls"}]}`, "[DONE]")
				}))
				defer server.Close()
				client := mustClient(t, Config{Model: "model", APIKey: "secret", BaseURL: server.URL, HTTPClient: server.Client(),
					Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}})
				request := textRequest()
				request.Tools = []llm.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}
				var response *llm.Response
				var err error
				if mode == "generate" {
					response, err = client.Generate(context.Background(), request)
				} else {
					stream, streamErr := client.Stream(context.Background(), request)
					if streamErr != nil {
						t.Fatal(streamErr)
					}
					response, err = llm.CollectStructural(stream)
				}
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
				if response.FinishReason != llm.FinishReasonToolCall || len(response.Message.ToolCalls) != 1 || string(response.Message.ToolCalls[0].Arguments) != "{}" {
					t.Fatalf("response = %#v", response)
				}
			})
		}
	}
}

func TestHistoricalToolIDsAreNormalizedAndPaired(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{{Text: "run"}}},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{ID: "call-with-an-invalid-long-id", Name: "read", Arguments: json.RawMessage(`{"n":1}`)},
			{ID: "second-invalid-long-id", Name: "read", Arguments: json.RawMessage(`{"n":2}`)},
		}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{
			{CallID: "call-with-an-invalid-long-id", Name: "read", Content: []llm.Part{{Text: "one"}}},
			{Name: "read", Content: []llm.Part{{Text: "two"}}},
		}},
	}}
	payload, prior, err := buildRequest("generate", "model", false, false, request)
	if err != nil {
		t.Fatal(err)
	}
	var wire chatRequest
	if err := json.Unmarshal(payload, &wire); err != nil {
		t.Fatal(err)
	}
	first, second := wire.Messages[1].ToolCalls[0].ID, wire.Messages[1].ToolCalls[1].ID
	if !regexp.MustCompile(`^[A-Za-z0-9]{9}$`).MatchString(first) || !regexp.MustCompile(`^[A-Za-z0-9]{9}$`).MatchString(second) || first == second {
		t.Fatalf("normalized IDs = %q, %q", first, second)
	}
	if wire.Messages[2].ToolCallID != first || wire.Messages[3].ToolCallID != second {
		t.Fatalf("result IDs = %q, %q; want %q, %q", wire.Messages[2].ToolCallID, wire.Messages[3].ToolCallID, first, second)
	}
	if len(prior) != 2 {
		t.Fatalf("prior IDs = %#v", prior)
	}
}

func TestProviderErrorsAndRetryAfter(t *testing.T) {
	tests := []struct {
		status int
		body   string
		kind   llm.ErrorKind
	}{
		{http.StatusUnauthorized, `{"message":"bad key"}`, llm.KindAuthentication},
		{http.StatusForbidden, `{"detail":"forbidden"}`, llm.KindPermission},
		{http.StatusTooManyRequests, `{"message":"slow down"}`, llm.KindRateLimit},
		{http.StatusBadRequest, `{"message":"context length exceeded"}`, llm.KindContextLimit},
		{http.StatusInternalServerError, `{"message":"broken"}`, llm.KindProvider},
	}
	for _, test := range tests {
		t.Run(http.StatusText(test.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Retry-After", "2")
				writer.WriteHeader(test.status)
				_, _ = io.WriteString(writer, test.body)
			}))
			defer server.Close()
			client := mustClient(t, Config{Provider: "custom-mistral", Model: "model", APIKey: "secret", BaseURL: server.URL, HTTPClient: server.Client()})
			response, err := client.Generate(context.Background(), textRequest())
			if response != nil || err == nil {
				t.Fatalf("Generate() = %#v, %v", response, err)
			}
			var modelErr *llm.Error
			if !errors.As(err, &modelErr) || modelErr.Kind != test.kind || modelErr.Provider != "custom-mistral" ||
				modelErr.HTTPStatus != test.status || modelErr.RetryAfter != 2*time.Second {
				t.Fatalf("error = %v (%#v)", err, modelErr)
			}
		})
	}
}

func TestValidationOccursBeforeIO(t *testing.T) {
	client := mustClient(t, Config{Model: "model", APIKey: "secret", HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		t.Fatal("unexpected HTTP request")
		return nil, nil
	})}})
	ctx, cancel := context.WithCancelCause(context.Background())
	cause := errors.New("stop")
	cancel(cause)
	if response, err := client.Generate(ctx, textRequest()); response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
	if stream, err := client.Stream(context.Background(), textRequest()); stream != nil || err == nil {
		t.Fatalf("Stream() = %#v, %v", stream, err)
	}
	request := textRequest()
	request.ReasoningEffort = llm.ReasoningEffortNone
	if _, _, err := client.prepare(context.Background(), "generate", request, false); err != nil {
		t.Fatalf("ReasoningEffortNone: %v", err)
	}
}

func TestContentFilterPreservesTextAndUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(w,
			`{"id":"resp","choices":[{"index":0,"delta":{"content":"Complete answer"},"finish_reason":null}]}`,
			`{"id":"resp","choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":1000,"completion_tokens":500,"total_tokens":1500}}`,
			"[DONE]")
	}))
	defer server.Close()
	client := mustClient(t, Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming}})
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("streaming=%t", streaming), func(t *testing.T) {
			var response *llm.Response
			var err error
			if streaming {
				stream, streamErr := client.Stream(context.Background(), textRequest())
				if streamErr != nil {
					t.Fatal(streamErr)
				}
				response, err = llm.CollectStructural(stream) // Always closes the stream.
			} else {
				response, err = client.Generate(context.Background(), textRequest())
			}
			if err != nil {
				t.Fatalf("response = %#v, error = %v", response, err)
			}
			if response.Text() != "Complete answer" || response.FinishReason != llm.FinishReasonContentFilter ||
				response.Usage == nil || response.Usage.InputTokens != 1000 || response.Usage.OutputTokens != 500 || response.Usage.TotalTokens != 1500 {
				t.Fatalf("response = %#v", response)
			}
		})
	}
}

func TestUnknownFinishReasonsAreUnsupported(t *testing.T) {
	for _, reason := range []string{"future_reason", "unknown_stop"} {
		t.Run(reason, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				writeSSE(w, fmt.Sprintf(`{"id":"resp","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":%q}]}`, reason))
			}))
			defer server.Close()
			client := mustClient(t, Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
				Capabilities: []llm.Capability{llm.CapabilityStreaming}})
			response, err := client.Generate(context.Background(), textRequest())
			var modelErr *llm.Error
			if response != nil || !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported {
				t.Fatalf("Generate = %#v, %v; want unsupported", response, err)
			}
			stream, err := client.Stream(context.Background(), textRequest())
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = stream.Close() }()
			for {
				chunk, err := stream.Recv()
				if err != nil {
					if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported {
						t.Fatalf("Recv = %v; want unsupported", err)
					}
					break
				}
				if chunk.FinishReason != "" {
					t.Fatalf("unknown finish reason produced success: %#v", chunk)
				}
			}
		})
	}
}

func TestMalformedStreamIsClassified(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writeSSE(writer, `{"id":"resp","choices":[{"index":0,"delta":{"content":[{"type":"unknown"}]},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	client := mustClient(t, Config{Model: "model", APIKey: "secret", BaseURL: server.URL, HTTPClient: server.Client()})
	response, err := client.Generate(context.Background(), textRequest())
	if response != nil || err == nil {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse || modelErr.Op != "generate" {
		t.Fatalf("error = %v (%#v)", err, modelErr)
	}
}

func mustClient(t *testing.T, config Config) *Client {
	t.Helper()
	client, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func textRequest() llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
}

func writeSSE(writer io.Writer, events ...string) {
	for _, event := range events {
		_, _ = io.WriteString(writer, "data: "+event+"\n\n")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func TestInfoOwnsCapabilities(t *testing.T) {
	configured := []llm.Capability{llm.CapabilityStreaming}
	client := mustClient(t, Config{Provider: "custom", Model: "model", Reasoning: true, APIKey: "secret", Capabilities: configured})
	configured[0] = llm.CapabilityAudio
	info := client.Info()
	if info.Provider != "custom" || info.Model != "model" || info.API != llm.APIMistralConversations || !info.Reasoning ||
		len(info.Capabilities) != 2 || info.Capabilities[1] != llm.CapabilityStreaming {
		t.Fatalf("Info() = %#v", info)
	}
	info.Capabilities[0] = llm.CapabilityAudio
	if client.Info().Capabilities[0] != llm.CapabilityGeneration {
		t.Fatal("Info returned aliased capabilities")
	}
}

func TestNewRejectsInvalidConfiguration(t *testing.T) {
	for _, config := range []Config{
		{APIKey: "key"},
		{Model: "model"},
		{Model: "model", APIKey: "key", BaseURL: "://bad"},
		{Model: "model", APIKey: "key", Capabilities: []llm.Capability{llm.CapabilityAudio}},
	} {
		client, err := New(config)
		if client != nil || err == nil {
			t.Fatalf("New(%#v) = %#v, %v", config, client, err)
		}
		var modelErr *llm.Error
		if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindInvalidRequest || modelErr.Op != "configure" {
			t.Fatalf("error = %v (%#v)", err, modelErr)
		}
	}
}

func TestHeaderStorageIsOwned(t *testing.T) {
	headers := http.Header{"X-Test": {"before"}}
	client := mustClient(t, Config{Model: "model", APIKey: "secret", Headers: headers})
	headers.Set("X-Test", "after")
	if got := client.headers.Get("X-Test"); !strings.EqualFold(got, "before") {
		t.Fatalf("stored header = %q", got)
	}
}
