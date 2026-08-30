package mistral

import (
	"context"
	"encoding/json"
	"errors"
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
	response, err := llm.Collect(stream, tools)
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
