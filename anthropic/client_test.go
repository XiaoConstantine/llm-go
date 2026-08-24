package anthropic

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
)

const validResponse = `{
	"id":"msg_123",
	"type":"message",
	"role":"assistant",
	"content":[{"type":"text","text":"hello"}],
	"model":"claude-served",
	"stop_reason":"end_turn",
	"stop_sequence":null,
	"usage":{"input_tokens":3,"output_tokens":2}
}`

func TestNewConfiguresClientWithoutMutatingHeaders(t *testing.T) {
	headers := http.Header{
		"X-Api-Key":    {"header-key"},
		"X-Trace":      {"trace"},
		"Content-Type": {"text/plain"},
	}
	headers["x-lower-case"] = []string{"preserved"}
	headers["X-Nil"] = nil
	client, err := New(Config{
		Model:                  " claude-model ",
		APIKey:                 "config-key",
		BaseURL:                "https://example.com/proxy/v1/",
		APIVersion:             "2026-08-01",
		DefaultMaxOutputTokens: 777,
		Headers:                headers,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if client.model != "claude-model" {
		t.Fatalf("Client = %#v", client)
	}
	if client.defaultMaxOutputTokens != 777 {
		t.Fatalf("Client defaults = %#v", client)
	}
	if client.headers.Get("X-Api-Key") != "header-key" || client.headers.Get("X-Trace") != "trace" {
		t.Fatalf("client headers = %#v", client.headers)
	}
	if client.headers.Get("Anthropic-Version") != "2026-08-01" || client.headers.Get("Content-Type") != "application/json" {
		t.Fatalf("protocol headers = %#v", client.headers)
	}
	if _, exists := client.headers["x-lower-case"]; !exists {
		t.Fatalf("Client changed custom header casing: %#v", client.headers)
	}
	if values, exists := client.headers["X-Nil"]; !exists || values != nil {
		t.Fatalf("Client changed nil custom header: %#v", client.headers)
	}
	if headers.Get("Content-Type") != "text/plain" || headers.Get("Anthropic-Version") != "" {
		t.Fatalf("New mutated input headers: %#v", headers)
	}
	headers.Set("X-Trace", "changed")
	if client.headers.Get("X-Trace") != "trace" {
		t.Fatalf("Client retained input headers: %#v", client.headers)
	}

	first := client.Info()
	second := client.Info()
	if first.Provider != "anthropic" || first.Model != "claude-model" ||
		len(first.Capabilities) != 1 || first.Capabilities[0] != llm.CapabilityGeneration {
		t.Fatalf("Info() = %#v", first)
	}
	first.Capabilities[0] = llm.CapabilityStreaming
	if second.Capabilities[0] != llm.CapabilityGeneration {
		t.Fatal("Info returned aliased capabilities")
	}
}

func TestClientUsesConfiguredProviderIdentity(t *testing.T) {
	client, err := New(Config{Provider: " gateway ", Model: "model"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if got := client.Info().Provider; got != "gateway" {
		t.Fatalf("Info().Provider = %q, want %q", got, "gateway")
	}

	temperature := 2.0
	response, err := client.Generate(context.Background(), llm.Request{
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
		Temperature: &temperature,
	})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Provider != "gateway" {
		t.Fatalf("Generate() error = %#v, want provider %q", modelErr, "gateway")
	}

	_, err = New(Config{Provider: " gateway "})
	if !errors.As(err, &modelErr) || modelErr.Provider != "gateway" {
		t.Fatalf("New(invalid) error = %#v, want provider %q", modelErr, "gateway")
	}
}

func TestRequestRejectsReasoningEffort(t *testing.T) {
	request := llm.Request{
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
		ReasoningEffort: llm.ReasoningEffortHigh,
	}
	err := checkRequest("generate", request)
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported {
		t.Fatalf("checkRequest() error = %#v", err)
	}
}

func TestRelabelProviderErrorPreservesJoinedCauses(t *testing.T) {
	closeErr := errors.New("close failed")
	err := relabelProviderError(errors.Join(malformedStream("bad event"), closeErr), "gateway")
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Provider != "gateway" {
		t.Fatalf("relabelProviderError() model error = %#v, want provider %q", modelErr, "gateway")
	}
	if !errors.Is(err, closeErr) {
		t.Fatalf("relabelProviderError() lost joined cause: %v", err)
	}
}

func TestNewDefaultsAndRejectsInvalidConfig(t *testing.T) {
	client, err := New(Config{Model: "model"})
	if err != nil {
		t.Fatalf("New(defaults) error = %v", err)
	}
	if client.defaultMaxOutputTokens != defaultMaxOutputTokens {
		t.Fatalf("New(defaults) = %#v", client)
	}
	if client.headers.Get("Anthropic-Version") != defaultAPIVersion {
		t.Fatalf("Anthropic-Version = %q", client.headers.Get("Anthropic-Version"))
	}

	for _, test := range []struct {
		name   string
		config Config
	}{
		{name: "empty model", config: Config{}},
		{name: "relative URL", config: Config{Model: "model", BaseURL: "/v1"}},
		{name: "unsupported scheme", config: Config{Model: "model", BaseURL: "file:///v1"}},
		{name: "query", config: Config{Model: "model", BaseURL: "https://example.com/v1?q=1"}},
		{name: "fragment", config: Config{Model: "model", BaseURL: "https://example.com/v1#fragment"}},
		{name: "invalid version", config: Config{Model: "model", APIVersion: "latest"}},
		{name: "negative default tokens", config: Config{Model: "model", DefaultMaxOutputTokens: -1}},
	} {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(test.config)
			if client != nil {
				t.Fatalf("New() client = %#v", client)
			}
			requireModelError(t, err, llm.KindInvalidRequest, "configure")
		})
	}
}

func TestNewPreservesConfigurationCause(t *testing.T) {
	client, err := New(Config{Model: "model", APIVersion: "latest"})
	if client != nil {
		t.Fatalf("New() client = %#v, want nil", client)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "configure")
	var parseErr *time.ParseError
	if !errors.As(err, &parseErr) {
		t.Fatalf("errors.As(%v, *time.ParseError) = false", err)
	}
}

func TestGenerateTranslatesTextRequestAndResponse(t *testing.T) {
	type record struct {
		path   string
		header http.Header
		body   []byte
	}
	records := make(chan record, 1)
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			panic(err)
		}
		records <- record{path: request.URL.Path, header: request.Header.Clone(), body: body}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{
			"id":"msg_response",
			"type":"message",
			"role":"assistant",
			"content":[{"type":"text","text":"first "},{"type":"text","text":"second"}],
			"model":"claude-served",
			"stop_reason":"stop_sequence",
			"stop_sequence":"END",
			"usage":{
				"input_tokens":10,
				"cache_creation_input_tokens":2,
				"cache_read_input_tokens":3,
				"output_tokens":4
			}
		}`)
	}))

	temperature := 0.25
	topP := 0.9
	client, err := New(Config{
		Model:      "claude-request",
		APIKey:     "secret",
		BaseURL:    server.URL + "/proxy/v1/",
		HTTPClient: server.Client(),
		Headers:    http.Header{"X-Trace": {"trace"}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.Part{{Text: "system one"}, {Text: "system two"}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello "}, {Text: "world"}}},
			{Role: llm.RoleAssistant, Content: []llm.Part{{Text: "prior answer"}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "continue"}}},
		},
		MaxOutputTokens: 123,
		Temperature:     &temperature,
		TopP:            &topP,
		Stop:            []string{"END"},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.ID != "msg_response" || response.Model != "claude-served" ||
		response.FinishReason != llm.FinishReasonStop || response.Text() != "first second" {
		t.Fatalf("Generate() response = %#v", response)
	}
	if response.Message.Role != llm.RoleAssistant || len(response.Message.Content) != 2 {
		t.Fatalf("response message = %#v", response.Message)
	}
	if response.Usage == nil || *response.Usage != (llm.Usage{InputTokens: 15, OutputTokens: 4, TotalTokens: 19}) {
		t.Fatalf("response usage = %#v", response.Usage)
	}

	got := <-records
	if got.path != "/proxy/v1/messages" {
		t.Fatalf("request path = %q", got.path)
	}
	if got.header.Get("X-Api-Key") != "secret" || got.header.Get("Anthropic-Version") != defaultAPIVersion ||
		got.header.Get("Content-Type") != "application/json" || got.header.Get("X-Trace") != "trace" {
		t.Fatalf("request headers = %#v", got.header)
	}
	if got.header.Get("User-Agent") != "Anthropic/Go 1.66.0" || got.header.Get("X-Stainless-Retry-Count") != "0" {
		t.Fatalf("SDK headers = %#v", got.header)
	}
	var payload map[string]any
	if err := jsonv2.Unmarshal(got.body, &payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if payload["model"] != "claude-request" || payload["max_tokens"] != float64(123) ||
		payload["temperature"] != 0.25 || payload["top_p"] != 0.9 {
		t.Fatalf("request options = %#v", payload)
	}
	stop, ok := payload["stop_sequences"].([]any)
	if !ok || len(stop) != 1 || stop[0] != "END" {
		t.Fatalf("stop_sequences = %#v", payload["stop_sequences"])
	}
	system, ok := payload["system"].([]any)
	if !ok || len(system) != 2 || system[0].(map[string]any)["text"] != "system one" ||
		system[1].(map[string]any)["text"] != "system two" {
		t.Fatalf("system = %#v", payload["system"])
	}
	messages, ok := payload["messages"].([]any)
	if !ok || len(messages) != 3 {
		t.Fatalf("messages = %#v", payload["messages"])
	}
	wants := []struct{ role, content string }{{"user", "hello world"}, {"assistant", "prior answer"}, {"user", "continue"}}
	for index, want := range wants {
		message := messages[index].(map[string]any)
		if message["role"] != want.role || message["content"] != want.content {
			t.Fatalf("messages[%d] = %#v", index, message)
		}
	}
}

func TestGenerateUsesHeaderOwnedAPIKeyCaseInsensitively(t *testing.T) {
	requests := make(chan *http.Request, 1)
	headers := http.Header{"x-api-key": {"header-key"}}
	client, err := New(Config{
		Model:      "model",
		APIKey:     "configured-key",
		BaseURL:    "http://example.com/prefix/v1",
		APIVersion: "2026-08-01",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests <- request.Clone(request.Context())
			return staticResponse(http.StatusOK, validResponse), nil
		})},
		Headers: headers,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	headers.Set("X-Api-Key", "mutated")

	response, err := client.Generate(context.Background(), textRequest("hello"))
	if err != nil || response == nil {
		t.Fatalf("Generate() = (%#v, %v)", response, err)
	}
	request := <-requests
	if values := request.Header["x-api-key"]; len(values) != 1 || values[0] != "header-key" {
		t.Fatalf("x-api-key = %#v, want header-owned key", values)
	}
	if request.URL.Path != "/prefix/v1/messages" {
		t.Fatalf("request path = %q", request.URL.Path)
	}
	if request.Header.Get("Anthropic-Version") != "2026-08-01" {
		t.Fatalf("Anthropic-Version = %q", request.Header.Get("Anthropic-Version"))
	}
}

func TestGeneratePreservesHeaderMapSemantics(t *testing.T) {
	requests := make(chan *http.Request, 1)
	client, err := New(Config{
		Model:   "model",
		APIKey:  "configured-key",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests <- request.Clone(request.Context())
			return staticResponse(http.StatusOK, validResponse), nil
		})},
		Headers: http.Header{
			"X-API-KEY":  {"header-key"},
			"X-Api-Key":  nil,
			"X-Multi":    {"one", "two"},
			"X-Nil":      nil,
			"User-Agent": nil,
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if err != nil || response == nil {
		t.Fatalf("Generate() = (%#v, %v)", response, err)
	}
	got := (<-requests).Header
	if values, exists := got["X-API-KEY"]; !exists || len(values) != 1 || values[0] != "header-key" {
		t.Fatalf("X-API-KEY = %#v, exists = %v", values, exists)
	}
	if values, exists := got["X-Api-Key"]; !exists || values != nil {
		t.Fatalf("X-Api-Key = %#v, exists = %v", values, exists)
	}
	if values := got["X-Multi"]; len(values) != 2 || values[0] != "one" || values[1] != "two" {
		t.Fatalf("X-Multi = %#v", values)
	}
	for _, key := range []string{"X-Nil", "User-Agent"} {
		if values, exists := got[key]; !exists || values != nil {
			t.Fatalf("%s = %#v, exists = %v", key, values, exists)
		}
	}
}

func TestGeneratePreservesEmptyQueryMarker(t *testing.T) {
	requests := make(chan *http.Request, 1)
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/prefix/v1?",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests <- request.Clone(request.Context())
			return staticResponse(http.StatusOK, validResponse), nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if err != nil || response == nil {
		t.Fatalf("Generate() = (%#v, %v)", response, err)
	}
	request := <-requests
	if request.URL.Path != "/prefix/v1/messages" || request.URL.RawQuery != "" || !request.URL.ForceQuery {
		t.Fatalf("request URL = %#v", request.URL)
	}
}

func TestGenerateIgnoresAmbientSDKConfiguration(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "ambient-key")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
	t.Setenv("ANTHROPIC_BASE_URL", "http://ambient.invalid/")
	t.Setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Ambient: leaked")
	t.Setenv("ANTHROPIC_PROFILE", "ambient-profile-must-not-load")

	requests := make(chan *http.Request, 1)
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://configured.example/proxy/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests <- request.Clone(request.Context())
			return staticResponse(http.StatusOK, validResponse), nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if err != nil || response == nil {
		t.Fatalf("Generate() = (%#v, %v)", response, err)
	}
	request := <-requests
	if request.URL.Host != "configured.example" || request.URL.Path != "/proxy/v1/messages" {
		t.Fatalf("request URL = %q", request.URL)
	}
	if request.Header.Get("X-Api-Key") != "" || request.Header.Get("Authorization") != "" || request.Header.Get("X-Ambient") != "" {
		t.Fatalf("ambient SDK configuration leaked into headers: %#v", request.Header)
	}
}

func TestGenerateDoesNotRetry(t *testing.T) {
	var calls atomic.Int64
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return staticResponse(http.StatusInternalServerError, `{"type":"error","error":{"message":"retryable"}}`), nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if response != nil {
		t.Fatalf("Generate() response = %#v", response)
	}
	requireModelError(t, err, llm.KindProvider, "generate")
	if calls.Load() != 1 {
		t.Fatalf("HTTP calls = %d, want one", calls.Load())
	}
}

func TestGenerateUsesConfiguredDefaultMaxOutputTokens(t *testing.T) {
	payloads := make(chan []byte, 1)
	client, err := New(Config{
		Model:                  "model",
		BaseURL:                "http://example.com/v1",
		DefaultMaxOutputTokens: 812,
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			body, err := io.ReadAll(request.Body)
			if err != nil {
				return nil, err
			}
			payloads <- body
			return staticResponse(http.StatusOK, validResponse), nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if err != nil || response == nil {
		t.Fatalf("Generate() = (%#v, %v)", response, err)
	}
	var payload map[string]any
	if err := jsonv2.Unmarshal(<-payloads, &payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if payload["max_tokens"] != float64(812) {
		t.Fatalf("max_tokens = %#v", payload["max_tokens"])
	}
	for _, field := range []string{"system", "temperature", "top_p", "stop_sequences"} {
		if _, exists := payload[field]; exists {
			t.Fatalf("default request contains %q: %#v", field, payload)
		}
	}
}

func TestGeneratePreflightOrderAndNoIO(t *testing.T) {
	var calls atomic.Int64
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, errors.New("unexpected request")
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	cause := errors.New("stop")
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(cause)
	response, err := client.Generate(ctx, llm.Request{})
	if response != nil {
		t.Fatalf("Generate(invalid) response = %#v", response)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "validate")

	valid := textRequest("hello")
	response, err = client.Generate(ctx, valid)
	if response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Generate(canceled) = (%#v, %v)", response, err)
	}

	temperature := 1.1
	penalty := 0.1
	tooMany := make([]llm.Message, maxMessages+1)
	for index := range tooMany {
		tooMany[index] = llm.Message{Role: llm.RoleUser}
	}
	for _, test := range []struct {
		name    string
		request llm.Request
		kind    llm.ErrorKind
		want    string
	}{
		{name: "tools", request: llm.Request{Messages: valid.Messages, Tools: []llm.Tool{{Name: "tool", InputSchema: []byte(`{"type":"object"}`)}}}, kind: llm.KindUnsupported, want: "tool capability"},
		{name: "tool history", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "tool", Arguments: []byte(`{}`)}}}, {Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call"}}}}}, kind: llm.KindUnsupported, want: "tool capability"},
		{name: "JSON", request: llm.Request{Messages: valid.Messages, ResponseFormat: llm.ResponseFormatJSON}, kind: llm.KindUnsupported, want: "JSON"},
		{name: "penalty", request: llm.Request{Messages: valid.Messages, PresencePenalty: &penalty}, kind: llm.KindUnsupported, want: "penalties"},
		{name: "temperature", request: llm.Request{Messages: valid.Messages, Temperature: &temperature}, kind: llm.KindInvalidRequest, want: "must not exceed 1"},
		{name: "late system", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, {Role: llm.RoleSystem}}}, kind: llm.KindInvalidRequest, want: "must precede"},
		{name: "system only", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleSystem}}}, kind: llm.KindInvalidRequest, want: "user or assistant"},
		{name: "image", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Kind: llm.PartImage, Data: []byte{1}, MediaType: "image/png"}}}}}, kind: llm.KindUnsupported, want: "binary content"},
		{name: "too many messages", request: llm.Request{Messages: tooMany}, kind: llm.KindInvalidRequest, want: "maximum"},
	} {
		t.Run(test.name, func(t *testing.T) {
			response, err := client.Generate(context.Background(), test.request)
			if response != nil {
				t.Fatalf("Generate() response = %#v", response)
			}
			requireModelError(t, err, test.kind, "generate")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Generate() error = %v, want %q", err, test.want)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls.Load())
	}

	atLimit := make([]llm.Message, maxMessages+1)
	atLimit[0] = llm.Message{Role: llm.RoleSystem}
	for index := 1; index < len(atLimit); index++ {
		atLimit[index] = llm.Message{Role: llm.RoleUser}
	}
	if err := checkRequest("generate", llm.Request{Messages: atLimit}); err != nil {
		t.Fatalf("checkRequest(system + maximum conversation messages) error = %v", err)
	}
}

func TestStreamPreflightOrder(t *testing.T) {
	client, err := New(Config{Model: "model"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{})
	if stream != nil {
		t.Fatalf("Stream(invalid) = %#v", stream)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "validate")

	cause := errors.New("stop")
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(cause)
	stream, err = client.Stream(ctx, textRequest("hello"))
	if stream != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Stream(canceled) = (%#v, %v)", stream, err)
	}
	stream, err = client.Stream(context.Background(), textRequest("hello"))
	if stream != nil {
		t.Fatalf("Stream(unsupported) = %#v", stream)
	}
	requireModelError(t, err, llm.KindUnsupported, "stream")
}

func TestGenerateRejectsMalformedResponse(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "invalid JSON", body: `{`, want: "decode response"},
		{name: "duplicate field", body: `{"id":"a","id":"b"}`, want: "duplicate object member"},
		{name: "missing ID", body: strings.Replace(validResponse, `"id":"msg_123",`, "", 1), want: "no ID"},
		{name: "wrong type", body: strings.Replace(validResponse, `"type":"message"`, `"type":"other"`, 1), want: "want message"},
		{name: "wrong role", body: strings.Replace(validResponse, `"role":"assistant"`, `"role":"user"`, 1), want: "want assistant"},
		{name: "missing model", body: strings.Replace(validResponse, `"model":"claude-served",`, "", 1), want: "no model"},
		{name: "missing stop", body: strings.Replace(validResponse, `"stop_reason":"end_turn",`, "", 1), want: "no stop reason"},
		{name: "null stop", body: strings.Replace(validResponse, `"stop_reason":"end_turn"`, `"stop_reason":null`, 1), want: "no stop reason"},
		{name: "unknown stop", body: strings.Replace(validResponse, `"stop_reason":"end_turn"`, `"stop_reason":"future_reason"`, 1), want: "unsupported stop reason"},
		{name: "missing content", body: strings.Replace(validResponse, `"content":[{"type":"text","text":"hello"}],`, "", 1), want: "no content"},
		{name: "null content", body: strings.Replace(validResponse, `[{"type":"text","text":"hello"}]`, `null`, 1), want: "no content"},
		{name: "unsupported content", body: strings.Replace(validResponse, `{"type":"text","text":"hello"}`, `{"type":"thinking"}`, 1), want: "unsupported type"},
		{name: "missing text", body: strings.Replace(validResponse, `,"text":"hello"`, "", 1), want: "no text"},
		{name: "null text", body: strings.Replace(validResponse, `"text":"hello"`, `"text":null`, 1), want: "no text"},
		{name: "missing usage", body: strings.Replace(validResponse, `,
	"usage":{"input_tokens":3,"output_tokens":2}`, "", 1), want: "incomplete usage"},
		{name: "incomplete usage", body: strings.Replace(validResponse, `"input_tokens":3,`, "", 1), want: "incomplete usage"},
		{name: "negative usage", body: strings.Replace(validResponse, `"input_tokens":3`, `"input_tokens":-1`, 1), want: "negative token usage"},
		{name: "negative cache usage", body: strings.Replace(validResponse, `"input_tokens":3`, `"input_tokens":3,"cache_read_input_tokens":-1`, 1), want: "negative token usage"},
		{name: "usage overflow", body: strings.Replace(validResponse, `"input_tokens":3,"output_tokens":2`, fmt.Sprintf(`"input_tokens":%d,"output_tokens":1`, maxInt), 1), want: "overflows int"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := newStaticClient(t, http.StatusOK, test.body, nil)
			response, err := client.Generate(context.Background(), textRequest("hello"))
			if response != nil {
				t.Fatalf("Generate() response = %#v", response)
			}
			requireModelError(t, err, llm.KindMalformedResponse, "generate")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Generate() error = %q, want substring %q", err, test.want)
			}
		})
	}
}

func TestGenerateMapsFinishReasons(t *testing.T) {
	for reason, want := range map[string]llm.FinishReason{
		"end_turn":                      llm.FinishReasonStop,
		"stop_sequence":                 llm.FinishReasonStop,
		"max_tokens":                    llm.FinishReasonLength,
		"model_context_window_exceeded": llm.FinishReasonLength,
		"refusal":                       llm.FinishReasonContentFilter,
	} {
		t.Run(reason, func(t *testing.T) {
			body := strings.Replace(validResponse, `"stop_reason":"end_turn"`, `"stop_reason":"`+reason+`"`, 1)
			client := newStaticClient(t, http.StatusOK, body, nil)
			response, err := client.Generate(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if response.FinishReason != want {
				t.Fatalf("FinishReason = %q, want %q", response.FinishReason, want)
			}
		})
	}
}

func TestGenerateAcceptsExplicitEmptyContent(t *testing.T) {
	for _, reason := range []string{"end_turn", "refusal"} {
		t.Run(reason, func(t *testing.T) {
			body := strings.Replace(validResponse, `[{"type":"text","text":"hello"}]`, `[]`, 1)
			body = strings.Replace(body, `"stop_reason":"end_turn"`, `"stop_reason":"`+reason+`"`, 1)
			client := newStaticClient(t, http.StatusOK, body, nil)
			response, err := client.Generate(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if response == nil || len(response.Message.Content) != 0 {
				t.Fatalf("Generate() response = %#v", response)
			}
		})
	}
}

func TestGenerateClassifiesKnownUnimplementedStopReasons(t *testing.T) {
	for _, reason := range []string{"pause_turn"} {
		t.Run(reason, func(t *testing.T) {
			body := strings.Replace(validResponse, `"stop_reason":"end_turn"`, `"stop_reason":"`+reason+`"`, 1)
			client := newStaticClient(t, http.StatusOK, body, nil)
			response, err := client.Generate(context.Background(), textRequest("hello"))
			if response != nil {
				t.Fatalf("Generate() response = %#v", response)
			}
			modelErr := requireModelError(t, err, llm.KindUnsupported, "generate")
			if !strings.Contains(modelErr.Error(), reason) {
				t.Fatalf("Generate() error = %v, want stop reason %q", err, reason)
			}
		})
	}
}

func TestGenerateClassifiesHTTPError(t *testing.T) {
	for _, test := range []struct {
		name   string
		status int
		body   string
		kind   llm.ErrorKind
	}{
		{name: "authentication", status: http.StatusUnauthorized, kind: llm.KindAuthentication},
		{name: "billing", status: http.StatusPaymentRequired, kind: llm.KindPermission},
		{name: "permission", status: http.StatusForbidden, kind: llm.KindPermission},
		{name: "rate limit", status: http.StatusTooManyRequests, kind: llm.KindRateLimit},
		{name: "invalid", status: http.StatusBadRequest, kind: llm.KindInvalidRequest},
		{name: "context", status: http.StatusBadRequest, body: "prompt is too long for the context window", kind: llm.KindContextLimit},
		{name: "too large", status: http.StatusRequestEntityTooLarge, kind: llm.KindInvalidRequest},
		{name: "conflict", status: http.StatusConflict, kind: llm.KindProvider},
		{name: "overloaded", status: 529, kind: llm.KindProvider},
		{name: "server", status: http.StatusInternalServerError, kind: llm.KindProvider},
	} {
		t.Run(test.name, func(t *testing.T) {
			message := test.body
			if message == "" {
				message = "provider failed"
			}
			body := fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":%s},"request_id":"req_body"}`, strconv.Quote(message))
			headers := http.Header{"Retry-After": {"2"}, "Request-Id": {"req_header"}}
			client := newStaticClient(t, test.status, body, headers)
			response, err := client.Generate(context.Background(), textRequest("hello"))
			if response != nil {
				t.Fatalf("Generate() response = %#v", response)
			}
			modelErr := requireModelError(t, err, test.kind, "generate")
			if modelErr.HTTPStatus != test.status || modelErr.RetryAfter != 2*time.Second {
				t.Fatalf("model error = %#v", modelErr)
			}
			var apiErr *APIError
			if !errors.As(err, &apiErr) || apiErr.Type != "api_error" || apiErr.Message != message || apiErr.RequestID != "req_body" {
				t.Fatalf("API error = %#v", apiErr)
			}
		})
	}
}

func TestGenerateFallsBackForMalformedHTTPError(t *testing.T) {
	client := newStaticClient(t, http.StatusBadGateway, " plain failure ", http.Header{"Request-Id": {"req_header"}})
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if response != nil {
		t.Fatalf("Generate() response = %#v", response)
	}
	modelErr := requireModelError(t, err, llm.KindProvider, "generate")
	if modelErr.HTTPStatus != http.StatusBadGateway {
		t.Fatalf("HTTPStatus = %d", modelErr.HTTPStatus)
	}
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Message != "plain failure" || apiErr.RequestID != "req_header" {
		t.Fatalf("API error = %#v", apiErr)
	}
}

func TestGenerateTransportAndCancellation(t *testing.T) {
	t.Run("transport", func(t *testing.T) {
		sentinel := errors.New("network down")
		client, err := New(Config{
			Model:   "model",
			BaseURL: "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, sentinel
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil || !errors.Is(err, sentinel) {
			t.Fatalf("Generate() = (%#v, %v)", response, err)
		}
		requireModelError(t, err, llm.KindTransport, "generate")
	})

	t.Run("cancellation cause", func(t *testing.T) {
		started := make(chan struct{})
		cause := errors.New("cancel generation")
		ctx, cancel := context.WithCancelCause(context.Background())
		client, err := New(Config{
			Model:   "model",
			BaseURL: "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				close(started)
				<-request.Context().Done()
				return nil, request.Context().Err()
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		go func() {
			<-started
			cancel(cause)
		}()
		response, err := client.Generate(ctx, textRequest("hello"))
		if response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
			t.Fatalf("Generate() = (%#v, %v)", response, err)
		}
	})

	t.Run("cancellation after HTTP response", func(t *testing.T) {
		cause := errors.New("cancel after transport")
		ctx, cancel := context.WithCancelCause(context.Background())
		closed := make(chan struct{})
		body := &errorBody{data: validResponse, onClose: func() { close(closed) }}
		client, err := New(Config{
			Model:   "model",
			BaseURL: "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				cancel(cause)
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     "200 OK",
					Header:     make(http.Header),
					Body:       body,
				}, nil
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		response, err := client.Generate(ctx, textRequest("hello"))
		if response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
			t.Fatalf("Generate() = (%#v, %v)", response, err)
		}
		select {
		case <-closed:
		default:
			t.Fatal("canceled response body was not closed")
		}
	})

	t.Run("redirect error after cancellation", func(t *testing.T) {
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
			Model:      "model",
			BaseURL:    server.URL,
			HTTPClient: httpClient,
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		response, err := client.Generate(ctx, textRequest("hello"))
		if response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || errors.Is(err, redirectErr) {
			t.Fatalf("Generate() = (%#v, %v)", response, err)
		}
		var modelErr *llm.Error
		if errors.As(err, &modelErr) {
			t.Fatalf("Generate() error contains model error: %#v", modelErr)
		}
	})
}

func TestGenerateHandlesBodyReadAndCloseErrors(t *testing.T) {
	readErr := errors.New("read failed")
	closeErr := errors.New("close failed")

	t.Run("read and close", func(t *testing.T) {
		client := clientWithBody(t, http.StatusOK, &errorBody{readErr: readErr, closeErr: closeErr})
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil || !errors.Is(err, readErr) || !errors.Is(err, closeErr) {
			t.Fatalf("Generate() = (%#v, %v)", response, err)
		}
		requireModelError(t, err, llm.KindTransport, "generate")
	})

	t.Run("successful response close", func(t *testing.T) {
		client := clientWithBody(t, http.StatusOK, &errorBody{data: validResponse, closeErr: closeErr})
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil || !errors.Is(err, closeErr) {
			t.Fatalf("Generate() = (%#v, %v)", response, err)
		}
		requireModelError(t, err, llm.KindTransport, "generate")
	})

	t.Run("provider response close", func(t *testing.T) {
		body := `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`
		client := clientWithBody(t, http.StatusTooManyRequests, &errorBody{data: body, closeErr: closeErr})
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil || !errors.Is(err, closeErr) {
			t.Fatalf("Generate() = (%#v, %v)", response, err)
		}
		requireModelError(t, err, llm.KindRateLimit, "generate")
	})

	t.Run("cancellation and close", func(t *testing.T) {
		cause := errors.New("cancel after response")
		ctx, cancel := context.WithCancelCause(context.Background())
		client := clientWithBody(t, http.StatusOK, &errorBody{
			data:     validResponse,
			closeErr: closeErr,
			onClose: func() {
				cancel(cause)
			},
		})
		response, err := client.Generate(ctx, textRequest("hello"))
		if response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || !errors.Is(err, closeErr) {
			t.Fatalf("Generate() = (%#v, %v)", response, err)
		}
		requireModelError(t, err, llm.KindTransport, "generate")
	})
}

func TestGenerateEnforcesBodyLimits(t *testing.T) {
	t.Run("request", func(t *testing.T) {
		var calls atomic.Int64
		client, err := New(Config{
			Model:   "model",
			BaseURL: "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				calls.Add(1)
				return staticResponse(http.StatusOK, validResponse), nil
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		response, err := client.Generate(context.Background(), textRequest(strings.Repeat("x", maxRequestBodyBytes)))
		if response != nil {
			t.Fatalf("Generate() response = %#v", response)
		}
		requireModelError(t, err, llm.KindInvalidRequest, "generate")
		if calls.Load() != 0 {
			t.Fatalf("HTTP calls = %d, want zero", calls.Load())
		}
	})

	t.Run("success response", func(t *testing.T) {
		client := clientWithBody(t, http.StatusOK, io.NopCloser(strings.NewReader(strings.Repeat("x", maxResponseBodyBytes+1))))
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil {
			t.Fatalf("Generate() response = %#v", response)
		}
		requireModelError(t, err, llm.KindMalformedResponse, "generate")
	})

	t.Run("error response", func(t *testing.T) {
		client := clientWithBody(t, http.StatusBadGateway, io.NopCloser(strings.NewReader(strings.Repeat("x", int(maxErrorBodyBytes)+1))))
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil {
			t.Fatalf("Generate() response = %#v", response)
		}
		modelErr := requireModelError(t, err, llm.KindProvider, "generate")
		var apiErr *APIError
		if modelErr.HTTPStatus != http.StatusBadGateway || !errors.As(err, &apiErr) ||
			!strings.Contains(apiErr.Message, "response body exceeds") {
			t.Fatalf("Generate() error = %#v, API error = %#v", modelErr, apiErr)
		}
	})
}

func textRequest(text string) llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: text}}}}}
}

func newStaticClient(t *testing.T, status int, body string, headers http.Header) *Client {
	t.Helper()
	client, err := New(Config{
		Model:      "model",
		BaseURL:    "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticTransport{status: status, body: body, headers: headers}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client
}

func clientWithBody(t *testing.T, status int, body io.ReadCloser) *Client {
	t.Helper()
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{StatusCode: status, Status: http.StatusText(status), Header: make(http.Header), Body: body}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client
}

func requireModelError(t *testing.T, err error, kind llm.ErrorKind, op string) *llm.Error {
	t.Helper()
	if err == nil {
		t.Fatal("error is nil")
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) {
		t.Fatalf("error type = %T, want *llm.Error: %v", err, err)
	}
	if modelErr.Kind != kind || modelErr.Op != op {
		t.Fatalf("model error = %#v, want kind %v op %q", modelErr, kind, op)
	}
	return modelErr
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

type staticTransport struct {
	status  int
	body    string
	headers http.Header
}

func (transport staticTransport) RoundTrip(*http.Request) (*http.Response, error) {
	response := staticResponse(transport.status, transport.body)
	response.Header = transport.headers.Clone()
	return response, nil
}

func staticResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Status:     strconv.Itoa(status) + " " + http.StatusText(status),
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

type errorBody struct {
	data     string
	offset   int
	readErr  error
	closeErr error
	onClose  func()
}

func (body *errorBody) Read(buffer []byte) (int, error) {
	if body.offset < len(body.data) {
		count := copy(buffer, body.data[body.offset:])
		body.offset += count
		return count, nil
	}
	if body.readErr != nil {
		err := body.readErr
		body.readErr = nil
		return 0, err
	}
	return 0, io.EOF
}

func (body *errorBody) Close() error {
	if body.onClose != nil {
		body.onClose()
	}
	return body.closeErr
}
