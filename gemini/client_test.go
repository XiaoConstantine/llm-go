package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestNewRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name      string
		config    Config
		wantError string
	}{
		{name: "empty model", config: Config{APIKey: "key"}, wantError: "model must not be empty"},
		{name: "invalid model", config: Config{Model: "models/../secret", APIKey: "key"}, wantError: "invalid path"},
		{name: "empty API key", config: Config{Model: "model"}, wantError: "API key must not be empty"},
		{name: "relative base URL", config: Config{Model: "model", APIKey: "key", BaseURL: "/api"}, wantError: "scheme must be http or https"},
		{name: "base URL query", config: Config{Model: "model", APIKey: "key", BaseURL: "https://example.com?x=1"}, wantError: "query or fragment"},
		{name: "invalid API version", config: Config{Model: "model", APIKey: "key", APIVersion: "v1/beta"}, wantError: "one path segment"},
		{name: "unsupported capability", config: Config{Model: "model", APIKey: "key", Capabilities: []llm.Capability{"future"}}, wantError: "not implemented"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(test.config)
			if client != nil {
				t.Fatalf("New() client = %#v, want nil", client)
			}
			modelErr := requireModelError(t, err, llm.KindInvalidRequest, "configure", defaultProvider)
			if !strings.Contains(modelErr.Error(), test.wantError) {
				t.Fatalf("New() error = %q, want substring %q", modelErr, test.wantError)
			}
		})
	}
}

func TestInfoCopiesCapabilitiesAndRelabelsProvider(t *testing.T) {
	configured := []llm.Capability{
		llm.CapabilityStreaming,
		llm.CapabilityTools,
		llm.CapabilityStreaming,
	}
	client, err := New(Config{
		Provider:     " google-gateway ",
		Model:        " gemini-model ",
		APIKey:       "key",
		Capabilities: configured,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	configured[0] = llm.CapabilityAudio

	wantCapabilities := []llm.Capability{
		llm.CapabilityGeneration,
		llm.CapabilityStreaming,
		llm.CapabilityTools,
	}
	info := client.Info()
	if info.Provider != "google-gateway" || info.Model != "gemini-model" ||
		!slices.Equal(info.Capabilities, wantCapabilities) {
		t.Fatalf("Info() = %#v, want provider google-gateway, model gemini-model, capabilities %v", info, wantCapabilities)
	}
	info.Capabilities[0] = llm.CapabilityAudio
	if got := client.Info().Capabilities; !slices.Equal(got, wantCapabilities) {
		t.Fatalf("Info().Capabilities after caller mutation = %v, want %v", got, wantCapabilities)
	}

	_, err = client.Generate(context.Background(), llm.Request{})
	_ = requireModelError(t, err, llm.KindInvalidRequest, "validate", "")
}

func TestRequestTranslatesReasoningEffort(t *testing.T) {
	request := llm.Request{
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
		ReasoningEffort: llm.ReasoningEffortXHigh,
	}
	_, config, err := requestToSDK("generate", request)
	if err != nil {
		t.Fatalf("requestToSDK() error = %v", err)
	}
	if config.ThinkingConfig == nil || config.ThinkingConfig.ThinkingLevel != "HIGH" || !config.ThinkingConfig.IncludeThoughts {
		t.Fatalf("ThinkingConfig = %#v", config.ThinkingConfig)
	}
}

func TestGenerateTranslatesMultimodalToolsAndResponse(t *testing.T) {
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
	t.Setenv("GOOGLE_GEMINI_BASE_URL", "http://ambient.invalid")
	t.Setenv("GOOGLE_API_KEY", "ambient-key")
	t.Setenv("GEMINI_API_KEY", "")

	var requestBody map[string]any
	var requestHeader http.Header
	var requestPath string
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requestHeader = request.Header.Clone()
		requestPath = request.URL.EscapedPath()
		if err := json.NewDecoder(request.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request: %v", err)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{
			"responseId":"response-1",
			"modelVersion":"gemini-model-001",
			"candidates":[{"content":{"role":"model","parts":[
				{"text":"hello"},
				{"inlineData":{"mimeType":"image/png","data":"aW1hZ2U="}},
				{"inlineData":{"mimeType":"audio/wav","data":"YXVkaW8="}}
			]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":4,"cachedContentTokenCount":2,"candidatesTokenCount":2,"thoughtsTokenCount":1,"totalTokenCount":7}
		}`)
	}))

	headers := http.Header{"X-Route": {"original"}}
	client, err := New(Config{
		Model:   "gemini-model",
		APIKey:  "secret",
		BaseURL: server.URL + "/proxy/",
		Capabilities: []llm.Capability{
			llm.CapabilityTools,
			llm.CapabilityVision,
			llm.CapabilityAudio,
		},
		HTTPClient: server.Client(),
		Headers:    headers,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	headers.Set("X-Route", "changed")

	temperature := 0.5
	topP := 0.8
	presencePenalty := 0.1
	frequencyPenalty := -0.1
	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.Part{{Text: "be concise"}}},
			{Role: llm.RoleUser, Content: []llm.Part{
				{Text: "describe"},
				{Kind: llm.PartImage, Data: []byte("image"), MediaType: "image/png"},
				{Kind: llm.PartAudio, Data: []byte("audio"), MediaType: "audio/wav"},
			}},
		},
		Tools: []llm.Tool{{
			Name:        "lookup_weather",
			Description: "looks up weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
			Strict:      true,
		}},
		MaxOutputTokens:  64,
		Temperature:      &temperature,
		TopP:             &topP,
		PresencePenalty:  &presencePenalty,
		FrequencyPenalty: &frequencyPenalty,
		Stop:             []string{"END"},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if requestPath != "/proxy/v1beta/models/gemini-model:generateContent" {
		t.Fatalf("request path = %q", requestPath)
	}
	if got := requestHeader.Get("X-Goog-Api-Key"); got != "secret" {
		t.Fatalf("X-Goog-Api-Key = %q, want secret", got)
	}
	if got := requestHeader.Get("X-Route"); got != "original" {
		t.Fatalf("X-Route = %q, want original", got)
	}
	assertGenerateRequest(t, requestBody)

	if response.ID != "response-1" || response.Model != "gemini-model-001" ||
		response.FinishReason != llm.FinishReasonStop {
		t.Fatalf("Generate() response metadata = %#v", response)
	}
	wantContent := []llm.Part{
		{Kind: llm.PartText, Text: "hello"},
		{Kind: llm.PartImage, Data: []byte("image"), MediaType: "image/png"},
		{Kind: llm.PartAudio, Data: []byte("audio"), MediaType: "audio/wav"},
	}
	if !slices.EqualFunc(response.Message.Content, wantContent, equalPart) {
		t.Fatalf("Generate() content = %#v, want %#v", response.Message.Content, wantContent)
	}
	if len(response.Message.ProviderData) == 0 {
		t.Fatal("Generate() ProviderData is empty")
	}
	if response.Usage == nil || *response.Usage != (llm.Usage{InputTokens: 2, OutputTokens: 3, CacheReadTokens: 2, ReasoningTokens: 1, TotalTokens: 7}) {
		t.Fatalf("Generate() usage = %#v", response.Usage)
	}
}

func assertGenerateRequest(t *testing.T, body map[string]any) {
	t.Helper()
	system := objectValue(t, body, "systemInstruction")
	systemParts := sliceValue(t, system, "parts")
	if got := fieldValue(t, systemParts[0], "text"); got != "be concise" {
		t.Fatalf("system text = %#v", got)
	}
	contents := sliceValue(t, body, "contents")
	if len(contents) != 1 || fieldValue(t, contents[0], "role") != "user" {
		t.Fatalf("contents = %#v", contents)
	}
	parts := sliceValue(t, contents[0], "parts")
	if len(parts) != 3 || fieldValue(t, parts[0], "text") != "describe" {
		t.Fatalf("user parts = %#v", parts)
	}
	if fieldValue(t, objectValue(t, parts[1], "inlineData"), "mimeType") != "image/png" ||
		fieldValue(t, objectValue(t, parts[2], "inlineData"), "mimeType") != "audio/wav" {
		t.Fatalf("inline data parts = %#v", parts)
	}
	generation := objectValue(t, body, "generationConfig")
	stops := sliceValue(t, generation, "stopSequences")
	if fieldValue(t, generation, "maxOutputTokens") != float64(64) ||
		len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("generation config = %#v", generation)
	}
	tools := sliceValue(t, body, "tools")
	declarations := sliceValue(t, tools[0], "functionDeclarations")
	if fieldValue(t, declarations[0], "name") != "lookup_weather" {
		t.Fatalf("function declarations = %#v", declarations)
	}
	toolConfig := objectValue(t, body, "toolConfig")
	calling := objectValue(t, toolConfig, "functionCallingConfig")
	if fieldValue(t, calling, "mode") != "VALIDATED" {
		t.Fatalf("function calling config = %#v", calling)
	}
}

func TestGeneratePreservesThoughtSignatureAcrossToolRoundTrip(t *testing.T) {
	var requestCount atomic.Int32
	var followup map[string]any
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		count := requestCount.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		if count == 1 {
			_, _ = io.WriteString(writer, `{
				"responseId":"tool-response",
				"candidates":[{"content":{"role":"model","parts":[{
					"functionCall":{"id":"call-1","name":"weather","args":{"city":"Boston"}},
					"thoughtSignature":"c2lnbmF0dXJl"
				},{"text":""}]},"finishReason":"STOP"}]
			}`)
			return
		}
		if err := json.NewDecoder(request.Body).Decode(&followup); err != nil {
			t.Errorf("decode follow-up request: %v", err)
		}
		_, _ = io.WriteString(writer, `{
			"candidates":[{"content":{"role":"model","parts":[{"text":"sunny"}]},"finishReason":"STOP"}]
		}`)
	}))

	client := mustTestClient(t, server, llm.CapabilityTools)
	tools := []llm.Tool{{Name: "weather", InputSchema: json.RawMessage(`{"type":"object"}`)}}
	user := llm.Message{Role: llm.RoleUser, Content: []llm.Part{{Text: "weather?"}}}
	first, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{user}, Tools: tools})
	if err != nil {
		t.Fatalf("first Generate() error = %v", err)
	}
	if first.FinishReason != llm.FinishReasonToolCall || len(first.Message.ToolCalls) != 1 || len(first.Message.Content) != 0 {
		t.Fatalf("first Generate() = %#v", first)
	}

	second, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{
			user,
			first.Message,
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{
				CallID:  "call-1",
				Content: []llm.Part{{Text: "72F"}},
			}}},
		},
		Tools: tools,
	})
	if err != nil {
		t.Fatalf("second Generate() error = %v", err)
	}
	if second.Text() != "sunny" {
		t.Fatalf("second Generate().Text() = %q, want sunny", second.Text())
	}

	contents := sliceValue(t, followup, "contents")
	modelParts := sliceValue(t, contents[1], "parts")
	if len(modelParts) != 1 {
		t.Fatalf("round-tripped model parts = %#v, want only the function call", modelParts)
	}
	if got := fieldValue(t, modelParts[0], "thoughtSignature"); got != "c2lnbmF0dXJl" {
		t.Fatalf("round-tripped thought signature = %#v", got)
	}
	functionCall := objectValue(t, modelParts[0], "functionCall")
	if fieldValue(t, functionCall, "id") != "call-1" || fieldValue(t, functionCall, "name") != "weather" {
		t.Fatalf("round-tripped function call = %#v", functionCall)
	}
	resultParts := sliceValue(t, contents[2], "parts")
	functionResponse := objectValue(t, resultParts[0], "functionResponse")
	if fieldValue(t, functionResponse, "id") != "call-1" || fieldValue(t, functionResponse, "name") != "weather" {
		t.Fatalf("function response = %#v", functionResponse)
	}
}

func TestGenerateClassifiesProviderErrors(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		status     string
		message    string
		wantKind   llm.ErrorKind
	}{
		{name: "authentication", statusCode: http.StatusUnauthorized, status: "UNAUTHENTICATED", wantKind: llm.KindAuthentication},
		{name: "permission", statusCode: http.StatusForbidden, status: "PERMISSION_DENIED", wantKind: llm.KindPermission},
		{name: "rate limit", statusCode: http.StatusTooManyRequests, status: "RESOURCE_EXHAUSTED", wantKind: llm.KindRateLimit},
		{name: "invalid request", statusCode: http.StatusBadRequest, status: "INVALID_ARGUMENT", wantKind: llm.KindInvalidRequest},
		{name: "context limit", statusCode: http.StatusBadRequest, status: "INVALID_ARGUMENT", message: "input token count exceeds the context window", wantKind: llm.KindContextLimit},
		{name: "provider", statusCode: http.StatusServiceUnavailable, status: "UNAVAILABLE", wantKind: llm.KindProvider},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "application/json")
				writer.WriteHeader(test.statusCode)
				_ = json.NewEncoder(writer).Encode(map[string]any{"error": map[string]any{
					"code": test.statusCode, "status": test.status, "message": test.message,
				}})
			}))

			client := mustTestClient(t, server)
			response, err := client.Generate(context.Background(), textRequest("hello"))
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			modelErr := requireModelError(t, err, test.wantKind, "generate", defaultProvider)
			if modelErr.HTTPStatus != test.statusCode {
				t.Fatalf("Generate() HTTP status = %d, want %d", modelErr.HTTPStatus, test.statusCode)
			}
		})
	}
}

func TestGenerateRejectsMalformedAndNonStrictJSONResponses(t *testing.T) {
	t.Run("malformed provider response", func(t *testing.T) {
		server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{`)
		}))
		client := mustTestClient(t, server)
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil {
			t.Fatalf("Generate() response = %#v, want nil", response)
		}
		_ = requireModelError(t, err, llm.KindMalformedResponse, "generate", defaultProvider)
	})

	t.Run("malformed provider shape", func(t *testing.T) {
		server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"candidates":{}}`)
		}))
		client := mustTestClient(t, server)
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil {
			t.Fatalf("Generate() response = %#v, want nil", response)
		}
		_ = requireModelError(t, err, llm.KindMalformedResponse, "generate", defaultProvider)
	})

	t.Run("cached tokens exceed prompt tokens", func(t *testing.T) {
		server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"toolUsePromptTokenCount":3,"cachedContentTokenCount":4,"candidatesTokenCount":1,"totalTokenCount":6}}`)
		}))
		client := mustTestClient(t, server)
		response, err := client.Generate(context.Background(), textRequest("hello"))
		if response != nil {
			t.Fatalf("Generate() response = %#v, want nil", response)
		}
		_ = requireModelError(t, err, llm.KindMalformedResponse, "generate", defaultProvider)
	})

	t.Run("non-strict JSON content", func(t *testing.T) {
		server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"candidates":[{"content":{"role":"model","parts":[{"text":"{\"x\":1,\"x\":2}"}]},"finishReason":"STOP"}]}`)
		}))
		client := mustTestClient(t, server, llm.CapabilityJSON)
		request := textRequest("hello")
		request.ResponseFormat = llm.ResponseFormatJSON
		response, err := client.Generate(context.Background(), request)
		if response != nil {
			t.Fatalf("Generate() response = %#v, want nil", response)
		}
		_ = requireModelError(t, err, llm.KindMalformedResponse, "generate", defaultProvider)
	})
}

func TestGeneratePreflightAvoidsProviderIO(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		writer.WriteHeader(http.StatusInternalServerError)
	}))
	client := mustTestClient(t, server)

	tests := []struct {
		name    string
		request llm.Request
		kind    llm.ErrorKind
	}{
		{name: "tools", request: llm.Request{
			Messages: textRequest("hello").Messages,
			Tools:    []llm.Tool{{Name: "tool", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		}, kind: llm.KindUnsupported},
		{name: "JSON", request: func() llm.Request {
			request := textRequest("hello")
			request.ResponseFormat = llm.ResponseFormatJSON
			return request
		}(), kind: llm.KindUnsupported},
		{name: "vision", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Kind: llm.PartImage, Data: []byte("x"), MediaType: "image/png"}}}}}, kind: llm.KindUnsupported},
		{name: "audio", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Kind: llm.PartAudio, Data: []byte("x"), MediaType: "audio/wav"}}}}}, kind: llm.KindUnsupported},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, err := client.Generate(context.Background(), test.request)
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			_ = requireModelError(t, err, test.kind, "generate", defaultProvider)
		})
	}

	cause := errors.New("stop")
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(cause)
	response, err := client.Generate(ctx, textRequest("hello"))
	if response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Generate(canceled) = (%#v, %v)", response, err)
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("provider requests = %d, want 0", got)
	}
}

func TestGenerateClassifiesCustomTransportError(t *testing.T) {
	transportErr := errors.New("dial failed")
	client, err := New(Config{
		Model:   "model",
		APIKey:  "key",
		BaseURL: "http://provider.example",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, transportErr
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	_ = requireModelError(t, err, llm.KindTransport, "generate", defaultProvider)
	if !errors.Is(err, transportErr) {
		t.Fatalf("Generate() error = %v, want transport cause", err)
	}
}

func TestGenerateClassifiesResponseReadError(t *testing.T) {
	readErr := errors.New("read failed")
	client, err := New(Config{
		Model:   "model",
		APIKey:  "key",
		BaseURL: "http://provider.example",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": {"application/json"}},
				Body:       &readErrorBody{err: readErr},
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	_ = requireModelError(t, err, llm.KindTransport, "generate", defaultProvider)
	if !errors.Is(err, readErr) {
		t.Fatalf("Generate() error = %v, want read cause", err)
	}
}

func mustTestClient(t *testing.T, server *httptest.Server, capabilities ...llm.Capability) *Client {
	t.Helper()
	client, err := New(Config{
		Model:        "gemini-model",
		APIKey:       "key",
		BaseURL:      server.URL,
		HTTPClient:   server.Client(),
		Capabilities: capabilities,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client
}

func textRequest(text string) llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: text}}}}}
}

func requireModelError(t *testing.T, err error, kind llm.ErrorKind, op, provider string) *llm.Error {
	t.Helper()
	if err == nil {
		t.Fatal("error = nil")
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) {
		t.Fatalf("errors.As(%v, *llm.Error) = false", err)
	}
	if modelErr.Kind != kind || modelErr.Op != op || modelErr.Provider != provider {
		t.Fatalf("model error = %#v, want kind %v op %q provider %q", modelErr, kind, op, provider)
	}
	return modelErr
}

func fieldValue(t *testing.T, value any, key string) any {
	t.Helper()
	object, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("value = %#v, want JSON object containing %q", value, key)
	}
	child, ok := object[key]
	if !ok {
		t.Fatalf("object %#v has no key %q", object, key)
	}
	return child
}

func objectValue(t *testing.T, value any, key string) map[string]any {
	t.Helper()
	child := fieldValue(t, value, key)
	result, ok := child.(map[string]any)
	if !ok {
		t.Fatalf("object[%q] = %#v, want JSON object", key, child)
	}
	return result
}

func sliceValue(t *testing.T, value any, key string) []any {
	t.Helper()
	object, ok := value.(map[string]any)
	if !ok {
		t.Fatalf("value = %#v, want JSON object containing %q", value, key)
	}
	child, ok := object[key]
	if !ok {
		t.Fatalf("object %#v has no key %q", object, key)
	}
	result, ok := child.([]any)
	if !ok {
		t.Fatalf("object[%q] = %#v, want JSON array", key, child)
	}
	return result
}

func equalPart(left, right llm.Part) bool {
	return left.Kind == right.Kind && left.Text == right.Text &&
		left.MediaType == right.MediaType && slices.Equal(left.Data, right.Data)
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}
