package vertex

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestGenerateWithVertexAPIKey(t *testing.T) {
	var path string
	var header http.Header
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		path = request.URL.EscapedPath()
		header = request.Header.Clone()
		if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
			t.Errorf("decode body: %v", err)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{
			"responseId":"vertex-response","modelVersion":"served-model",
			"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}
		}`)
	}))
	defer server.Close()

	headers := http.Header{"X-Route": {"owned"}}
	client, err := New(Config{Model: "gemini-model", Reasoning: true, APIKey: "vertex-key", BaseURL: server.URL + "/proxy",
		HTTPClient: server.Client(), Headers: headers})
	if err != nil {
		t.Fatal(err)
	}
	headers.Set("X-Route", "mutated")
	if info := client.Info(); info.Provider != "google-vertex" || info.Model != "gemini-model" || info.API != llm.APIGoogleVertex || !info.Reasoning {
		t.Fatalf("Info() = %#v", info)
	}
	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.ID != "vertex-response" || response.Model != "served-model" || response.Text() != "hello" ||
		response.FinishReason != llm.FinishReasonStop {
		t.Fatalf("response = %#v", response)
	}
	if path != "/proxy/v1/publishers/google/models/gemini-model:generateContent" {
		t.Fatalf("path = %q", path)
	}
	if header.Get("X-Goog-Api-Key") != "vertex-key" || header.Get("X-Route") != "owned" {
		t.Fatalf("headers = %#v", header)
	}
	contents, ok := body["contents"].([]any)
	if !ok || len(contents) != 1 {
		t.Fatalf("body = %#v", body)
	}
}

func TestProjectLocationWithAuthenticatedHTTPClient(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if got := request.Header.Get("Authorization"); got != "Bearer supplied" {
			t.Errorf("Authorization = %q", got)
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
	}))
	defer server.Close()
	transport := roundTripFunc(func(request *http.Request) (*http.Response, error) {
		request.Header.Set("Authorization", "Bearer supplied")
		return server.Client().Transport.RoundTrip(request)
	})
	client, err := New(Config{Model: "model", Project: "project", Location: "us-central1", BaseURL: server.URL,
		HTTPClient: &http.Client{Transport: transport}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}}); err != nil {
		t.Fatal(err)
	}
}

func TestStreamUsesVertexEndpoint(t *testing.T) {
	var path, query string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		path, query = request.URL.EscapedPath(), request.URL.RawQuery
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer,
			`data: {"responseId":"id","modelVersion":"served","candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`+"\n\n")
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming}})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	response, err := llm.Collect(stream, nil)
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "ok" || response.ID != "id" || response.Model != "served" {
		t.Fatalf("response = %#v", response)
	}
	if path != "/v1/publishers/google/models/model:streamGenerateContent" || query != "alt=sse" {
		t.Fatalf("endpoint = %q?%s", path, query)
	}
}

func TestReasoningControlsRequireDeclaredSupport(t *testing.T) {
	client, err := New(Config{Model: "model", APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}},
		ReasoningEffort: llm.ReasoningEffortHigh}
	if response, err := client.Generate(context.Background(), request); response != nil || err == nil {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
}

func TestConfigurationValidationAndProviderRelabeling(t *testing.T) {
	for _, config := range []Config{
		{},
		{Model: "model"},
		{Model: "model", Project: "project"},
		{Model: "model", Location: "location"},
		{Model: "model", APIKey: "key", BaseURL: "://bad"},
	} {
		client, err := New(config)
		if client != nil || err == nil {
			t.Fatalf("New(%#v) = %#v, %v", config, client, err)
		}
		var modelErr *llm.Error
		if !errors.As(err, &modelErr) || modelErr.Provider != "google-vertex" || modelErr.Kind != llm.KindInvalidRequest {
			t.Fatalf("error = %v (%#v)", err, modelErr)
		}
	}

	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(writer, `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"slow down"}}`)
	}))
	defer server.Close()
	client, err := New(Config{Provider: "vertex-gateway", Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}})
	if response != nil || err == nil {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Provider != "vertex-gateway" || modelErr.Kind != llm.KindRateLimit {
		t.Fatalf("error = %v (%#v)", err, modelErr)
	}
}

func TestVertexConfigIgnoresRegionalTemplateBaseURL(t *testing.T) {
	client, err := New(Config{Model: "model", APIKey: "key", BaseURL: "https://{location}-aiplatform.googleapis.com"})
	if client == nil || err != nil {
		t.Fatalf("New() = %#v, %v", client, err)
	}
}

func TestCustomBaseURLContainingVersionIsNotDuplicated(t *testing.T) {
	base, version, ok := normalizeCustomBaseURL("https://gateway.example/proxy/v1", "v1")
	if !ok || base != "https://gateway.example" || version != "proxy/v1" {
		t.Fatalf("normalizeCustomBaseURL() = %q, %q, %v", base, version, ok)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}
