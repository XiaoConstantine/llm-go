package azure

import (
	"context"
	jsonv2 "encoding/json/v2"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestGenerateUsesAzureResponsesProtocol(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/openai/v1/responses" {
			t.Errorf("path = %q, want /openai/v1/responses", request.URL.Path)
		}
		if got := request.URL.Query().Get("api-version"); got != "2026-01-01" {
			t.Errorf("api-version = %q", got)
		}
		if got := request.Header.Get("Api-Key"); got != "azure-secret" {
			t.Errorf("Api-Key = %q", got)
		}
		if got := request.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization = %q, want empty", got)
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
		if payload["model"] != "prod-gpt-deployment" || payload["stream"] != true {
			t.Errorf("model/stream = %#v/%#v", payload["model"], payload["stream"])
		}
		reasoning, _ := payload["reasoning"].(map[string]any)
		if reasoning["summary"] != "auto" {
			t.Errorf("reasoning = %#v, want summary auto", payload["reasoning"])
		}
		text, _ := payload["text"].(map[string]any)
		if text["verbosity"] != "low" {
			t.Errorf("text = %#v, want low verbosity", payload["text"])
		}
		include, _ := payload["include"].([]any)
		if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
			t.Errorf("include = %#v", payload["include"])
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"delta\":\"hello\"}\n\n")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_text.done\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"text\":\"hello\"}\n\n")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}]}}\n\n")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"served-deployment\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n")
	}))
	defer server.Close()

	client, err := New(Config{Provider: "azure", Model: "gpt-5-catalog-model", DeploymentName: "prod-gpt-deployment",
		APIKey: "azure-secret", APIVersion: "2026-01-01", BaseURL: server.URL + "/openai/v1",
		HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if info := client.Info(); info.Provider != "azure" || info.Model != "gpt-5-catalog-model" ||
		info.API != llm.APIAzureOpenAIResponses || !slices.Contains(info.Capabilities, llm.CapabilityStreaming) {
		t.Fatalf("Info() = %#v", info)
	}
	response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{
		Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}},
	}}})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.ID != "resp_1" || response.Model != "served-deployment" || response.Text() != "hello" {
		t.Fatalf("Generate() = %#v", response)
	}
}

func TestStreamUsesDefaultAPIVersionAndEmitsEvents(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if got := request.URL.Query().Get("api-version"); got != "v1" {
			t.Errorf("api-version = %q, want v1", got)
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"delta\":\"hello\"}\n\n")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_text.done\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"text\":\"hello\"}\n\n")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}]}}\n\n")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"deployment\",\"status\":\"completed\",\"output\":[]}}\n\n")
	}))
	defer server.Close()
	client, err := New(Config{Model: "gpt-4o", APIKey: "key", BaseURL: server.URL,
		HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming}})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	t.Cleanup(func() {
		if closeErr := stream.Close(); closeErr != nil {
			t.Errorf("Close() error = %v", closeErr)
		}
	})
	var text strings.Builder
	var kinds []llm.StreamEventKind
	var finish llm.FinishReason
	for {
		chunk, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			t.Fatalf("Recv() error = %v", recvErr)
		}
		for _, part := range chunk.Content {
			text.WriteString(part.Text)
		}
		for _, event := range chunk.Events {
			kinds = append(kinds, event.Kind)
		}
		if chunk.FinishReason != "" {
			finish = chunk.FinishReason
		}
	}
	if text.String() != "hello" || finish != llm.FinishReasonStop ||
		!slices.Contains(kinds, llm.StreamEventTextDelta) || !slices.Contains(kinds, llm.StreamEventDone) {
		t.Fatalf("stream = text %q, finish %q, events %#v", text.String(), finish, kinds)
	}
}

func TestCompatibilityMetadataIsOwned(t *testing.T) {
	compatibility := &llm.OpenAIResponsesCompatibility{StrictTools: llm.CompatibilityDisabled}
	client, err := NewWithCompatibility(Config{Model: "model", APIKey: "key", ResourceName: "resource"}, compatibility)
	if err != nil {
		t.Fatal(err)
	}
	compatibility.StrictTools = llm.CompatibilityEnabled
	info := client.Info()
	if info.Compatibility == nil || info.Compatibility.OpenAIResponses == nil ||
		info.Compatibility.OpenAIResponses.StrictTools != llm.CompatibilityDisabled {
		t.Fatalf("Info().Compatibility = %#v", info.Compatibility)
	}
}

func TestHTTPErrorClassification(t *testing.T) {
	tests := []struct {
		status int
		kind   llm.ErrorKind
	}{
		{status: http.StatusUnauthorized, kind: llm.KindAuthentication},
		{status: http.StatusForbidden, kind: llm.KindPermission},
		{status: http.StatusNotFound, kind: llm.KindInvalidRequest},
		{status: http.StatusTooManyRequests, kind: llm.KindRateLimit},
	}
	for _, test := range tests {
		t.Run(http.StatusText(test.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "application/json")
				writer.WriteHeader(test.status)
				_, _ = io.WriteString(writer, `{"error":{"message":"request failed","type":"request_error","code":"failed"}}`)
			}))
			defer server.Close()
			client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()})
			if err != nil {
				t.Fatal(err)
			}
			response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			var modelErr *llm.Error
			if !errors.As(err, &modelErr) || modelErr.Kind != test.kind || modelErr.HTTPStatus != test.status || modelErr.Provider != defaultProvider {
				t.Fatalf("Generate() error = %#v, want kind %v and status %d", err, test.kind, test.status)
			}
		})
	}
}

func TestResolveBaseURL(t *testing.T) {
	tests := []struct {
		name     string
		baseURL  string
		resource string
		want     string
		wantErr  string
	}{
		{name: "resource", resource: "my-resource", want: "https://my-resource.openai.azure.com/openai/v1"},
		{name: "Azure host root", baseURL: "https://my-resource.openai.azure.com/", want: "https://my-resource.openai.azure.com/openai/v1"},
		{name: "Azure host openai", baseURL: "https://my-resource.cognitiveservices.azure.com/openai/", want: "https://my-resource.cognitiveservices.azure.com/openai/v1"},
		{name: "custom proxy", baseURL: "http://localhost:8080/custom/", want: "http://localhost:8080/custom"},
		{name: "missing", wantErr: "base URL is required"},
		{name: "bad resource", resource: "bad/name", wantErr: "resource name"},
		{name: "query", baseURL: "https://example.com/openai?x=1", wantErr: "query or fragment"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := resolveBaseURL("azure", test.baseURL, test.resource)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("resolveBaseURL() = (%q, %v), want %q", got, err, test.wantErr)
				}
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("resolveBaseURL() = (%q, %v), want %q", got, err, test.want)
			}
		})
	}
}

func TestNewValidatesAzureConfiguration(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		want   string
	}{
		{name: "model", config: Config{APIKey: "key", ResourceName: "resource"}, want: "model must not be empty"},
		{name: "key", config: Config{Model: "model", ResourceName: "resource"}, want: "API key must not be empty"},
		{name: "API version", config: Config{Model: "model", APIKey: "key", ResourceName: "resource", APIVersion: "v1?bad"}, want: "API version"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(test.config)
			if client != nil || err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("New() = (%#v, %v), want %q", client, err, test.want)
			}
			var classified *llm.Error
			if !errors.As(err, &classified) || classified.Kind != llm.KindInvalidRequest {
				t.Fatalf("New() error = %#v", err)
			}
		})
	}
}
