package models

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestNewRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name      string
		configs   []ProviderConfig
		provider  string
		wantError string
	}{
		{name: "empty collection", wantError: "at least one provider"},
		{name: "empty ID", configs: []ProviderConfig{{API: OpenAIChatCompletions}}, wantError: "ID must not be empty"},
		{name: "empty API", configs: []ProviderConfig{{ID: "openai"}}, provider: "openai", wantError: "API must not be empty"},
		{name: "unsupported API", configs: []ProviderConfig{{ID: "custom", API: "future-api"}}, provider: "custom", wantError: "not supported"},
		{
			name: "duplicate ID",
			configs: []ProviderConfig{
				{ID: "openai", API: OpenAIChatCompletions},
				{ID: " openai ", API: AnthropicMessages},
			},
			provider:  "openai",
			wantError: "more than once",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			collection, err := New(test.configs...)
			if collection != nil {
				t.Fatalf("New() collection = %#v, want nil", collection)
			}
			modelErr := requireModelError(t, err, llm.KindInvalidRequest, "configure", test.provider)
			if !strings.Contains(modelErr.Error(), test.wantError) {
				t.Fatalf("New() error = %q, want substring %q", modelErr, test.wantError)
			}
		})
	}
}

func TestGeneratorSelectsConfiguredAPI(t *testing.T) {
	collection, err := New(
		ProviderConfig{ID: " openai ", API: OpenAIChatCompletions},
		ProviderConfig{ID: "anthropic-gateway", API: AnthropicMessages},
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	tests := []struct {
		name string
		info llm.ModelInfo
	}{
		{
			name: "OpenAI",
			info: llm.ModelInfo{
				Provider:     "openai",
				Model:        " gpt-model ",
				Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
			},
		},
		{
			name: "Anthropic",
			info: llm.ModelInfo{
				Provider:     "anthropic-gateway",
				Model:        " claude-model ",
				Capabilities: []llm.Capability{llm.CapabilityStreaming},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator, err := collection.Generator(test.info)
			if err != nil {
				t.Fatalf("Generator() error = %v", err)
			}
			got := generator.Info()
			if got.Provider != test.info.Provider || got.Model != strings.TrimSpace(test.info.Model) {
				t.Fatalf("Generator().Info() = %#v", got)
			}
			wantCapabilities := append([]llm.Capability{llm.CapabilityGeneration}, test.info.Capabilities...)
			if !slices.Equal(got.Capabilities, wantCapabilities) {
				t.Fatalf("Generator().Info().Capabilities = %v, want %v", got.Capabilities, wantCapabilities)
			}
		})
	}
}

func TestGeneratorSupportsOpenAICompatibleProvider(t *testing.T) {
	receivedHeaders := make(chan http.Header, 1)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		receivedHeaders <- request.Header.Clone()
		writer.Header().Set("Content-Type", "application/json")
		writer.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(writer, `{"error":{"message":"denied","type":"authentication_error"}}`)
	}))
	defer server.Close()

	headers := http.Header{"X-Route": {"original"}}
	collection, err := New(ProviderConfig{
		ID:         "ollama",
		API:        OpenAIChatCompletions,
		BaseURL:    server.URL + "/v1",
		HTTPClient: server.Client(),
		Headers:    headers,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	headers.Set("X-Route", "changed")

	generator, err := collection.Generator(llm.ModelInfo{Provider: "ollama", Model: "local-model"})
	if err != nil {
		t.Fatalf("Generator() error = %v", err)
	}
	response, err := generator.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
	})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	modelErr := requireModelError(t, err, llm.KindAuthentication, "generate", "ollama")
	if modelErr.Kind != llm.KindAuthentication || modelErr.HTTPStatus != http.StatusUnauthorized {
		t.Fatalf("Generate() error = %#v", modelErr)
	}
	if got := (<-receivedHeaders).Get("X-Route"); got != "original" {
		t.Fatalf("request X-Route = %q, want %q", got, "original")
	}
}

func TestGeneratorRejectsInvalidIdentity(t *testing.T) {
	collection, err := New(ProviderConfig{ID: "openai", API: OpenAIChatCompletions})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	tests := []struct {
		name     string
		info     llm.ModelInfo
		provider string
		want     string
	}{
		{name: "empty provider", info: llm.ModelInfo{Model: "model"}, want: "provider must not be empty"},
		{name: "empty model", info: llm.ModelInfo{Provider: "openai"}, provider: "openai", want: "model name must not be empty"},
		{name: "unknown provider", info: llm.ModelInfo{Provider: "anthropic", Model: "model"}, provider: "anthropic", want: "not configured"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			generator, err := collection.Generator(test.info)
			if generator != nil {
				t.Fatalf("Generator() = %#v, want nil", generator)
			}
			modelErr := requireModelError(t, err, llm.KindInvalidRequest, "resolve", test.provider)
			if !strings.Contains(modelErr.Error(), test.want) {
				t.Fatalf("Generator() error = %q, want substring %q", modelErr, test.want)
			}
		})
	}
}

func TestNilCollectionReturnsError(t *testing.T) {
	var collection *Collection
	generator, err := collection.Generator(llm.ModelInfo{Provider: "openai", Model: "model"})
	if generator != nil {
		t.Fatalf("Generator() = %#v, want nil", generator)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "resolve", "openai")
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
