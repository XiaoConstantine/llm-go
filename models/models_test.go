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
		{name: "Codex API key", configs: []ProviderConfig{{ID: "openai-codex", API: OpenAICodexResponses, APIKey: "key"}}, provider: "openai-codex", wantError: "APIKey is not used"},
		{name: "token on API-key protocol", configs: []ProviderConfig{{ID: "openai", API: OpenAIChatCompletions, Credentials: Credentials{AccessToken: "token"}}}, provider: "openai", wantError: "token-based protocols"},
		{name: "ambiguous Codex credentials", configs: []ProviderConfig{{ID: "openai-codex", API: OpenAICodexResponses, Credentials: Credentials{AccessToken: "token"}, ResolveCredentials: func(context.Context, string) (Credentials, error) { return Credentials{}, nil }}}, provider: "openai-codex", wantError: "must be empty"},
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
		ProviderConfig{ID: " openai ", API: OpenAIResponses, APIKey: "key"},
		ProviderConfig{ID: "openai-compatible", API: OpenAIChatCompletions},
		ProviderConfig{ID: "openai-codex", API: OpenAICodexResponses, Credentials: Credentials{AccessToken: "token", AccountID: "account"}},
		ProviderConfig{ID: "anthropic-gateway", API: AnthropicMessages},
		ProviderConfig{ID: "google", API: GeminiGenerateContent, APIKey: "key"},
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
			name: "OpenAI compatible",
			info: llm.ModelInfo{
				Provider:     "openai-compatible",
				Model:        " local-model ",
				Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
			},
		},
		{
			name: "OpenAI Codex",
			info: llm.ModelInfo{
				Provider:     "openai-codex",
				Model:        " gpt-codex ",
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
		{
			name: "Gemini",
			info: llm.ModelInfo{
				Provider:     "google",
				Model:        " gemini-model ",
				Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision},
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

func TestGeneratorForCatalogModel(t *testing.T) {
	collection, err := New(ProviderConfig{ID: "openai-compatible", API: OpenAIChatCompletions})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	catalog, err := NewCatalog(llm.Model{
		Provider:     "openai-compatible",
		ID:           "local-model",
		Name:         "Local Model",
		API:          llm.APIOpenAIChatCompletions,
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
	})
	if err != nil {
		t.Fatalf("NewCatalog() error = %v", err)
	}
	model, ok := catalog.Model("openai-compatible", "local-model")
	if !ok {
		t.Fatal("Catalog.Model() did not find local-model")
	}
	generator, err := collection.GeneratorFor(model)
	if err != nil {
		t.Fatalf("GeneratorFor() error = %v", err)
	}
	info := generator.Info()
	if info.Provider != model.Provider || info.Model != model.ID || !slices.Equal(info.Capabilities, model.Capabilities) {
		t.Fatalf("GeneratorFor().Info() = %#v, want model %#v", info, model)
	}

	model.API = llm.APIAnthropicMessages
	generator, err = collection.GeneratorFor(model)
	if generator != nil {
		t.Fatalf("GeneratorFor(mismatched API) = %#v, want nil", generator)
	}
	modelErr := requireModelError(t, err, llm.KindInvalidRequest, "resolve", "openai-compatible")
	if !strings.Contains(modelErr.Error(), "does not match") {
		t.Fatalf("GeneratorFor(mismatched API) error = %q", modelErr)
	}
}

func TestGeneratorSupportsOpenAICodexSubscription(t *testing.T) {
	requests := make(chan *http.Request, 1)
	rejectedTokens := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		requests <- request.Clone(request.Context())
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
	}))
	defer server.Close()

	collection, err := New(ProviderConfig{
		ID:  "openai-codex",
		API: OpenAICodexResponses,
		ResolveCredentials: func(_ context.Context, rejectedAccessToken string) (Credentials, error) {
			rejectedTokens <- rejectedAccessToken
			return Credentials{AccessToken: "subscription-token", AccountID: "account-123"}, nil
		},
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	generator, err := collection.Generator(llm.ModelInfo{Provider: "openai-codex", Model: "gpt-codex"})
	if err != nil {
		t.Fatalf("Generator() error = %v", err)
	}
	response, err := generator.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.FinishReason != llm.FinishReasonStop {
		t.Fatalf("Generate().FinishReason = %q", response.FinishReason)
	}
	request := <-requests
	if request.URL.Path != "/codex/responses" {
		t.Fatalf("request path = %q", request.URL.Path)
	}
	if got := request.Header.Get("Authorization"); got != "Bearer subscription-token" {
		t.Fatalf("Authorization = %q", got)
	}
	if got := request.Header.Get("ChatGPT-Account-ID"); got != "account-123" {
		t.Fatalf("ChatGPT-Account-ID = %q", got)
	}
	if got := <-rejectedTokens; got != "" {
		t.Fatalf("rejected access token = %q, want empty", got)
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

func TestGeneratorReturnsNilOnProviderConfigurationError(t *testing.T) {
	tests := []struct {
		name     string
		config   ProviderConfig
		model    string
		contains string
	}{
		{name: "OpenAI", config: ProviderConfig{ID: "openai", API: OpenAIChatCompletions, BaseURL: ":"}, model: "gpt", contains: "base URL"},
		{name: "OpenAI Responses", config: ProviderConfig{ID: "openai-responses", API: OpenAIResponses}, model: "gpt", contains: "API key"},
		{name: "OpenAI Codex", config: ProviderConfig{ID: "openai-codex", API: OpenAICodexResponses}, model: "gpt-codex", contains: "access token"},
		{name: "Anthropic", config: ProviderConfig{ID: "anthropic", API: AnthropicMessages, BaseURL: ":"}, model: "claude", contains: "base URL"},
		{name: "Gemini", config: ProviderConfig{ID: "google", API: GeminiGenerateContent}, model: "gemini", contains: "API key"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			collection, err := New(test.config)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			generator, err := collection.Generator(llm.ModelInfo{Provider: test.config.ID, Model: test.model})
			if generator != nil {
				t.Fatalf("Generator() = %#v, want nil", generator)
			}
			modelErr := requireModelError(t, err, llm.KindInvalidRequest, "configure", test.config.ID)
			if !strings.Contains(modelErr.Error(), test.contains) {
				t.Fatalf("Generator() error = %v", modelErr)
			}
		})
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
