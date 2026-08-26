package models

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/openai"
)

func TestBuiltinProviders(t *testing.T) {
	tests := []struct {
		id      string
		api     API
		baseURL string
	}{
		{ProviderOpenRouter, OpenAIChatCompletions, "https://openrouter.ai/api/v1"},
		{ProviderGroq, OpenAIChatCompletions, "https://api.groq.com/openai/v1"},
		{ProviderDeepSeek, OpenAIChatCompletions, "https://api.deepseek.com"},
		{ProviderXAI, OpenAIResponses, "https://api.x.ai/v1"},
		{ProviderCerebras, OpenAIChatCompletions, "https://api.cerebras.ai/v1"},
		{ProviderFireworks, OpenAIChatCompletions, "https://api.fireworks.ai/inference/v1"},
	}

	profiles := BuiltinProviders()
	if len(profiles) != len(tests) {
		t.Fatalf("BuiltinProviders() length = %d, want %d", len(profiles), len(tests))
	}
	for index, test := range tests {
		profile, ok := BuiltinProvider(" " + test.id + " ")
		if !ok {
			t.Fatalf("BuiltinProvider(%q) was not found", test.id)
		}
		if profile != profiles[index] || profile.ID() != test.id || profile.API() != test.api || profile.BaseURL() != test.baseURL {
			t.Fatalf("profile %q = %#v", test.id, profile)
		}
		config := profile.Config("secret")
		if config.ID != test.id || config.API != test.api || config.APIKey != "secret" || config.BaseURL != test.baseURL {
			t.Fatalf("Config(%q) = %#v", test.id, config)
		}
	}
	if _, ok := BuiltinProvider("GROQ"); ok {
		t.Fatal("BuiltinProvider accepted noncanonical case")
	}
	if _, ok := BuiltinProvider("missing"); ok {
		t.Fatal("BuiltinProvider found an unknown provider")
	}

	profiles[0] = ProviderProfile{}
	again := BuiltinProviders()
	if again[0].ID() != ProviderOpenRouter {
		t.Fatalf("caller mutation changed built-in profiles: %#v", again)
	}
}

func TestBuiltinProfileConfigsConstructGenerators(t *testing.T) {
	profiles := BuiltinProviders()
	configs := make([]ProviderConfig, len(profiles))
	for index, profile := range profiles {
		configs[index] = profile.Config("key")
	}
	collection, err := New(configs...)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	for _, profile := range profiles {
		model := llm.Model{
			Provider:     profile.ID(),
			ID:           "model",
			API:          llm.API(profile.API()),
			Capabilities: []llm.Capability{llm.CapabilityStreaming},
		}
		generator, err := collection.GeneratorFor(model)
		if err != nil {
			t.Fatalf("GeneratorFor(%q) error = %v", profile.ID(), err)
		}
		if info := generator.Info(); info.Provider != profile.ID() || info.Model != "model" ||
			!slices.Equal(info.Capabilities, []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming}) {
			t.Fatalf("GeneratorFor(%q).Info() = %#v", profile.ID(), info)
		}
	}
}

func TestXAIProfileRequestsEncryptedReasoning(t *testing.T) {
	payloads := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer func() { _ = request.Body.Close() }()
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		var payload map[string]any
		if err := jsonv2.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		payloads <- payload
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"response\",\"model\":\"grok-model\",\"status\":\"completed\",\"output\":[]}}\n\n")
	}))
	defer server.Close()

	profile, _ := BuiltinProvider(ProviderXAI)
	provider := profile.Config("key")
	provider.BaseURL = server.URL + "/v1"
	provider.HTTPClient = server.Client()
	collection, err := New(provider)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	generator, err := collection.Generator(llm.ModelInfo{Provider: ProviderXAI, Model: "grok-model"})
	if err != nil {
		t.Fatalf("Generator() error = %v", err)
	}
	if _, err := generator.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	payload := <-payloads
	include, _ := payload["include"].([]any)
	if len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %#v", payload["include"])
	}
	if reasoning, _ := payload["reasoning"].(map[string]any); reasoning["summary"] != nil {
		t.Fatalf("reasoning = %#v, want no summary request", reasoning)
	}
}

func TestDeepSeekProfileUsesLegacyLimitAndPreservesReasoning(t *testing.T) {
	requests := make(chan map[string]any, 2)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer func() { _ = request.Body.Close() }()
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
			return
		}
		var payload map[string]any
		if err := jsonv2.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		requests <- payload
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chat","model":"deepseek-model","choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning_content":"working"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}`)
	}))
	defer server.Close()

	config, ok := BuiltinProvider(ProviderDeepSeek)
	if !ok {
		t.Fatal("DeepSeek profile not found")
	}
	provider := config.Config("key")
	provider.BaseURL = server.URL + "/v1"
	provider.HTTPClient = server.Client()
	collection, err := New(provider)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	generator, err := collection.Generator(llm.ModelInfo{Provider: ProviderDeepSeek, Model: "deepseek-model"})
	if err != nil {
		t.Fatalf("Generator() error = %v", err)
	}
	request := llm.Request{
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "question"}}}},
		MaxOutputTokens: 123,
	}
	response, err := generator.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.ReasoningSummary != "" || response.Text() != "answer" {
		t.Fatalf("Generate() response = %#v", response)
	}
	data, err := openai.ParseReasoningData(response.Message)
	if err != nil || data.ReasoningContent == nil || *data.ReasoningContent != "working" {
		t.Fatalf("ParseReasoningData() = (%#v, %v)", data, err)
	}
	first := <-requests
	if first["max_tokens"] != float64(123) {
		t.Fatalf("max_tokens = %#v", first["max_tokens"])
	}
	if _, exists := first["max_completion_tokens"]; exists {
		t.Fatalf("request contains max_completion_tokens: %#v", first)
	}

	request.Messages = []llm.Message{request.Messages[0], response.Message, {Role: llm.RoleUser, Content: []llm.Part{{Text: "continue"}}}}
	request.MaxOutputTokens = 0
	if _, err := generator.Generate(context.Background(), request); err != nil {
		t.Fatalf("continuation Generate() error = %v", err)
	}
	second := <-requests
	messages := second["messages"].([]any)
	assistant := messages[1].(map[string]any)
	if assistant["reasoning_content"] != "working" {
		t.Fatalf("continued assistant message = %#v", assistant)
	}
}
