package modelsdev

import (
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func floatPtr(v float64) *float64 { return &v }
func intPtr(v int) *int           { return &v }

func TestDefaultProviderAPI(t *testing.T) {
	tests := []struct {
		provider string
		want     llm.API
	}{
		{"anthropic", llm.APIAnthropicMessages},
		{"openai", llm.APIOpenAIResponses},
		{"xai", llm.APIOpenAIResponses},
		{"google", llm.APIGeminiGenerateContent},
		{"openai-codex", llm.APIOpenAICodexResponses},
		{"openrouter", llm.APIOpenAIChatCompletions},
		{"deepseek", llm.APIOpenAIChatCompletions},
		{"groq", llm.APIOpenAIChatCompletions},
		{"unknown", llm.APIOpenAIChatCompletions},
	}
	for _, tt := range tests {
		got := DefaultProviderAPI(tt.provider)
		if got != tt.want {
			t.Errorf("DefaultProviderAPI(%q) = %q, want %q", tt.provider, got, tt.want)
		}
	}
}

func TestDefaultProviderCompatibilityMatchesResolvedAPI(t *testing.T) {
	// Anthropic with AnthropicMessages API -> valid
	compat := DefaultProviderCompatibility("anthropic", ModelEntry{Reasoning: true}, llm.APIAnthropicMessages)
	if compat == nil || compat.Anthropic == nil || compat.Anthropic.AdaptiveThinking != llm.CompatibilityEnabled {
		t.Errorf("expected Anthropic compatibility with AdaptiveThinking, got %+v", compat)
	}

	// Anthropic routed through OpenAIChatCompletions -> should not attach Anthropic compatibility
	compatWrongAPI := DefaultProviderCompatibility("anthropic", ModelEntry{Reasoning: true}, llm.APIOpenAIChatCompletions)
	if compatWrongAPI != nil {
		t.Errorf("expected nil compatibility when Anthropic is routed via Chat Completions, got %+v", compatWrongAPI)
	}

	// OpenAI with OpenAIResponses -> valid
	openaiCompat := DefaultProviderCompatibility("openai", ModelEntry{}, llm.APIOpenAIResponses)
	if openaiCompat == nil || openaiCompat.OpenAIResponses == nil {
		t.Errorf("expected OpenAIResponses compatibility, got %+v", openaiCompat)
	}

	// OpenAI routed via OpenAIChatCompletions -> nil
	openaiChatCompat := DefaultProviderCompatibility("openai", ModelEntry{}, llm.APIOpenAIChatCompletions)
	if openaiChatCompat != nil {
		t.Errorf("expected nil compatibility when OpenAI is routed via Chat Completions, got %+v", openaiChatCompat)
	}
}

func TestConvertModelBasic(t *testing.T) {
	entry := ModelEntry{
		ID:               "gpt-4o",
		Name:             "GPT-4o",
		ToolCall:         true,
		StructuredOutput: true,
		Reasoning:        false,
		Modalities: &Modalities{
			Input:  []string{"text", "image"},
			Output: []string{"text"},
		},
		Limit: &Limit{
			Context: 128000,
			Output:  16384,
		},
		Cost: &Cost{
			Input:      floatPtr(2.5),
			Output:     floatPtr(10.0),
			CacheRead:  floatPtr(1.25),
			CacheWrite: floatPtr(0),
		},
	}

	model, err := ConvertModel("openai", entry, llm.APIOpenAIResponses, nil)
	if err != nil {
		t.Fatalf("ConvertModel failed: %v", err)
	}

	if model.ID != "gpt-4o" || model.Provider != "openai" || model.Name != "GPT-4o" {
		t.Errorf("unexpected model identity: %+v", model)
	}
	if model.ContextWindow != 128000 || model.MaxOutputTokens != 16384 {
		t.Errorf("unexpected limits: context=%d max_out=%d", model.ContextWindow, model.MaxOutputTokens)
	}
	if model.Cost == nil || model.Cost.Input != 2.5 || model.Cost.Output != 10.0 || model.Cost.CacheRead != 1.25 {
		t.Errorf("unexpected cost: %+v", model.Cost)
	}

	// Verify capabilities
	expectedCaps := []llm.Capability{
		llm.CapabilityStreaming,
		llm.CapabilityTools,
		llm.CapabilityJSON,
		llm.CapabilityVision,
	}
	if len(model.Capabilities) != len(expectedCaps) {
		t.Fatalf("capabilities len = %d, want %d: %v", len(model.Capabilities), len(expectedCaps), model.Capabilities)
	}
	for i, cap := range expectedCaps {
		if model.Capabilities[i] != cap {
			t.Errorf("capability[%d] = %v, want %v", i, model.Capabilities[i], cap)
		}
	}
}

func TestConvertModelTiers(t *testing.T) {
	entry := ModelEntry{
		ID:   "gpt-5.5",
		Name: "GPT-5.5",
		Limit: &Limit{
			Context: 1000000,
			Output:  128000,
		},
		Cost: &Cost{
			Input:     floatPtr(5.0),
			Output:    floatPtr(30.0),
			CacheRead: floatPtr(0.5),
			Tiers: []CostTierEntry{
				{
					Input:     floatPtr(10.0),
					Output:    floatPtr(45.0),
					CacheRead: floatPtr(1.0),
					Tier: &TierCondition{
						Type: "context",
						Size: 272000,
					},
				},
			},
		},
	}

	model, err := ConvertModel("openai", entry, llm.APIOpenAIResponses, nil)
	if err != nil {
		t.Fatalf("ConvertModel failed: %v", err)
	}

	if model.Cost == nil || len(model.Cost.Tiers) != 1 {
		t.Fatalf("expected 1 tier, got %+v", model.Cost)
	}
	tier := model.Cost.Tiers[0]
	if tier.InputTokensAbove != 272000 || tier.Input != 10.0 || tier.Output != 45.0 || tier.CacheRead != 1.0 {
		t.Errorf("unexpected tier: %+v", tier)
	}
}

func TestConvertModelTierOnlyPricing(t *testing.T) {
	entry := ModelEntry{
		ID:   "tiered-model",
		Name: "Tiered Model",
		Limit: &Limit{
			Context: 1000000,
			Output:  8192,
		},
		Cost: &Cost{
			Tiers: []CostTierEntry{
				{
					Input:     floatPtr(1.0),
					Output:    floatPtr(2.0),
					CacheRead: floatPtr(0.1),
					Tier: &TierCondition{
						Type: "context",
						Size: 100000,
					},
				},
			},
		},
	}

	model, err := ConvertModel("openrouter", entry, llm.APIOpenAIChatCompletions, nil)
	if err != nil {
		t.Fatalf("ConvertModel failed: %v", err)
	}

	if model.Cost == nil {
		t.Fatal("expected non-nil cost for tier-only pricing")
	}
	if len(model.Cost.Tiers) != 1 || model.Cost.Tiers[0].InputTokensAbove != 100000 {
		t.Errorf("unexpected cost tiers: %+v", model.Cost)
	}
}

func TestConvertModelContextOver200kFallback(t *testing.T) {
	entry := ModelEntry{
		ID:   "gemini-2.5-pro",
		Name: "Gemini 2.5 Pro",
		Limit: &Limit{
			Context: 1000000,
			Output:  8192,
		},
		Cost: &Cost{
			Input:     floatPtr(1.25),
			Output:    floatPtr(10.0),
			CacheRead: floatPtr(0.125),
			ContextOver200k: &CostRates{
				Input:     floatPtr(2.5),
				Output:    floatPtr(15.0),
				CacheRead: floatPtr(0.25),
			},
		},
	}

	model, err := ConvertModel("google", entry, llm.APIGeminiGenerateContent, nil)
	if err != nil {
		t.Fatalf("ConvertModel failed: %v", err)
	}

	if model.Cost == nil || len(model.Cost.Tiers) != 1 {
		t.Fatalf("expected 1 tier from context_over_200k, got %+v", model.Cost)
	}
	tier := model.Cost.Tiers[0]
	if tier.InputTokensAbove != 200000 || tier.Input != 2.5 || tier.Output != 15.0 || tier.CacheRead != 0.25 {
		t.Errorf("unexpected tier: %+v", tier)
	}
}

func TestConvertModelValidation(t *testing.T) {
	_, err := ConvertModel("", ModelEntry{ID: "m"}, llm.APIOpenAIResponses, nil)
	if err == nil {
		t.Error("expected error for empty provider")
	}

	_, err = ConvertModel("p", ModelEntry{}, llm.APIOpenAIResponses, nil)
	if err == nil {
		t.Error("expected error for empty ID")
	}
}
