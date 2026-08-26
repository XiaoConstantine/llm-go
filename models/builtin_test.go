package models

import (
	"slices"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestBuiltinCatalogSnapshot(t *testing.T) {
	catalog := BuiltinCatalog()
	if catalog == nil {
		t.Fatal("BuiltinCatalog() = nil")
	}
	models := catalog.Models("")
	if len(models) != 442 {
		t.Fatalf("len(BuiltinCatalog().Models()) = %d, want 442", len(models))
	}

	deepseek, ok := catalog.Model(ProviderDeepSeek, "deepseek-v4-flash")
	if !ok {
		t.Fatal("DeepSeek representative model is missing")
	}
	if deepseek.API != llm.APIOpenAIChatCompletions || !deepseek.Reasoning ||
		deepseek.ContextWindow != 1_000_000 || deepseek.MaxOutputTokens != 384_000 || deepseek.Cost == nil {
		t.Fatalf("DeepSeek model = %#v", deepseek)
	}
	compatibility := deepseek.Compatibility
	if compatibility == nil || compatibility.OpenAIChat == nil ||
		compatibility.OpenAIChat.MaxTokensField != llm.MaxTokensFieldLegacy ||
		compatibility.OpenAIChat.ReasoningContentReplay != llm.CompatibilityEnabled ||
		compatibility.OpenAIChat.ThinkingFormat != llm.ThinkingFormatDeepSeek {
		t.Fatalf("DeepSeek compatibility = %#v", compatibility)
	}

	automatic, ok := catalog.Model(ProviderOpenRouter, "openrouter/auto")
	if !ok || automatic.Cost != nil {
		t.Fatalf("OpenRouter automatic model = (%#v, %t), want unknown cost", automatic, ok)
	}

	audio, ok := catalog.Model(ProviderOpenRouter, "openai/gpt-audio")
	if !ok {
		t.Fatal("OpenRouter audio model is missing")
	}
	hasAudio := false
	for _, capability := range audio.Capabilities {
		hasAudio = hasAudio || capability == llm.CapabilityAudio
	}
	if !hasAudio {
		t.Fatalf("OpenRouter audio capabilities = %#v, want audio", audio.Capabilities)
	}
	if audio.Cost != nil {
		t.Fatalf("OpenRouter audio cost = %#v, want unknown because audio and text token rates differ", audio.Cost)
	}
}

func TestBuiltinCatalogProtocolCapabilitiesAreUsable(t *testing.T) {
	catalog := BuiltinCatalog()
	for _, model := range catalog.Models("anthropic") {
		if model.API == llm.APIAnthropicMessages && !hasCapability(model.Capabilities, llm.CapabilityVision) {
			t.Errorf("Anthropic model %q lacks vision", model.ID)
		}
	}
	confirmedJSON := map[string]bool{
		"gpt-4-turbo": true, "gpt-4.1": true, "gpt-4.1-mini": true, "gpt-4.1-nano": true,
		"gpt-4o": true, "gpt-4o-2024-05-13": true, "gpt-4o-2024-08-06": true,
		"gpt-4o-2024-11-20": true, "gpt-4o-mini": true,
		"o1": true, "o1-pro": true, "o3": true, "o3-mini": true, "o3-pro": true, "o4-mini": true,
	}
	for _, model := range catalog.Models("") {
		if model.API != llm.APIOpenAIResponses {
			continue
		}
		hasJSON := hasCapability(model.Capabilities, llm.CapabilityJSON)
		wantJSON := model.Provider == "openai" && confirmedJSON[model.ID]
		if hasJSON != wantJSON {
			t.Errorf("Responses model %s/%s JSON capability = %v, want %v", model.Provider, model.ID, hasJSON, wantJSON)
		}
	}
	disabledLongCache := map[string]bool{
		"accounts/fireworks/models/glm-5p2":       true,
		"accounts/fireworks/models/kimi-k3":       true,
		"accounts/fireworks/routers/glm-5p2-fast": true,
		"accounts/fireworks/routers/kimi-k3-fast": true,
	}
	for _, model := range catalog.Models("") {
		if model.API != llm.APIOpenAIChatCompletions {
			continue
		}
		if model.Compatibility == nil || model.Compatibility.OpenAIChat == nil {
			t.Errorf("Chat model %s/%s lacks compatibility metadata", model.Provider, model.ID)
			continue
		}
		compatibility := model.Compatibility.OpenAIChat
		wantLong := llm.CompatibilityEnabled
		if model.Provider == ProviderFireworks && disabledLongCache[model.ID] {
			wantLong = llm.CompatibilityDisabled
		}
		if compatibility.LongCacheRetention != wantLong {
			t.Errorf("Chat model %s/%s long cache = %q, want %q", model.Provider, model.ID, compatibility.LongCacheRetention, wantLong)
		}
		wantFallback := hasCapability(model.Capabilities, llm.CapabilityVision)
		if (compatibility.ToolResultImageFallback == llm.CompatibilityEnabled) != wantFallback {
			t.Errorf("Chat model %s/%s image fallback = %q, vision %v", model.Provider, model.ID, compatibility.ToolResultImageFallback, wantFallback)
		}
		if model.Provider == ProviderOpenRouter {
			if compatibility.SessionAffinity != llm.CompatibilityEnabled || compatibility.SessionAffinityFormat != llm.SessionAffinityOpenRouter {
				t.Errorf("OpenRouter model %s session affinity = %q/%q", model.ID, compatibility.SessionAffinity, compatibility.SessionAffinityFormat)
			}
		} else if compatibility.SessionAffinity != llm.CompatibilityDefault || compatibility.SessionAffinityFormat != llm.SessionAffinityDefault {
			t.Errorf("Chat model %s/%s has unexpected session affinity = %q/%q", model.Provider, model.ID, compatibility.SessionAffinity, compatibility.SessionAffinityFormat)
		}
	}
	collection, err := New(ProviderConfig{ID: "openai", API: OpenAIResponses, APIKey: "test"})
	if err != nil {
		t.Fatal(err)
	}
	model, ok := catalog.Model("openai", "gpt-4.1")
	if !ok {
		t.Fatal("gpt-4.1 missing")
	}
	generator, err := collection.GeneratorFor(model)
	if err != nil {
		t.Fatal(err)
	}
	if !hasCapability(generator.Info().Capabilities, llm.CapabilityJSON) {
		t.Fatal("gpt-4.1 Collection generator lacks JSON capability")
	}
}

func hasCapability(capabilities []llm.Capability, target llm.Capability) bool {
	return slices.Contains(capabilities, target)
}

func TestBuiltinCatalogMatchesProviderProfiles(t *testing.T) {
	catalog := BuiltinCatalog()
	for _, profile := range BuiltinProviders() {
		entries := catalog.Models(profile.ID())
		if len(entries) == 0 {
			t.Fatalf("catalog has no models for built-in provider %q", profile.ID())
		}
		collection, err := New(profile.Config("test-key"))
		if err != nil {
			t.Fatalf("New(%s) error = %v", profile.ID(), err)
		}
		for _, model := range entries {
			if model.API != llm.API(profile.API()) {
				t.Errorf("catalog model %s/%s uses API %q, built-in profile uses %q", model.Provider, model.ID, model.API, profile.API())
				continue
			}
			generator, err := collection.GeneratorFor(model)
			if err != nil {
				t.Errorf("GeneratorFor(%s/%s) error = %v", model.Provider, model.ID, err)
				continue
			}
			if info := generator.Info(); info.Provider != model.Provider || info.Model != model.ID {
				t.Errorf("GeneratorFor(%s/%s).Info() = %#v", model.Provider, model.ID, info)
			}
		}
	}
}

func TestBuiltinCatalogReturnsOwnedMetadata(t *testing.T) {
	first, ok := BuiltinCatalog().Model(ProviderDeepSeek, "deepseek-v4-flash")
	if !ok {
		t.Fatal("representative model is missing")
	}
	first.Capabilities[0] = llm.CapabilityAudio
	first.Cost.Input = 999
	first.Compatibility.OpenAIChat.MaxTokensField = llm.MaxTokensFieldCompletion

	second, ok := BuiltinCatalog().Model(ProviderDeepSeek, "deepseek-v4-flash")
	if !ok || second.Capabilities[0] == llm.CapabilityAudio || second.Cost.Input == 999 ||
		second.Compatibility.OpenAIChat.MaxTokensField != llm.MaxTokensFieldLegacy {
		t.Fatalf("BuiltinCatalog() retained caller mutation: %#v", second)
	}
}
