package models

import (
	"errors"
	"slices"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestCatalogNormalizesAndLooksUpModels(t *testing.T) {
	catalog, err := NewCatalog(
		llm.Model{
			Provider:        " openai ",
			ID:              " gpt-test ",
			API:             llm.APIOpenAIResponses,
			Capabilities:    []llm.Capability{llm.CapabilityTools, llm.CapabilityGeneration, llm.CapabilityTools},
			ContextWindow:   128_000,
			MaxOutputTokens: 16_384,
		},
		llm.Model{
			Provider: "anthropic",
			ID:       "claude-test",
			Name:     " Claude Test ",
			API:      llm.APIAnthropicMessages,
		},
	)
	if err != nil {
		t.Fatalf("NewCatalog() error = %v", err)
	}

	all := catalog.Models("")
	if len(all) != 2 || all[0].Provider != "openai" || all[1].Provider != "anthropic" {
		t.Fatalf("Models() = %#v", all)
	}
	openAI := catalog.Models(" openai ")
	if len(openAI) != 1 {
		t.Fatalf("Models(openai) = %#v", openAI)
	}
	got := openAI[0]
	if got.ID != "gpt-test" || got.Name != "gpt-test" || got.API != llm.APIOpenAIResponses ||
		got.ContextWindow != 128_000 || got.MaxOutputTokens != 16_384 {
		t.Fatalf("Models(openai)[0] = %#v", got)
	}
	if want := []llm.Capability{llm.CapabilityGeneration, llm.CapabilityTools}; !slices.Equal(got.Capabilities, want) {
		t.Fatalf("Capabilities = %v, want %v", got.Capabilities, want)
	}

	lookup, ok := catalog.Model(" anthropic ", " claude-test ")
	if !ok || lookup.Name != "Claude Test" {
		t.Fatalf("Model() = (%#v, %t)", lookup, ok)
	}
	if _, ok := catalog.Model("anthropic", "missing"); ok {
		t.Fatal("Model(missing) found, want false")
	}
}

func TestCatalogOwnsModelStorage(t *testing.T) {
	capabilities := []llm.Capability{llm.CapabilityStreaming}
	tiers := []llm.ModelCostTier{{InputTokensAbove: 1_000, Input: 2}}
	compatibility := &llm.OpenAIResponsesCompatibility{StrictTools: llm.CompatibilityEnabled}
	catalog, err := NewCatalog(llm.Model{
		Provider:      "openai",
		ID:            "gpt-test",
		API:           llm.APIOpenAIResponses,
		Capabilities:  capabilities,
		Cost:          &llm.ModelCost{Input: 1, Tiers: tiers},
		Compatibility: &llm.ModelCompatibility{OpenAIResponses: compatibility},
	})
	if err != nil {
		t.Fatalf("NewCatalog() error = %v", err)
	}
	capabilities[0] = llm.CapabilityAudio
	tiers[0].Input = 99
	compatibility.StrictTools = llm.CompatibilityDisabled

	first, _ := catalog.Model("openai", "gpt-test")
	first.Capabilities[0] = llm.CapabilityAudio
	first.Cost.Input = 99
	first.Cost.Tiers[0].Input = 99
	first.Compatibility.OpenAIResponses.StrictTools = llm.CompatibilityDisabled
	listed := catalog.Models("openai")
	listed[0].Capabilities[0] = llm.CapabilityAudio
	listed[0].Cost.Tiers[0].Input = 99
	listed[0].Compatibility.OpenAIResponses.StrictTools = llm.CompatibilityDisabled

	second, _ := catalog.Model("openai", "gpt-test")
	want := []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming}
	if !slices.Equal(second.Capabilities, want) {
		t.Fatalf("second Capabilities = %v, want %v", second.Capabilities, want)
	}
	if second.Cost == nil || second.Cost.Input != 1 || second.Cost.Tiers[0].Input != 2 {
		t.Fatalf("second Cost = %#v", second.Cost)
	}
	if second.Compatibility == nil || second.Compatibility.OpenAIResponses == nil ||
		second.Compatibility.OpenAIResponses.StrictTools != llm.CompatibilityEnabled {
		t.Fatalf("second Compatibility = %#v", second.Compatibility)
	}
	if listedAgain := catalog.Models("openai"); len(listedAgain) != 1 || !slices.Equal(listedAgain[0].Capabilities, want) || listedAgain[0].Cost.Tiers[0].Input != 2 {
		t.Fatalf("second Models() = %#v, want capabilities %v", listedAgain, want)
	}
}

func TestNewCatalogRejectsInvalidModels(t *testing.T) {
	valid := llm.Model{Provider: "provider", ID: "model", API: "custom-api"}
	tests := []struct {
		name     string
		models   []llm.Model
		provider string
		want     string
	}{
		{name: "empty provider", models: []llm.Model{{ID: "model", API: "api"}}, want: "provider must not be empty"},
		{name: "empty model", models: []llm.Model{{Provider: "provider", API: "api"}}, provider: "provider", want: "model ID must not be empty"},
		{name: "empty API", models: []llm.Model{{Provider: "provider", ID: "model"}}, provider: "provider", want: "API must not be empty"},
		{name: "negative context", models: []llm.Model{{Provider: "provider", ID: "model", API: "api", ContextWindow: -1}}, provider: "provider", want: "context window"},
		{name: "negative output", models: []llm.Model{{Provider: "provider", ID: "model", API: "api", MaxOutputTokens: -1}}, provider: "provider", want: "max output tokens"},
		{name: "output exceeds context", models: []llm.Model{{Provider: "provider", ID: "model", API: "api", ContextWindow: 10, MaxOutputTokens: 11}}, provider: "provider", want: "must not exceed"},
		{name: "unknown capability", models: []llm.Model{{Provider: "provider", ID: "model", API: "api", Capabilities: []llm.Capability{"future"}}}, provider: "provider", want: "is invalid"},
		{name: "invalid cost", models: []llm.Model{{Provider: "provider", ID: "model", API: "api", Cost: &llm.ModelCost{Input: -1}}}, provider: "provider", want: "cost"},
		{name: "compatibility API mismatch", models: []llm.Model{{Provider: "provider", ID: "model", API: llm.APIAnthropicMessages, Compatibility: &llm.ModelCompatibility{OpenAIChat: &llm.OpenAIChatCompatibility{}}}}, provider: "provider", want: "compatibility"},
		{name: "invalid compatibility", models: []llm.Model{{Provider: "provider", ID: "model", API: llm.APIOpenAIChatCompletions, Compatibility: &llm.ModelCompatibility{OpenAIChat: &llm.OpenAIChatCompatibility{ThinkingFormat: "future"}}}}, provider: "provider", want: "thinking format"},
		{name: "duplicate", models: []llm.Model{valid, valid}, provider: "provider", want: "more than once"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			catalog, err := NewCatalog(test.models...)
			if catalog != nil {
				t.Fatalf("NewCatalog() = %#v, want nil", catalog)
			}
			var modelErr *llm.Error
			if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindInvalidRequest ||
				modelErr.Op != "catalog" || modelErr.Provider != test.provider {
				t.Fatalf("NewCatalog() error = %#v", err)
			}
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("NewCatalog() error = %q, want substring %q", err, test.want)
			}
		})
	}
}

func TestNilCatalogReadsAreEmpty(t *testing.T) {
	var catalog *Catalog
	if models := catalog.Models(""); models != nil {
		t.Fatalf("Models() = %#v, want nil", models)
	}
	if model, ok := catalog.Model("provider", "model"); ok || model.Provider != "" || model.ID != "" {
		t.Fatalf("Model() = (%#v, %t), want zero, false", model, ok)
	}
}
