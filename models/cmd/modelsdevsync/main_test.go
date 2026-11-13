package main

import (
	"bytes"
	jsonv2 "encoding/json/v2"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/models/modelsdev"
)

func floatPtr(v float64) *float64 { return &v }
func intPtr(v int) *int           { return &v }
func boolPtr(v bool) *bool        { return &v }

func TestSyncCatalogUpdatesLimitsAndCost(t *testing.T) {
	catalog := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      1,
		Models: []sourceModel{
			{
				Provider:        "anthropic",
				ID:              "claude-3-5-sonnet",
				Source:          "models.dev",
				Name:            "Claude 3.5 Sonnet",
				API:             llm.APIAnthropicMessages,
				Reasoning:       boolPtr(false),
				Capabilities:    &[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
				ContextWindow:   intPtr(100000),
				MaxOutputTokens: intPtr(4096),
				Cost: &sourceCost{
					Input:      floatPtr(3.0),
					Output:     floatPtr(15.0),
					CacheRead:  floatPtr(0.3),
					CacheWrite: floatPtr(3.75),
				},
				Compatibility: &sourceCompatibility{
					Anthropic: &sourceAnthropic{
						StrictTools: boolPtr(true),
					},
				},
			},
		},
	}

	dataset := modelsdev.Dataset{
		"anthropic": modelsdev.ProviderEntry{
			ID:   "anthropic",
			Name: "Anthropic",
			Models: map[string]modelsdev.ModelEntry{
				"claude-3-5-sonnet": {
					ID:        "claude-3-5-sonnet",
					ToolCall:  true,
					Name:      "Claude 3.5 Sonnet",
					Reasoning: true,
					Limit: &modelsdev.Limit{
						Context: 200000,
						Output:  8192,
					},
					Cost: &modelsdev.Cost{
						Input:      floatPtr(3.0),
						Output:     floatPtr(15.0),
						CacheRead:  floatPtr(0.3),
						CacheWrite: floatPtr(3.75),
					},
					Modalities: &modelsdev.Modalities{
						Input:  []string{"text", "image"},
						Output: []string{"text"},
					},
				},
			},
		},
	}

	result, err := syncCatalog(catalog, dataset, SyncOptions{
		AddNew: false,
		DryRun: false,
	})
	if err != nil {
		t.Fatalf("syncCatalog failed: %v", err)
	}

	if len(result.UpdatedModels) != 1 {
		t.Fatalf("len(result.UpdatedModels) = %d, want 1", len(result.UpdatedModels))
	}
	if catalog.Revision != 2 {
		t.Errorf("catalog.Revision = %d, want 2", catalog.Revision)
	}

	m := catalog.Models[0]
	if *m.ContextWindow != 200000 || *m.MaxOutputTokens != 8192 {
		t.Errorf("unexpected limits: context=%d max=%d", *m.ContextWindow, *m.MaxOutputTokens)
	}
	if !*m.Reasoning {
		t.Errorf("expected reasoning=true")
	}
	// Verify vision capability was added
	var hasVision bool
	for _, c := range *m.Capabilities {
		if c == llm.CapabilityVision {
			hasVision = true
			break
		}
	}
	if !hasVision {
		t.Errorf("expected CapabilityVision to be added")
	}

	// Verify compatibility is preserved!
	if m.Compatibility == nil || m.Compatibility.Anthropic == nil || !*m.Compatibility.Anthropic.StrictTools {
		t.Errorf("expected compatibility to be preserved, got %+v", m.Compatibility)
	}
}

func TestApplyModelUpdateKeepsCapabilitiesAndPricingUsable(t *testing.T) {
	chat := sourceModel{
		Provider: "deepseek", ID: "vision", Name: "Vision", API: llm.APIOpenAIChatCompletions,
		Capabilities:  &[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
		Compatibility: &sourceCompatibility{OpenAIChat: &sourceOpenAIChat{}},
	}
	if !applyModelUpdate(&chat, modelsdev.ModelEntry{Modalities: &modelsdev.Modalities{Input: []string{"text", "image"}}}) {
		t.Fatal("vision update reported no change")
	}
	if !slices.Contains(*chat.Capabilities, llm.CapabilityVision) || chat.Compatibility.OpenAIChat.ToolResultImageFallback == nil || !*chat.Compatibility.OpenAIChat.ToolResultImageFallback {
		t.Fatalf("vision model = %#v", chat)
	}

	audio := sourceModel{
		Provider: "openrouter", ID: "audio", Name: "Audio", API: llm.APIOpenAIChatCompletions,
		Capabilities: &[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityAudio},
	}
	if applyModelUpdate(&audio, modelsdev.ModelEntry{Cost: &modelsdev.Cost{Input: floatPtr(1), Output: floatPtr(2)}}) {
		t.Fatal("audio-only pricing changed the model")
	}
	if audio.Cost != nil {
		t.Fatalf("audio cost = %#v, want nil", audio.Cost)
	}
}

func TestSyncCatalogIsIdempotentOnTiers(t *testing.T) {
	threshold := 272000
	catalog := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      1,
		Models: []sourceModel{
			{
				Provider:        "openai",
				ID:              "gpt-5.5",
				Source:          "models.dev",
				Name:            "GPT-5.5",
				API:             llm.APIOpenAIResponses,
				Reasoning:       boolPtr(true),
				Capabilities:    &[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
				ContextWindow:   intPtr(1000000),
				MaxOutputTokens: intPtr(128000),
				Cost: &sourceCost{
					Input:      floatPtr(5.0),
					Output:     floatPtr(30.0),
					CacheRead:  floatPtr(0.5),
					CacheWrite: floatPtr(0),
					Tiers: []sourceTier{
						{
							InputTokensAbove: &threshold,
							Input:            floatPtr(10.0),
							Output:           floatPtr(45.0),
							CacheRead:        floatPtr(1.0),
							CacheWrite:       floatPtr(0),
						},
					},
				},
			},
		},
	}

	dataset := modelsdev.Dataset{
		"openai": modelsdev.ProviderEntry{
			ID:   "openai",
			Name: "OpenAI",
			Models: map[string]modelsdev.ModelEntry{
				"gpt-5.5": {
					ID:         "gpt-5.5",
					ToolCall:   true,
					Modalities: &modelsdev.Modalities{Output: []string{"text"}},
					Name:       "GPT-5.5",
					Reasoning:  true,
					Limit: &modelsdev.Limit{
						Context: 1000000,
						Output:  128000,
					},
					Cost: &modelsdev.Cost{
						Input:      floatPtr(5.0),
						Output:     floatPtr(30.0),
						CacheRead:  floatPtr(0.5),
						CacheWrite: floatPtr(0),
						Tiers: []modelsdev.CostTierEntry{
							{
								Input:      floatPtr(10.0),
								Output:     floatPtr(45.0),
								CacheRead:  floatPtr(1.0),
								CacheWrite: floatPtr(0),
								Tier: &modelsdev.TierCondition{
									Type: "context",
									Size: 272000,
								},
							},
						},
					},
				},
			},
		},
	}

	// First run: already up to date
	result1, err := syncCatalog(catalog, dataset, SyncOptions{})
	if err != nil {
		t.Fatalf("first sync failed: %v", err)
	}
	if len(result1.UpdatedModels) != 0 {
		t.Fatalf("expected 0 updates on identical tiers, got %d: %v", len(result1.UpdatedModels), result1.UpdatedModels)
	}
	if catalog.Revision != 1 {
		t.Errorf("revision changed to %d on identical data", catalog.Revision)
	}

	// Second run: still idempotent
	result2, err := syncCatalog(catalog, dataset, SyncOptions{})
	if err != nil {
		t.Fatalf("second sync failed: %v", err)
	}
	if len(result2.UpdatedModels) != 0 {
		t.Fatalf("expected 0 updates on second run, got %d", len(result2.UpdatedModels))
	}
	if catalog.Revision != 1 {
		t.Errorf("revision changed to %d on second run", catalog.Revision)
	}
}

func TestSyncCatalogNoChangesDoesNotIncrementRevision(t *testing.T) {
	catalog := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      5,
		Models: []sourceModel{
			{
				Provider:        "openai",
				ID:              "gpt-4o",
				Source:          "models.dev",
				Name:            "GPT-4o",
				API:             llm.APIOpenAIResponses,
				Reasoning:       boolPtr(false),
				Capabilities:    &[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision},
				ContextWindow:   intPtr(128000),
				MaxOutputTokens: intPtr(16384),
				Cost: &sourceCost{
					Input:      floatPtr(2.5),
					Output:     floatPtr(10.0),
					CacheRead:  floatPtr(1.25),
					CacheWrite: floatPtr(0),
				},
			},
		},
	}

	dataset := modelsdev.Dataset{
		"openai": modelsdev.ProviderEntry{
			ID:   "openai",
			Name: "OpenAI",
			Models: map[string]modelsdev.ModelEntry{
				"gpt-4o": {
					ID:        "gpt-4o",
					ToolCall:  true,
					Name:      "GPT-4o",
					Reasoning: false,
					Limit: &modelsdev.Limit{
						Context: 128000,
						Output:  16384,
					},
					Cost: &modelsdev.Cost{
						Input:      floatPtr(2.5),
						Output:     floatPtr(10.0),
						CacheRead:  floatPtr(1.25),
						CacheWrite: floatPtr(0),
					},
					Modalities: &modelsdev.Modalities{
						Input:  []string{"text", "image"},
						Output: []string{"text"},
					},
				},
			},
		},
	}

	result, err := syncCatalog(catalog, dataset, SyncOptions{
		AddNew: false,
		DryRun: false,
	})
	if err != nil {
		t.Fatalf("syncCatalog failed: %v", err)
	}

	if len(result.UpdatedModels) != 0 {
		t.Fatalf("expected 0 updated models, got %d", len(result.UpdatedModels))
	}
	if catalog.Revision != 5 {
		t.Errorf("catalog.Revision = %d, want 5", catalog.Revision)
	}
	if result.Unchanged != 1 {
		t.Errorf("result.Unchanged = %d, want 1", result.Unchanged)
	}
}

func TestSyncCatalogAddNewModelWithContextOver200k(t *testing.T) {
	catalog := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      1,
		Models:        []sourceModel{},
	}

	dataset := modelsdev.Dataset{
		"google": modelsdev.ProviderEntry{
			ID:   "google",
			Name: "Google",
			Models: map[string]modelsdev.ModelEntry{
				"gemini-2.5-pro": {
					ID:         "gemini-2.5-pro",
					Name:       "Gemini 2.5 Pro",
					Reasoning:  true,
					ToolCall:   true,
					Modalities: &modelsdev.Modalities{Output: []string{"text"}},
					Limit: &modelsdev.Limit{
						Context: 1000000,
						Output:  8192,
					},
					Cost: &modelsdev.Cost{
						Input:     floatPtr(1.25),
						Output:    floatPtr(10.0),
						CacheRead: floatPtr(0.125),
						ContextOver200k: &modelsdev.CostRates{
							Input:     floatPtr(2.5),
							Output:    floatPtr(15.0),
							CacheRead: floatPtr(0.25),
						},
					},
				},
			},
		},
	}

	result, err := syncCatalog(catalog, dataset, SyncOptions{
		Providers: []string{"google"},
		AddNew:    true,
		DryRun:    false,
	})
	if err != nil {
		t.Fatalf("syncCatalog failed: %v", err)
	}

	if len(result.AddedModels) != 1 {
		t.Fatalf("len(result.AddedModels) = %d, want 1", len(result.AddedModels))
	}
	if catalog.Revision != 2 {
		t.Errorf("catalog.Revision = %d, want 2", catalog.Revision)
	}
	if len(catalog.Models) != 1 {
		t.Fatalf("len(catalog.Models) = %d, want 1", len(catalog.Models))
	}

	added := catalog.Models[0]
	if added.Provider != "google" || added.ID != "gemini-2.5-pro" {
		t.Errorf("unexpected added model: %+v", added)
	}
	if added.API != llm.APIGeminiGenerateContent {
		t.Errorf("added.API = %q, want %q", added.API, llm.APIGeminiGenerateContent)
	}
	if added.Cost == nil || len(added.Cost.Tiers) != 1 {
		t.Fatalf("expected 1 cost tier from context_over_200k, got: %+v", added.Cost)
	}
	tier := added.Cost.Tiers[0]
	if *tier.InputTokensAbove != 200000 || *tier.Input != 2.5 || *tier.Output != 15.0 || *tier.CacheRead != 0.25 || *tier.CacheWrite != 0.0 {
		t.Errorf("unexpected tier rates: %+v", tier)
	}

	if err := validateCatalog(catalog); err != nil {
		t.Fatalf("validateCatalog failed on added model: %v", err)
	}
}

func TestCreateSourceModelNormalizesIdentityAndKeepsProviderCompatibility(t *testing.T) {
	entry := modelsdev.ModelEntry{
		ID:         " deepseek-v4 ",
		Name:       " DeepSeek V4 ",
		ToolCall:   true,
		Reasoning:  true,
		Modalities: &modelsdev.Modalities{Output: []string{"text"}},
		Limit:      &modelsdev.Limit{Context: 1_000_000, Output: 384_000},
	}
	model, ok := createSourceModel("deepseek", entry)
	if !ok {
		t.Fatal("createSourceModel rejected valid model")
	}
	if model.ID != "deepseek-v4" || model.Name != "DeepSeek V4" {
		t.Fatalf("identity = %q/%q", model.ID, model.Name)
	}
	if model.Compatibility == nil || model.Compatibility.OpenAIChat == nil || model.Compatibility.OpenAIChat.ThinkingFormat != llm.ThinkingFormatDeepSeek {
		t.Fatalf("compatibility = %#v", model.Compatibility)
	}
	if err := validateCatalog(&sourceCatalog{SchemaVersion: schemaVersion, Revision: 1, Models: []sourceModel{model}}); err != nil {
		t.Fatalf("validate normalized model: %v", err)
	}

	entry.ID = "router/model"
	entry.Name = "Router Model"
	model, ok = createSourceModel("openrouter", entry)
	if !ok || model.Compatibility == nil || model.Compatibility.OpenAIChat == nil || model.Compatibility.OpenAIChat.ThinkingFormat != llm.ThinkingFormatOpenRouter || model.Compatibility.OpenAIChat.SessionAffinityFormat != llm.SessionAffinityOpenRouter {
		t.Fatalf("OpenRouter compatibility = %#v, ok = %t", model.Compatibility, ok)
	}
}

func TestSyncCatalogFiltersNewModelsAndUsesProviderAPI(t *testing.T) {
	for _, test := range []struct {
		provider string
		api      llm.API
	}{
		{"mistral", llm.APIMistralConversations},
		{"google-vertex", llm.APIGoogleVertex},
		{"amazon-bedrock", llm.APIBedrockConverseStream},
	} {
		t.Run(test.provider, func(t *testing.T) {
			entries := map[string]modelsdev.ModelEntry{}
			for _, candidate := range []struct {
				id     string
				tools  bool
				output []string
			}{
				{"generation", true, []string{"text"}},
				{"embedding", false, []string{"text"}},
				{"image", true, []string{"image"}},
				{"multimodal", true, []string{"text", "image"}},
				{"unknown", true, nil},
				{"existing", false, []string{"image"}},
			} {
				entries[candidate.id] = modelsdev.ModelEntry{ID: candidate.id, ToolCall: candidate.tools,
					Modalities: &modelsdev.Modalities{Output: candidate.output}, Limit: &modelsdev.Limit{Context: 1000, Output: 123}}
			}
			catalog := &sourceCatalog{SchemaVersion: schemaVersion, Revision: 1, Models: []sourceModel{{
				Provider: test.provider, ID: "existing", Name: "Existing", API: test.api,
				Reasoning: boolPtr(false), Capabilities: &[]llm.Capability{llm.CapabilityStreaming},
				ContextWindow: intPtr(500), MaxOutputTokens: intPtr(100),
			}}}
			result, err := syncCatalog(catalog, modelsdev.Dataset{test.provider: {Models: entries}}, SyncOptions{Providers: []string{test.provider}, AddNew: true})
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(result.AddedModels, []string{test.provider + "/generation"}) || len(result.UpdatedModels) != 0 || result.Unchanged != 1 || len(catalog.Models) != 2 {
				t.Fatalf("sync result = %#v, model count = %d", result, len(catalog.Models))
			}
			if !slices.Equal(result.ExcludedModels, []string{test.provider + "/embedding", test.provider + "/image", test.provider + "/multimodal", test.provider + "/unknown"}) {
				t.Fatalf("excluded models = %v", result.ExcludedModels)
			}
			if err := validateCatalog(catalog); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestValidateCatalogRejectsInvalidData(t *testing.T) {
	// Missing name
	bad1 := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      1,
		Models: []sourceModel{
			{
				Provider:        "openai",
				ID:              "m",
				ContextWindow:   intPtr(100),
				MaxOutputTokens: intPtr(50),
			},
		},
	}
	if err := validateCatalog(bad1); err == nil {
		t.Error("expected error for missing name and API")
	}

	// Output > context
	bad2 := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      1,
		Models: []sourceModel{
			{
				Provider:        "openai",
				ID:              "m",
				Name:            "M",
				API:             llm.APIOpenAIResponses,
				ContextWindow:   intPtr(100),
				MaxOutputTokens: intPtr(200),
			},
		},
	}
	if err := validateCatalog(bad2); err == nil {
		t.Error("expected error for max output > context window")
	}

	// Tier above context window
	threshold := 500
	bad3 := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      1,
		Models: []sourceModel{
			{
				Provider:        "openai",
				ID:              "m",
				Name:            "M",
				API:             llm.APIOpenAIResponses,
				ContextWindow:   intPtr(100),
				MaxOutputTokens: intPtr(50),
				Cost: &sourceCost{
					Input:      floatPtr(1),
					Output:     floatPtr(1),
					CacheRead:  floatPtr(0),
					CacheWrite: floatPtr(0),
					Tiers: []sourceTier{
						{
							InputTokensAbove: &threshold,
							Input:            floatPtr(2),
							Output:           floatPtr(2),
							CacheRead:        floatPtr(0),
							CacheWrite:       floatPtr(0),
						},
					},
				},
			},
		},
	}
	if err := validateCatalog(bad3); err == nil {
		t.Error("expected error for tier threshold exceeding context window")
	}
}

func TestSyncCatalogSkipsInvalidCacheReadPrices(t *testing.T) {
	for _, cost := range []*modelsdev.Cost{
		{Input: floatPtr(.1), CacheRead: floatPtr(.3)}, // Explicit invalid rates are not clamped.
		{Input: floatPtr(5), Tiers: []modelsdev.CostTierEntry{{InputTokensAbove: intPtr(200), Input: floatPtr(.1), CacheRead: floatPtr(.2)}}},
		// A discarded cost must not report its provisional base-price clamp.
		{Input: floatPtr(.1), Tiers: []modelsdev.CostTierEntry{{InputTokensAbove: intPtr(200), Input: floatPtr(.1), CacheRead: floatPtr(.2)}}},
	} {
		catalog := &sourceCatalog{SchemaVersion: schemaVersion, Revision: 1, Models: []sourceModel{{
			Provider: "openrouter", ID: "existing", Source: modelsDevSource, Name: "Existing", API: llm.APIOpenAIChatCompletions,
			Reasoning: boolPtr(false), Capabilities: &[]llm.Capability{llm.CapabilityStreaming},
			ContextWindow: intPtr(1000), MaxOutputTokens: intPtr(100),
			Cost: &sourceCost{Input: floatPtr(3), Output: floatPtr(7), CacheRead: floatPtr(.3), CacheWrite: floatPtr(4)},
		}}}
		before := catalog.Models[0]
		beforeCost := *before.Cost
		before.Cost = &beforeCost
		dataset := modelsdev.Dataset{"openrouter": {Models: map[string]modelsdev.ModelEntry{
			"existing": {ID: "existing", ToolCall: true, Limit: &modelsdev.Limit{Context: 2000, Output: 200}, Reasoning: true, Modalities: &modelsdev.Modalities{Input: []string{"text", "image"}, Output: []string{"text"}}, Cost: cost},
			"bad-new":  {ID: "bad-new", ToolCall: true, Modalities: &modelsdev.Modalities{Output: []string{"text"}}, Limit: &modelsdev.Limit{Context: 1000, Output: 100}, Cost: &modelsdev.Cost{Input: floatPtr(.1), CacheRead: floatPtr(.2)}},
			"good-new": {ID: "good-new", ToolCall: true, Modalities: &modelsdev.Modalities{Output: []string{"text"}}, Limit: &modelsdev.Limit{Context: 1000, Output: 100}, Cost: &modelsdev.Cost{Input: floatPtr(.2), CacheRead: floatPtr(.2)}},
		}}}
		result, err := syncCatalog(catalog, dataset, SyncOptions{Providers: []string{"openrouter"}, AddNew: true})
		if err != nil {
			t.Fatal(err)
		}
		got := catalog.Models[0]
		if !slices.Equal(result.UpdatedModels, []string{"openrouter/existing"}) || !slices.Equal(result.AddedModels, []string{"openrouter/good-new"}) || !reflect.DeepEqual(got.Cost, before.Cost) || *got.ContextWindow != 2000 || *got.MaxOutputTokens != 200 || !*got.Reasoning || !slices.Contains(*got.Capabilities, llm.CapabilityVision) {
			t.Fatalf("cost skip lost metadata or applied invalid prices: result=%#v, model=%#v", result, got)
		}
		if len(result.ClampedModels) != 0 {
			t.Fatalf("discarded prices reported as clamped: %v", result.ClampedModels)
		}
		if len(result.SkippedModels) != 2 || !strings.Contains(result.SkippedModels[0], "openrouter/bad-new:") || !strings.Contains(result.SkippedModels[1], "openrouter/existing: cost skipped:") || !strings.Contains(result.SkippedModels[1], "cache_read") {
			t.Fatalf("skipped models = %v", result.SkippedModels)
		}
		if err := validateCatalog(catalog); err != nil {
			t.Fatal(err)
		}
		// The written output must remain acceptable on the next sync as well.
		path := filepath.Join(t.TempDir(), "catalog.json")
		if err := writeCatalog(path, catalog); err != nil {
			t.Fatal(err)
		}
		if _, err := loadCatalog(path); err != nil {
			t.Fatal(err)
		}
		// Hand-edited catalogs must obey the same rule, for base and tier rates.
		catalog.Models[0].Cost.CacheRead = floatPtr(4)
		if cost.Tiers != nil {
			catalog.Models[0].Cost.CacheRead = floatPtr(.3)
			catalog.Models[0].Cost.Tiers = []sourceTier{{InputTokensAbove: intPtr(200), Input: floatPtr(.1), Output: floatPtr(1), CacheRead: floatPtr(.2), CacheWrite: floatPtr(0)}}
		}
		if err := validateCatalog(catalog); err == nil || !strings.Contains(err.Error(), "cache_read") {
			t.Fatalf("invalid cache price validation = %v", err)
		}
	}
}

func TestSyncCatalogRecordedPartialCost(t *testing.T) {
	catalog, err := loadCatalog(filepath.Join("..", "..", "catalogsource", "catalog.json"))
	if err != nil {
		t.Fatal(err)
	}
	// Explicitly opt this fixture model into synchronization; the canonical
	// reviewed snapshot remains curated.
	for i := range catalog.Models {
		if catalog.Models[i].Provider == "openrouter" && catalog.Models[i].ID == "meta-llama/llama-3.1-8b-instruct" {
			catalog.Models[i].Source = modelsDevSource
		}
	}
	dataset, err := loadDataset(filepath.Join("..", "..", "modelsdev", "testdata", "partial-cost.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	for run := range 2 {
		result, err := syncCatalog(catalog, dataset, SyncOptions{Providers: []string{"openrouter"}})
		if err != nil {
			t.Fatal(err)
		}
		if len(result.SkippedModels) != 0 {
			t.Fatalf("skipped prices: %v", result.SkippedModels)
		}
		if slices.Contains(result.UpdatedModels, "openrouter/meta-llama/llama-3.1-8b-instruct") != (run == 0) {
			t.Fatalf("updates on run %d = %v", run, result.UpdatedModels)
		}
		var wantClamped []string
		if run == 0 {
			wantClamped = []string{"openrouter/meta-llama/llama-3.1-8b-instruct"}
		}
		if !slices.Equal(result.ClampedModels, wantClamped) {
			t.Fatalf("clamped models on run %d = %v, want %v", run, result.ClampedModels, wantClamped)
		}
		for _, model := range catalog.Models {
			if model.Provider == "openrouter" && model.ID == "meta-llama/llama-3.1-8b-instruct" {
				if model.Cost == nil || *model.Cost.Input != .02 || *model.Cost.Output != .04 || *model.Cost.CacheRead != .02 {
					t.Fatalf("merged cost = %#v", model.Cost)
				}
			}
		}
		if err := validateCatalog(catalog); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSyncCatalogRetainsTierRates(t *testing.T) {
	for _, test := range []struct {
		name string
		cost *modelsdev.Cost
		want llm.ModelCostTier
	}{
		{"omitted", &modelsdev.Cost{Input: floatPtr(5)}, llm.ModelCostTier{InputTokensAbove: 200000, Input: 6, Output: 14, CacheRead: .6, CacheWrite: 8}},
		{"moved", &modelsdev.Cost{Tiers: []modelsdev.CostTierEntry{{InputTokensAbove: intPtr(300000), Output: floatPtr(19)}}}, llm.ModelCostTier{InputTokensAbove: 300000, Input: 3, Output: 19, CacheRead: .3, CacheWrite: 4}},
		{"matching", &modelsdev.Cost{Tiers: []modelsdev.CostTierEntry{{InputTokensAbove: intPtr(200000), Output: floatPtr(19)}}}, llm.ModelCostTier{InputTokensAbove: 200000, Input: 6, Output: 19, CacheRead: .6, CacheWrite: 8}},
		{"lower input", &modelsdev.Cost{Tiers: []modelsdev.CostTierEntry{{InputTokensAbove: intPtr(200000), Input: floatPtr(.2)}}}, llm.ModelCostTier{InputTokensAbove: 200000, Input: .2, Output: 14, CacheRead: .2, CacheWrite: 8}},
		{"legacy", &modelsdev.Cost{ContextOver200k: &modelsdev.CostRates{Output: floatPtr(19)}}, llm.ModelCostTier{InputTokensAbove: 200000, Input: 6, Output: 19, CacheRead: .6, CacheWrite: 8}},
		{"filtered", &modelsdev.Cost{Tiers: []modelsdev.CostTierEntry{{InputTokensAbove: intPtr(1000000), Input: floatPtr(20)}}}, llm.ModelCostTier{InputTokensAbove: 200000, Input: 6, Output: 14, CacheRead: .6, CacheWrite: 8}},
	} {
		t.Run(test.name, func(t *testing.T) {
			catalog := &sourceCatalog{SchemaVersion: schemaVersion, Revision: 1, Models: []sourceModel{{Provider: "openrouter", ID: "m", Source: modelsDevSource, Name: "M", API: llm.APIOpenAIChatCompletions,
				Reasoning: boolPtr(false), Capabilities: &[]llm.Capability{llm.CapabilityStreaming}, ContextWindow: intPtr(1000000), MaxOutputTokens: intPtr(1000),
				Cost: &sourceCost{Input: floatPtr(3), Output: floatPtr(7), CacheRead: floatPtr(.3), CacheWrite: floatPtr(4), Tiers: []sourceTier{
					{InputTokensAbove: intPtr(200000), Input: floatPtr(6), Output: floatPtr(14), CacheRead: floatPtr(.6), CacheWrite: floatPtr(8)},
					{InputTokensAbove: intPtr(500000), Input: floatPtr(8), Output: floatPtr(20), CacheRead: floatPtr(.8), CacheWrite: floatPtr(10)},
				}},
			}}}
			dataset := modelsdev.Dataset{"openrouter": {Models: map[string]modelsdev.ModelEntry{"m": {
				ID: "m", ToolCall: true, Modalities: &modelsdev.Modalities{Output: []string{"text"}},
				Limit: &modelsdev.Limit{Context: 1000000, Output: 1000}, Cost: test.cost,
			}}}}
			for run := range 2 {
				result, err := syncCatalog(catalog, dataset, SyncOptions{Providers: []string{"openrouter"}})
				if err != nil || len(result.SkippedModels) > 0 {
					t.Fatalf("sync = %#v, %v", result, err)
				}
				if run == 1 && len(result.UpdatedModels) > 0 {
					t.Fatalf("non-idempotent tier update: %v", result.UpdatedModels)
				}
				got := map[int]llm.ModelCostTier{}
				for _, tier := range catalog.Models[0].Cost.Tiers {
					got[*tier.InputTokensAbove] = llm.ModelCostTier{InputTokensAbove: *tier.InputTokensAbove, Input: *tier.Input, Output: *tier.Output, CacheRead: *tier.CacheRead, CacheWrite: *tier.CacheWrite}
				}
				want := map[int]llm.ModelCostTier{test.want.InputTokensAbove: test.want}
				if test.name == "filtered" || test.name == "omitted" {
					want[500000] = llm.ModelCostTier{InputTokensAbove: 500000, Input: 8, Output: 20, CacheRead: .8, CacheWrite: 10}
				}
				var wantClamped []string
				if run == 0 && test.name == "lower input" {
					wantClamped = []string{"openrouter/m"}
				}
				if !slices.Equal(result.ClampedModels, wantClamped) {
					t.Fatalf("clamped models = %v, want %v", result.ClampedModels, wantClamped)
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("tiers = %#v, want %#v", got, want)
				}
				if err := validateCatalog(catalog); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

func TestSyncCatalogDropsTiersOutsideMergedContext(t *testing.T) {
	for _, window := range []int{200000, 272000, 272001} {
		for _, variant := range []string{"no cost", "invalid price", "retained upstream", "omitted upstream"} {
			t.Run(fmt.Sprintf("%s/%d", variant, window), func(t *testing.T) {
				catalog, err := loadCatalog(filepath.Join("..", "..", "catalogsource", "catalog.json"))
				if err != nil {
					t.Fatal(err)
				}
				entry := modelsdev.ModelEntry{ID: "openai/gpt-5.6-sol", ToolCall: true,
					Modalities: &modelsdev.Modalities{Output: []string{"text"}},
					Limit:      &modelsdev.Limit{Context: window, Output: 128000}}
				for i := range catalog.Models {
					if catalog.Models[i].Provider == "openrouter" && catalog.Models[i].ID == entry.ID {
						catalog.Models[i].Source = modelsDevSource
					}
				}
				switch variant {
				case "invalid price":
					entry.Cost = &modelsdev.Cost{Input: floatPtr(.1), CacheRead: floatPtr(.3)}
				case "retained upstream":
					entry.Cost = &modelsdev.Cost{Tiers: []modelsdev.CostTierEntry{{InputTokensAbove: intPtr(272000), Output: floatPtr(20)}}}
				case "omitted upstream":
					entry.Cost = &modelsdev.Cost{Input: floatPtr(3)}
				}
				dataset := modelsdev.Dataset{"openrouter": {Models: map[string]modelsdev.ModelEntry{entry.ID: entry}}}
				for run := range 2 {
					result, err := syncCatalog(catalog, dataset, SyncOptions{Providers: []string{"openrouter"}})
					if err != nil {
						t.Fatal(err)
					}
					if err := validateCatalog(catalog); err != nil {
						t.Fatal(err)
					}
					if run == 1 && len(result.UpdatedModels) > 0 {
						t.Fatalf("second sync changed models: %v", result.UpdatedModels)
					}
					for _, model := range catalog.Models {
						if model.Provider != "openrouter" || model.ID != entry.ID {
							continue
						}
						wantTiers := 0
						if window > 272000 {
							wantTiers = 1
						}
						if *model.ContextWindow != window || len(model.Cost.Tiers) != wantTiers {
							t.Fatalf("context=%d tiers=%d, want %d/%d", *model.ContextWindow, len(model.Cost.Tiers), window, wantTiers)
						}
						if variant == "invalid price" && (*model.Cost.Input != 2 || *model.Cost.CacheRead != .2 || len(result.SkippedModels) != 1) {
							t.Fatalf("invalid base cost not skipped: %+v, %v", model.Cost, result.SkippedModels)
						}
					}
				}
			})
		}
	}
}

func TestSyncCatalogLeavesManualCodexRoutesUnchanged(t *testing.T) {
	catalog := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      1,
		Models: []sourceModel{{
			Provider:        "openai-codex",
			ID:              "gpt-5.5",
			Name:            "GPT-5.5",
			API:             llm.APIOpenAICodexResponses,
			Reasoning:       boolPtr(true),
			Capabilities:    &[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision},
			ContextWindow:   intPtr(272_000),
			MaxOutputTokens: intPtr(128_000),
		}},
	}
	dataset := modelsdev.Dataset{
		"openai": {
			Models: map[string]modelsdev.ModelEntry{
				"gpt-5.5": {ID: "gpt-5.5", Limit: &modelsdev.Limit{Context: 1_050_000, Output: 128_000}},
			},
		},
	}

	result, err := syncCatalog(catalog, dataset, SyncOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.UpdatedModels) != 0 || result.Unchanged != 1 || catalog.Revision != 1 {
		t.Fatalf("sync result = %#v, revision %d", result, catalog.Revision)
	}
	if got := *catalog.Models[0].ContextWindow; got != 272_000 {
		t.Fatalf("Codex context window = %d, want 272000", got)
	}
}

func syncTestEntry(id string) modelsdev.ModelEntry {
	return modelsdev.ModelEntry{
		ID: id, Name: id, ToolCall: true,
		Modalities: &modelsdev.Modalities{Output: []string{"text"}},
		Limit:      &modelsdev.Limit{Context: 1000, Output: 100},
	}
}

func syncTestModel(t *testing.T, provider, id string) sourceModel {
	t.Helper()
	model, ok := createSourceModel(provider, syncTestEntry(id))
	if !ok {
		t.Fatal("createSourceModel rejected test entry")
	}
	model.Source = "models.dev"
	return model
}

func TestSyncCatalogReconcilesManagedMembership(t *testing.T) {
	for _, reason := range []string{"missing", "deprecated", "no tools", "non-text output", "missing modalities", "invalid limits"} {
		t.Run(reason, func(t *testing.T) {
			managed := syncTestModel(t, "openai", "old")
			curated := syncTestModel(t, "openai", "curated")
			curated.Source = ""
			otherRoute := syncTestModel(t, "openrouter", "old")
			catalog := &sourceCatalog{SchemaVersion: 2, Revision: 7, Models: []sourceModel{managed, curated, otherRoute}}
			entry := syncTestEntry("old")
			switch reason {
			case "deprecated":
				entry.Status = "deprecated"
			case "no tools":
				entry.ToolCall = false
			case "non-text output":
				entry.Modalities.Output = []string{"text", "image"}
			case "missing modalities":
				entry.Modalities = nil
			case "invalid limits":
				entry.Limit.Output = entry.Limit.Context + 1
			}
			entries := map[string]modelsdev.ModelEntry{"old": entry}
			if reason == "missing" {
				delete(entries, "old")
			} else {
				// Even a matching deprecated/ineligible upstream entry must not
				// alter a curated entry or claim ownership of it.
				entry.ID, entry.Name = "curated", "Upstream name"
				entry.Limit = &modelsdev.Limit{Context: 500, Output: 50}
				entries["curated"] = entry
			}
			dataset := modelsdev.Dataset{"openai": {Models: entries}}
			// AddNew must not re-add a just-retired entry. Removals also apply
			// without AddNew, including a removal-only revision bump.
			for _, addNew := range []bool{false, true} {
				candidate := *catalog
				candidate.Models = slices.Clone(catalog.Models)
				result, err := syncCatalog(&candidate, dataset, SyncOptions{Providers: []string{"openai"}, AddNew: addNew})
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(candidate.Models, []sourceModel{curated, otherRoute}) {
					t.Fatalf("reconciled models = %#v, want only unchanged curated and other route", candidate.Models)
				}
				if !slices.Equal(result.RemovedModels, []string{"openai/old"}) || len(result.AddedModels) != 0 || len(result.UpdatedModels) != 0 || result.Total != 2 || result.Unchanged != 2 || candidate.Revision != 8 {
					t.Fatalf("result = %#v, revision = %d", result, candidate.Revision)
				}
			}
		})
	}
}

func TestSyncCatalogRejectsIncompleteSnapshotsBeforeMutation(t *testing.T) {
	for _, dataset := range []modelsdev.Dataset{
		nil,
		{},
		{"openai": {}},
		{"openai": {Models: map[string]modelsdev.ModelEntry{"wrong-key": syncTestEntry("old")}}},
	} {
		catalog := &sourceCatalog{SchemaVersion: 2, Revision: 1, Models: []sourceModel{syncTestModel(t, "openai", "old")}}
		before, _ := jsonv2.Marshal(catalog)
		if _, err := syncCatalog(catalog, dataset, SyncOptions{}); err == nil {
			t.Errorf("accepted incomplete or inconsistent snapshot: %#v", dataset)
		}
		after, _ := jsonv2.Marshal(catalog)
		if !bytes.Equal(before, after) {
			t.Fatal("invalid snapshot mutated catalog")
		}
	}
}

func TestSyncCatalogMembershipIsDeterministicAndPersistent(t *testing.T) {
	var want []byte
	var wantResult SyncResult
	for run := range 10 {
		catalog := &sourceCatalog{SchemaVersion: 2, Revision: 1, Models: []sourceModel{
			syncTestModel(t, "openai", "removed-z"), syncTestModel(t, "openai", "removed-a"),
		}}
		dataset := modelsdev.Dataset{"openai": {Models: map[string]modelsdev.ModelEntry{
			"new-z": syncTestEntry("new-z"), "new-a": syncTestEntry("new-a"),
		}}}
		result, err := syncCatalog(catalog, dataset, SyncOptions{Providers: []string{"openai"}, AddNew: true})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(result.AddedModels, []string{"openai/new-a", "openai/new-z"}) || !slices.Equal(result.RemovedModels, []string{"openai/removed-a", "openai/removed-z"}) {
			t.Fatalf("unsorted or incorrect report: %#v", result)
		}
		path := filepath.Join(t.TempDir(), "catalog.json")
		if err := writeCatalog(path, catalog); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if run == 0 {
			want, wantResult = data, result
		} else if !bytes.Equal(data, want) || !reflect.DeepEqual(result, wantResult) {
			t.Fatal("catalog or report depends on map/input order")
		}
		loaded, err := loadCatalog(path)
		if err != nil {
			t.Fatal(err)
		}
		for _, model := range loaded.Models {
			if model.Source != "models.dev" {
				t.Fatalf("new model lost ownership: %#v", model)
			}
		}
		if _, err := syncCatalog(loaded, dataset, SyncOptions{AddNew: true}); err != nil {
			t.Fatal(err)
		}
		if err := writeCatalog(path, loaded); err != nil {
			t.Fatal(err)
		}
		again, _ := os.ReadFile(path)
		if !bytes.Equal(data, again) {
			t.Fatal("repeat sync changed catalog or revision")
		}
		// Explicitly empty models is authoritative; absent/null models is not.
		dataset["openai"] = modelsdev.ProviderEntry{Models: map[string]modelsdev.ModelEntry{}}
		removed, err := syncCatalog(loaded, dataset, SyncOptions{})
		if err != nil || len(loaded.Models) != 0 || len(removed.RemovedModels) != 2 || loaded.Revision != 3 {
			t.Fatalf("empty desired set: result %#v, catalog %#v, err %v", removed, loaded, err)
		}
	}
}

func TestSyncCatalogDryRunReportsRetirement(t *testing.T) {
	catalog := &sourceCatalog{SchemaVersion: 2, Revision: 1, Models: []sourceModel{syncTestModel(t, "openai", "old")}}
	result, err := syncCatalog(catalog, modelsdev.Dataset{"openai": {Models: map[string]modelsdev.ModelEntry{}}}, SyncOptions{DryRun: true})
	if err != nil || !slices.Equal(result.RemovedModels, []string{"openai/old"}) || result.Total != 0 || catalog.Revision != 1 {
		t.Fatalf("dry run = %#v, revision = %d, err = %v", result, catalog.Revision, err)
	}
}

func TestSyncCatalogEligibilityAndProviderMapping(t *testing.T) {
	entries := make(map[string]modelsdev.ModelEntry)
	for _, status := range []string{"alpha", "beta", "deprecated"} {
		entry := syncTestEntry(status)
		entry.Status = status
		entry.ReleaseDate = "2020-01-01" // Age is not a retirement signal.
		entries[status] = entry
	}
	noTools := syncTestEntry("no-tools")
	noTools.ToolCall = false
	entries[noTools.ID] = noTools
	image := syncTestEntry("image")
	image.Modalities.Output = []string{"image"}
	entries[image.ID] = image
	catalog := &sourceCatalog{SchemaVersion: schemaVersion, Revision: 1}
	result, err := syncCatalog(catalog, modelsdev.Dataset{"fireworks-ai": {Models: entries}}, SyncOptions{Providers: []string{"fireworks"}, AddNew: true})
	if err != nil || !slices.Equal(result.AddedModels, []string{"fireworks/alpha", "fireworks/beta"}) {
		t.Fatalf("eligible discoveries = %#v, err = %v", result, err)
	}
	delete(entries, "alpha")
	result, err = syncCatalog(catalog, modelsdev.Dataset{"fireworks-ai": {Models: entries}}, SyncOptions{Providers: []string{"fireworks"}})
	if err != nil || !slices.Equal(result.RemovedModels, []string{"fireworks/alpha"}) || len(catalog.Models) != 1 || catalog.Models[0].ID != "beta" {
		t.Fatalf("mapped route retirement = %#v, err = %v", result, err)
	}
	if _, err := syncCatalog(catalog, modelsdev.Dataset{"openai-codex": {Models: entries}}, SyncOptions{Providers: []string{"openai-codex"}, AddNew: true}); err == nil {
		t.Fatal("Codex accepted models.dev synchronization")
	}
}

func TestSyncCommandWritesRemovalOnlyAndHonorsDryRun(t *testing.T) {
	dir := t.TempDir()
	catalogPath := filepath.Join(dir, "catalog.json")
	datasetPath := filepath.Join(dir, "modelsdev.json")
	catalog := &sourceCatalog{SchemaVersion: schemaVersion, Revision: 1, Models: []sourceModel{syncTestModel(t, "openai", "old")}}
	if err := writeCatalog(catalogPath, catalog); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(datasetPath, []byte(`{"openai":{"models":{}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(catalogPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, dryRun := range []bool{true, false} {
		args := []string{"run", ".", "-catalog", catalogPath, "-file", datasetPath, "-providers", "openai"}
		if dryRun {
			args = append(args, "-dry-run")
		}
		if output, err := exec.CommandContext(t.Context(), "go", args...).CombinedOutput(); err != nil {
			t.Fatalf("sync command: %v\n%s", err, output)
		}
		if dryRun {
			after, err := os.ReadFile(catalogPath)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatalf("dry run changed source file: %v", err)
			}
		} else {
			loaded, err := loadCatalog(catalogPath)
			if err != nil {
				t.Fatal(err)
			}
			if len(loaded.Models) != 0 || loaded.Revision != 2 {
				t.Fatalf("removal-only sync was not persisted: %#v", loaded)
			}
		}
	}
}

func TestValidateCatalogRejectsInvalidOwnership(t *testing.T) {
	model := syncTestModel(t, "openai", "test")
	model.Source = "unknown"
	catalog := &sourceCatalog{SchemaVersion: schemaVersion, Revision: 1, Models: []sourceModel{model}}
	if err := validateCatalog(catalog); err == nil {
		t.Fatal("accepted unknown source")
	}
	catalog.Models[0] = syncTestModel(t, "openai-codex", "test")
	if err := validateCatalog(catalog); err == nil {
		t.Fatal("accepted managed Codex entry")
	}
}

func TestLoadDatasetPreservesLifecycleStatus(t *testing.T) {
	path := filepath.Join(t.TempDir(), "modelsdev.json")
	if err := os.WriteFile(path, []byte(`{"openai":{"models":{"old":{"id":"old","status":"deprecated"}}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	dataset, err := loadDataset(path, "")
	if err != nil || dataset["openai"].Models["old"].Status != "deprecated" {
		t.Fatalf("dataset = %#v, err = %v", dataset, err)
	}
}

func TestLoadCanonicalCatalog(t *testing.T) {
	catalog, err := loadCatalog(filepath.Join("..", "..", "catalogsource", "catalog.json"))
	if err != nil {
		t.Fatalf("load canonical catalog: %v", err)
	}
	for _, model := range catalog.Models {
		if model.Provider == "google" && model.ID == "gemini-3.8-flash" {
			if model.Compatibility == nil || model.Compatibility.Gemini == nil ||
				model.Compatibility.Gemini.ThinkingLevelsOnly == nil || !*model.Compatibility.Gemini.ThinkingLevelsOnly {
				t.Fatalf("Gemini 3.8 compatibility = %#v", model.Compatibility)
			}
			return
		}
	}
	t.Fatal("google/gemini-3.8-flash is missing")
}

func TestWriteCatalogProducesValidFile(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "catalog.json")
	catalog := &sourceCatalog{
		SchemaVersion: schemaVersion,
		Revision:      1,
		Models: []sourceModel{
			{
				Provider:        "openai",
				ID:              "gpt-4o",
				Name:            "GPT-4o",
				API:             llm.APIOpenAIResponses,
				Reasoning:       boolPtr(false),
				Capabilities:    &[]llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
				ContextWindow:   intPtr(128000),
				MaxOutputTokens: intPtr(16384),
			},
		},
	}

	if err := writeCatalog(tmpFile, catalog); err != nil {
		t.Fatalf("writeCatalog failed: %v", err)
	}

	loaded, err := loadCatalog(tmpFile)
	if err != nil {
		t.Fatalf("loadCatalog failed: %v", err)
	}
	if loaded.Revision != 1 || len(loaded.Models) != 1 || loaded.Models[0].ID != "gpt-4o" {
		t.Errorf("unexpected loaded catalog: %+v", loaded)
	}
}
