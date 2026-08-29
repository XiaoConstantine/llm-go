package main

import (
	"path/filepath"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/models/modelsdev"
)

func floatPtr(v float64) *float64 { return &v }
func intPtr(v int) *int           { return &v }
func boolPtr(v bool) *bool        { return &v }

func TestSyncCatalogUpdatesLimitsAndCost(t *testing.T) {
	catalog := &sourceCatalog{
		SchemaVersion: 1,
		Revision:      1,
		Models: []sourceModel{
			{
				Provider:        "anthropic",
				ID:              "claude-3-5-sonnet",
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
						Input: []string{"text", "image"},
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

func TestSyncCatalogIsIdempotentOnTiers(t *testing.T) {
	threshold := 272000
	catalog := &sourceCatalog{
		SchemaVersion: 1,
		Revision:      1,
		Models: []sourceModel{
			{
				Provider:        "openai",
				ID:              "gpt-5.5",
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
					ID:        "gpt-5.5",
					Name:      "GPT-5.5",
					Reasoning: true,
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
		SchemaVersion: 1,
		Revision:      5,
		Models: []sourceModel{
			{
				Provider:        "openai",
				ID:              "gpt-4o",
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
						Input: []string{"text", "image"},
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
		SchemaVersion: 1,
		Revision:      1,
		Models:        []sourceModel{},
	}

	dataset := modelsdev.Dataset{
		"google": modelsdev.ProviderEntry{
			ID:   "google",
			Name: "Google",
			Models: map[string]modelsdev.ModelEntry{
				"gemini-2.5-pro": {
					ID:        "gemini-2.5-pro",
					Name:      "Gemini 2.5 Pro",
					Reasoning: true,
					ToolCall:  true,
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

func TestValidateCatalogRejectsInvalidData(t *testing.T) {
	// Missing name
	bad1 := &sourceCatalog{
		SchemaVersion: 1,
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
		SchemaVersion: 1,
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
		SchemaVersion: 1,
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

func TestSyncCatalogLeavesManualCodexRoutesUnchanged(t *testing.T) {
	catalog := &sourceCatalog{
		SchemaVersion: 1,
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

func TestWriteCatalogProducesValidFile(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "catalog.json")
	catalog := &sourceCatalog{
		SchemaVersion: 1,
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
