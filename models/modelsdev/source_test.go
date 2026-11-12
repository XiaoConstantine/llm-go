package modelsdev

import (
	"context"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/models"
)

const sampleDataset = `{
  "anthropic": {
    "id": "anthropic",
    "name": "Anthropic",
    "models": {
      "claude-3-7-sonnet": {
        "id": "claude-3-7-sonnet",
        "name": "Claude 3.7 Sonnet",
        "reasoning": true,
        "tool_call": true,
        "structured_output": true,
        "modalities": {
          "input": ["text", "image"],
          "output": ["text"]
        },
        "limit": {
          "context": 200000,
          "output": 64000
        },
        "cost": {
          "input": 3.0,
          "output": 15.0,
          "cache_read": 0.3,
          "cache_write": 3.75
        }
      }
    }
  },
  "fireworks-ai": {
    "id": "fireworks-ai",
    "name": "Fireworks AI",
    "models": {
      "accounts/fireworks/models/deepseek-v3": {
        "id": "accounts/fireworks/models/deepseek-v3",
        "name": "DeepSeek V3",
        "reasoning": false,
        "tool_call": true,
        "limit": {
          "context": 128000,
          "output": 8192
        },
        "cost": {
          "input": 0.9,
          "output": 0.9
        }
      }
    }
  }
}`

func TestSourceFetchSuccess(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"etag-123"`)
		w.Header().Set("Last-Modified", "Fri, 28 Aug 2026 12:00:00 GMT")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sampleDataset))
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	resp, err := source.Fetch(context.Background(), models.CatalogFetchRequest{
		Provider: "anthropic",
	})
	if err != nil {
		t.Fatalf("Fetch() error = %v", err)
	}

	if resp.NotModified {
		t.Fatal("expected NotModified=false")
	}
	if resp.ETag != `"etag-123"` {
		t.Errorf("resp.ETag = %q, want %q", resp.ETag, `"etag-123"`)
	}
	if len(resp.Models) != 1 {
		t.Fatalf("len(resp.Models) = %d, want 1", len(resp.Models))
	}

	model := resp.Models[0]
	if model.ID != "claude-3-7-sonnet" || model.Provider != "anthropic" {
		t.Errorf("unexpected model identity: %+v", model)
	}
	if model.API != llm.APIAnthropicMessages {
		t.Errorf("model.API = %q, want %q", model.API, llm.APIAnthropicMessages)
	}
	if model.ContextWindow != 200000 || model.MaxOutputTokens != 64000 {
		t.Errorf("model limits = context:%d, max:%d", model.ContextWindow, model.MaxOutputTokens)
	}
	if model.Cost == nil || model.Cost.Input != 3.0 || model.Cost.Output != 15.0 {
		t.Errorf("model cost = %+v", model.Cost)
	}
}

func TestSourceProviderAlias(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"etag-fw"`)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sampleDataset))
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	// "fireworks" should resolve to "fireworks-ai"
	resp, err := source.Fetch(context.Background(), models.CatalogFetchRequest{
		Provider: "fireworks",
	})
	if err != nil {
		t.Fatalf("Fetch() error = %v", err)
	}
	if len(resp.Models) != 1 {
		t.Fatalf("len(resp.Models) = %d, want 1", len(resp.Models))
	}
	if resp.Models[0].Provider != "fireworks" {
		t.Errorf("model provider = %q, want fireworks", resp.Models[0].Provider)
	}
}

func TestSourceNotModified(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("If-None-Match") == `"etag-123"` {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		w.Header().Set("ETag", `"etag-123"`)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sampleDataset))
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	resp, err := source.Fetch(context.Background(), models.CatalogFetchRequest{
		Provider: "anthropic",
		ETag:     `"etag-123"`,
	})
	if err != nil {
		t.Fatalf("Fetch() error = %v", err)
	}
	if !resp.NotModified {
		t.Fatal("expected NotModified=true")
	}
	if len(resp.Models) != 0 {
		t.Fatalf("expected 0 models on NotModified, got %d", len(resp.Models))
	}
}

func TestSourceLastModified304PreservesPublishedModels(t *testing.T) {
	lastModTime := time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("If-Modified-Since") != "" {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		w.Header().Set("Last-Modified", lastModTime.Format(http.TimeFormat))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sampleDataset))
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	// Pre-populate store with a snapshot containing only LastModified (no ETag)
	cachedModel := llm.Model{
		Provider:        "anthropic",
		ID:              "claude-3-7-sonnet",
		Name:            "Claude 3.7 Sonnet",
		API:             llm.APIAnthropicMessages,
		Capabilities:    []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
		ContextWindow:   200000,
		MaxOutputTokens: 64000,
	}
	store, err := models.NewMemoryCatalogStore(map[string]models.CatalogStoreEntry{
		"anthropic": {
			Models:       []llm.Model{cachedModel},
			LastModified: lastModTime,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	manager, err := models.NewCatalogManager(models.CatalogManagerConfig{
		Store: store,
		Providers: []models.CatalogProvider{
			{Provider: "anthropic", Source: source},
		},
	})
	if err != nil {
		t.Fatalf("NewCatalogManager error = %v", err)
	}

	// Refresh should conditionally validate and preserve restored models
	result := manager.Refresh(context.Background(), models.CatalogRefreshOptions{})
	if len(result.Providers) != 1 {
		t.Fatalf("len(result.Providers) = %d, want 1", len(result.Providers))
	}
	pResult := result.Providers[0]
	if pResult.Err != nil {
		t.Fatalf("refresh failed: %v", pResult.Err)
	}
	if !pResult.Restored || !pResult.NotModified || !pResult.Published {
		t.Fatalf("unexpected provider refresh state: %+v", pResult)
	}

	modelsList := manager.Models("anthropic")
	if len(modelsList) != 1 || modelsList[0].ID != "claude-3-7-sonnet" {
		t.Fatalf("expected preserved published models, got: %+v", modelsList)
	}
}

func TestSourceMultiProviderInitialRefreshWithSharedCache(t *testing.T) {
	var requestCount atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)
		if r.Header.Get("If-None-Match") == `"v1"` {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		w.Header().Set("ETag", `"v1"`)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sampleDataset))
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	store, err := models.NewMemoryCatalogStore(nil)
	if err != nil {
		t.Fatal(err)
	}

	manager, err := models.NewCatalogManager(models.CatalogManagerConfig{
		Store: store,
		Providers: []models.CatalogProvider{
			{Provider: "anthropic", Source: source},
			{Provider: "fireworks", Source: source},
		},
	})
	if err != nil {
		t.Fatalf("NewCatalogManager error = %v", err)
	}

	// Initial refresh for two providers sharing one source
	result := manager.Refresh(context.Background(), models.CatalogRefreshOptions{})
	if len(result.Providers) != 2 {
		t.Fatalf("len(result.Providers) = %d, want 2", len(result.Providers))
	}
	for _, p := range result.Providers {
		if p.Err != nil {
			t.Fatalf("provider %s failed refresh: %v", p.Provider, p.Err)
		}
		if !p.Published {
			t.Fatalf("provider %s not published", p.Provider)
		}
	}

	anthropicModels := manager.Models("anthropic")
	if len(anthropicModels) != 1 || anthropicModels[0].ID != "claude-3-7-sonnet" {
		t.Fatalf("unexpected anthropic models: %+v", anthropicModels)
	}

	fireworksModels := manager.Models("fireworks")
	if len(fireworksModels) != 1 || fireworksModels[0].ID != "accounts/fireworks/models/deepseek-v3" {
		t.Fatalf("unexpected fireworks models: %+v", fireworksModels)
	}
}

func TestSourceUnknownProvider(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"etag-1"`)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sampleDataset))
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	resp, err := source.Fetch(context.Background(), models.CatalogFetchRequest{
		Provider: "non-existent-provider",
	})
	if err != nil {
		t.Fatalf("Fetch() error = %v", err)
	}
	if len(resp.Models) != 0 {
		t.Errorf("len(resp.Models) = %d, want 0", len(resp.Models))
	}
}

func TestSourceSkipsMalformedModels(t *testing.T) {
	malformed := `{
		"anthropic": {
			"id": "anthropic",
			"name": "Anthropic",
			"models": {
				"bad-model": {
					"id": ""
				},
				"bad-limit": {"id":"bad-limit","limit":{"context":100,"output":101}},
				"bad-cost": {"id":"bad-cost","cost":{"input":-1}},
				"good-model": {"id":"good-model","limit":{"context":200,"output":100}}
			}
		}
	}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(malformed))
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	response, err := source.Fetch(context.Background(), models.CatalogFetchRequest{
		Provider: "anthropic",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Models) != 1 || response.Models[0].ID != "good-model" {
		t.Fatalf("models = %#v, want only good-model", response.Models)
	}
}

func TestSourceBaselineRefreshPreservesCuratedMetadata(t *testing.T) {
	baseline := models.BuiltinCatalog()
	dataset := Dataset{}
	for _, model := range baseline.Models("") {
		provider := model.Provider
		if provider == "fireworks" {
			provider = "fireworks-ai"
		}
		entry := dataset[provider]
		if entry.Models == nil {
			entry.Models = map[string]ModelEntry{}
		}
		entry.Models[model.ID] = ModelEntry{ID: model.ID, Name: "upstream name", Reasoning: !model.Reasoning,
			ToolCall: true, StructuredOutput: true, Modalities: &Modalities{Input: []string{"text", "image", "audio"}},
			Limit: &Limit{Context: 2_000_000, Output: 12345}, Cost: &Cost{Input: floatPtr(1.23), Output: floatPtr(4.56)}}
		dataset[provider] = entry
	}
	dataset["mistral"].Models["new-model"] = ModelEntry{ID: "new-model", ToolCall: true}
	payload, err := jsonv2.Marshal(dataset)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(payload)
	}))
	defer server.Close()
	source := NewSource(WithURL(server.URL), WithHTTPClient(server.Client()))
	var providers []models.CatalogProvider
	for provider := range dataset {
		if provider == "fireworks-ai" {
			provider = "fireworks"
		}
		providers = append(providers, models.CatalogProvider{Provider: provider, Source: source})
	}
	store, err := models.NewMemoryCatalogStore(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := models.CatalogManagerConfig{Baseline: baseline, Store: store, Providers: providers}
	for _, offline := range []bool{false, true} {
		manager, err := models.NewCatalogManager(config)
		if err != nil {
			t.Fatal(err)
		}
		for _, result := range manager.Refresh(context.Background(), models.CatalogRefreshOptions{NoNetwork: offline}).Providers {
			if result.Err != nil {
				t.Fatalf("refresh %s (offline=%t): %v", result.Provider, offline, result.Err)
			}
		}
		for _, want := range baseline.Models("") {
			got, found := manager.Model(want.Provider, want.ID)
			if !found || got.ContextWindow != 2_000_000 || got.MaxOutputTokens != 12345 || got.Cost == nil || got.Cost.Input != 1.23 || got.Cost.Output != 4.56 {
				t.Fatalf("limits/cost not refreshed for %s/%s: %#v", want.Provider, want.ID, got)
			}
			got.ContextWindow, got.MaxOutputTokens, got.Cost = want.ContextWindow, want.MaxOutputTokens, want.Cost
			if !reflect.DeepEqual(got, want) {
				t.Errorf("curated metadata changed for %s/%s (offline=%t): got %#v, want %#v", want.Provider, want.ID, offline, got, want)
			}
		}
		added, ok := manager.Model("mistral", "new-model")
		if !ok || added.API != llm.APIMistralConversations {
			t.Fatalf("new model = %#v", added)
		}
		profile, _ := models.BuiltinProvider("google-vertex")
		collection, err := models.New(profile.Config("test-key"))
		if err != nil {
			t.Fatal(err)
		}
		vertex, _ := manager.Model("google-vertex", "gemini-2.5-pro")
		if _, err := collection.GeneratorFor(vertex); err != nil {
			t.Fatalf("GeneratorFor refreshed Vertex model: %v", err)
		}
	}
}

func TestSourceBaselineKeepsMissingMetadataAndValidatesMergedLimits(t *testing.T) {
	baseline, err := models.NewCatalog(
		llm.Model{Provider: "custom", ID: "missing", API: llm.APIOpenAIResponses, ContextWindow: 1000, MaxOutputTokens: 200, Cost: &llm.ModelCost{Input: 3, Output: 7}},
		llm.Model{Provider: "custom", ID: "partial", API: llm.APIOpenAIResponses, ContextWindow: 1000, MaxOutputTokens: 200},
		llm.Model{Provider: "custom", ID: "invalid", API: llm.APIOpenAIResponses, ContextWindow: 1000, MaxOutputTokens: 200},
	)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"custom":{"models":{
			"missing":{"id":"missing"},
			"partial":{"id":"partial","limit":{"output":300}},
			"invalid":{"id":"invalid","limit":{"context":100}}
		}}}`))
	}))
	defer server.Close()
	source := NewSource(WithURL(server.URL), WithHTTPClient(server.Client()), WithBaseline(baseline),
		WithAPIResolver(func(string, ModelEntry) llm.API { return llm.APIAnthropicMessages }),
		WithCompatibilityResolver(func(string, ModelEntry, llm.API) *llm.ModelCompatibility {
			return &llm.ModelCompatibility{Anthropic: &llm.AnthropicCompatibility{AdaptiveThinking: llm.CompatibilityEnabled}}
		}))
	for range 2 {
		response, err := source.Fetch(context.Background(), models.CatalogFetchRequest{Provider: "custom"})
		if err != nil {
			t.Fatal(err)
		}
		if len(response.Models) != 2 {
			t.Fatalf("models = %#v, want missing and partial only", response.Models)
		}
		want, _ := baseline.Model("custom", "missing")
		if !reflect.DeepEqual(response.Models[0], want) {
			t.Fatalf("missing metadata erased baseline: %#v", response.Models[0])
		}
		want, _ = baseline.Model("custom", "partial")
		want.MaxOutputTokens = 300
		if !reflect.DeepEqual(response.Models[1], want) {
			t.Fatalf("partial limits or resolver changed baseline: %#v", response.Models[1])
		}
		response.Models[0].Cost.Input = 999
		response.Models[1].Capabilities[0] = llm.CapabilityAudio
	}
}

func TestSourceResolversWithoutExplicitBaseline(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"openai":{"models":{"gpt-5":{"id":"gpt-5"}}}}`))
	}))
	defer server.Close()
	baseline := models.BuiltinCatalog()
	curated, _ := baseline.Model("openai", "gpt-5")
	for _, resolver := range []string{"api", "compatibility"} {
		for _, policy := range []string{"implicit", "before", "after", "nil"} {
			t.Run(resolver+"/"+policy, func(t *testing.T) {
				opts := []Option{WithURL(server.URL), WithHTTPClient(server.Client())}
				if policy == "before" {
					opts = append(opts, WithBaseline(baseline))
				}
				if resolver == "api" {
					opts = append(opts, WithAPIResolver(func(string, ModelEntry) llm.API { return llm.APIOpenAIChatCompletions }))
				} else {
					opts = append(opts, WithCompatibilityResolver(func(string, ModelEntry, llm.API) *llm.ModelCompatibility { return nil }))
				}
				if policy == "after" {
					opts = append(opts, WithBaseline(baseline))
				}
				if policy == "nil" {
					opts = append(opts, WithBaseline(nil))
				}
				resp, err := NewSource(opts...).Fetch(context.Background(), models.CatalogFetchRequest{Provider: "openai"})
				if err != nil || len(resp.Models) != 1 {
					t.Fatalf("Fetch = %#v, %v", resp, err)
				}
				got := resp.Models[0]
				if policy == "before" || policy == "after" {
					if !reflect.DeepEqual(got, curated) {
						t.Fatalf("explicit baseline lost: %#v", got)
					}
				} else if resolver == "api" {
					if got.API != llm.APIOpenAIChatCompletions {
						t.Fatalf("resolver API = %q", got.API)
					}
				} else if got.Compatibility != nil {
					t.Fatalf("resolver compatibility = %#v, want nil", got.Compatibility)
				}
			})
		}
	}
}

func TestSourceRecordedSnapshotBaselineOptions(t *testing.T) {
	// Recorded from https://models.dev/api.json on 2026-11-12, snapshot SHA256
	// fb66e34f3882207a4c7bb9d58f2429334a13d994c55b914718e763c2f62974f7.
	// In particular, the OpenRouter record has no cache_read field.
	payload, err := os.ReadFile("testdata/partial-cost.json")
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(payload) }))
	defer server.Close()
	baseline := models.BuiltinCatalog()
	for _, mode := range []string{"default", "explicit", "opt-out"} {
		t.Run(mode, func(t *testing.T) {
			opts := []Option{WithURL(server.URL), WithHTTPClient(server.Client())}
			switch mode {
			case "explicit":
				opts = append(opts, WithBaseline(baseline))
			case "opt-out":
				opts = append(opts, WithBaseline(nil))
			}
			source := NewSource(opts...)
			manager, err := models.NewCatalogManager(models.CatalogManagerConfig{Baseline: baseline, Providers: []models.CatalogProvider{
				{Provider: "anthropic", Source: source}, {Provider: "openai", Source: source}, {Provider: "openrouter", Source: source},
			}})
			if err != nil {
				t.Fatal(err)
			}
			for _, result := range manager.Refresh(context.Background(), models.CatalogRefreshOptions{}).Providers {
				if result.Err != nil {
					t.Fatal(result.Err)
				}
			}
			for _, id := range []struct{ provider, model string }{{"anthropic", "claude-opus-4-5"}, {"openai", "gpt-5"}} {
				want, _ := baseline.Model(id.provider, id.model)
				got, _ := manager.Model(id.provider, id.model)
				preserved := got.API == want.API && reflect.DeepEqual(got.Compatibility, want.Compatibility) && reflect.DeepEqual(got.Capabilities, want.Capabilities)
				if preserved != (mode != "opt-out") {
					t.Errorf("%s curated metadata preserved = %t", id.model, preserved)
				}
			}
			got, _ := manager.Model("openrouter", "deepseek/deepseek-v3.1-terminus")
			wantRead := 0.135
			if mode == "opt-out" {
				wantRead = 0
			}
			if got.Cost == nil || got.Cost.Input != 0.27 || got.Cost.Output != 1 || got.Cost.CacheRead != wantRead {
				t.Fatalf("partial upstream cost = %#v, want .27/1/%g", got.Cost, wantRead)
			}
			got, _ = manager.Model("openrouter", "meta-llama/llama-3.1-8b-instruct")
			wantRead = .02
			if mode == "opt-out" {
				wantRead = 0
			}
			if got.Cost == nil || got.Cost.Input != .02 || got.Cost.Output != .04 || got.Cost.CacheRead != wantRead {
				t.Fatalf("lower input with omitted cache_read = %#v, want .02/.04/%g", got.Cost, wantRead)
			}
		})
	}
}

func TestSourceMergesPartialCostsAndRetainsTiers(t *testing.T) {
	for _, test := range []struct {
		name string
		cost string
		want llm.ModelCost
	}{
		{"partial", `{"input":5}`, llm.ModelCost{Input: 5, Output: 7, CacheRead: .3, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 200000, Input: 6, Output: 14, CacheRead: .6, CacheWrite: 8}}}},
		{"explicit zero", `{"cache_read":0}`, llm.ModelCost{Input: 3, Output: 7, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 200000, Input: 6, Output: 14, CacheRead: .6, CacheWrite: 8}}}},
		{"new tier", `{"input":5,"tiers":[{"input_tokens_above":300000,"output":9}]}`, llm.ModelCost{Input: 5, Output: 7, CacheRead: .3, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 300000, Input: 5, Output: 9, CacheRead: .3, CacheWrite: 4}}}},
		{"matching tier", `{"tiers":[{"input_tokens_above":200000,"output":19}]}`, llm.ModelCost{Input: 3, Output: 7, CacheRead: .3, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 200000, Input: 6, Output: 19, CacheRead: .6, CacheWrite: 8}}}},
		{"lower tier input", `{"tiers":[{"input_tokens_above":200000,"input":0.2}]}`, llm.ModelCost{Input: 3, Output: 7, CacheRead: .3, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 200000, Input: .2, Output: 14, CacheRead: .2, CacheWrite: 8}}}},
		{"context over tier", `{"context_over_200k":{"output":19}}`, llm.ModelCost{Input: 3, Output: 7, CacheRead: .3, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 200000, Input: 6, Output: 19, CacheRead: .6, CacheWrite: 8}}}},
		{"filtered tiers", `{"tiers":[{"input_tokens_above":1000000,"input":20}]}`, llm.ModelCost{Input: 3, Output: 7, CacheRead: .3, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 200000, Input: 6, Output: 14, CacheRead: .6, CacheWrite: 8}}}},
		{"lower base input", `{"input":0.2}`, llm.ModelCost{Input: .2, Output: 7, CacheRead: .2, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 200000, Input: 6, Output: 14, CacheRead: .6, CacheWrite: 8}}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			baseline, err := models.NewCatalog(llm.Model{Provider: "custom", ID: "m", API: llm.APIOpenAIResponses, ContextWindow: 1000000, MaxOutputTokens: 1000,
				Cost: &llm.ModelCost{Input: 3, Output: 7, CacheRead: .3, CacheWrite: 4, Tiers: []llm.ModelCostTier{{InputTokensAbove: 200000, Input: 6, Output: 14, CacheRead: .6, CacheWrite: 8}}}})
			if err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(`{"custom":{"models":{"m":{"id":"m","cost":` + test.cost + `}}}}`))
			}))
			defer server.Close()
			source := NewSource(WithBaseline(baseline), WithURL(server.URL), WithHTTPClient(server.Client()))
			for range 2 {
				resp, err := source.Fetch(context.Background(), models.CatalogFetchRequest{Provider: "custom"})
				if err != nil {
					t.Fatal(err)
				}
				if len(resp.Models) != 1 || !reflect.DeepEqual(resp.Models[0].Cost, &test.want) {
					t.Fatalf("cost = %#v, want %#v", resp.Models, test.want)
				}
				resp.Models[0].Cost.Input = 999
				resp.Models[0].Cost.Tiers[0].Input = 999
			}
		})
	}
}

func TestSourceDropsTiersOutsideMergedContext(t *testing.T) {
	for _, window := range []int{200000, 272000, 272001} {
		for _, cost := range []string{``, `,"cost":{"input":3}`, `,"cost":{"tiers":[{"input_tokens_above":272000,"output":20}]}`} {
			t.Run(fmt.Sprintf("%d/%s", window, cost), func(t *testing.T) {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
					_, _ = fmt.Fprintf(w, `{"openrouter":{"models":{"openai/gpt-5.6-sol":{"id":"openai/gpt-5.6-sol","limit":{"context":%d}%s}}}}`, window, cost)
				}))
				defer server.Close()
				response, err := NewSource(WithURL(server.URL)).Fetch(context.Background(), models.CatalogFetchRequest{Provider: "openrouter"})
				if err != nil || len(response.Models) != 1 {
					t.Fatalf("Fetch = %#v, %v", response, err)
				}
				got := response.Models[0]
				wantTiers := 0
				if window > 272000 {
					wantTiers = 1
				}
				if got.ContextWindow != window || got.Cost == nil || len(got.Cost.Tiers) != wantTiers {
					t.Fatalf("context=%d cost=%+v, want context=%d tiers=%d", got.ContextWindow, got.Cost, window, wantTiers)
				}
			})
		}
	}
}

func TestSourceRejectsExplicitInvalidCachePrices(t *testing.T) {
	for _, cost := range []string{`{"input":0.1,"cache_read":0.3}`, `{"tiers":[{"input_tokens_above":200000,"input":0.1,"cache_read":0.3}]}`} {
		t.Run(cost, func(t *testing.T) {
			baseline := models.BuiltinCatalog()
			want, _ := baseline.Model("openrouter", "openai/gpt-5.6-sol")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = fmt.Fprintf(w, `{"openrouter":{"models":{"openai/gpt-5.6-sol":{"id":"openai/gpt-5.6-sol","limit":{"output":1000},"cost":%s},"valid":{"id":"valid"}}}}`, cost)
			}))
			defer server.Close()
			source := NewSource(WithURL(server.URL))
			manager, err := models.NewCatalogManager(models.CatalogManagerConfig{Baseline: baseline, Providers: []models.CatalogProvider{{Provider: "openrouter", Source: source}}})
			if err != nil {
				t.Fatal(err)
			}
			if result := manager.Refresh(context.Background(), models.CatalogRefreshOptions{}); result.Providers[0].Err != nil {
				t.Fatal(result.Providers[0].Err)
			}
			got, _ := manager.Model("openrouter", want.ID)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("invalid record changed baseline: cost=%+v output=%d", got.Cost, got.MaxOutputTokens)
			}
			if _, ok := manager.Model("openrouter", "valid"); !ok {
				t.Fatal("valid peer was dropped")
			}
			// Without the valid peer, the same rejection fails the whole provider.
			filtered := NewSource(WithURL(server.URL), WithFilter(func(_ string, entry ModelEntry) bool { return entry.ID != "valid" }))
			_, err = filtered.Fetch(context.Background(), models.CatalogFetchRequest{Provider: "openrouter"})
			var modelErr *llm.Error
			if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse || !strings.Contains(err.Error(), "cache_read must not exceed input") {
				t.Fatalf("all-invalid error = %v", err)
			}
		})
	}
}

func TestSourceAllInvalidPreservesPublishedOverlayAndStore(t *testing.T) {
	var payload atomic.Value
	payload.Store(`{"custom":{"models":{"dynamic":{"id":"dynamic","limit":{"context":1000,"output":123}}}}}`)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(payload.Load().(string))) }))
	defer server.Close()
	source := NewSource(WithURL(server.URL), WithHTTPClient(server.Client()))
	store, err := models.NewMemoryCatalogStore(nil)
	if err != nil {
		t.Fatal(err)
	}
	config := models.CatalogManagerConfig{Store: store, Providers: []models.CatalogProvider{{Provider: "custom", Source: source}}}
	manager, err := models.NewCatalogManager(config)
	if err != nil {
		t.Fatal(err)
	}
	if result := manager.Refresh(context.Background(), models.CatalogRefreshOptions{Force: true}); result.Providers[0].Err != nil {
		t.Fatal(result.Providers[0].Err)
	}
	before, _, err := store.Read(context.Background(), "custom")
	if err != nil {
		t.Fatal(err)
	}
	payload.Store(`{"custom":{"models":{"bad-limit":{"id":"bad-limit","limit":{"context":1,"output":2}},"bad-cost":{"id":"bad-cost","cost":{"input":-1}}}}}`)
	result := manager.Refresh(context.Background(), models.CatalogRefreshOptions{Force: true})
	var llmErr *llm.Error
	if !errors.As(result.Providers[0].Err, &llmErr) || llmErr.Kind != llm.KindMalformedResponse {
		t.Fatalf("refresh error = %v, want malformed response", result.Providers[0].Err)
	}
	for _, id := range []string{"bad-limit", "bad-cost"} {
		if !strings.Contains(llmErr.Error(), id) {
			t.Errorf("error does not identify %s: %v", id, llmErr)
		}
	}
	after, _, err := store.Read(context.Background(), "custom")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("failed refresh replaced persisted overlay")
	}
	if model, ok := manager.Model("custom", "dynamic"); !ok || model.MaxOutputTokens != 123 {
		t.Fatalf("published dynamic model = %#v", model)
	}
	// Deliberately filtering everything is not a conversion failure.
	filtered := NewSource(WithURL(server.URL), WithHTTPClient(server.Client()), WithFilter(func(string, ModelEntry) bool { return false }))
	if resp, err := filtered.Fetch(context.Background(), models.CatalogFetchRequest{Provider: "custom"}); err != nil || len(resp.Models) != 0 {
		t.Fatalf("filtered response = %#v, %v", resp, err)
	}
}

func TestSourceAllInvalidErrorIsBounded(t *testing.T) {
	for _, count := range []int{10, 11, 5000} {
		t.Run(fmt.Sprint(count), func(t *testing.T) {
			entries := map[string]ModelEntry{}
			for i := range count {
				id := fmt.Sprintf("bad-%05d", i)
				entries[id] = ModelEntry{ID: id, Limit: &Limit{Context: 1, Output: 2}}
			}
			payload, err := jsonv2.Marshal(Dataset{"custom": {Models: entries}})
			if err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(payload) }))
			defer server.Close()
			_, err = NewSource(WithURL(server.URL), WithHTTPClient(server.Client())).Fetch(context.Background(), models.CatalogFetchRequest{Provider: "custom"})
			var llmErr *llm.Error
			if !errors.As(err, &llmErr) || llmErr.Kind != llm.KindMalformedResponse {
				t.Fatalf("error = %v", err)
			}
			message := err.Error()
			t.Logf("rejected=%d error_bytes=%d", count, len(message))
			if len(message) > 2000 {
				t.Fatalf("error too large: %d bytes", len(message))
			}
			last := -1
			for i := range 10 {
				index := strings.Index(message, fmt.Sprintf("bad-%05d", i))
				if index <= last {
					t.Fatalf("missing or unsorted rejection ID %d", i)
				}
				last = index
			}
			if strings.Contains(message, "bad-00010") {
				t.Fatal("more than ten IDs reported")
			}
			if count > 10 && !strings.Contains(message, fmt.Sprintf("and %d more", count-10)) {
				t.Fatalf("missing remainder: %s", message)
			}
			if count == 10 && strings.Contains(message, " more") {
				t.Fatalf("unexpected remainder: %s", message)
			}
		})
	}
}

func TestSourceAllInvalidErrorTruncatesIDsAndIncludesCause(t *testing.T) {
	id := strings.Repeat("x", 100000)
	payload, err := jsonv2.Marshal(Dataset{"custom": {Models: map[string]ModelEntry{id: {ID: id, Limit: &Limit{Context: 1, Output: 2}}}}})
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(payload) }))
	defer server.Close()
	_, err = NewSource(WithURL(server.URL)).Fetch(context.Background(), models.CatalogFetchRequest{Provider: "custom"})
	if err == nil {
		t.Fatal("invalid record accepted")
	}
	t.Logf("id_bytes=%d error_bytes=%d", len(id), len(err.Error()))
	if len(err.Error()) > 2000 || !strings.Contains(err.Error(), strings.Repeat("x", 32)) || !strings.Contains(err.Error(), "max output tokens (2) must not exceed context window (1)") {
		t.Fatalf("error bytes=%d; want bounded ID and conversion cause", len(err.Error()))
	}
}

func TestSourceOversizedPayloadRejected(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		// Write 33MB of zeroes
		chunk := make([]byte, 1024*1024)
		for i := 0; i < 33; i++ {
			_, _ = w.Write(chunk)
		}
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	_, err := source.Fetch(context.Background(), models.CatalogFetchRequest{
		Provider: "anthropic",
	})
	if err == nil {
		t.Fatal("expected error for oversized payload")
	}
}

func TestSourceContextCancellation(t *testing.T) {
	source := NewSource(
		WithURL("http://127.0.0.1:0"),
	)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := source.Fetch(ctx, models.CatalogFetchRequest{
		Provider: "openai",
	})
	if err == nil {
		t.Fatal("expected error on cancelled context")
	}
}

func TestSourceConcurrentFetch(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"v1"`)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sampleDataset))
	}))
	defer ts.Close()

	source := NewSource(
		WithURL(ts.URL),
		WithHTTPClient(ts.Client()),
	)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			_, _ = source.Fetch(context.Background(), models.CatalogFetchRequest{Provider: "anthropic"})
		}()
		go func() {
			defer wg.Done()
			_, _ = source.Fetch(context.Background(), models.CatalogFetchRequest{Provider: "fireworks"})
		}()
	}
	wg.Wait()
}
