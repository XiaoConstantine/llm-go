package modelsdev

import (
	"context"
	"net/http"
	"net/http/httptest"
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

func TestSourceMalformedModelFailsRefresh(t *testing.T) {
	malformed := `{
		"anthropic": {
			"id": "anthropic",
			"name": "Anthropic",
			"models": {
				"bad-model": {
					"id": ""
				}
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

	_, err := source.Fetch(context.Background(), models.CatalogFetchRequest{
		Provider: "anthropic",
	})
	if err == nil {
		t.Fatal("expected error for malformed model with empty ID")
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
