package models

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
)

func catalogTestModel(provider, id string, api llm.API) llm.Model {
	return llm.Model{Provider: provider, ID: id, API: api, Capabilities: []llm.Capability{llm.CapabilityStreaming}}
}

type testCatalogStore struct {
	mu                sync.Mutex
	entries           map[string]CatalogStoreEntry
	readErr, writeErr error
	reads, writes     int
}

func (s *testCatalogStore) Read(ctx context.Context, provider string) (CatalogStoreEntry, bool, error) {
	if err := ctx.Err(); err != nil {
		return CatalogStoreEntry{}, false, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.reads++
	if s.readErr != nil {
		return CatalogStoreEntry{}, false, s.readErr
	}
	entry, ok := s.entries[provider]
	return cloneCatalogEntry(entry), ok, nil
}
func (s *testCatalogStore) Write(ctx context.Context, provider string, entry CatalogStoreEntry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.writes++
	if s.writeErr != nil {
		return s.writeErr
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.entries == nil {
		s.entries = make(map[string]CatalogStoreEntry)
	}
	s.entries[provider] = cloneCatalogEntry(entry)
	return nil
}

func TestMemoryCatalogStoreOwnsInputsAndOutputs(t *testing.T) {
	entry := CatalogStoreEntry{Models: []llm.Model{catalogTestModel("provider", "model", llm.APIOpenAIResponses)}, ETag: `"v1"`}
	store, err := NewMemoryCatalogStore(map[string]CatalogStoreEntry{"provider": entry})
	if err != nil {
		t.Fatal(err)
	}
	entry.Models[0].ID = "input-mutated"
	first, found, err := store.Read(context.Background(), "provider")
	if err != nil || !found || first.Models[0].ID != "model" {
		t.Fatalf("Read() = %#v/%v/%v", first, found, err)
	}
	first.Models[0].ID = "output-mutated"
	second, _, _ := store.Read(context.Background(), "provider")
	if second.Models[0].ID != "model" {
		t.Fatal("Read output aliases store")
	}
	written := CatalogStoreEntry{Models: []llm.Model{catalogTestModel("provider", "written", llm.APIOpenAIResponses)}}
	if err := store.Write(context.Background(), "provider", written); err != nil {
		t.Fatal(err)
	}
	written.Models[0].ID = "write-mutated"
	stored, _, _ := store.Read(context.Background(), "provider")
	if stored.Models[0].ID != "written" {
		t.Fatal("Write input aliases store")
	}
}

func TestCatalogManagerRestoreBeforeConditionalNotModified(t *testing.T) {
	baseline, _ := NewCatalog(catalogTestModel("provider", "base", llm.APIOpenAIResponses))
	stored := CatalogStoreEntry{Models: []llm.Model{catalogTestModel("provider", "cached", llm.APIOpenAIChatCompletions)}, ETag: `"v1"`, LastModified: time.Unix(10, 0), CheckedAt: time.Unix(20, 0)}
	store := &testCatalogStore{entries: map[string]CatalogStoreEntry{"provider": stored}}
	var manager *CatalogManager
	source := CatalogSourceFunc(func(_ context.Context, request CatalogFetchRequest) (CatalogFetchResponse, error) {
		if _, ok := manager.Model("provider", "cached"); !ok {
			t.Error("persisted model was not restored before fetch")
		}
		if request.ETag != stored.ETag || !request.LastModified.Equal(stored.LastModified) || !request.CheckedAt.Equal(stored.CheckedAt) {
			t.Errorf("validators/freshness = %q/%v/%v", request.ETag, request.LastModified, request.CheckedAt)
		}
		return CatalogFetchResponse{NotModified: true, ETag: `"v2"`}, nil
	})
	manager, _ = NewCatalogManager(CatalogManagerConfig{Baseline: baseline, Store: store, Now: func() time.Time { return time.Unix(30, 0) }, Providers: []CatalogProvider{{Provider: "provider", Source: source}}})
	result := manager.Refresh(context.Background(), CatalogRefreshOptions{})
	if len(result.Providers) != 1 || result.Providers[0].Err != nil || !result.Providers[0].Restored || !result.Providers[0].NotModified || !result.Providers[0].Published {
		t.Fatalf("Refresh() = %#v", result)
	}
	if _, ok := manager.Model("provider", "base"); !ok {
		t.Fatal("baseline model lost")
	}
	store.mu.Lock()
	persisted := store.entries["provider"]
	store.mu.Unlock()
	if persisted.ETag != `"v2"` || !persisted.CheckedAt.Equal(time.Unix(30, 0)) || len(persisted.Models) != 1 {
		t.Fatalf("persisted = %#v", persisted)
	}
	persisted.Models[0].ID = "mutated"
	if model, _ := manager.Model("provider", "cached"); model.ID != "cached" {
		t.Fatal("store output aliases manager")
	}
}

func TestCatalogManagerOverlayReplacesMatchingBaselineOnly(t *testing.T) {
	baseline, _ := NewCatalog(
		llm.Model{Provider: "provider", ID: "same", Name: "static", API: llm.APIOpenAIResponses},
		catalogTestModel("provider", "untouched", llm.APIOpenAIResponses),
	)
	manager, _ := NewCatalogManager(CatalogManagerConfig{Baseline: baseline, Providers: []CatalogProvider{{Provider: "provider", Source: CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
		return CatalogFetchResponse{Models: []llm.Model{{Provider: "provider", ID: "same", Name: "dynamic", API: llm.APIOpenAIChatCompletions}}}, nil
	})}}})
	result := manager.Refresh(context.Background(), CatalogRefreshOptions{})
	if result.Providers[0].Err != nil {
		t.Fatal(result.Providers[0].Err)
	}
	model, _ := manager.Model("provider", "same")
	if model.Name != "dynamic" || model.API != llm.APIOpenAIChatCompletions {
		t.Fatalf("overlay model = %#v", model)
	}
	if _, ok := manager.Model("provider", "untouched"); !ok {
		t.Fatal("nonmatching baseline model removed")
	}
}

func TestCatalogManagerNoNetworkRestoreAndMalformedStateRetention(t *testing.T) {
	good := CatalogStoreEntry{Models: []llm.Model{catalogTestModel("provider", "good", llm.APIOpenAIResponses)}}
	store := &testCatalogStore{entries: map[string]CatalogStoreEntry{"provider": good}}
	fetches := 0
	manager, _ := NewCatalogManager(CatalogManagerConfig{Store: store, Providers: []CatalogProvider{{Provider: "provider", Source: CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
		fetches++
		return CatalogFetchResponse{}, nil
	})}}})
	result := manager.Refresh(context.Background(), CatalogRefreshOptions{NoNetwork: true})
	if fetches != 0 || result.Providers[0].Err != nil {
		t.Fatalf("offline refresh = %#v fetches=%d", result, fetches)
	}
	if _, ok := manager.Model("provider", "good"); !ok {
		t.Fatal("offline restore missing")
	}

	manager.providers["provider"].restored = false
	store.mu.Lock()
	store.entries["provider"] = CatalogStoreEntry{Models: []llm.Model{catalogTestModel("wrong", "bad", llm.APIOpenAIResponses)}}
	store.mu.Unlock()
	result = manager.Refresh(context.Background(), CatalogRefreshOptions{NoNetwork: true})
	if result.Providers[0].Err == nil {
		t.Fatal("malformed persisted state succeeded")
	}
	if _, ok := manager.Model("provider", "good"); !ok {
		t.Fatal("malformed restore replaced good state")
	}
}

func TestCatalogManagerRejectsNotModifiedWithoutValidator(t *testing.T) {
	store := &testCatalogStore{}
	manager, _ := NewCatalogManager(CatalogManagerConfig{Store: store, Providers: []CatalogProvider{{Provider: "provider", Source: CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
		return CatalogFetchResponse{NotModified: true, ETag: `"unexpected"`}, nil
	})}}})
	result := manager.Refresh(context.Background(), CatalogRefreshOptions{})
	if result.Providers[0].Err == nil || result.Providers[0].Published || result.Providers[0].NotModified {
		t.Fatalf("Refresh() = %#v", result)
	}
	store.mu.Lock()
	writes := store.writes
	store.mu.Unlock()
	if writes != 0 {
		t.Fatalf("store writes = %d, want 0", writes)
	}
	if models := manager.Models("provider"); len(models) != 0 {
		t.Fatalf("published models = %#v", models)
	}
}

func TestCatalogManagerFetchAndStoreFailuresRetainLastKnown(t *testing.T) {
	store := &testCatalogStore{}
	mode := "good"
	source := CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
		switch mode {
		case "good":
			return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", "good", llm.APIOpenAIResponses)}}, nil
		case "malformed":
			return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("wrong", "bad", llm.APIOpenAIResponses)}}, nil
		case "duplicate":
			return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", "same", llm.APIOpenAIResponses), catalogTestModel("provider", "same", llm.APIOpenAIChatCompletions)}}, nil
		default:
			return CatalogFetchResponse{}, errors.New("fetch failed")
		}
	})
	manager, _ := NewCatalogManager(CatalogManagerConfig{Store: store, Providers: []CatalogProvider{{Provider: "provider", Source: source}}})
	if result := manager.Refresh(context.Background(), CatalogRefreshOptions{}); result.Providers[0].Err != nil {
		t.Fatal(result.Providers[0].Err)
	}
	for _, next := range []string{"malformed", "duplicate", "error"} {
		mode = next
		result := manager.Refresh(context.Background(), CatalogRefreshOptions{Force: true})
		if result.Providers[0].Err == nil {
			t.Fatalf("%s refresh succeeded", next)
		}
		if _, ok := manager.Model("provider", "good"); !ok {
			t.Fatalf("%s replaced last known state", next)
		}
	}
	mode = "good"
	store.writeErr = errors.New("disk full")
	result := manager.Refresh(context.Background(), CatalogRefreshOptions{Force: true})
	if result.Providers[0].Err == nil {
		t.Fatal("store failure succeeded")
	}
	if _, ok := manager.Model("provider", "good"); !ok {
		t.Fatal("store failure replaced state")
	}
}

func TestCatalogManagerCallEntryOrderWinsBeforeOlderProviderBegin(t *testing.T) {
	olderBeforeBegin := make(chan struct{})
	releaseOlder := make(chan struct{})
	var sourceCalls atomic.Int32
	source := CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
		call := sourceCalls.Add(1)
		id := "new"
		if call > 1 {
			id = "old"
		}
		return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", id, llm.APIOpenAIResponses)}}, nil
	})
	manager, _ := NewCatalogManager(CatalogManagerConfig{Providers: []CatalogProvider{{Provider: "provider", Source: source}}})
	manager.beforeProviderBegin = func(_ string, invocation uint64) {
		if invocation == 1 {
			close(olderBeforeBegin)
			<-releaseOlder
		}
	}
	olderDone := make(chan CatalogRefreshResult, 1)
	go func() { olderDone <- manager.Refresh(context.Background(), CatalogRefreshOptions{}) }()
	<-olderBeforeBegin
	newer := manager.Refresh(context.Background(), CatalogRefreshOptions{Force: true})
	if newer.Providers[0].Err != nil || !newer.Providers[0].Published {
		t.Fatalf("newer refresh = %#v", newer)
	}
	close(releaseOlder)
	older := <-olderDone
	if !errors.Is(older.Providers[0].Err, context.Canceled) || older.Providers[0].Published {
		t.Fatalf("older refresh = %#v", older)
	}
	if calls := sourceCalls.Load(); calls != 1 {
		t.Fatalf("source calls = %d, want 1", calls)
	}
	if _, ok := manager.Model("provider", "new"); !ok {
		t.Fatal("newer invocation did not win")
	}
	if _, ok := manager.Model("provider", "old"); ok {
		t.Fatal("older invocation published")
	}
}

func TestCatalogManagerDuplicateProvidersAreSkipped(t *testing.T) {
	var calls atomic.Int32
	manager, _ := NewCatalogManager(CatalogManagerConfig{Providers: []CatalogProvider{{Provider: "provider", Source: CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
		calls.Add(1)
		return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", "model", llm.APIOpenAIResponses)}}, nil
	})}}})
	result := manager.Refresh(context.Background(), CatalogRefreshOptions{Providers: []string{"provider", "provider", "provider"}})
	if calls.Load() != 1 || len(result.Providers) != 3 || result.Providers[0].Skipped || !result.Providers[1].Skipped || !result.Providers[2].Skipped {
		t.Fatalf("duplicate refresh = %#v, calls = %d", result, calls.Load())
	}
}

func TestCatalogManagerOverlappingRefreshLatestWins(t *testing.T) {
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	secondStarted := make(chan struct{})
	var mu sync.Mutex
	calls := 0
	source := CatalogSourceFunc(func(ctx context.Context, _ CatalogFetchRequest) (CatalogFetchResponse, error) {
		mu.Lock()
		calls++
		call := calls
		mu.Unlock()
		if call == 1 {
			close(firstStarted)
			<-releaseFirst
			return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", "old", llm.APIOpenAIResponses)}}, nil
		}
		close(secondStarted)
		return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", "new", llm.APIOpenAIResponses)}}, nil
	})
	store := &testCatalogStore{}
	manager, _ := NewCatalogManager(CatalogManagerConfig{Store: store, Providers: []CatalogProvider{{Provider: "provider", Source: source}}})
	firstDone := make(chan CatalogRefreshResult, 1)
	go func() { firstDone <- manager.Refresh(context.Background(), CatalogRefreshOptions{}) }()
	<-firstStarted
	secondDone := make(chan CatalogRefreshResult, 1)
	go func() { secondDone <- manager.Refresh(context.Background(), CatalogRefreshOptions{Force: true}) }()
	<-secondStarted
	second := <-secondDone
	if second.Providers[0].Err != nil {
		t.Fatal(second.Providers[0].Err)
	}
	close(releaseFirst)
	first := <-firstDone
	if first.Providers[0].Published {
		t.Fatalf("stale refresh published: %#v", first)
	}
	if _, ok := manager.Model("provider", "new"); !ok {
		t.Fatal("latest model missing")
	}
	if _, ok := manager.Model("provider", "old"); ok {
		t.Fatal("stale model published")
	}
	store.mu.Lock()
	persisted := store.entries["provider"]
	store.mu.Unlock()
	if len(persisted.Models) != 1 || persisted.Models[0].ID != "new" {
		t.Fatalf("persisted stale state = %#v", persisted)
	}
}

func TestCatalogManagerProvidersRefreshConcurrentlyAndSelectiveSkips(t *testing.T) {
	started := make(chan string, 2)
	release := make(chan struct{})
	provider := func(id string) CatalogProvider {
		return CatalogProvider{Provider: id, Source: CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
			started <- id
			<-release
			return CatalogFetchResponse{Models: []llm.Model{catalogTestModel(id, "model", llm.APIOpenAIResponses)}}, nil
		})}
	}
	manager, _ := NewCatalogManager(CatalogManagerConfig{Providers: []CatalogProvider{provider("one"), provider("two"), {Provider: "static"}}})
	done := make(chan CatalogRefreshResult, 1)
	go func() { done <- manager.Refresh(context.Background(), CatalogRefreshOptions{}) }()
	seen := map[string]bool{<-started: true, <-started: true}
	if !seen["one"] || !seen["two"] {
		t.Fatalf("started=%v", seen)
	}
	close(release)
	result := <-done
	if len(result.Providers) != 2 {
		t.Fatalf("result=%#v", result)
	}
	selective := manager.Refresh(context.Background(), CatalogRefreshOptions{Providers: []string{"static", "unknown"}, NoNetwork: true})
	if len(selective.Providers) != 2 || !selective.Providers[0].Skipped || !selective.Providers[1].Skipped {
		t.Fatalf("selective=%#v", selective)
	}
}

func TestCatalogManagerRefreshReturnsProviderScopedPartialErrors(t *testing.T) {
	manager, _ := NewCatalogManager(CatalogManagerConfig{Providers: []CatalogProvider{
		{Provider: "good", Source: CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
			return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("good", "model", llm.APIOpenAIResponses)}}, nil
		})},
		{Provider: "bad", Source: CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
			return CatalogFetchResponse{}, errors.New("failed")
		})},
	}})
	result := manager.Refresh(context.Background(), CatalogRefreshOptions{})
	if len(result.Providers) != 2 || result.Providers[0].Err != nil || result.Providers[1].Err == nil {
		t.Fatalf("Refresh() = %#v", result)
	}
	if _, ok := manager.Model("good", "model"); !ok {
		t.Fatal("successful provider publication was discarded")
	}
	if len(result.ErrorMap()) != 1 {
		t.Fatalf("ErrorMap() = %#v", result.ErrorMap())
	}
}

func TestCatalogManagerSourcePanicDoesNotWedgeFutureRefresh(t *testing.T) {
	calls := 0
	source := CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
		calls++
		if calls == 1 {
			panic("boom")
		}
		return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", "ok", llm.APIOpenAIResponses)}}, nil
	})
	manager, _ := NewCatalogManager(CatalogManagerConfig{Providers: []CatalogProvider{{Provider: "provider", Source: source}}})
	if result := manager.Refresh(context.Background(), CatalogRefreshOptions{}); result.Providers[0].Err == nil {
		t.Fatal("panic succeeded")
	} else {
		var modelErr *llm.Error
		if !errors.As(result.Providers[0].Err, &modelErr) || modelErr.Kind != llm.KindProvider || modelErr.Provider != "provider" {
			t.Fatalf("panic classification = %#v", modelErr)
		}
	}
	if result := manager.Refresh(context.Background(), CatalogRefreshOptions{}); result.Providers[0].Err != nil {
		t.Fatal(result.Providers[0].Err)
	}
	if _, ok := manager.Model("provider", "ok"); !ok {
		t.Fatal("future refresh wedged")
	}
}

func TestCatalogManagerRefreshUsesCurrentCredential(t *testing.T) {
	store, _ := NewMemoryCredentialStore(map[string]StoredCredential{"provider": {Type: CredentialAPIKey, APIKey: "first"}})
	credentials, _ := NewCredentialManager(store)
	seen := make(chan string, 2)
	source := CatalogSourceFunc(func(_ context.Context, request CatalogFetchRequest) (CatalogFetchResponse, error) {
		if request.Credential == nil {
			t.Fatal("missing refresh credential")
		}
		seen <- request.Credential.APIKey
		request.Credential.APIKey = "source-mutated"
		return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", "model", llm.APIOpenAIResponses)}}, nil
	})
	manager, _ := NewCatalogManager(CatalogManagerConfig{Credentials: credentials, Providers: []CatalogProvider{{Provider: "provider", Source: source}}})
	manager.Refresh(context.Background(), CatalogRefreshOptions{})
	_ = store.Modify(context.Background(), "provider", func(current StoredCredential, _ bool) (*StoredCredential, error) {
		current.APIKey = "second"
		return &current, nil
	})
	manager.Refresh(context.Background(), CatalogRefreshOptions{Force: true})
	if first, second := <-seen, <-seen; first != "first" || second != "second" {
		t.Fatalf("refresh credentials = %q/%q", first, second)
	}
	stored, _, _ := store.Read(context.Background(), "provider")
	if stored.APIKey != "second" {
		t.Fatal("source mutated stored credential")
	}
}

func TestCatalogManagerAvailabilityCredentialFilterOwnership(t *testing.T) {
	baseline, _ := NewCatalog(catalogTestModel("one", "a", llm.APIOpenAIResponses), catalogTestModel("one", "b", llm.APIOpenAIChatCompletions), catalogTestModel("two", "c", llm.APIOpenAIResponses))
	credentials, _ := NewMemoryCredentialStore(map[string]StoredCredential{"one": {Type: CredentialAPIKey, APIKey: "secret", Attributes: map[string]string{"tier": "basic"}}})
	managerCredentials, _ := NewCredentialManager(credentials)
	filter := func(ctx context.Context, metadata AvailableCredential, models []llm.Model) ([]llm.Model, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if metadata.Attributes["tier"] != "basic" {
			t.Fatalf("metadata=%#v", metadata)
		}
		metadata.Attributes["tier"] = "mutated"
		models[0].ID = "mutated"
		return models[1:], nil
	}
	manager, _ := NewCatalogManager(CatalogManagerConfig{Baseline: baseline, Credentials: managerCredentials, Providers: []CatalogProvider{{Provider: "one", Filter: filter}}})
	available, err := manager.Available(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(available) != 1 || available[0].ID != "b" {
		t.Fatalf("available=%#v", available)
	}
	stored, _, _ := credentials.Read(context.Background(), "one")
	if stored.Attributes["tier"] != "basic" {
		t.Fatal("filter mutated credential metadata")
	}
	available[0].ID = "caller"
	if model, _ := manager.Model("one", "b"); model.ID != "b" {
		t.Fatal("availability aliases catalog")
	}
}

func TestCatalogManagerTypedNilsAndCancellation(t *testing.T) {
	var nilStore *testCatalogStore
	if manager, err := NewCatalogManager(CatalogManagerConfig{Store: nilStore}); err == nil || manager != nil {
		t.Fatalf("typed nil store = %#v, %v", manager, err)
	}
	var source *typedNilCatalogSource
	if manager, err := NewCatalogManager(CatalogManagerConfig{Providers: []CatalogProvider{{Provider: "provider", Source: source}}}); err == nil || manager != nil {
		t.Fatalf("typed nil source=%#v,%v", manager, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	manager, _ := NewCatalogManager(CatalogManagerConfig{Providers: []CatalogProvider{{Provider: "provider", Source: CatalogSourceFunc(func(ctx context.Context, _ CatalogFetchRequest) (CatalogFetchResponse, error) {
		return CatalogFetchResponse{}, ctx.Err()
	})}}})
	result := manager.Refresh(ctx, CatalogRefreshOptions{})
	if !errors.Is(result.Providers[0].Err, context.Canceled) {
		t.Fatalf("canceled=%#v", result)
	}
}

type panicCatalogStore struct{ panicRead bool }

func (s *panicCatalogStore) Read(context.Context, string) (CatalogStoreEntry, bool, error) {
	if s.panicRead {
		s.panicRead = false
		panic("read boom")
	}
	return CatalogStoreEntry{}, false, nil
}
func (*panicCatalogStore) Write(context.Context, string, CatalogStoreEntry) error { return nil }

func TestCatalogManagerStorePanicDoesNotWedge(t *testing.T) {
	store := &panicCatalogStore{panicRead: true}
	manager, _ := NewCatalogManager(CatalogManagerConfig{Store: store, Providers: []CatalogProvider{{Provider: "provider", Source: CatalogSourceFunc(func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
		return CatalogFetchResponse{Models: []llm.Model{catalogTestModel("provider", "ok", llm.APIOpenAIResponses)}}, nil
	})}}})
	first := manager.Refresh(context.Background(), CatalogRefreshOptions{NoNetwork: true})
	if first.Providers[0].Err == nil {
		t.Fatal("store panic succeeded")
	}
	second := manager.Refresh(context.Background(), CatalogRefreshOptions{})
	if second.Providers[0].Err != nil {
		t.Fatal(second.Providers[0].Err)
	}
}

type typedNilCatalogSource struct{}

func (*typedNilCatalogSource) Fetch(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error) {
	return CatalogFetchResponse{}, nil
}
