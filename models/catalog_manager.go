package models

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	llm "github.com/XiaoConstantine/llm-go"
)

// CatalogStoreEntry is one provider-scoped persisted dynamic catalog snapshot.
type CatalogStoreEntry struct {
	Models       []llm.Model
	ETag         string
	LastModified time.Time
	CheckedAt    time.Time
	FetchedAt    time.Time
}

// CatalogStore persists provider-scoped catalog snapshots. Implementations must
// own inputs/outputs, honor ctx, and never commit after cancellation.
type CatalogStore interface {
	Read(context.Context, string) (CatalogStoreEntry, bool, error)
	Write(context.Context, string, CatalogStoreEntry) error
}

// MemoryCatalogStore is a concurrency-safe in-memory CatalogStore.
type MemoryCatalogStore struct {
	mu      sync.RWMutex
	entries map[string]CatalogStoreEntry
}

// NewMemoryCatalogStore returns a store that owns a validated copy of initial.
func NewMemoryCatalogStore(initial map[string]CatalogStoreEntry) (*MemoryCatalogStore, error) {
	store := &MemoryCatalogStore{entries: make(map[string]CatalogStoreEntry, len(initial))}
	for rawProvider, entry := range initial {
		provider := strings.TrimSpace(rawProvider)
		if provider == "" {
			return nil, fmt.Errorf("catalog store provider must not be empty")
		}
		if provider != rawProvider {
			return nil, fmt.Errorf("catalog store provider %q contains surrounding whitespace", rawProvider)
		}
		normalized, err := validateProviderEntry(provider, entry)
		if err != nil {
			return nil, err
		}
		store.entries[provider] = normalized
	}
	return store, nil
}

// Read returns an owned copy of one provider entry.
func (s *MemoryCatalogStore) Read(ctx context.Context, provider string) (CatalogStoreEntry, bool, error) {
	if err := ctx.Err(); err != nil {
		return CatalogStoreEntry{}, false, err
	}
	s.mu.RLock()
	entry, ok := s.entries[provider]
	s.mu.RUnlock()
	if err := ctx.Err(); err != nil {
		return CatalogStoreEntry{}, false, err
	}
	return cloneCatalogEntry(entry), ok, nil
}

// Write validates and stores an owned copy of entry.
func (s *MemoryCatalogStore) Write(ctx context.Context, provider string, entry CatalogStoreEntry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	normalized, err := validateProviderEntry(provider, entry)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	s.entries[provider] = normalized
	return nil
}

// CatalogFetchRequest supplies prior validators and current credential material.
// Sources must not retain Credential and must honor ctx.
type CatalogFetchRequest struct {
	Provider     string
	ETag         string
	LastModified time.Time
	CheckedAt    time.Time
	FetchedAt    time.Time
	Force        bool
	Credential   *StoredCredential
}

// CatalogFetchResponse is either NotModified or a complete provider model list.
type CatalogFetchResponse struct {
	Models       []llm.Model
	NotModified  bool
	ETag         string
	LastModified time.Time
}

// CatalogSource fetches one provider's complete dynamic model list. Implementations
// must honor ctx, be safe for concurrent calls across manager operations, and
// transfer ownership of returned response storage to the caller.
type CatalogSource interface {
	Fetch(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error)
}

// CatalogSourceFunc adapts a function to CatalogSource.
type CatalogSourceFunc func(context.Context, CatalogFetchRequest) (CatalogFetchResponse, error)

// Fetch calls f with the supplied context and request.
func (f CatalogSourceFunc) Fetch(ctx context.Context, request CatalogFetchRequest) (CatalogFetchResponse, error) {
	return f(ctx, request)
}

// AvailableCredential contains no credential secrets.
type AvailableCredential struct {
	Provider   string
	Type       CredentialType
	ExpiresAt  time.Time
	Attributes map[string]string
}

// CatalogAvailabilityFilter selects available models using secret-free metadata.
// It must honor ctx, may be called concurrently, and receives owned inputs.
type CatalogAvailabilityFilter func(context.Context, AvailableCredential, []llm.Model) ([]llm.Model, error)

// CatalogProvider configures a dynamic source and/or availability filter.
type CatalogProvider struct {
	Provider string
	Source   CatalogSource
	Filter   CatalogAvailabilityFilter
}

// CatalogManagerConfig configures immutable snapshot publication.
type CatalogManagerConfig struct {
	Baseline    *Catalog
	Store       CatalogStore
	Credentials *CredentialManager
	Providers   []CatalogProvider
	Now         func() time.Time
}

// CatalogManager atomically publishes immutable catalog snapshots. It is safe
// for concurrent use. Refresh performs all work before returning, leaves no
// background goroutines, and therefore requires no Close method.
type CatalogManager struct {
	baseline    []llm.Model
	store       CatalogStore
	credentials *CredentialManager
	now         func() time.Time
	providers   map[string]*catalogProviderState
	order       []string

	catalogMu       sync.Mutex
	snapshot        atomic.Pointer[Catalog]
	refreshSequence atomic.Uint64

	// beforeProviderBegin is an internal scheduling hook used by adversarial tests.
	beforeProviderBegin func(string, uint64)
}

type catalogProviderState struct {
	provider string
	source   CatalogSource
	filter   CatalogAvailabilityFilter

	mu         sync.Mutex
	invocation uint64
	cancel     context.CancelFunc
	restored   bool
	entry      CatalogStoreEntry
	publishMu  sync.Mutex
}

// NewCatalogManager validates config and owns snapshots of its baseline and
// provider metadata. It retains configured stores, managers, sources, filters,
// and callbacks, which must remain concurrency-safe while in use.
func NewCatalogManager(config CatalogManagerConfig) (*CatalogManager, error) {
	baseline := config.Baseline
	if baseline == nil {
		var err error
		baseline, err = NewCatalog()
		if err != nil {
			return nil, err
		}
	}
	if typedNil(config.Store) {
		return nil, fmt.Errorf("catalog store must not be typed nil")
	}
	now := config.Now
	if now == nil {
		now = time.Now
	}
	manager := &CatalogManager{baseline: baseline.Models(""), store: config.Store, credentials: config.Credentials,
		now: now, providers: make(map[string]*catalogProviderState, len(config.Providers))}
	for index, providerConfig := range config.Providers {
		provider := strings.TrimSpace(providerConfig.Provider)
		if provider == "" {
			return nil, fmt.Errorf("catalog providers[%d].Provider must not be empty", index)
		}
		if _, exists := manager.providers[provider]; exists {
			return nil, fmt.Errorf("catalog provider %q is configured more than once", provider)
		}
		if typedNil(providerConfig.Source) {
			return nil, fmt.Errorf("catalog provider %q source must not be typed nil", provider)
		}
		if nilFunction(providerConfig.Filter) {
			providerConfig.Filter = nil
		}
		manager.providers[provider] = &catalogProviderState{provider: provider, source: providerConfig.Source, filter: providerConfig.Filter}
		manager.order = append(manager.order, provider)
	}
	snapshot, err := NewCatalog(manager.baseline...)
	if err != nil {
		return nil, err
	}
	manager.snapshot.Store(snapshot)
	return manager, nil
}

// Snapshot returns the current immutable catalog snapshot.
func (m *CatalogManager) Snapshot() *Catalog {
	if m == nil {
		return nil
	}
	return m.snapshot.Load()
}

// Models returns caller-owned models from the current snapshot. An empty
// provider returns all models.
func (m *CatalogManager) Models(provider string) []llm.Model {
	if snapshot := m.Snapshot(); snapshot != nil {
		return snapshot.Models(provider)
	}
	return nil
}

// Model returns a caller-owned model from the current snapshot.
func (m *CatalogManager) Model(provider, model string) (llm.Model, bool) {
	if snapshot := m.Snapshot(); snapshot != nil {
		return snapshot.Model(provider, model)
	}
	return llm.Model{}, false
}

// CatalogRefreshOptions controls selective, force, and restore-only refreshes.
type CatalogRefreshOptions struct {
	Providers []string
	Force     bool
	NoNetwork bool
}

// ProviderRefreshResult describes one selected provider refresh.
type ProviderRefreshResult struct {
	Provider    string
	Restored    bool
	Fetched     bool
	NotModified bool
	Published   bool
	Skipped     bool
	Err         error
}

// CatalogRefreshResult contains provider results in selection order.
type CatalogRefreshResult struct{ Providers []ProviderRefreshResult }

// Refresh restores persisted state before optional network work. Omitted
// Providers refreshes configured dynamic providers; selective unknown/static
// providers and duplicate provider entries return deterministic skipped results.
// It runs selected providers concurrently and waits for all of them. A newer
// refresh supersedes and cancels older work for the same provider.
func (m *CatalogManager) Refresh(ctx context.Context, options CatalogRefreshOptions) CatalogRefreshResult {
	if m == nil || ctx == nil {
		return CatalogRefreshResult{Providers: []ProviderRefreshResult{{Err: errors.New("catalog manager or context is nil")}}}
	}
	invocation := m.refreshSequence.Add(1)
	ids := append([]string(nil), options.Providers...)
	if len(ids) == 0 {
		for _, provider := range m.order {
			if m.providers[provider].source != nil {
				ids = append(ids, provider)
			}
		}
	}
	results := make([]ProviderRefreshResult, len(ids))
	seen := make(map[string]struct{}, len(ids))
	var wait sync.WaitGroup
	for index, raw := range ids {
		index, provider := index, strings.TrimSpace(raw)
		if _, duplicate := seen[provider]; duplicate {
			results[index] = ProviderRefreshResult{Provider: provider, Skipped: true}
			continue
		}
		seen[provider] = struct{}{}
		state := m.providers[provider]
		if provider == "" || state == nil || state.source == nil {
			results[index] = ProviderRefreshResult{Provider: provider, Skipped: true}
			continue
		}
		if !state.reserve(invocation) {
			results[index] = ProviderRefreshResult{Provider: provider, Err: context.Canceled}
			continue
		}
		wait.Go(func() {
			if m.beforeProviderBegin != nil {
				m.beforeProviderBegin(provider, invocation)
			}
			results[index] = m.refreshProvider(ctx, state, options, invocation)
		})
	}
	wait.Wait()
	return CatalogRefreshResult{Providers: results}
}

func (m *CatalogManager) refreshProvider(parent context.Context, state *catalogProviderState, options CatalogRefreshOptions, invocation uint64) ProviderRefreshResult {
	result := ProviderRefreshResult{Provider: state.provider}
	ctx, finish, started := state.begin(parent, invocation)
	if !started {
		result.Err = context.Canceled
		return result
	}
	defer finish()
	var errs []error
	if restored, err := m.restore(ctx, state, invocation); err != nil {
		errs = append(errs, err)
	} else {
		result.Restored = restored
	}
	if options.NoNetwork || ctx.Err() != nil {
		if ctx.Err() != nil {
			errs = append(errs, ctx.Err())
		}
		result.Err = errors.Join(errs...)
		return result
	}
	entry := state.current()
	var credential *StoredCredential
	if m.credentials != nil {
		value, found, err := m.credentials.Resolve(ctx, state.provider)
		if err != nil {
			errs = append(errs, catalogOperationError("catalog refresh", state.provider, err))
			result.Err = errors.Join(errs...)
			return result
		}
		if found {
			cloned := cloneStoredCredential(value)
			credential = &cloned
		}
	}
	request := CatalogFetchRequest{Provider: state.provider, ETag: entry.ETag, LastModified: entry.LastModified,
		CheckedAt: entry.CheckedAt, FetchedAt: entry.FetchedAt, Force: options.Force, Credential: credential}
	response, err := callCatalogSource(ctx, state.provider, state.source, request)
	if err != nil {
		errs = append(errs, err)
		result.Err = errors.Join(errs...)
		return result
	}
	result.Fetched = true
	now := m.now()
	if response.NotModified {
		if request.ETag == "" && request.LastModified.IsZero() {
			errs = append(errs, catalogOperationError("catalog refresh", state.provider, errors.New("NotModified response requires a prior ETag or Last-Modified validator")))
			result.Err = errors.Join(errs...)
			return result
		}
		if len(response.Models) != 0 {
			errs = append(errs, catalogOperationError("catalog refresh", state.provider, errors.New("NotModified response must not contain models")))
			result.Err = errors.Join(errs...)
			return result
		}
		result.NotModified = true
		entry.CheckedAt = now
		if response.ETag != "" {
			entry.ETag = response.ETag
		}
		if !response.LastModified.IsZero() {
			entry.LastModified = response.LastModified
		}
	} else {
		entry = CatalogStoreEntry{Models: response.Models, ETag: response.ETag, LastModified: response.LastModified, CheckedAt: now, FetchedAt: now}
	}
	normalized, err := validateProviderEntry(state.provider, entry)
	if err != nil {
		errs = append(errs, catalogOperationError("catalog refresh", state.provider, err))
		result.Err = errors.Join(errs...)
		return result
	}
	published, err := m.publish(ctx, state, invocation, normalized, true)
	if err != nil {
		errs = append(errs, err)
	}
	result.Published = published
	result.Err = errors.Join(errs...)
	return result
}

func (s *catalogProviderState) reserve(invocation uint64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if invocation <= s.invocation {
		return false
	}
	s.invocation = invocation
	if s.cancel != nil {
		s.cancel()
		s.cancel = nil
	}
	return true
}

func (s *catalogProviderState) begin(parent context.Context, invocation uint64) (context.Context, func(), bool) {
	s.mu.Lock()
	if s.invocation != invocation {
		s.mu.Unlock()
		return nil, nil, false
	}
	ctx, cancel := context.WithCancel(parent)
	s.cancel = cancel
	s.mu.Unlock()
	return ctx, func() {
		s.mu.Lock()
		if s.invocation == invocation {
			s.cancel = nil
		}
		s.mu.Unlock()
		cancel()
	}, true
}

func (s *catalogProviderState) current() CatalogStoreEntry {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneCatalogEntry(s.entry)
}

func (m *CatalogManager) restore(ctx context.Context, state *catalogProviderState, invocation uint64) (bool, error) {
	state.mu.Lock()
	restored := state.restored
	state.mu.Unlock()
	if restored || m.store == nil {
		return false, nil
	}
	entry, found, err := callCatalogStoreRead(ctx, state.provider, m.store)
	if err != nil {
		return false, catalogOperationError("catalog restore", state.provider, err)
	}
	if !found {
		state.mu.Lock()
		if state.invocation == invocation {
			state.restored = true
		}
		state.mu.Unlock()
		return false, nil
	}
	normalized, err := validateProviderEntry(state.provider, entry)
	if err != nil {
		return false, catalogOperationError("catalog restore", state.provider, err)
	}
	published, err := m.publish(ctx, state, invocation, normalized, false)
	return published, err
}

func (m *CatalogManager) publish(ctx context.Context, state *catalogProviderState, invocation uint64, entry CatalogStoreEntry, persist bool) (bool, error) {
	state.publishMu.Lock()
	defer state.publishMu.Unlock()
	if !state.isCurrent(invocation) {
		return false, context.Canceled
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	if persist && m.store != nil {
		if err := callCatalogStoreWrite(ctx, state.provider, m.store, cloneCatalogEntry(entry)); err != nil {
			return false, catalogOperationError("catalog store", state.provider, err)
		}
	}
	if !state.isCurrent(invocation) {
		return false, context.Canceled
	}
	state.mu.Lock()
	state.entry = cloneCatalogEntry(entry)
	state.restored = true
	state.mu.Unlock()
	m.catalogMu.Lock()
	models := append([]llm.Model(nil), m.baseline...)
	for _, provider := range m.order {
		providerEntry := m.providers[provider].current()
		models = mergeProviderOverlay(models, provider, providerEntry.Models)
	}
	snapshot, err := NewCatalog(models...)
	if err == nil {
		m.snapshot.Store(snapshot)
	}
	m.catalogMu.Unlock()
	if err != nil {
		return false, catalogOperationError("catalog publish", state.provider, err)
	}
	return true, nil
}

func (s *catalogProviderState) isCurrent(invocation uint64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.invocation == invocation
}

func mergeProviderOverlay(models []llm.Model, provider string, overlay []llm.Model) []llm.Model {
	override := make(map[string]struct{}, len(overlay))
	for _, model := range overlay {
		override[model.ID] = struct{}{}
	}
	result := make([]llm.Model, 0, len(models)+len(overlay))
	for _, model := range models {
		if model.Provider == provider {
			if _, exists := override[model.ID]; exists {
				continue
			}
		}
		result = append(result, model)
	}
	return append(result, overlay...)
}

func callCatalogStoreRead(ctx context.Context, provider string, store CatalogStore) (entry CatalogStoreEntry, found bool, err error) {
	defer func() {
		if value := recover(); value != nil {
			err = fmt.Errorf("store panicked: %v", value)
		}
	}()
	return store.Read(ctx, provider)
}

func callCatalogStoreWrite(ctx context.Context, provider string, store CatalogStore, entry CatalogStoreEntry) (err error) {
	defer func() {
		if value := recover(); value != nil {
			err = fmt.Errorf("store panicked: %v", value)
		}
	}()
	return store.Write(ctx, provider, entry)
}

func callCatalogSource(ctx context.Context, provider string, source CatalogSource, request CatalogFetchRequest) (response CatalogFetchResponse, err error) {
	defer func() {
		if value := recover(); value != nil {
			err = catalogOperationError("catalog refresh", provider, fmt.Errorf("source panicked: %v", value))
		}
	}()
	response, err = source.Fetch(ctx, request)
	if err != nil {
		err = catalogOperationError("catalog refresh", provider, err)
	}
	return
}

func validateProviderEntry(provider string, entry CatalogStoreEntry) (CatalogStoreEntry, error) {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return CatalogStoreEntry{}, fmt.Errorf("provider must not be empty")
	}
	if !utf8.ValidString(entry.ETag) {
		return CatalogStoreEntry{}, fmt.Errorf("ETag must be valid UTF-8")
	}
	for index, model := range entry.Models {
		if strings.TrimSpace(model.Provider) != provider {
			return CatalogStoreEntry{}, fmt.Errorf("models[%d] provider %q does not match %q", index, model.Provider, provider)
		}
	}
	catalog, err := NewCatalog(entry.Models...)
	if err != nil {
		return CatalogStoreEntry{}, err
	}
	entry.Models = catalog.Models(provider)
	return cloneCatalogEntry(entry), nil
}

func cloneCatalogEntry(entry CatalogStoreEntry) CatalogStoreEntry {
	entry.Models = cloneModels(entry.Models)
	return entry
}

func cloneModels(models []llm.Model) []llm.Model {
	result := make([]llm.Model, len(models))
	for index, model := range models {
		result[index] = cloneModel(model)
	}
	return result
}

func typedNil(value any) bool {
	if value == nil {
		return false
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	}
	return false
}

func catalogOperationError(op, provider string, err error) error {
	if err == nil {
		return nil
	}
	return &llm.Error{Kind: llm.KindProvider, Op: op, Provider: provider, Err: err}
}

// Available returns caller-owned models whose providers have a usable managed
// credential. Filter callbacks receive only secret-free credential metadata.
func (m *CatalogManager) Available(ctx context.Context, provider string) ([]llm.Model, error) {
	if m == nil || m.credentials == nil {
		return nil, nil
	}
	models := m.Models(provider)
	grouped := make(map[string][]llm.Model)
	var order []string
	for _, model := range models {
		if _, ok := grouped[model.Provider]; !ok {
			order = append(order, model.Provider)
		}
		grouped[model.Provider] = append(grouped[model.Provider], model)
	}
	var available []llm.Model
	var errs []error
	for _, id := range order {
		credential, found, err := m.credentials.Resolve(ctx, id)
		if err != nil {
			errs = append(errs, catalogOperationError("catalog availability", id, err))
			continue
		}
		if !found {
			continue
		}
		metadata := AvailableCredential{Provider: id, Type: credential.Type, ExpiresAt: credential.ExpiresAt,
			Attributes: cloneStringMap(credential.Attributes)}
		selected := cloneModels(grouped[id])
		if state := m.providers[id]; state != nil && state.filter != nil {
			selected, err = callAvailabilityFilter(ctx, id, state.filter, metadata, selected)
			if err != nil {
				errs = append(errs, err)
				continue
			}
		}
		canonical := make(map[string]llm.Model, len(grouped[id]))
		for _, model := range grouped[id] {
			canonical[model.ID] = model
		}
		selectedIDs := make(map[string]struct{}, len(selected))
		for _, model := range selected {
			original, ok := canonical[model.ID]
			if !ok || model.Provider != id {
				errs = append(errs, catalogOperationError("catalog availability", id, fmt.Errorf("filter returned unknown model %q", model.ID)))
				continue
			}
			if _, duplicate := selectedIDs[model.ID]; duplicate {
				errs = append(errs, catalogOperationError("catalog availability", id, fmt.Errorf("filter returned model %q more than once", model.ID)))
				continue
			}
			selectedIDs[model.ID] = struct{}{}
			available = append(available, cloneModel(original))
		}
	}
	if err := ctx.Err(); err != nil {
		errs = append(errs, err)
	}
	return available, errors.Join(errs...)
}

func callAvailabilityFilter(ctx context.Context, provider string, filter CatalogAvailabilityFilter, credential AvailableCredential, models []llm.Model) (result []llm.Model, err error) {
	defer func() {
		if value := recover(); value != nil {
			err = catalogOperationError("catalog availability", provider, fmt.Errorf("filter panicked: %v", value))
		}
	}()
	result, err = filter(ctx, credential, cloneModels(models))
	if err != nil {
		err = catalogOperationError("catalog availability", provider, err)
	}
	return
}

func cloneStringMap(input map[string]string) map[string]string {
	if input == nil {
		return nil
	}
	result := make(map[string]string, len(input))
	maps.Copy(result, input)
	return result
}

// ErrorMap returns the last non-nil error for each provider in the result.
func (r CatalogRefreshResult) ErrorMap() map[string]error {
	result := make(map[string]error)
	for _, provider := range r.Providers {
		if provider.Err != nil {
			result[provider.Provider] = provider.Err
		}
	}
	return result
}
