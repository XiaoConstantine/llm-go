package models

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

type registryGenerator struct {
	info     llm.ModelInfo
	generate func(context.Context, llm.Request) (*llm.Response, error)
}

func (g *registryGenerator) Info() llm.ModelInfo { return cloneModelInfo(g.info) }
func (g *registryGenerator) Generate(ctx context.Context, request llm.Request) (*llm.Response, error) {
	if g.generate != nil {
		return g.generate(ctx, request)
	}
	return &llm.Response{Message: llm.Message{Role: llm.RoleAssistant}}, nil
}
func (*registryGenerator) Stream(context.Context, llm.Request) (llm.Stream, error) {
	return nil, errors.New("unused")
}

func customFactory(api llm.API, seen chan<- GeneratorFactoryConfig) GeneratorFactory {
	return func(ctx context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if seen != nil {
			seen <- config
		}
		return &registryGenerator{info: llm.ModelInfo{Provider: config.Provider, Model: config.Model.Model, API: api, Capabilities: []llm.Capability{llm.CapabilityGeneration}}}, nil
	}
}

func TestFactoryRegistryValidationAndOwnership(t *testing.T) {
	var nilFactory GeneratorFactory
	for _, registrations := range [][]FactoryRegistration{
		{{Factory: customFactory("x", nil)}},
		{{API: "x", Factory: nilFactory}},
		{{API: llm.APIOpenAIResponses, Factory: customFactory(llm.APIOpenAIResponses, nil)}},
		{{API: "x", Factory: customFactory("x", nil)}, {API: "x", Factory: customFactory("x", nil)}},
	} {
		if registry, err := NewFactoryRegistry(registrations...); err == nil || registry != nil {
			t.Fatalf("NewFactoryRegistry(%#v) = %#v, %v", registrations, registry, err)
		}
	}
	registry, err := NewFactoryRegistry(FactoryRegistration{API: "custom", Factory: customFactory("custom", nil)})
	if err != nil {
		t.Fatal(err)
	}
	apis := registry.APIs()
	apis[0] = "mutated"
	found := false
	for _, api := range registry.APIs() {
		found = found || api == "custom"
	}
	if !found {
		t.Fatal("registry APIs aliased caller storage")
	}
}

func TestCollectionMixedProtocolRoutingAndOwnedFactoryConfig(t *testing.T) {
	const firstAPI llm.API = "custom-one"
	const secondAPI llm.API = "custom-two"
	seen := make(chan GeneratorFactoryConfig, 2)
	registry, err := NewFactoryRegistry(
		FactoryRegistration{API: firstAPI, Factory: customFactory(firstAPI, seen)},
		FactoryRegistration{API: secondAPI, Factory: customFactory(secondAPI, seen)},
	)
	if err != nil {
		t.Fatal(err)
	}
	headers := http.Header{"X-Route": {"original"}}
	additional := []ProviderAPIConfig{{API: secondAPI, Headers: headers}}
	collection, err := NewWithRegistry(registry, ProviderConfig{ID: "provider", API: API(firstAPI), Headers: headers,
		AdditionalAPIs: additional})
	if err != nil {
		t.Fatal(err)
	}
	headers.Set("X-Route", "mutated")
	additional[0].API = "mutated"
	additional[0].Headers.Set("X-Route", "also-mutated")
	for _, api := range []llm.API{firstAPI, secondAPI} {
		info := llm.ModelInfo{Provider: "provider", Model: "model-" + string(api), API: api,
			Capabilities: []llm.Capability{llm.CapabilityStreaming}, ContextWindow: 100, MaxOutputTokens: 20,
			Cost: &llm.ModelCost{Input: 1, Tiers: []llm.ModelCostTier{{InputTokensAbove: 10, Input: 2}}}}
		generator, err := collection.Generator(info)
		if err != nil {
			t.Fatal(err)
		}
		if info := generator.Info(); info.API != api || info.ContextWindow != 100 || info.MaxOutputTokens != 20 {
			t.Fatalf("Info() = %#v", info)
		}
		config := <-seen
		if config.Headers.Get("X-Route") != "original" {
			t.Fatalf("factory headers = %#v", config.Headers)
		}
		config.Headers.Set("X-Route", "factory-mutated")
		config.Model.Capabilities[0] = llm.CapabilityAudio
		config.Model.Cost.Tiers[0].Input = 99
		if info.Capabilities[0] != llm.CapabilityStreaming || info.Cost.Tiers[0].Input != 2 {
			t.Fatal("factory config aliases caller model info")
		}
	}
	if _, err := collection.Generator(llm.ModelInfo{Provider: "provider", Model: "model", API: "not-enabled"}); err == nil {
		t.Fatal("unconfigured model API succeeded")
	}
	generator, err := collection.Generator(llm.ModelInfo{Provider: "provider", Model: "fallback"})
	if err != nil || generator.Info().API != firstAPI {
		t.Fatalf("default route = %#v, %v", generator, err)
	}
}

type panicOnSecondInfoGenerator struct {
	registryGenerator
	calls int
}

func (g *panicOnSecondInfoGenerator) Info() llm.ModelInfo {
	g.calls++
	if g.calls > 1 {
		panic("Info called more than once")
	}
	return cloneModelInfo(g.info)
}

func TestCollectionUsesValidatedFactoryInfoOnlyOnce(t *testing.T) {
	const api llm.API = "custom-info-once"
	var base *panicOnSecondInfoGenerator
	factory := func(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
		base = &panicOnSecondInfoGenerator{info: llm.ModelInfo{
			Provider: config.Provider, Model: config.Model.Model, API: api,
			Capabilities: []llm.Capability{llm.CapabilityGeneration},
		}}
		return base, nil
	}
	registry, _ := NewFactoryRegistry(FactoryRegistration{API: api, Factory: factory})
	collection, _ := NewWithRegistry(registry, ProviderConfig{ID: "provider", API: API(api)})
	generator, err := collection.Generator(llm.ModelInfo{Provider: "provider", Model: "model", API: api, ContextWindow: 100})
	if err != nil {
		t.Fatal(err)
	}
	if base.calls != 1 {
		t.Fatalf("underlying Info calls = %d, want 1", base.calls)
	}
	for range 2 {
		info := generator.Info()
		if info.Provider != "provider" || info.Model != "model" || info.API != api || info.ContextWindow != 100 {
			t.Fatalf("wrapped Info() = %#v", info)
		}
	}
	if base.calls != 1 {
		t.Fatalf("underlying Info calls after wrapped Info = %d, want 1", base.calls)
	}
}

func TestCollectionFactoryInfoNilPanicAndCancellation(t *testing.T) {
	const api llm.API = "custom"
	tests := []struct {
		name    string
		factory GeneratorFactory
	}{
		{name: "nil", factory: func(context.Context, GeneratorFactoryConfig) (llm.Generator, error) { return nil, nil }},
		{name: "typed nil", factory: func(context.Context, GeneratorFactoryConfig) (llm.Generator, error) {
			var g *registryGenerator
			return g, nil
		}},
		{name: "mismatch", factory: func(context.Context, GeneratorFactoryConfig) (llm.Generator, error) {
			return &registryGenerator{info: llm.ModelInfo{Provider: "wrong", Model: "model", API: api}}, nil
		}},
		{name: "panic", factory: func(context.Context, GeneratorFactoryConfig) (llm.Generator, error) { panic("boom") }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			registry, _ := NewFactoryRegistry(FactoryRegistration{API: api, Factory: test.factory})
			collection, _ := NewWithRegistry(registry, ProviderConfig{ID: "provider", API: API(api)})
			if generator, err := collection.Generator(llm.ModelInfo{Provider: "provider", Model: "model", API: api}); err == nil || generator != nil {
				t.Fatalf("Generator() = %#v, %v", generator, err)
			}
		})
	}
	registry, _ := NewFactoryRegistry(FactoryRegistration{API: api, Factory: customFactory(api, nil)})
	collection, _ := NewWithRegistry(registry, ProviderConfig{ID: "provider", API: API(api)})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := collection.GeneratorContext(ctx, llm.ModelInfo{Provider: "provider", Model: "model", API: api}); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled error = %v", err)
	}
}

func TestCustomFactoryManagedOAuthRotation(t *testing.T) {
	const api llm.API = "custom-oauth"
	store, _ := NewMemoryCredentialStore(map[string]StoredCredential{"provider": {Type: CredentialOAuth, AccessToken: "first"}})
	manager, _ := NewCredentialManager(store)
	seen := make(chan string, 2)
	factory := func(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
		return &registryGenerator{info: llm.ModelInfo{Provider: config.Provider, Model: config.Model.Model, API: api}, generate: func(ctx context.Context, _ llm.Request) (*llm.Response, error) {
			credential, found, err := config.ResolveCredential(ctx)
			if err != nil || !found {
				return nil, err
			}
			seen <- credential.AccessToken
			return &llm.Response{Message: llm.Message{Role: llm.RoleAssistant}}, nil
		}}, nil
	}
	registry, _ := NewFactoryRegistry(FactoryRegistration{API: api, Factory: factory})
	collection, _ := NewWithCredentialManagerAndRegistry(manager, registry, ProviderConfig{ID: "provider", API: API(api)})
	generator, err := collection.Generator(llm.ModelInfo{Provider: "provider", Model: "model", API: api})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	_, _ = generator.Generate(context.Background(), request)
	_ = store.Modify(context.Background(), "provider", func(current StoredCredential, _ bool) (*StoredCredential, error) {
		current.AccessToken = "second"
		return &current, nil
	})
	_, _ = generator.Generate(context.Background(), request)
	if first, second := <-seen, <-seen; first != "first" || second != "second" {
		t.Fatalf("OAuth tokens = %q/%q", first, second)
	}
}

func TestCustomFactoryManagedCredentialRotation(t *testing.T) {
	const api llm.API = "custom"
	store, _ := NewMemoryCredentialStore(map[string]StoredCredential{"provider": {Type: CredentialAPIKey, APIKey: "first"}})
	manager, _ := NewCredentialManager(store)
	var mu sync.Mutex
	var seen []string
	factory := func(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
		return &registryGenerator{info: llm.ModelInfo{Provider: config.Provider, Model: config.Model.Model, API: api}, generate: func(ctx context.Context, _ llm.Request) (*llm.Response, error) {
			credential, found, err := config.ResolveCredential(ctx)
			if err != nil || !found {
				return nil, err
			}
			mu.Lock()
			seen = append(seen, credential.APIKey)
			mu.Unlock()
			return &llm.Response{Message: llm.Message{Role: llm.RoleAssistant}}, nil
		}}, nil
	}
	registry, _ := NewFactoryRegistry(FactoryRegistration{API: api, Factory: factory})
	collection, _ := NewWithCredentialManagerAndRegistry(manager, registry, ProviderConfig{ID: "provider", API: API(api)})
	generator, err := collection.Generator(llm.ModelInfo{Provider: "provider", Model: "model", API: api})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	if _, err := generator.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	_ = store.Modify(context.Background(), "provider", func(current StoredCredential, _ bool) (*StoredCredential, error) {
		current.APIKey = "second"
		return &current, nil
	})
	if _, err := generator.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 2 || seen[0] != "first" || seen[1] != "second" {
		t.Fatalf("credentials = %v", seen)
	}
}
