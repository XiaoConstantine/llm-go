package models_test

import (
	"context"
	"fmt"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/models"
)

func ExampleCollection() {
	collection, err := models.New(models.ProviderConfig{
		ID:     "openai",
		API:    models.OpenAIResponses,
		APIKey: "key",
	})
	if err != nil {
		panic(err)
	}

	generator, err := collection.Generator(llm.ModelInfo{
		Provider:     "openai",
		Model:        "gpt-model",
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
	})
	if err != nil {
		panic(err)
	}

	info := generator.Info()
	fmt.Println(info.Provider, info.Model)
	// Output: openai gpt-model
}

type exampleGenerator struct{ info llm.ModelInfo }

func (g *exampleGenerator) Info() llm.ModelInfo { return g.info }
func (g *exampleGenerator) Generate(context.Context, llm.Request) (*llm.Response, error) {
	return &llm.Response{Message: llm.Message{Role: llm.RoleAssistant}}, nil
}
func (g *exampleGenerator) Stream(context.Context, llm.Request) (llm.Stream, error) {
	return nil, fmt.Errorf("streaming is not implemented")
}

func ExampleNewFactoryRegistry() {
	const customAPI llm.API = "acme-generate"
	registry, _ := models.NewFactoryRegistry(models.FactoryRegistration{API: customAPI, Factory: func(ctx context.Context, config models.GeneratorFactoryConfig) (llm.Generator, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return &exampleGenerator{info: llm.ModelInfo{Provider: config.Provider, Model: config.Model.Model, API: customAPI, Capabilities: []llm.Capability{llm.CapabilityGeneration}}}, nil
	}})
	collection, _ := models.NewWithRegistry(registry, models.ProviderConfig{
		ID: "acme", API: models.OpenAIChatCompletions,
		AdditionalAPIs: []models.ProviderAPIConfig{{API: customAPI}},
	})
	generator, _ := collection.Generator(llm.ModelInfo{Provider: "acme", Model: "special", API: customAPI})
	fmt.Println(generator.Info().API)
	// Output: acme-generate
}

func ExampleBuiltinProvider() {
	profile, ok := models.BuiltinProvider(models.ProviderDeepSeek)
	if !ok {
		panic("missing provider profile")
	}
	config := profile.Config("api-key")
	collection, err := models.New(config)
	if err != nil {
		panic(err)
	}
	generator, err := collection.Generator(llm.ModelInfo{Provider: models.ProviderDeepSeek, Model: "deepseek-model"})
	if err != nil {
		panic(err)
	}
	fmt.Println(generator.Info().Provider, profile.BaseURL())
	// Output: deepseek https://api.deepseek.com
}

func ExampleBuiltinCatalog() {
	model, ok := models.BuiltinCatalog().Model(models.ProviderDeepSeek, "deepseek-v4-flash")
	fmt.Println(ok, model.Name, model.ContextWindow, model.Reasoning)
	// Output: true DeepSeek V4 Flash 1000000 true
}

func ExampleCatalogManager() {
	baseline, _ := models.NewCatalog(llm.Model{Provider: "acme", ID: "static", API: llm.APIOpenAIResponses})
	store, _ := models.NewMemoryCatalogStore(nil)
	manager, _ := models.NewCatalogManager(models.CatalogManagerConfig{
		Baseline: baseline, Store: store,
		Providers: []models.CatalogProvider{{Provider: "acme", Source: models.CatalogSourceFunc(func(ctx context.Context, request models.CatalogFetchRequest) (models.CatalogFetchResponse, error) {
			if err := ctx.Err(); err != nil {
				return models.CatalogFetchResponse{}, err
			}
			if request.ETag == `"v1"` {
				return models.CatalogFetchResponse{NotModified: true}, nil
			}
			return models.CatalogFetchResponse{Models: []llm.Model{{Provider: "acme", ID: "dynamic", API: llm.APIOpenAIChatCompletions}}, ETag: `"v1"`}, nil
		})}},
		Now: func() time.Time { return time.Unix(1, 0) },
	})
	manager.Refresh(context.Background(), models.CatalogRefreshOptions{})
	_, dynamic := manager.Model("acme", "dynamic")
	fmt.Println(dynamic)
	// Output: true
}

func ExampleCatalogManager_Available() {
	baseline, _ := models.NewCatalog(llm.Model{Provider: "acme", ID: "basic", API: llm.APIOpenAIResponses})
	credentials, _ := models.NewMemoryCredentialStore(map[string]models.StoredCredential{
		"acme": {Type: models.CredentialAPIKey, APIKey: "secret", Attributes: map[string]string{"tier": "basic"}},
	})
	managerCredentials, _ := models.NewCredentialManager(credentials)
	manager, _ := models.NewCatalogManager(models.CatalogManagerConfig{Baseline: baseline, Credentials: managerCredentials})
	available, _ := manager.Available(context.Background(), "acme")
	fmt.Println(available[0].ID)
	// Output: basic
}

func ExampleCatalog() {
	catalog, err := models.NewCatalog(llm.Model{
		Provider:        "openai",
		ID:              "gpt-model",
		Name:            "GPT Model",
		API:             llm.APIOpenAIResponses,
		Capabilities:    []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
		ContextWindow:   128_000,
		MaxOutputTokens: 16_384,
		Reasoning:       true,
		Compatibility: &llm.ModelCompatibility{OpenAIResponses: &llm.OpenAIResponsesCompatibility{
			StrictTools: llm.CompatibilityEnabled,
		}},
	})
	if err != nil {
		panic(err)
	}

	info, ok := catalog.Model("openai", "gpt-model")
	fmt.Println(ok, info.Name, info.ContextWindow)
	// Output: true GPT Model 128000
}
