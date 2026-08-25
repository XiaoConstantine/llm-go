package models_test

import (
	"fmt"

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
