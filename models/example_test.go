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

func ExampleCatalog() {
	catalog, err := models.NewCatalog(llm.Model{
		Provider:        "openai",
		ID:              "gpt-model",
		Name:            "GPT Model",
		API:             llm.APIOpenAIResponses,
		Capabilities:    []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
		ContextWindow:   128_000,
		MaxOutputTokens: 16_384,
	})
	if err != nil {
		panic(err)
	}

	info, ok := catalog.Model("openai", "gpt-model")
	fmt.Println(ok, info.Name, info.ContextWindow)
	// Output: true GPT Model 128000
}
