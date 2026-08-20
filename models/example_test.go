package models_test

import (
	"fmt"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/models"
)

func Example() {
	collection, err := models.New(models.ProviderConfig{
		ID:  "openai",
		API: models.OpenAIChatCompletions,
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
