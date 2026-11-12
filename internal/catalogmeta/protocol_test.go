package catalogmeta

import (
	"slices"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestProviderProtocolsAndCapabilities(t *testing.T) {
	for _, test := range []struct {
		provider     string
		api          llm.API
		capabilities []llm.Capability
	}{
		{"amazon-bedrock", llm.APIBedrockConverseStream, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision}},
		{"anthropic", llm.APIAnthropicMessages, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision}},
		{"mistral", llm.APIMistralConversations, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision}},
		{"google", llm.APIGeminiGenerateContent, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON, llm.CapabilityAudio}},
		{"google-vertex", llm.APIGoogleVertex, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON, llm.CapabilityAudio}},
		{"openai", llm.APIOpenAIResponses, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON}},
		{"xai", llm.APIOpenAIResponses, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON}},
		{"openai-codex", llm.APIOpenAICodexResponses, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityAudio}},
		{"cerebras", llm.APIOpenAIChatCompletions, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON, llm.CapabilityAudio}},
		{"deepseek", llm.APIOpenAIChatCompletions, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON, llm.CapabilityAudio}},
		{"fireworks", llm.APIOpenAIChatCompletions, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON, llm.CapabilityAudio}},
		{"groq", llm.APIOpenAIChatCompletions, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON, llm.CapabilityAudio}},
		{"openrouter", llm.APIOpenAIChatCompletions, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON, llm.CapabilityAudio}},
		{"unknown", "", nil},
	} {
		t.Run(test.provider, func(t *testing.T) {
			api, known := ProviderAPI(test.provider)
			if api != test.api || known != (test.api != "") {
				t.Fatalf("ProviderAPI = %q, %t", api, known)
			}
			for _, capability := range []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON, llm.CapabilityAudio, "unknown"} {
				if got := SupportsCapability(api, capability); got != slices.Contains(test.capabilities, capability) {
					t.Errorf("SupportsCapability(%q, %q) = %t", api, capability, got)
				}
				if SupportsCapability("custom-api", capability) {
					t.Errorf("unknown API accepts %q", capability)
				}
			}
		})
	}
}
