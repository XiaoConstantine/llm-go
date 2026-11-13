// Package catalogmeta holds protocol rules shared by catalog import and generation.
package catalogmeta

import llm "github.com/XiaoConstantine/llm-go"

// ProviderAPI looks up the protocol of a canonical built-in catalog provider.
// Unknown providers are not assumed to implement any particular protocol.
func ProviderAPI(provider string) (llm.API, bool) {
	api, ok := providerAPIs[provider]
	return api, ok
}

var providerAPIs = map[string]llm.API{
	"amazon-bedrock": llm.APIBedrockConverseStream,
	"anthropic":      llm.APIAnthropicMessages,
	"cerebras":       llm.APIOpenAIChatCompletions,
	"deepseek":       llm.APIOpenAIChatCompletions,
	"fireworks":      llm.APIOpenAIChatCompletions,
	"google":         llm.APIGeminiGenerateContent,
	"google-vertex":  llm.APIGoogleVertex,
	"groq":           llm.APIOpenAIChatCompletions,
	"mistral":        llm.APIMistralConversations,
	"openai":         llm.APIOpenAIResponses,
	"openai-codex":   llm.APIOpenAICodexResponses,
	"openrouter":     llm.APIOpenAIChatCompletions,
	"xai":            llm.APIOpenAIResponses,
}

// SupportsCapability reports protocol support, not support by a particular model.
// Unknown APIs and capabilities are not supported.
func SupportsCapability(api llm.API, capability llm.Capability) bool {
	switch api {
	case llm.APIOpenAIChatCompletions:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityJSON || capability == llm.CapabilityVision || capability == llm.CapabilityAudio ||
			capability == llm.CapabilityDocuments
	case llm.APIOpenAIResponses:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityJSON || capability == llm.CapabilityVision || capability == llm.CapabilityDocuments
	case llm.APIOpenAICodexResponses:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityVision || capability == llm.CapabilityAudio
	case llm.APIGeminiGenerateContent, llm.APIGoogleVertex:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityJSON || capability == llm.CapabilityVision || capability == llm.CapabilityAudio
	case llm.APIAnthropicMessages:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityVision || capability == llm.CapabilityDocuments
	case llm.APIMistralConversations, llm.APIBedrockConverseStream:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools || capability == llm.CapabilityVision
	default:
		return false
	}
}
