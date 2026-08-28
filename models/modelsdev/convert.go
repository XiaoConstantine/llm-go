package modelsdev

import (
	"fmt"
	"slices"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
)

// DefaultProviderAPI returns the default wire protocol for a known provider ID.
func DefaultProviderAPI(provider string) llm.API {
	switch strings.TrimSpace(provider) {
	case "anthropic":
		return llm.APIAnthropicMessages
	case "openai", "xai":
		return llm.APIOpenAIResponses
	case "google":
		return llm.APIGeminiGenerateContent
	case "openai-codex", "codex":
		return llm.APIOpenAICodexResponses
	default:
		return llm.APIOpenAIChatCompletions
	}
}

// DefaultProviderCompatibility returns default compatibility settings for a model
// validated against the resolved API protocol.
func DefaultProviderCompatibility(provider string, model ModelEntry, api llm.API) *llm.ModelCompatibility {
	provider = strings.TrimSpace(provider)
	switch api {
	case llm.APIAnthropicMessages:
		if provider == "anthropic" {
			compat := &llm.AnthropicCompatibility{
				StrictTools: llm.CompatibilityEnabled,
			}
			if model.Reasoning {
				compat.AdaptiveThinking = llm.CompatibilityEnabled
			}
			return &llm.ModelCompatibility{Anthropic: compat}
		}
	case llm.APIOpenAIResponses:
		if provider == "openai" {
			return &llm.ModelCompatibility{
				OpenAIResponses: &llm.OpenAIResponsesCompatibility{
					StrictTools:     llm.CompatibilityEnabled,
					AdditionalTools: llm.CompatibilityEnabled,
					ToolSearch:      llm.CompatibilityEnabled,
				},
			}
		}
	case llm.APIOpenAIChatCompletions:
		switch provider {
		case "deepseek":
			return &llm.ModelCompatibility{
				OpenAIChat: &llm.OpenAIChatCompatibility{
					ThinkingFormat: llm.ThinkingFormatDeepSeek,
				},
			}
		case "openrouter":
			return &llm.ModelCompatibility{
				OpenAIChat: &llm.OpenAIChatCompatibility{
					ThinkingFormat:        llm.ThinkingFormatOpenRouter,
					SessionAffinityFormat: llm.SessionAffinityOpenRouter,
				},
			}
		}
	}
	return nil
}

func cloneCompatibility(compatibility *llm.ModelCompatibility) *llm.ModelCompatibility {
	if compatibility == nil {
		return nil
	}
	clone := *compatibility
	if compatibility.OpenAIChat != nil {
		value := *compatibility.OpenAIChat
		clone.OpenAIChat = &value
	}
	if compatibility.OpenAIResponses != nil {
		value := *compatibility.OpenAIResponses
		clone.OpenAIResponses = &value
	}
	if compatibility.Anthropic != nil {
		value := *compatibility.Anthropic
		clone.Anthropic = &value
	}
	return &clone
}

// ConvertModel maps a models.dev ModelEntry into an llm.Model.
func ConvertModel(provider string, entry ModelEntry, api llm.API, compat *llm.ModelCompatibility) (llm.Model, error) {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return llm.Model{}, fmt.Errorf("provider must not be empty")
	}
	id := strings.TrimSpace(entry.ID)
	if id == "" {
		return llm.Model{}, fmt.Errorf("model ID must not be empty")
	}
	name := strings.TrimSpace(entry.Name)
	if name == "" {
		name = id
	}
	if api == "" {
		api = DefaultProviderAPI(provider)
	}

	capabilities := deriveCapabilities(entry, api)

	var contextWindow, maxOutputTokens int
	if entry.Limit != nil {
		contextWindow = entry.Limit.Context
		maxOutputTokens = entry.Limit.Output
	}

	if contextWindow < 0 {
		return llm.Model{}, fmt.Errorf("context window must not be negative: %d", contextWindow)
	}
	if maxOutputTokens < 0 {
		return llm.Model{}, fmt.Errorf("max output tokens must not be negative: %d", maxOutputTokens)
	}
	if contextWindow > 0 && maxOutputTokens > contextWindow {
		return llm.Model{}, fmt.Errorf("max output tokens (%d) must not exceed context window (%d)", maxOutputTokens, contextWindow)
	}

	cost := convertCost(entry.Cost, contextWindow)
	if cost != nil {
		if err := cost.Validate(); err != nil {
			return llm.Model{}, fmt.Errorf("cost: %w", err)
		}
	}

	clonedCompat := cloneCompatibility(compat)
	if clonedCompat != nil {
		if err := clonedCompat.Validate(api); err != nil {
			return llm.Model{}, fmt.Errorf("compatibility for API %q: %w", api, err)
		}
	}

	model := llm.Model{
		Provider:        provider,
		ID:              id,
		Name:            name,
		API:             api,
		Reasoning:       entry.Reasoning,
		Capabilities:    capabilities,
		ContextWindow:   contextWindow,
		MaxOutputTokens: maxOutputTokens,
		Cost:            cost,
		Compatibility:   clonedCompat,
	}

	return model, nil
}

func deriveCapabilities(entry ModelEntry, api llm.API) []llm.Capability {
	var caps []llm.Capability

	// Streaming is standard on generation endpoints
	caps = append(caps, llm.CapabilityStreaming)

	if entry.ToolCall && supportsCapability(api, llm.CapabilityTools) {
		caps = append(caps, llm.CapabilityTools)
	}

	if entry.StructuredOutput && supportsCapability(api, llm.CapabilityJSON) {
		caps = append(caps, llm.CapabilityJSON)
	}

	if entry.Modalities != nil {
		if slices.Contains(entry.Modalities.Input, "image") && supportsCapability(api, llm.CapabilityVision) {
			caps = append(caps, llm.CapabilityVision)
		}
		if slices.Contains(entry.Modalities.Input, "audio") && supportsCapability(api, llm.CapabilityAudio) {
			caps = append(caps, llm.CapabilityAudio)
		}
	}

	return caps
}

func supportsCapability(api llm.API, capability llm.Capability) bool {
	switch api {
	case llm.APIOpenAIChatCompletions:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityJSON || capability == llm.CapabilityVision || capability == llm.CapabilityAudio
	case llm.APIOpenAIResponses:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityJSON || capability == llm.CapabilityVision
	case llm.APIOpenAICodexResponses:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityVision || capability == llm.CapabilityAudio
	case llm.APIAnthropicMessages:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools || capability == llm.CapabilityVision
	case llm.APIGeminiGenerateContent:
		return capability == llm.CapabilityStreaming || capability == llm.CapabilityTools ||
			capability == llm.CapabilityJSON || capability == llm.CapabilityVision || capability == llm.CapabilityAudio
	default:
		return true
	}
}

func convertCost(entry *Cost, contextWindow int) *llm.ModelCost {
	if entry == nil {
		return nil
	}

	hasBase := entry.Input != nil || entry.Output != nil || entry.CacheRead != nil || entry.CacheWrite != nil
	hasTiers := len(entry.Tiers) > 0 || entry.ContextOver200k != nil

	if !hasBase && !hasTiers {
		return nil
	}

	cost := &llm.ModelCost{}
	if entry.Input != nil {
		cost.Input = *entry.Input
	}
	if entry.Output != nil {
		cost.Output = *entry.Output
	}
	if entry.CacheRead != nil {
		cost.CacheRead = *entry.CacheRead
	}
	if entry.CacheWrite != nil {
		cost.CacheWrite = *entry.CacheWrite
	}

	for _, t := range entry.Tiers {
		threshold := 0
		if t.InputTokensAbove != nil {
			threshold = *t.InputTokensAbove
		} else if t.Tier != nil && t.Tier.Size > 0 {
			threshold = t.Tier.Size
		}
		if threshold <= 0 {
			continue
		}
		if contextWindow > 0 && threshold >= contextWindow {
			continue
		}
		tierCost := llm.ModelCostTier{
			InputTokensAbove: threshold,
			Input:            cost.Input,
			Output:           cost.Output,
			CacheRead:        cost.CacheRead,
			CacheWrite:       cost.CacheWrite,
		}
		if t.Input != nil {
			tierCost.Input = *t.Input
		}
		if t.Output != nil {
			tierCost.Output = *t.Output
		}
		if t.CacheRead != nil {
			tierCost.CacheRead = *t.CacheRead
		}
		if t.CacheWrite != nil {
			tierCost.CacheWrite = *t.CacheWrite
		}
		cost.Tiers = append(cost.Tiers, tierCost)
	}

	if len(cost.Tiers) == 0 && entry.ContextOver200k != nil {
		threshold := 200000
		if contextWindow == 0 || threshold < contextWindow {
			rates := entry.ContextOver200k
			tierCost := llm.ModelCostTier{
				InputTokensAbove: threshold,
				Input:            cost.Input,
				Output:           cost.Output,
				CacheRead:        cost.CacheRead,
				CacheWrite:       cost.CacheWrite,
			}
			if rates.Input != nil {
				tierCost.Input = *rates.Input
			}
			if rates.Output != nil {
				tierCost.Output = *rates.Output
			}
			if rates.CacheRead != nil {
				tierCost.CacheRead = *rates.CacheRead
			}
			if rates.CacheWrite != nil {
				tierCost.CacheWrite = *rates.CacheWrite
			}
			cost.Tiers = append(cost.Tiers, tierCost)
		}
	}

	return cost
}
