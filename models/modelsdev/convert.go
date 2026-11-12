package modelsdev

import (
	"fmt"
	"slices"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/catalogmeta"
)

// DefaultProviderAPI returns the default wire protocol for a known provider ID.
// Unknown providers retain the OpenAI Chat Completions fallback; use
// WithAPIResolver when a custom provider uses another protocol.
func DefaultProviderAPI(provider string) llm.API {
	provider = strings.TrimSpace(provider)
	if provider == "codex" {
		provider = "openai-codex"
	}
	if api, ok := catalogmeta.ProviderAPI(provider); ok {
		return api
	}
	return llm.APIOpenAIChatCompletions
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
		chat := &llm.OpenAIChatCompatibility{LongCacheRetention: llm.CompatibilityEnabled}
		if model.Modalities != nil && slices.Contains(model.Modalities.Input, "image") {
			chat.ToolResultImageFallback = llm.CompatibilityEnabled
		}
		switch provider {
		case "deepseek":
			chat.MaxTokensField = llm.MaxTokensFieldLegacy
			chat.InstructionRole = llm.InstructionRoleSystem
			chat.ReasoningContentReplay = llm.CompatibilityEnabled
			chat.ThinkingFormat = llm.ThinkingFormatDeepSeek
		case "openrouter":
			chat.InstructionRole = llm.InstructionRoleSystem
			chat.ThinkingFormat = llm.ThinkingFormatOpenRouter
			chat.SessionAffinity = llm.CompatibilityEnabled
			chat.SessionAffinityFormat = llm.SessionAffinityOpenRouter
		}
		return &llm.ModelCompatibility{OpenAIChat: chat}
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
	if compatibility.Gemini != nil {
		value := *compatibility.Gemini
		clone.Gemini = &value
	}
	if compatibility.Anthropic != nil {
		value := *compatibility.Anthropic
		clone.Anthropic = &value
	}
	return &clone
}

// ConvertModel maps a models.dev ModelEntry into an llm.Model. It rejects invalid
// prices, including base or tier cache-read rates above their input rates.
func ConvertModel(provider string, entry ModelEntry, api llm.API, compat *llm.ModelCompatibility) (llm.Model, error) {
	return convertModel(provider, entry, api, compat, nil)
}

func convertModel(provider string, entry ModelEntry, api llm.API, compat *llm.ModelCompatibility, baselineCost *llm.ModelCost) (llm.Model, error) {
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

	cost := convertCost(entry.Cost, contextWindow, baselineCost)
	if cost != nil {
		if err := cost.Validate(); err != nil {
			return llm.Model{}, fmt.Errorf("cost: %w", err)
		}
		if err := catalogmeta.ValidateCacheReadRate(cost.Input, cost.CacheRead); err != nil {
			return llm.Model{}, fmt.Errorf("cost: %w", err)
		}
		for index, tier := range cost.Tiers {
			if err := catalogmeta.ValidateCacheReadRate(tier.Input, tier.CacheRead); err != nil {
				return llm.Model{}, fmt.Errorf("cost tier %d: %w", index, err)
			}
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

	if entry.ToolCall && catalogmeta.SupportsCapability(api, llm.CapabilityTools) {
		caps = append(caps, llm.CapabilityTools)
	}

	if entry.StructuredOutput && catalogmeta.SupportsCapability(api, llm.CapabilityJSON) {
		caps = append(caps, llm.CapabilityJSON)
	}

	if entry.Modalities != nil {
		if slices.Contains(entry.Modalities.Input, "image") && catalogmeta.SupportsCapability(api, llm.CapabilityVision) {
			caps = append(caps, llm.CapabilityVision)
		}
		if slices.Contains(entry.Modalities.Input, "audio") && catalogmeta.SupportsCapability(api, llm.CapabilityAudio) {
			caps = append(caps, llm.CapabilityAudio)
		}
	}

	return caps
}

func convertCost(entry *Cost, contextWindow int, baseline *llm.ModelCost) *llm.ModelCost {
	var cost *llm.ModelCost
	if baseline != nil {
		clone := *baseline
		clone.Tiers = slices.DeleteFunc(slices.Clone(baseline.Tiers), func(tier llm.ModelCostTier) bool {
			return contextWindow > 0 && tier.InputTokensAbove >= contextWindow
		})
		cost = &clone
	}
	if entry == nil {
		return cost
	}

	hasBase := entry.Input != nil || entry.Output != nil || entry.CacheRead != nil || entry.CacheWrite != nil
	hasTiers := len(entry.Tiers) > 0 || entry.ContextOver200k != nil

	if !hasBase && !hasTiers {
		return cost
	}

	if cost == nil {
		cost = &llm.ModelCost{}
	}
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
	cost.CacheRead = catalogmeta.ClampRetainedCacheRead(cost.Input, cost.CacheRead, entry.CacheRead)

	// Omitted tiers are not a request to remove curated tiered pricing.
	if !hasTiers {
		return cost
	}
	var tiers []llm.ModelCostTier
	mergeTier := func(threshold int, input, output, cacheRead, cacheWrite *float64) {
		if threshold <= 0 || (contextWindow > 0 && threshold >= contextWindow) {
			return
		}
		tier := llm.ModelCostTier{InputTokensAbove: threshold, Input: cost.Input, Output: cost.Output, CacheRead: cost.CacheRead, CacheWrite: cost.CacheWrite}
		for _, baselineTier := range cost.Tiers {
			if baselineTier.InputTokensAbove == threshold {
				tier = baselineTier
				break
			}
		}
		if input != nil {
			tier.Input = *input
		}
		if output != nil {
			tier.Output = *output
		}
		if cacheRead != nil {
			tier.CacheRead = *cacheRead
		}
		if cacheWrite != nil {
			tier.CacheWrite = *cacheWrite
		}
		tier.CacheRead = catalogmeta.ClampRetainedCacheRead(tier.Input, tier.CacheRead, cacheRead)
		tiers = append(tiers, tier)
	}
	for _, tier := range entry.Tiers {
		threshold := 0
		if tier.InputTokensAbove != nil {
			threshold = *tier.InputTokensAbove
		} else if tier.Tier != nil {
			threshold = tier.Tier.Size
		}
		mergeTier(threshold, tier.Input, tier.Output, tier.CacheRead, tier.CacheWrite)
	}
	if len(tiers) == 0 && entry.ContextOver200k != nil {
		rates := entry.ContextOver200k
		mergeTier(200000, rates.Input, rates.Output, rates.CacheRead, rates.CacheWrite)
	}
	if len(tiers) > 0 {
		cost.Tiers = tiers
	}

	return cost
}
