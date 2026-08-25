package models

import (
	"fmt"
	"strings"
	"unicode/utf8"

	llm "github.com/XiaoConstantine/llm-go"
)

// Catalog is an immutable collection of model metadata. It is safe for
// concurrent use.
type Catalog struct {
	models     []llm.Model
	byProvider map[string]map[string]int
}

// NewCatalog constructs a Catalog. Provider, ID, and API are required and each
// provider/model pair must be unique. Name defaults to ID. Generation is added
// as the first capability when omitted.
func NewCatalog(models ...llm.Model) (*Catalog, error) {
	catalog := &Catalog{
		models:     make([]llm.Model, 0, len(models)),
		byProvider: make(map[string]map[string]int),
	}
	for index, model := range models {
		normalized, err := normalizeModel(model)
		if err != nil {
			return nil, catalogError(strings.TrimSpace(model.Provider), "models[%d]: %v", index, err)
		}
		providerModels := catalog.byProvider[normalized.Provider]
		if providerModels == nil {
			providerModels = make(map[string]int)
			catalog.byProvider[normalized.Provider] = providerModels
		}
		if _, exists := providerModels[normalized.ID]; exists {
			return nil, catalogError(normalized.Provider, "models[%d]: model %q is configured more than once", index, normalized.ID)
		}
		providerModels[normalized.ID] = len(catalog.models)
		catalog.models = append(catalog.models, normalized)
	}
	return catalog, nil
}

// Models returns catalog entries for provider in catalog order. An empty
// provider returns all entries. The returned values and their capability slices
// are owned by the caller.
func (c *Catalog) Models(provider string) []llm.Model {
	if c == nil {
		return nil
	}
	provider = strings.TrimSpace(provider)
	models := make([]llm.Model, 0, len(c.models))
	for _, model := range c.models {
		if provider == "" || model.Provider == provider {
			models = append(models, cloneModel(model))
		}
	}
	return models
}

// Model looks up one model by provider and model ID. The returned value and its
// capability slice are owned by the caller.
func (c *Catalog) Model(provider, model string) (llm.Model, bool) {
	if c == nil {
		return llm.Model{}, false
	}
	providerModels := c.byProvider[strings.TrimSpace(provider)]
	index, exists := providerModels[strings.TrimSpace(model)]
	if !exists {
		return llm.Model{}, false
	}
	return cloneModel(c.models[index]), true
}

func normalizeModel(model llm.Model) (llm.Model, error) {
	model.Provider = strings.TrimSpace(model.Provider)
	model.ID = strings.TrimSpace(model.ID)
	model.Name = strings.TrimSpace(model.Name)
	model.API = llm.API(strings.TrimSpace(string(model.API)))
	if model.Provider == "" {
		return llm.Model{}, fmt.Errorf("provider must not be empty")
	}
	if !utf8.ValidString(model.Provider) {
		return llm.Model{}, fmt.Errorf("provider must be valid UTF-8")
	}
	if model.ID == "" {
		return llm.Model{}, fmt.Errorf("model ID must not be empty")
	}
	if !utf8.ValidString(model.ID) {
		return llm.Model{}, fmt.Errorf("model ID must be valid UTF-8")
	}
	if model.Name == "" {
		model.Name = model.ID
	}
	if !utf8.ValidString(model.Name) {
		return llm.Model{}, fmt.Errorf("name must be valid UTF-8")
	}
	if model.API == "" {
		return llm.Model{}, fmt.Errorf("API must not be empty")
	}
	if !utf8.ValidString(string(model.API)) {
		return llm.Model{}, fmt.Errorf("API must be valid UTF-8")
	}
	if model.ContextWindow < 0 {
		return llm.Model{}, fmt.Errorf("context window must not be negative")
	}
	if model.MaxOutputTokens < 0 {
		return llm.Model{}, fmt.Errorf("max output tokens must not be negative")
	}
	if model.ContextWindow != 0 && model.MaxOutputTokens > model.ContextWindow {
		return llm.Model{}, fmt.Errorf("max output tokens must not exceed context window")
	}
	if model.Cost != nil {
		if err := model.Cost.Validate(); err != nil {
			return llm.Model{}, fmt.Errorf("cost: %w", err)
		}
		cost := *model.Cost
		cost.Tiers = append([]llm.ModelCostTier(nil), model.Cost.Tiers...)
		model.Cost = &cost
	}
	if err := model.Compatibility.Validate(model.API); err != nil {
		return llm.Model{}, fmt.Errorf("compatibility: %w", err)
	}
	model.Compatibility = cloneCompatibility(model.Compatibility)

	capabilities := make([]llm.Capability, 0, len(model.Capabilities)+1)
	seen := make(map[llm.Capability]struct{}, len(model.Capabilities)+1)
	add := func(capability llm.Capability) {
		if _, exists := seen[capability]; exists {
			return
		}
		seen[capability] = struct{}{}
		capabilities = append(capabilities, capability)
	}
	add(llm.CapabilityGeneration)
	for index, capability := range model.Capabilities {
		switch capability {
		case llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools,
			llm.CapabilityJSON, llm.CapabilityVision, llm.CapabilityAudio:
		default:
			return llm.Model{}, fmt.Errorf("capabilities[%d] %q is invalid", index, capability)
		}
		add(capability)
	}
	model.Capabilities = capabilities
	return model, nil
}

func cloneModel(model llm.Model) llm.Model {
	model.Capabilities = append([]llm.Capability(nil), model.Capabilities...)
	if model.Cost != nil {
		cost := *model.Cost
		cost.Tiers = append([]llm.ModelCostTier(nil), model.Cost.Tiers...)
		model.Cost = &cost
	}
	model.Compatibility = cloneCompatibility(model.Compatibility)
	return model
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

func catalogError(provider, format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindInvalidRequest,
		Op:       "catalog",
		Provider: provider,
		Err:      fmt.Errorf(format, args...),
	}
}
