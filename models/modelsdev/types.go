package modelsdev

// ProviderEntry represents one provider record from models.dev.
type ProviderEntry struct {
	ID     string                `json:"id"`
	Name   string                `json:"name"`
	API    *string               `json:"api,omitempty"`
	NPM    string                `json:"npm,omitempty"`
	Doc    string                `json:"doc,omitempty"`
	Env    []string              `json:"env,omitempty"`
	Models map[string]ModelEntry `json:"models"`
}

// ModelEntry represents one model record from models.dev.
type ModelEntry struct {
	ID               string            `json:"id"`
	Name             string            `json:"name"`
	Description      string            `json:"description,omitempty"`
	Family           string            `json:"family,omitempty"`
	Attachment       bool              `json:"attachment,omitempty"`
	Reasoning        bool              `json:"reasoning,omitempty"`
	ReasoningOptions []ReasoningOption `json:"reasoning_options,omitempty"`
	ToolCall         bool              `json:"tool_call,omitempty"`
	StructuredOutput bool              `json:"structured_output,omitempty"`
	Temperature      bool              `json:"temperature,omitempty"`
	Knowledge        string            `json:"knowledge,omitempty"`
	ReleaseDate      string            `json:"release_date,omitempty"`
	LastUpdated      string            `json:"last_updated,omitempty"`
	Modalities       *Modalities       `json:"modalities,omitempty"`
	OpenWeights      bool              `json:"open_weights,omitempty"`
	Limit            *Limit            `json:"limit,omitempty"`
	Cost             *Cost             `json:"cost,omitempty"`
}

// Modalities describes input and output modality support.
type Modalities struct {
	Input  []string `json:"input,omitempty"`
	Output []string `json:"output,omitempty"`
}

// Limit describes model context and output token limits.
type Limit struct {
	Context int `json:"context"`
	Output  int `json:"output"`
}

// Cost describes token pricing in USD per million tokens.
type Cost struct {
	Input           *float64        `json:"input,omitempty"`
	Output          *float64        `json:"output,omitempty"`
	CacheRead       *float64        `json:"cache_read,omitempty"`
	CacheWrite      *float64        `json:"cache_write,omitempty"`
	InputAudio      *float64        `json:"input_audio,omitempty"`
	OutputAudio     *float64        `json:"output_audio,omitempty"`
	Reasoning       *float64        `json:"reasoning,omitempty"`
	Tiers           []CostTierEntry `json:"tiers,omitempty"`
	ContextOver200k *CostRates      `json:"context_over_200k,omitempty"`
}

// CostTierEntry describes tiered pricing based on context thresholds.
type CostTierEntry struct {
	InputTokensAbove *int           `json:"input_tokens_above,omitempty"`
	Input            *float64       `json:"input,omitempty"`
	Output           *float64       `json:"output,omitempty"`
	CacheRead        *float64       `json:"cache_read,omitempty"`
	CacheWrite       *float64       `json:"cache_write,omitempty"`
	Tier             *TierCondition `json:"tier,omitempty"`
}

// TierCondition describes the condition for a pricing tier.
type TierCondition struct {
	Type string `json:"type"`
	Size int    `json:"size"`
}

// CostRates describes rates for a context tier.
type CostRates struct {
	Input      *float64 `json:"input,omitempty"`
	Output     *float64 `json:"output,omitempty"`
	CacheRead  *float64 `json:"cache_read,omitempty"`
	CacheWrite *float64 `json:"cache_write,omitempty"`
}

// ReasoningOption describes reasoning configuration options.
type ReasoningOption struct {
	Type   string   `json:"type"`
	Values []string `json:"values,omitempty"`
}

// Dataset is the top-level payload from models.dev containing all providers.
type Dataset map[string]ProviderEntry
