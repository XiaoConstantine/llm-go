package main

import (
	"context"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/models/modelsdev"
)

var defaultProviderMapping = map[string]string{
	"anthropic":  "anthropic",
	"cerebras":   "cerebras",
	"deepseek":   "deepseek",
	"fireworks":  "fireworks-ai",
	"google":     "google",
	"groq":       "groq",
	"openai":     "openai",
	"openrouter": "openrouter",
	"xai":        "xai",
}

var providerAPIs = map[string]llm.API{
	"anthropic":  llm.APIAnthropicMessages,
	"cerebras":   llm.APIOpenAIChatCompletions,
	"deepseek":   llm.APIOpenAIChatCompletions,
	"fireworks":  llm.APIOpenAIChatCompletions,
	"google":     llm.APIGeminiGenerateContent,
	"groq":       llm.APIOpenAIChatCompletions,
	"openai":     llm.APIOpenAIResponses,
	"openrouter": llm.APIOpenAIChatCompletions,
	"xai":        llm.APIOpenAIResponses,
}

const maxResponseSize = 32 << 20

type sourceCatalog struct {
	SchemaVersion int           `json:"schema_version"`
	Revision      int           `json:"revision"`
	Models        []sourceModel `json:"models"`
}

type sourceModel struct {
	Provider        string               `json:"provider"`
	ID              string               `json:"id"`
	Name            string               `json:"name"`
	API             llm.API              `json:"api"`
	Reasoning       *bool                `json:"reasoning"`
	Capabilities    *[]llm.Capability    `json:"capabilities"`
	ContextWindow   *int                 `json:"context_window"`
	MaxOutputTokens *int                 `json:"max_output_tokens"`
	Cost            *sourceCost          `json:"cost,omitempty"`
	Compatibility   *sourceCompatibility `json:"compatibility,omitempty"`
}

type sourceCost struct {
	Input      *float64     `json:"input"`
	Output     *float64     `json:"output"`
	CacheRead  *float64     `json:"cache_read"`
	CacheWrite *float64     `json:"cache_write"`
	Tiers      []sourceTier `json:"tiers,omitempty"`
}

type sourceTier struct {
	InputTokensAbove *int     `json:"input_tokens_above"`
	Input            *float64 `json:"input"`
	Output           *float64 `json:"output"`
	CacheRead        *float64 `json:"cache_read"`
	CacheWrite       *float64 `json:"cache_write"`
}

type sourceCompatibility struct {
	OpenAIChat      *sourceOpenAIChat      `json:"openai_chat,omitempty"`
	OpenAIResponses *sourceOpenAIResponses `json:"openai_responses,omitempty"`
	Anthropic       *sourceAnthropic       `json:"anthropic,omitempty"`
}

type sourceOpenAIChat struct {
	MaxTokensField          llm.MaxTokensField        `json:"max_tokens_field,omitempty"`
	InstructionRole         llm.InstructionRole       `json:"instruction_role,omitempty"`
	ReasoningEffort         *bool                     `json:"reasoning_effort,omitempty"`
	StreamingUsage          *bool                     `json:"streaming_usage,omitempty"`
	FinishReason            *bool                     `json:"finish_reason,omitempty"`
	ToolResultName          *bool                     `json:"tool_result_name,omitempty"`
	ToolResultImageFallback *bool                     `json:"tool_result_image_fallback,omitempty"`
	AssistantAfterTool      *bool                     `json:"assistant_after_tool_result,omitempty"`
	ReasoningContentReplay  *bool                     `json:"reasoning_content_replay,omitempty"`
	StrictTools             *bool                     `json:"strict_tools,omitempty"`
	LongCacheRetention      *bool                     `json:"long_cache_retention,omitempty"`
	SessionAffinity         *bool                     `json:"session_affinity,omitempty"`
	SessionAffinityFormat   llm.SessionAffinityFormat `json:"session_affinity_format,omitempty"`
	ThinkingFormat          llm.ThinkingFormat        `json:"thinking_format,omitempty"`
	CacheControlFormat      llm.CacheControlFormat    `json:"cache_control_format,omitempty"`
}

type sourceOpenAIResponses struct {
	DeveloperRole           *bool                     `json:"developer_role,omitempty"`
	EncryptedReasoning      *bool                     `json:"encrypted_reasoning,omitempty"`
	StrictTools             *bool                     `json:"strict_tools,omitempty"`
	AdditionalTools         *bool                     `json:"additional_tools,omitempty"`
	ToolSearch              *bool                     `json:"tool_search,omitempty"`
	LongCacheRetention      *bool                     `json:"long_cache_retention,omitempty"`
	ExplicitPromptCacheMode *bool                     `json:"explicit_prompt_cache_mode,omitempty"`
	SessionAffinityFormat   llm.SessionAffinityFormat `json:"session_affinity_format,omitempty"`
}

type sourceAnthropic struct {
	EagerToolInputStreaming *bool `json:"eager_tool_input_streaming,omitempty"`
	LongCacheRetention      *bool `json:"long_cache_retention,omitempty"`
	SessionAffinity         *bool `json:"session_affinity,omitempty"`
	CacheControlOnTools     *bool `json:"cache_control_on_tools,omitempty"`
	Temperature             *bool `json:"temperature,omitempty"`
	AdaptiveThinking        *bool `json:"adaptive_thinking,omitempty"`
	EmptyThinkingSignature  *bool `json:"empty_thinking_signature,omitempty"`
	StrictTools             *bool `json:"strict_tools,omitempty"`
	ToolReferences          *bool `json:"tool_references,omitempty"`
}

// SyncOptions configures catalog synchronization.
type SyncOptions struct {
	Providers []string
	AddNew    bool
	DryRun    bool
}

// SyncResult summarizes modifications during synchronization.
type SyncResult struct {
	UpdatedModels []string
	AddedModels   []string
	Unchanged     int
	Total         int
}

func main() {
	catalogPath := flag.String("catalog", "catalogsource/catalog.json", "path to catalog.json")
	url := flag.String("url", modelsdev.DefaultURL, "models.dev API URL")
	file := flag.String("file", "", "path to local models.dev JSON file (optional)")
	providersFlag := flag.String("providers", "", "comma-separated list of providers to sync (default: all built-in)")
	addNew := flag.Bool("add-new", false, "add newly discovered models from models.dev")
	dryRun := flag.Bool("dry-run", false, "report changes without writing to catalog file")
	outputPath := flag.String("output", "", "output catalog file (default: same as -catalog)")
	flag.Parse()

	if *outputPath == "" {
		*outputPath = *catalogPath
	}

	var providers []string
	if *providersFlag != "" {
		for _, p := range strings.Split(*providersFlag, ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				providers = append(providers, p)
			}
		}
	}

	dataset, err := loadDataset(*file, *url)
	if err != nil {
		fatal(fmt.Errorf("load dataset: %w", err))
	}

	catalog, err := loadCatalog(*catalogPath)
	if err != nil {
		fatal(fmt.Errorf("load catalog: %w", err))
	}

	result, err := syncCatalog(catalog, dataset, SyncOptions{
		Providers: providers,
		AddNew:    *addNew,
		DryRun:    *dryRun,
	})
	if err != nil {
		fatal(fmt.Errorf("sync catalog: %w", err))
	}

	if err := validateCatalog(catalog); err != nil {
		fatal(fmt.Errorf("validate catalog: %w", err))
	}

	fmt.Printf("Sync complete:\n")
	fmt.Printf("  Updated models: %d\n", len(result.UpdatedModels))
	for _, m := range result.UpdatedModels {
		fmt.Printf("    - %s\n", m)
	}
	fmt.Printf("  Added models:   %d\n", len(result.AddedModels))
	for _, m := range result.AddedModels {
		fmt.Printf("    + %s\n", m)
	}
	fmt.Printf("  Unchanged:      %d\n", result.Unchanged)
	fmt.Printf("  Total models:   %d\n", result.Total)

	if !*dryRun && (len(result.UpdatedModels) > 0 || len(result.AddedModels) > 0) {
		if err := writeCatalog(*outputPath, catalog); err != nil {
			fatal(fmt.Errorf("write catalog: %w", err))
		}
		fmt.Printf("Updated catalog written to %s (revision %d)\n", *outputPath, catalog.Revision)
	}
}

func loadDataset(filePath, url string) (modelsdev.Dataset, error) {
	var body []byte
	var err error

	if filePath != "" {
		body, err = os.ReadFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("read file %s: %w", filePath, err)
		}
	} else {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Accept", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
		}
		body, err = io.ReadAll(io.LimitReader(resp.Body, maxResponseSize+1))
		if err != nil {
			return nil, err
		}
		if len(body) > maxResponseSize {
			return nil, fmt.Errorf("payload exceeded maximum size of %d bytes", maxResponseSize)
		}
	}

	var dataset modelsdev.Dataset
	if err := jsonv2.Unmarshal(body, &dataset); err != nil {
		return nil, fmt.Errorf("decode JSON: %w", err)
	}
	return dataset, nil
}

func loadCatalog(path string) (*sourceCatalog, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var catalog sourceCatalog
	if err := jsonv2.Unmarshal(data, &catalog, jsonv2.RejectUnknownMembers(true)); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	if catalog.SchemaVersion != 1 {
		return nil, fmt.Errorf("schema_version is %d, want 1", catalog.SchemaVersion)
	}
	return &catalog, nil
}

func validateCatalog(catalog *sourceCatalog) error {
	if catalog.SchemaVersion != 1 {
		return fmt.Errorf("schema_version is %d, want 1", catalog.SchemaVersion)
	}
	if catalog.Revision <= 0 {
		return errors.New("revision must be positive")
	}
	seen := make(map[string]map[string]struct{})
	for i, m := range catalog.Models {
		if m.Provider == "" || !utf8.ValidString(m.Provider) || strings.TrimSpace(m.Provider) != m.Provider {
			return fmt.Errorf("models[%d]: provider must be nonempty valid UTF-8 without surrounding whitespace", i)
		}
		if m.ID == "" || !utf8.ValidString(m.ID) || strings.TrimSpace(m.ID) != m.ID {
			return fmt.Errorf("models[%d]: ID must be nonempty valid UTF-8 without surrounding whitespace", i)
		}
		if m.Name == "" || !utf8.ValidString(m.Name) || strings.TrimSpace(m.Name) != m.Name {
			return fmt.Errorf("models[%d]: name must be nonempty valid UTF-8 without surrounding whitespace", i)
		}
		expectedAPI, knownProvider := providerAPIs[m.Provider]
		if !knownProvider {
			return fmt.Errorf("models[%d]: unsupported built-in provider %q", i, m.Provider)
		}
		if m.API != expectedAPI {
			return fmt.Errorf("models[%d]: provider %q requires API %q, got %q", i, m.Provider, expectedAPI, m.API)
		}
		if m.Reasoning == nil || m.Capabilities == nil || m.ContextWindow == nil || m.MaxOutputTokens == nil {
			return fmt.Errorf("models[%d] %s/%s: reasoning, capabilities, context_window, and max_output_tokens are required and must not be null", i, m.Provider, m.ID)
		}
		if *m.ContextWindow <= 0 || *m.MaxOutputTokens <= 0 || *m.MaxOutputTokens > *m.ContextWindow {
			return fmt.Errorf("models[%d] %s/%s: context and output-token limits must be positive and output must not exceed context", i, m.Provider, m.ID)
		}
		capSeen := make(map[llm.Capability]struct{}, len(*m.Capabilities))
		for _, cap := range *m.Capabilities {
			if !supportsCapability(m.API, cap) {
				return fmt.Errorf("models[%d] %s/%s: API %q does not support capability %q", i, m.Provider, m.ID, m.API, cap)
			}
			if _, exists := capSeen[cap]; exists {
				return fmt.Errorf("models[%d] %s/%s: duplicate capability %q", i, m.Provider, m.ID, cap)
			}
			capSeen[cap] = struct{}{}
		}
		if m.Cost != nil {
			if m.Cost.Input == nil || m.Cost.Output == nil || m.Cost.CacheRead == nil || m.Cost.CacheWrite == nil {
				return fmt.Errorf("models[%d] %s/%s: cost rates must not be null", i, m.Provider, m.ID)
			}
			for _, rate := range []*float64{m.Cost.Input, m.Cost.Output, m.Cost.CacheRead, m.Cost.CacheWrite} {
				if rate == nil || math.IsNaN(*rate) || math.IsInf(*rate, 0) || *rate < 0 {
					return fmt.Errorf("models[%d] %s/%s: cost rates must be finite and nonnegative", i, m.Provider, m.ID)
				}
			}
			tierThresholds := make(map[int]struct{}, len(m.Cost.Tiers))
			for ti, t := range m.Cost.Tiers {
				if t.InputTokensAbove == nil || *t.InputTokensAbove <= 0 || *t.InputTokensAbove >= *m.ContextWindow {
					return fmt.Errorf("models[%d] %s/%s: tier %d threshold must be positive and below context window", i, m.Provider, m.ID, ti)
				}
				if _, exists := tierThresholds[*t.InputTokensAbove]; exists {
					return fmt.Errorf("models[%d] %s/%s: duplicate tier threshold %d", i, m.Provider, m.ID, *t.InputTokensAbove)
				}
				tierThresholds[*t.InputTokensAbove] = struct{}{}
				for _, rate := range []*float64{t.Input, t.Output, t.CacheRead, t.CacheWrite} {
					if rate == nil || math.IsNaN(*rate) || math.IsInf(*rate, 0) || *rate < 0 {
						return fmt.Errorf("models[%d] %s/%s: tier %d rates must be finite and nonnegative", i, m.Provider, m.ID, ti)
					}
				}
			}
		}
		if m.Compatibility != nil {
			compatModel := convertSourceCompatibility(m.Compatibility)
			if err := compatModel.Validate(m.API); err != nil {
				return fmt.Errorf("models[%d] %s/%s: compatibility: %w", i, m.Provider, m.ID, err)
			}
		}
		providerModels := seen[m.Provider]
		if providerModels == nil {
			providerModels = make(map[string]struct{})
			seen[m.Provider] = providerModels
		}
		if _, exists := providerModels[m.ID]; exists {
			return fmt.Errorf("duplicate model %s/%s", m.Provider, m.ID)
		}
		providerModels[m.ID] = struct{}{}
	}
	return nil
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
		return false
	}
}

func convertSourceCompatibility(source *sourceCompatibility) *llm.ModelCompatibility {
	if source == nil {
		return nil
	}
	compat := &llm.ModelCompatibility{}
	if value := source.OpenAIChat; value != nil {
		compat.OpenAIChat = &llm.OpenAIChatCompatibility{
			MaxTokensField:           value.MaxTokensField,
			InstructionRole:          value.InstructionRole,
			ReasoningEffort:          toggle(value.ReasoningEffort),
			StreamingUsage:           toggle(value.StreamingUsage),
			FinishReason:             toggle(value.FinishReason),
			ToolResultName:           toggle(value.ToolResultName),
			ToolResultImageFallback:  toggle(value.ToolResultImageFallback),
			AssistantAfterToolResult: toggle(value.AssistantAfterTool),
			ReasoningContentReplay:   toggle(value.ReasoningContentReplay),
			StrictTools:              toggle(value.StrictTools),
			LongCacheRetention:       toggle(value.LongCacheRetention),
			SessionAffinity:          toggle(value.SessionAffinity),
			SessionAffinityFormat:    value.SessionAffinityFormat,
			ThinkingFormat:           value.ThinkingFormat,
			CacheControlFormat:       value.CacheControlFormat,
		}
	}
	if value := source.OpenAIResponses; value != nil {
		compat.OpenAIResponses = &llm.OpenAIResponsesCompatibility{
			DeveloperRole:           toggle(value.DeveloperRole),
			EncryptedReasoning:      toggle(value.EncryptedReasoning),
			StrictTools:             toggle(value.StrictTools),
			AdditionalTools:         toggle(value.AdditionalTools),
			ToolSearch:              toggle(value.ToolSearch),
			LongCacheRetention:      toggle(value.LongCacheRetention),
			ExplicitPromptCacheMode: toggle(value.ExplicitPromptCacheMode),
			SessionAffinityFormat:   value.SessionAffinityFormat,
		}
	}
	if value := source.Anthropic; value != nil {
		compat.Anthropic = &llm.AnthropicCompatibility{
			EagerToolInputStreaming: toggle(value.EagerToolInputStreaming),
			LongCacheRetention:      toggle(value.LongCacheRetention),
			SessionAffinity:         toggle(value.SessionAffinity),
			CacheControlOnTools:     toggle(value.CacheControlOnTools),
			Temperature:             toggle(value.Temperature),
			AdaptiveThinking:        toggle(value.AdaptiveThinking),
			EmptyThinkingSignature:  toggle(value.EmptyThinkingSignature),
			StrictTools:             toggle(value.StrictTools),
			ToolReferences:          toggle(value.ToolReferences),
		}
	}
	return compat
}

func toggle(value *bool) llm.CompatibilityToggle {
	if value == nil {
		return llm.CompatibilityDefault
	}
	if *value {
		return llm.CompatibilityEnabled
	}
	return llm.CompatibilityDisabled
}

func syncCatalog(catalog *sourceCatalog, dataset modelsdev.Dataset, opts SyncOptions) (SyncResult, error) {
	targetProviders := make(map[string]struct{})
	if len(opts.Providers) > 0 {
		for _, p := range opts.Providers {
			targetProviders[p] = struct{}{}
		}
	} else {
		for p := range defaultProviderMapping {
			targetProviders[p] = struct{}{}
		}
	}

	var result SyncResult
	var changesMade bool

	existingByProvider := make(map[string]map[string]int)
	for i, m := range catalog.Models {
		if existingByProvider[m.Provider] == nil {
			existingByProvider[m.Provider] = make(map[string]int)
		}
		existingByProvider[m.Provider][m.ID] = i
	}

	for i := range catalog.Models {
		m := &catalog.Models[i]
		if _, targeted := targetProviders[m.Provider]; !targeted {
			result.Unchanged++
			continue
		}

		modelsDevProvider := defaultProviderMapping[m.Provider]
		if modelsDevProvider == "" {
			modelsDevProvider = m.Provider
		}

		providerData, ok := dataset[modelsDevProvider]
		if !ok {
			result.Unchanged++
			continue
		}

		modelEntry, ok := providerData.Models[m.ID]
		if !ok {
			result.Unchanged++
			continue
		}

		updated := applyModelUpdate(m, modelEntry)
		if updated {
			changesMade = true
			result.UpdatedModels = append(result.UpdatedModels, fmt.Sprintf("%s/%s", m.Provider, m.ID))
		} else {
			result.Unchanged++
		}
	}

	if opts.AddNew {
		for provider := range targetProviders {
			modelsDevProvider := defaultProviderMapping[provider]
			if modelsDevProvider == "" {
				modelsDevProvider = provider
			}

			providerData, ok := dataset[modelsDevProvider]
			if !ok {
				continue
			}

			existing := existingByProvider[provider]
			for modelID, entry := range providerData.Models {
				if _, exists := existing[modelID]; exists {
					continue
				}

				newModel, ok := createSourceModel(provider, entry)
				if !ok {
					continue
				}
				catalog.Models = append(catalog.Models, newModel)
				changesMade = true
				result.AddedModels = append(result.AddedModels, fmt.Sprintf("%s/%s", provider, modelID))
			}
		}
	}

	sort.Slice(catalog.Models, func(i, j int) bool {
		if catalog.Models[i].Provider != catalog.Models[j].Provider {
			return catalog.Models[i].Provider < catalog.Models[j].Provider
		}
		return catalog.Models[i].ID < catalog.Models[j].ID
	})

	if changesMade && !opts.DryRun {
		catalog.Revision++
	}

	result.Total = len(catalog.Models)
	return result, nil
}

func applyModelUpdate(m *sourceModel, entry modelsdev.ModelEntry) bool {
	var changed bool

	if entry.Limit != nil {
		if entry.Limit.Context > 0 && (m.ContextWindow == nil || *m.ContextWindow != entry.Limit.Context) {
			val := entry.Limit.Context
			m.ContextWindow = &val
			changed = true
		}
		if entry.Limit.Output > 0 && (m.MaxOutputTokens == nil || *m.MaxOutputTokens != entry.Limit.Output) {
			val := entry.Limit.Output
			m.MaxOutputTokens = &val
			changed = true
		}
	}

	if entry.Reasoning && (m.Reasoning == nil || !*m.Reasoning) {
		val := true
		m.Reasoning = &val
		changed = true
	}

	if entry.Modalities != nil && slices.Contains(entry.Modalities.Input, "image") {
		if m.Capabilities != nil && !slices.Contains(*m.Capabilities, llm.CapabilityVision) {
			caps := append(*m.Capabilities, llm.CapabilityVision)
			m.Capabilities = &caps
			changed = true
		}
	}

	if entry.Cost != nil {
		costUpdated := updateSourceCost(m, entry.Cost)
		if costUpdated {
			changed = true
		}
	}

	return changed
}

func updateSourceCost(m *sourceModel, cost *modelsdev.Cost) bool {
	var changed bool
	if m.Cost == nil {
		zero := 0.0
		m.Cost = &sourceCost{
			Input:      &zero,
			Output:     &zero,
			CacheRead:  &zero,
			CacheWrite: &zero,
		}
		changed = true
	}

	if cost.Input != nil && *cost.Input >= 0 && (m.Cost.Input == nil || *m.Cost.Input != *cost.Input) {
		m.Cost.Input = cost.Input
		changed = true
	}
	if cost.Output != nil && *cost.Output >= 0 && (m.Cost.Output == nil || *m.Cost.Output != *cost.Output) {
		m.Cost.Output = cost.Output
		changed = true
	}
	if cost.CacheRead != nil && *cost.CacheRead >= 0 && (m.Cost.CacheRead == nil || *m.Cost.CacheRead != *cost.CacheRead) {
		m.Cost.CacheRead = cost.CacheRead
		changed = true
	}
	if cost.CacheWrite != nil && *cost.CacheWrite >= 0 && (m.Cost.CacheWrite == nil || *m.Cost.CacheWrite != *cost.CacheWrite) {
		m.Cost.CacheWrite = cost.CacheWrite
		changed = true
	}

	baseInput := m.Cost.Input
	baseOutput := m.Cost.Output
	baseRead := m.Cost.CacheRead
	baseWrite := m.Cost.CacheWrite

	// Update tiers idempotently with non-null values
	var newTiers []sourceTier
	if len(cost.Tiers) > 0 {
		for _, t := range cost.Tiers {
			threshold := 0
			if t.InputTokensAbove != nil {
				threshold = *t.InputTokensAbove
			} else if t.Tier != nil && t.Tier.Size > 0 {
				threshold = t.Tier.Size
			}
			if threshold > 0 && (m.ContextWindow == nil || threshold < *m.ContextWindow) {
				tInput := baseInput
				if t.Input != nil && *t.Input >= 0 {
					tInput = t.Input
				}
				tOutput := baseOutput
				if t.Output != nil && *t.Output >= 0 {
					tOutput = t.Output
				}
				tRead := baseRead
				if t.CacheRead != nil && *t.CacheRead >= 0 {
					tRead = t.CacheRead
				}
				tWrite := baseWrite
				if t.CacheWrite != nil && *t.CacheWrite >= 0 {
					tWrite = t.CacheWrite
				}

				newTiers = append(newTiers, sourceTier{
					InputTokensAbove: &threshold,
					Input:            tInput,
					Output:           tOutput,
					CacheRead:        tRead,
					CacheWrite:       tWrite,
				})
			}
		}
	} else if cost.ContextOver200k != nil {
		threshold := 200000
		if m.ContextWindow == nil || threshold < *m.ContextWindow {
			tInput := baseInput
			if cost.ContextOver200k.Input != nil && *cost.ContextOver200k.Input >= 0 {
				tInput = cost.ContextOver200k.Input
			}
			tOutput := baseOutput
			if cost.ContextOver200k.Output != nil && *cost.ContextOver200k.Output >= 0 {
				tOutput = cost.ContextOver200k.Output
			}
			tRead := baseRead
			if cost.ContextOver200k.CacheRead != nil && *cost.ContextOver200k.CacheRead >= 0 {
				tRead = cost.ContextOver200k.CacheRead
			}
			tWrite := baseWrite
			if cost.ContextOver200k.CacheWrite != nil && *cost.ContextOver200k.CacheWrite >= 0 {
				tWrite = cost.ContextOver200k.CacheWrite
			}

			newTiers = append(newTiers, sourceTier{
				InputTokensAbove: &threshold,
				Input:            tInput,
				Output:           tOutput,
				CacheRead:        tRead,
				CacheWrite:       tWrite,
			})
		}
	}

	if !tiersEqual(m.Cost.Tiers, newTiers) {
		m.Cost.Tiers = newTiers
		changed = true
	}

	return changed
}

func tiersEqual(a, b []sourceTier) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !equalIntPtr(a[i].InputTokensAbove, b[i].InputTokensAbove) {
			return false
		}
		if !equalFloatPtr(a[i].Input, b[i].Input) {
			return false
		}
		if !equalFloatPtr(a[i].Output, b[i].Output) {
			return false
		}
		if !equalFloatPtr(a[i].CacheRead, b[i].CacheRead) {
			return false
		}
		if !equalFloatPtr(a[i].CacheWrite, b[i].CacheWrite) {
			return false
		}
	}
	return true
}

func equalIntPtr(a, b *int) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

func equalFloatPtr(a, b *float64) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}

func createSourceModel(provider string, entry modelsdev.ModelEntry) (sourceModel, bool) {
	// Only add model if it has authoritative context limits and output <= context
	if entry.Limit == nil || entry.Limit.Context <= 0 || entry.Limit.Output <= 0 || entry.Limit.Output > entry.Limit.Context {
		return sourceModel{}, false
	}

	name := entry.Name
	if name == "" {
		name = entry.ID
	}
	api := modelsdev.DefaultProviderAPI(provider)
	reasoning := entry.Reasoning
	caps := []llm.Capability{llm.CapabilityStreaming}
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

	ctxWindow := entry.Limit.Context
	maxOut := entry.Limit.Output

	var cost *sourceCost
	if entry.Cost != nil {
		zero := 0.0
		input := &zero
		if entry.Cost.Input != nil && *entry.Cost.Input >= 0 {
			input = entry.Cost.Input
		}
		output := &zero
		if entry.Cost.Output != nil && *entry.Cost.Output >= 0 {
			output = entry.Cost.Output
		}
		cacheRead := &zero
		if entry.Cost.CacheRead != nil && *entry.Cost.CacheRead >= 0 {
			cacheRead = entry.Cost.CacheRead
		}
		cacheWrite := &zero
		if entry.Cost.CacheWrite != nil && *entry.Cost.CacheWrite >= 0 {
			cacheWrite = entry.Cost.CacheWrite
		}

		cost = &sourceCost{
			Input:      input,
			Output:     output,
			CacheRead:  cacheRead,
			CacheWrite: cacheWrite,
		}

		if len(entry.Cost.Tiers) > 0 {
			for _, t := range entry.Cost.Tiers {
				threshold := 0
				if t.InputTokensAbove != nil {
					threshold = *t.InputTokensAbove
				} else if t.Tier != nil && t.Tier.Size > 0 {
					threshold = t.Tier.Size
				}
				if threshold > 0 && threshold < ctxWindow {
					tInput := input
					if t.Input != nil && *t.Input >= 0 {
						tInput = t.Input
					}
					tOutput := output
					if t.Output != nil && *t.Output >= 0 {
						tOutput = t.Output
					}
					tRead := cacheRead
					if t.CacheRead != nil && *t.CacheRead >= 0 {
						tRead = t.CacheRead
					}
					tWrite := cacheWrite
					if t.CacheWrite != nil && *t.CacheWrite >= 0 {
						tWrite = t.CacheWrite
					}

					cost.Tiers = append(cost.Tiers, sourceTier{
						InputTokensAbove: &threshold,
						Input:            tInput,
						Output:           tOutput,
						CacheRead:        tRead,
						CacheWrite:       tWrite,
					})
				}
			}
		} else if entry.Cost.ContextOver200k != nil {
			threshold := 200000
			if threshold < ctxWindow {
				tInput := input
				if entry.Cost.ContextOver200k.Input != nil && *entry.Cost.ContextOver200k.Input >= 0 {
					tInput = entry.Cost.ContextOver200k.Input
				}
				tOutput := output
				if entry.Cost.ContextOver200k.Output != nil && *entry.Cost.ContextOver200k.Output >= 0 {
					tOutput = entry.Cost.ContextOver200k.Output
				}
				tRead := cacheRead
				if entry.Cost.ContextOver200k.CacheRead != nil && *entry.Cost.ContextOver200k.CacheRead >= 0 {
					tRead = entry.Cost.ContextOver200k.CacheRead
				}
				tWrite := cacheWrite
				if entry.Cost.ContextOver200k.CacheWrite != nil && *entry.Cost.ContextOver200k.CacheWrite >= 0 {
					tWrite = entry.Cost.ContextOver200k.CacheWrite
				}

				cost.Tiers = append(cost.Tiers, sourceTier{
					InputTokensAbove: &threshold,
					Input:            tInput,
					Output:           tOutput,
					CacheRead:        tRead,
					CacheWrite:       tWrite,
				})
			}
		}
	}

	return sourceModel{
		Provider:        provider,
		ID:              entry.ID,
		Name:            name,
		API:             api,
		Reasoning:       &reasoning,
		Capabilities:    &caps,
		ContextWindow:   &ctxWindow,
		MaxOutputTokens: &maxOut,
		Cost:            cost,
	}, true
}

func writeCatalog(path string, catalog *sourceCatalog) error {
	data, err := jsonv2.Marshal(catalog, jsontext.Multiline(true), jsontext.WithIndent("  "))
	if err != nil {
		return fmt.Errorf("encode json: %w", err)
	}
	data = append(data, '\n')

	mode := os.FileMode(0o644)
	if info, err := os.Stat(path); err == nil {
		mode = info.Mode().Perm()
	}

	dir := filepath.Dir(path)
	tmpFile, err := os.CreateTemp(dir, "catalog-*.json.tmp")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	tmpName := tmpFile.Name()
	defer func() { _ = os.Remove(tmpName) }()

	if err := tmpFile.Chmod(mode); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("chmod temp file: %w", err)
	}
	if _, err := tmpFile.Write(data); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("write temp file: %w", err)
	}
	if err := tmpFile.Sync(); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("sync temp file: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("close temp file: %w", err)
	}

	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("rename to %s: %w", path, err)
	}
	return nil
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "modelsdevsync: %v\n", err)
	os.Exit(1)
}
