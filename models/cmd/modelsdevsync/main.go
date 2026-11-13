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
	"github.com/XiaoConstantine/llm-go/internal/catalogmeta"
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

const (
	maxResponseSize = 32 << 20
	schemaVersion   = 2
	modelsDevSource = "models.dev"
)

type sourceCatalog struct {
	SchemaVersion int           `json:"schema_version"`
	Revision      int           `json:"revision"`
	Models        []sourceModel `json:"models"`
}

type sourceModel struct {
	Provider        string               `json:"provider"`
	ID              string               `json:"id"`
	Source          string               `json:"source,omitempty"`
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
	Gemini          *sourceGemini          `json:"gemini,omitempty"`
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

type sourceGemini struct {
	ThinkingLevelsOnly *bool `json:"thinking_levels_only,omitempty"`
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
	UpdatedModels  []string
	AddedModels    []string
	RemovedModels  []string
	ExcludedModels []string
	SkippedModels  []string
	ClampedModels  []string
	Unchanged      int
	Total          int
}

func main() {
	catalogPath := flag.String("catalog", "catalogsource/catalog.json", "path to catalog.json")
	url := flag.String("url", modelsdev.DefaultURL, "models.dev API URL")
	file := flag.String("file", "", "path to local models.dev JSON file (optional)")
	providersFlag := flag.String("providers", "", "comma-separated list of providers to sync (default: all built-in)")
	addNew := flag.Bool("add-new", false, "add new tool-calling models with text-only output from models.dev")
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
	fmt.Printf("  Excluded new models: %d\n", len(result.ExcludedModels))
	for _, m := range result.ExcludedModels {
		fmt.Printf("    - %s (discovery requirements)\n", m)
	}
	fmt.Printf("  Skipped costs: %d\n", len(result.SkippedModels))
	for _, m := range result.SkippedModels {
		fmt.Printf("    ! %s\n", m)
	}
	fmt.Printf("  Clamped cache-read prices: %d\n", len(result.ClampedModels))
	for _, m := range result.ClampedModels {
		fmt.Printf("    ! %s (retained cache_read capped at input)\n", m)
	}
	fmt.Printf("  Removed models: %d\n", len(result.RemovedModels))
	for _, m := range result.RemovedModels {
		fmt.Printf("    - %s\n", m)
	}
	fmt.Printf("  Unchanged:      %d\n", result.Unchanged)
	fmt.Printf("  Total models:   %d\n", result.Total)

	if !*dryRun && (len(result.UpdatedModels) > 0 || len(result.AddedModels) > 0 || len(result.RemovedModels) > 0) {
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
		defer func() { _ = resp.Body.Close() }()

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
	if catalog.SchemaVersion != schemaVersion {
		return nil, fmt.Errorf("schema_version is %d, want %d", catalog.SchemaVersion, schemaVersion)
	}
	return &catalog, nil
}

func validateCatalog(catalog *sourceCatalog) error {
	if catalog.SchemaVersion != schemaVersion {
		return fmt.Errorf("schema_version is %d, want %d", catalog.SchemaVersion, schemaVersion)
	}
	if catalog.Revision <= 0 {
		return errors.New("revision must be positive")
	}
	seen := make(map[string]map[string]struct{})
	for i, m := range catalog.Models {
		if m.Source != "" && m.Source != modelsDevSource {
			return fmt.Errorf("models[%d]: unknown source %q", i, m.Source)
		}
		if m.Provider == "openai-codex" && m.Source != "" {
			return fmt.Errorf("models[%d]: openai-codex must be curated", i)
		}
		if m.Provider == "" || !utf8.ValidString(m.Provider) || strings.TrimSpace(m.Provider) != m.Provider {
			return fmt.Errorf("models[%d]: provider must be nonempty valid UTF-8 without surrounding whitespace", i)
		}
		if m.ID == "" || !utf8.ValidString(m.ID) || strings.TrimSpace(m.ID) != m.ID {
			return fmt.Errorf("models[%d]: ID must be nonempty valid UTF-8 without surrounding whitespace", i)
		}
		if m.Name == "" || !utf8.ValidString(m.Name) || strings.TrimSpace(m.Name) != m.Name {
			return fmt.Errorf("models[%d]: name must be nonempty valid UTF-8 without surrounding whitespace", i)
		}
		expectedAPI, knownProvider := catalogmeta.ProviderAPI(m.Provider)
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
			if !catalogmeta.SupportsCapability(m.API, cap) {
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
		if err := validateCacheReadPrices(m.Cost); err != nil {
			return fmt.Errorf("models[%d] %s/%s: %w", i, m.Provider, m.ID, err)
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

func validateCacheReadPrices(cost *sourceCost) error {
	if cost == nil {
		return nil
	}
	if cost.Input != nil && cost.CacheRead != nil {
		if err := catalogmeta.ValidateCacheReadRate(*cost.Input, *cost.CacheRead); err != nil {
			return err
		}
	}
	for index, tier := range cost.Tiers {
		if tier.Input != nil && tier.CacheRead != nil {
			if err := catalogmeta.ValidateCacheReadRate(*tier.Input, *tier.CacheRead); err != nil {
				return fmt.Errorf("tier %d: %w", index, err)
			}
		}
	}
	return nil
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
	if value := source.Gemini; value != nil {
		compat.Gemini = &llm.GeminiCompatibility{ThinkingLevelsOnly: toggle(value.ThinkingLevelsOnly)}
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
			if _, known := catalogmeta.ProviderAPI(p); !known || p == "openai-codex" {
				return SyncResult{}, fmt.Errorf("provider %q is not a models.dev sync route", p)
			}
			targetProviders[p] = struct{}{}
		}
	} else {
		for p := range defaultProviderMapping {
			targetProviders[p] = struct{}{}
		}
	}

	// Resolve upstream names per route, and validate snapshot completeness before
	// modifying anything. A missing provider/null models map is not evidence of
	// retirement; an explicitly empty models object is an authoritative empty set.
	snapshots := make(map[string]modelsdev.ProviderEntry)
	for provider := range targetProviders {
		upstream := defaultProviderMapping[provider]
		if upstream == "" {
			upstream = provider
		}
		data, ok := dataset[upstream]
		if !ok {
			if len(opts.Providers) > 0 {
				return SyncResult{}, fmt.Errorf("snapshot is missing provider %q", upstream)
			}
			continue
		}
		if data.Models == nil {
			return SyncResult{}, fmt.Errorf("snapshot provider %q has no models object", upstream)
		}
		for id, entry := range data.Models {
			if id == "" || strings.TrimSpace(id) != id || id != entry.ID {
				return SyncResult{}, fmt.Errorf("snapshot provider %q has inconsistent model key %q or ID %q", upstream, id, entry.ID)
			}
		}
		snapshots[provider] = data
	}
	for _, model := range catalog.Models {
		if _, targeted := targetProviders[model.Provider]; targeted && model.Source == modelsDevSource {
			if _, ok := snapshots[model.Provider]; !ok {
				return SyncResult{}, fmt.Errorf("snapshot is missing managed provider %q", model.Provider)
			}
		}
	}

	var result SyncResult
	existingByProvider := make(map[string]map[string]struct{})
	retained := make([]sourceModel, 0, len(catalog.Models))
	for _, model := range catalog.Models {
		if existingByProvider[model.Provider] == nil {
			existingByProvider[model.Provider] = make(map[string]struct{})
		}
		existingByProvider[model.Provider][model.ID] = struct{}{}
		data, targeted := snapshots[model.Provider]
		if !targeted || model.Source != modelsDevSource {
			retained = append(retained, model)
			result.Unchanged++
			continue
		}
		entry, found := data.Models[model.ID]
		if !found || !eligibleModel(entry) {
			result.RemovedModels = append(result.RemovedModels, model.Provider+"/"+model.ID)
			continue
		}
		// Check prices before applying them, without discarding independent metadata.
		// Audio pricing is not imported by applyModelUpdate either.
		if entry.Cost != nil && (model.Capabilities == nil || !slices.Contains(*model.Capabilities, llm.CapabilityAudio)) {
			candidate := model
			if model.Cost != nil {
				cost := *model.Cost
				candidate.Cost = &cost
			}
			if entry.Limit != nil && entry.Limit.Context > 0 {
				candidate.ContextWindow = &entry.Limit.Context
			}
			_, clamped := updateSourceCost(&candidate, entry.Cost)
			if err := validateCacheReadPrices(candidate.Cost); err != nil {
				result.SkippedModels = append(result.SkippedModels, fmt.Sprintf("%s/%s: cost skipped: %v", model.Provider, model.ID, err))
				entry.Cost = nil
			} else if clamped {
				result.ClampedModels = append(result.ClampedModels, fmt.Sprintf("%s/%s", model.Provider, model.ID))
			}
		}

		if applyModelUpdate(&model, entry) {
			result.UpdatedModels = append(result.UpdatedModels, model.Provider+"/"+model.ID)
		} else {
			result.Unchanged++
		}
		retained = append(retained, model)
	}
	catalog.Models = retained

	if opts.AddNew {
		for provider, data := range snapshots {
			for id, entry := range data.Models {
				if _, exists := existingByProvider[provider][id]; exists {
					continue
				}
				model, ok := createSourceModel(provider, entry)
				if !ok {
					result.ExcludedModels = append(result.ExcludedModels, fmt.Sprintf("%s/%s", provider, id))
					continue
				}
				if err := validateCacheReadPrices(model.Cost); err != nil {
					result.SkippedModels = append(result.SkippedModels, fmt.Sprintf("%s/%s: new model skipped (cost): %v", provider, id, err))
					continue
				}
				catalog.Models = append(catalog.Models, model)
				result.AddedModels = append(result.AddedModels, provider+"/"+model.ID)
			}
		}
	}

	sort.Slice(catalog.Models, func(i, j int) bool {
		if catalog.Models[i].Provider != catalog.Models[j].Provider {
			return catalog.Models[i].Provider < catalog.Models[j].Provider
		}
		return catalog.Models[i].ID < catalog.Models[j].ID
	})
	slices.Sort(result.AddedModels)
	slices.Sort(result.UpdatedModels)
	slices.Sort(result.RemovedModels)
	if !opts.DryRun && (len(result.AddedModels) > 0 || len(result.UpdatedModels) > 0 || len(result.RemovedModels) > 0) {
		catalog.Revision++
	}
	slices.Sort(result.ExcludedModels)
	slices.Sort(result.SkippedModels)
	slices.Sort(result.ClampedModels)
	result.Total = len(catalog.Models)
	return result, nil
}

// eligibleModel is the membership policy for models.dev-managed entries, both
// existing and new. Curated entries are outside this policy. No age/name rule or
// cross-provider retirement inference belongs here.
func eligibleModel(entry modelsdev.ModelEntry) bool {
	return strings.TrimSpace(entry.ID) != "" && entry.Status != "deprecated" &&
		entry.ToolCall && entry.Modalities != nil && slices.Equal(entry.Modalities.Output, []string{"text"}) &&
		entry.Limit != nil && entry.Limit.Context > 0 && entry.Limit.Output > 0 && entry.Limit.Output <= entry.Limit.Context
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
		if m.Compatibility != nil && m.Compatibility.OpenAIChat != nil && (m.Compatibility.OpenAIChat.ToolResultImageFallback == nil || !*m.Compatibility.OpenAIChat.ToolResultImageFallback) {
			enabled := true
			m.Compatibility.OpenAIChat.ToolResultImageFallback = &enabled
			changed = true
		}
	}

	// Audio-capable models may bill text and audio tokens at different rates,
	// which ModelCost cannot represent without undercounting one modality.
	if entry.Cost != nil && (m.Capabilities == nil || !slices.Contains(*m.Capabilities, llm.CapabilityAudio)) {
		costUpdated, _ := updateSourceCost(m, entry.Cost)
		if costUpdated {
			changed = true
		}
	}
	// Limits still apply when upstream pricing is absent or rejected.
	if pruneSourceCostTiers(m) {
		changed = true
	}

	return changed
}

func pruneSourceCostTiers(m *sourceModel) bool {
	if m.Cost == nil || m.ContextWindow == nil || *m.ContextWindow <= 0 {
		return false
	}
	// Preflight price validation uses a shallow cost copy; do not mutate its slice.
	tiers := slices.DeleteFunc(slices.Clone(m.Cost.Tiers), func(tier sourceTier) bool {
		return *tier.InputTokensAbove >= *m.ContextWindow
	})
	if len(tiers) == len(m.Cost.Tiers) {
		return false
	}
	m.Cost.Tiers = tiers
	return true
}

func updateSourceCost(m *sourceModel, cost *modelsdev.Cost) (changed, clamped bool) {
	changed = pruneSourceCostTiers(m)
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

	read := catalogmeta.ClampRetainedCacheRead(*m.Cost.Input, *m.Cost.CacheRead, cost.CacheRead)
	if read != *m.Cost.CacheRead {
		m.Cost.CacheRead = &read
		changed, clamped = true, true
	}

	// Seed matching thresholds from their curated tier, new thresholds from base rates.
	var newTiers []sourceTier
	mergeTier := func(threshold int, input, output, cacheRead, cacheWrite *float64) {
		if threshold <= 0 || (m.ContextWindow != nil && threshold >= *m.ContextWindow) {
			return
		}
		tier := sourceTier{InputTokensAbove: &threshold, Input: m.Cost.Input, Output: m.Cost.Output, CacheRead: m.Cost.CacheRead, CacheWrite: m.Cost.CacheWrite}
		for _, baselineTier := range m.Cost.Tiers {
			if *baselineTier.InputTokensAbove == threshold {
				tier = baselineTier
				break
			}
		}
		if input != nil && *input >= 0 {
			tier.Input = input
		}
		if output != nil && *output >= 0 {
			tier.Output = output
		}
		if cacheRead != nil && *cacheRead >= 0 {
			tier.CacheRead = cacheRead
		}
		if cacheWrite != nil && *cacheWrite >= 0 {
			tier.CacheWrite = cacheWrite
		}
		read := catalogmeta.ClampRetainedCacheRead(*tier.Input, *tier.CacheRead, cacheRead)
		clamped = clamped || read != *tier.CacheRead
		tier.CacheRead = &read
		newTiers = append(newTiers, tier)
	}
	for _, tier := range cost.Tiers {
		threshold := 0
		if tier.InputTokensAbove != nil {
			threshold = *tier.InputTokensAbove
		} else if tier.Tier != nil {
			threshold = tier.Tier.Size
		}
		mergeTier(threshold, tier.Input, tier.Output, tier.CacheRead, tier.CacheWrite)
	}
	if len(newTiers) == 0 && cost.ContextOver200k != nil {
		rates := cost.ContextOver200k
		mergeTier(200000, rates.Input, rates.Output, rates.CacheRead, rates.CacheWrite)
	}
	if len(newTiers) > 0 && !tiersEqual(m.Cost.Tiers, newTiers) {
		m.Cost.Tiers = newTiers
		changed = true
	}

	return changed, clamped
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
	api, known := catalogmeta.ProviderAPI(provider)
	if !known {
		return sourceModel{}, false
	}
	if !eligibleModel(entry) {
		return sourceModel{}, false
	}

	id := strings.TrimSpace(entry.ID)
	name := strings.TrimSpace(entry.Name)
	if name == "" {
		name = id
	}
	reasoning := entry.Reasoning
	caps := []llm.Capability{llm.CapabilityStreaming}
	if entry.ToolCall && catalogmeta.SupportsCapability(api, llm.CapabilityTools) {
		caps = append(caps, llm.CapabilityTools)
	}
	// Responses JSON support is model-specific and must be reviewed before it is
	// advertised by the built-in catalog.
	if entry.StructuredOutput && api != llm.APIOpenAIResponses && catalogmeta.SupportsCapability(api, llm.CapabilityJSON) {
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
		ID:              id,
		Source:          modelsDevSource,
		Name:            name,
		API:             api,
		Reasoning:       &reasoning,
		Capabilities:    &caps,
		ContextWindow:   &ctxWindow,
		MaxOutputTokens: &maxOut,
		Cost:            cost,
		Compatibility:   sourceCompatibilityFromModel(modelsdev.DefaultProviderCompatibility(provider, entry, api)),
	}, true
}

func sourceCompatibilityFromModel(compat *llm.ModelCompatibility) *sourceCompatibility {
	if compat == nil {
		return nil
	}
	source := &sourceCompatibility{}
	if value := compat.OpenAIChat; value != nil {
		source.OpenAIChat = &sourceOpenAIChat{
			MaxTokensField:          value.MaxTokensField,
			InstructionRole:         value.InstructionRole,
			ReasoningEffort:         sourceToggle(value.ReasoningEffort),
			StreamingUsage:          sourceToggle(value.StreamingUsage),
			FinishReason:            sourceToggle(value.FinishReason),
			ToolResultName:          sourceToggle(value.ToolResultName),
			ToolResultImageFallback: sourceToggle(value.ToolResultImageFallback),
			AssistantAfterTool:      sourceToggle(value.AssistantAfterToolResult),
			ReasoningContentReplay:  sourceToggle(value.ReasoningContentReplay),
			StrictTools:             sourceToggle(value.StrictTools),
			LongCacheRetention:      sourceToggle(value.LongCacheRetention),
			SessionAffinity:         sourceToggle(value.SessionAffinity),
			SessionAffinityFormat:   value.SessionAffinityFormat,
			ThinkingFormat:          value.ThinkingFormat,
			CacheControlFormat:      value.CacheControlFormat,
		}
	}
	if value := compat.OpenAIResponses; value != nil {
		source.OpenAIResponses = &sourceOpenAIResponses{
			DeveloperRole:           sourceToggle(value.DeveloperRole),
			EncryptedReasoning:      sourceToggle(value.EncryptedReasoning),
			StrictTools:             sourceToggle(value.StrictTools),
			AdditionalTools:         sourceToggle(value.AdditionalTools),
			ToolSearch:              sourceToggle(value.ToolSearch),
			LongCacheRetention:      sourceToggle(value.LongCacheRetention),
			ExplicitPromptCacheMode: sourceToggle(value.ExplicitPromptCacheMode),
			SessionAffinityFormat:   value.SessionAffinityFormat,
		}
	}
	if value := compat.Gemini; value != nil {
		source.Gemini = &sourceGemini{ThinkingLevelsOnly: sourceToggle(value.ThinkingLevelsOnly)}
	}
	if value := compat.Anthropic; value != nil {
		source.Anthropic = &sourceAnthropic{
			EagerToolInputStreaming: sourceToggle(value.EagerToolInputStreaming),
			LongCacheRetention:      sourceToggle(value.LongCacheRetention),
			SessionAffinity:         sourceToggle(value.SessionAffinity),
			CacheControlOnTools:     sourceToggle(value.CacheControlOnTools),
			Temperature:             sourceToggle(value.Temperature),
			AdaptiveThinking:        sourceToggle(value.AdaptiveThinking),
			EmptyThinkingSignature:  sourceToggle(value.EmptyThinkingSignature),
			StrictTools:             sourceToggle(value.StrictTools),
			ToolReferences:          sourceToggle(value.ToolReferences),
		}
	}
	return source
}

func sourceToggle(value llm.CompatibilityToggle) *bool {
	switch value {
	case llm.CompatibilityEnabled:
		value := true
		return &value
	case llm.CompatibilityDisabled:
		value := false
		return &value
	default:
		return nil
	}
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
