package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/format"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	llm "github.com/XiaoConstantine/llm-go"
)

const schemaVersion = 1

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
}

func main() {
	source := flag.String("source", "catalogsource/catalog.json", "catalog source JSON")
	output := flag.String("output", "catalog_generated.go", "generated Go output")
	check := flag.Bool("check", false, "fail if output is stale instead of writing it")
	flag.Parse()

	generated, err := generate(*source)
	if err != nil {
		fatal(err)
	}
	if *check {
		current, err := os.ReadFile(*output)
		if err != nil {
			fatal(err)
		}
		if !bytes.Equal(current, generated) {
			fatal(errors.New("generated catalog is stale; run go generate ./models"))
		}
		return
	}
	if err := os.WriteFile(*output, generated, 0o644); err != nil {
		fatal(err)
	}
}

func generate(path string) ([]byte, error) {
	revision, models, err := load(path)
	if err != nil {
		return nil, err
	}
	return render(revision, models)
}

func load(path string) (int, []llm.Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, nil, fmt.Errorf("read %s: %w", path, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var source sourceCatalog
	if err := decoder.Decode(&source); err != nil {
		return 0, nil, fmt.Errorf("decode %s: %w", path, err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return 0, nil, errors.New("catalog source must contain exactly one JSON value")
	}
	if source.SchemaVersion != schemaVersion {
		return 0, nil, fmt.Errorf("schema_version is %d, want %d", source.SchemaVersion, schemaVersion)
	}
	if source.Revision <= 0 {
		return 0, nil, errors.New("revision must be positive")
	}
	models := make([]llm.Model, len(source.Models))
	seen := make(map[string]map[string]struct{})
	for index, value := range source.Models {
		model, err := value.model()
		if err != nil {
			return 0, nil, fmt.Errorf("models[%d] %s/%s: %w", index, value.Provider, value.ID, err)
		}
		providerModels := seen[model.Provider]
		if providerModels == nil {
			providerModels = make(map[string]struct{})
			seen[model.Provider] = providerModels
		}
		if _, exists := providerModels[model.ID]; exists {
			return 0, nil, fmt.Errorf("models[%d]: duplicate model %s/%s", index, model.Provider, model.ID)
		}
		providerModels[model.ID] = struct{}{}
		models[index] = model
	}
	sort.Slice(models, func(i, j int) bool {
		if models[i].Provider != models[j].Provider {
			return models[i].Provider < models[j].Provider
		}
		return models[i].ID < models[j].ID
	})
	return source.Revision, models, nil
}

func (source sourceModel) model() (llm.Model, error) {
	if source.Provider == "" || !utf8.ValidString(source.Provider) || strings.TrimSpace(source.Provider) != source.Provider {
		return llm.Model{}, errors.New("provider must be nonempty valid UTF-8 without surrounding whitespace")
	}
	if source.ID == "" || !utf8.ValidString(source.ID) || strings.TrimSpace(source.ID) != source.ID {
		return llm.Model{}, errors.New("ID must be nonempty valid UTF-8 without surrounding whitespace")
	}
	if source.Name == "" || !utf8.ValidString(source.Name) || strings.TrimSpace(source.Name) != source.Name {
		return llm.Model{}, errors.New("name must be nonempty valid UTF-8 without surrounding whitespace")
	}
	expectedAPI, knownProvider := providerAPIs[source.Provider]
	if !knownProvider {
		return llm.Model{}, fmt.Errorf("unsupported built-in provider %q", source.Provider)
	}
	if source.API != expectedAPI {
		return llm.Model{}, fmt.Errorf("provider %q requires API %q, got %q", source.Provider, expectedAPI, source.API)
	}
	if source.Reasoning == nil || source.Capabilities == nil || source.ContextWindow == nil || source.MaxOutputTokens == nil {
		return llm.Model{}, errors.New("reasoning, capabilities, context_window, and max_output_tokens are required and must not be null")
	}
	if *source.ContextWindow <= 0 || *source.MaxOutputTokens <= 0 || *source.MaxOutputTokens > *source.ContextWindow {
		return llm.Model{}, errors.New("context and output-token limits must be positive and output must not exceed context")
	}
	capabilities := append([]llm.Capability(nil), (*source.Capabilities)...)
	seen := make(map[llm.Capability]struct{}, len(capabilities))
	for _, capability := range capabilities {
		if !supportsCapability(source.API, capability) {
			return llm.Model{}, fmt.Errorf("API %q does not support capability %q", source.API, capability)
		}
		if _, exists := seen[capability]; exists {
			return llm.Model{}, fmt.Errorf("duplicate capability %q", capability)
		}
		seen[capability] = struct{}{}
	}
	cost, err := source.Cost.model()
	if err != nil {
		return llm.Model{}, fmt.Errorf("cost: %w", err)
	}
	model := llm.Model{Provider: source.Provider, ID: source.ID, Name: source.Name, API: source.API,
		Reasoning: *source.Reasoning, Capabilities: capabilities, ContextWindow: *source.ContextWindow,
		MaxOutputTokens: *source.MaxOutputTokens, Cost: cost, Compatibility: source.Compatibility.model()}
	if model.Cost != nil {
		if err := model.Cost.Validate(); err != nil {
			return llm.Model{}, fmt.Errorf("cost: %w", err)
		}
		if model.ContextWindow != 0 {
			for index, tier := range model.Cost.Tiers {
				if tier.InputTokensAbove >= model.ContextWindow {
					return llm.Model{}, fmt.Errorf("cost tier %d threshold must be below context window", index)
				}
			}
		}
	}
	if err := model.Compatibility.Validate(model.API); err != nil {
		return llm.Model{}, fmt.Errorf("compatibility: %w", err)
	}
	return model, nil
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

func (source *sourceCost) model() (*llm.ModelCost, error) {
	if source == nil {
		return nil, nil
	}
	if source.Input == nil || source.Output == nil || source.CacheRead == nil || source.CacheWrite == nil {
		return nil, errors.New("input, output, cache_read, and cache_write are required and must not be null")
	}
	cost := &llm.ModelCost{Input: *source.Input, Output: *source.Output, CacheRead: *source.CacheRead,
		CacheWrite: *source.CacheWrite, Tiers: make([]llm.ModelCostTier, len(source.Tiers))}
	for index, tier := range source.Tiers {
		if tier.InputTokensAbove == nil || tier.Input == nil || tier.Output == nil || tier.CacheRead == nil || tier.CacheWrite == nil {
			return nil, fmt.Errorf("tier %d input_tokens_above, input, output, cache_read, and cache_write are required and must not be null", index)
		}
		cost.Tiers[index] = llm.ModelCostTier{InputTokensAbove: *tier.InputTokensAbove, Input: *tier.Input,
			Output: *tier.Output, CacheRead: *tier.CacheRead, CacheWrite: *tier.CacheWrite}
	}
	return cost, nil
}

func (source *sourceCompatibility) model() *llm.ModelCompatibility {
	if source == nil {
		return nil
	}
	compatibility := &llm.ModelCompatibility{}
	if value := source.OpenAIChat; value != nil {
		compatibility.OpenAIChat = &llm.OpenAIChatCompatibility{MaxTokensField: value.MaxTokensField,
			InstructionRole: value.InstructionRole, ReasoningEffort: toggle(value.ReasoningEffort),
			StreamingUsage: toggle(value.StreamingUsage), FinishReason: toggle(value.FinishReason),
			ToolResultName: toggle(value.ToolResultName), ToolResultImageFallback: toggle(value.ToolResultImageFallback),
			AssistantAfterToolResult: toggle(value.AssistantAfterTool),
			ReasoningContentReplay:   toggle(value.ReasoningContentReplay), StrictTools: toggle(value.StrictTools),
			LongCacheRetention: toggle(value.LongCacheRetention), SessionAffinity: toggle(value.SessionAffinity),
			SessionAffinityFormat: value.SessionAffinityFormat, ThinkingFormat: value.ThinkingFormat,
			CacheControlFormat: value.CacheControlFormat}
	}
	if value := source.OpenAIResponses; value != nil {
		compatibility.OpenAIResponses = &llm.OpenAIResponsesCompatibility{DeveloperRole: toggle(value.DeveloperRole),
			EncryptedReasoning: toggle(value.EncryptedReasoning), StrictTools: toggle(value.StrictTools),
			LongCacheRetention: toggle(value.LongCacheRetention), ExplicitPromptCacheMode: toggle(value.ExplicitPromptCacheMode),
			SessionAffinityFormat: value.SessionAffinityFormat}
	}
	if value := source.Anthropic; value != nil {
		compatibility.Anthropic = &llm.AnthropicCompatibility{EagerToolInputStreaming: toggle(value.EagerToolInputStreaming),
			LongCacheRetention: toggle(value.LongCacheRetention), SessionAffinity: toggle(value.SessionAffinity),
			CacheControlOnTools: toggle(value.CacheControlOnTools), Temperature: toggle(value.Temperature),
			AdaptiveThinking: toggle(value.AdaptiveThinking), EmptyThinkingSignature: toggle(value.EmptyThinkingSignature),
			StrictTools: toggle(value.StrictTools)}
	}
	return compatibility
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

func render(revision int, models []llm.Model) ([]byte, error) {
	var output bytes.Buffer
	output.WriteString("// Code generated by go generate ./models; DO NOT EDIT.\n\npackage models\n\n")
	output.WriteString("import llm \"github.com/XiaoConstantine/llm-go\"\n\n")
	fmt.Fprintf(&output, "const generatedBuiltinCatalogRevision = %d\n\n", revision)
	output.WriteString("var generatedBuiltinModels = []llm.Model{\n")
	for _, model := range models {
		renderModel(&output, model)
	}
	output.WriteString("}\n")
	formatted, err := format.Source(output.Bytes())
	if err != nil {
		return nil, fmt.Errorf("format generated source: %w", err)
	}
	return formatted, nil
}

func renderModel(output *bytes.Buffer, model llm.Model) {
	fmt.Fprintf(output, "{Provider:%q,ID:%q,Name:%q,API:llm.API(%q),Reasoning:%t,ContextWindow:%d,MaxOutputTokens:%d",
		model.Provider, model.ID, model.Name, model.API, model.Reasoning, model.ContextWindow, model.MaxOutputTokens)
	if len(model.Capabilities) != 0 {
		output.WriteString(",Capabilities:[]llm.Capability{")
		for _, capability := range model.Capabilities {
			fmt.Fprintf(output, "llm.Capability(%q),", capability)
		}
		output.WriteString("}")
	}
	if model.Cost != nil {
		output.WriteString(",Cost:")
		renderCost(output, model.Cost)
	}
	if model.Compatibility != nil {
		output.WriteString(",Compatibility:")
		renderCompatibility(output, model.Compatibility)
	}
	output.WriteString("},\n")
}

func renderCost(output *bytes.Buffer, cost *llm.ModelCost) {
	fmt.Fprintf(output, "&llm.ModelCost{Input:%s,Output:%s,CacheRead:%s,CacheWrite:%s",
		number(cost.Input), number(cost.Output), number(cost.CacheRead), number(cost.CacheWrite))
	if len(cost.Tiers) != 0 {
		output.WriteString(",Tiers:[]llm.ModelCostTier{")
		for _, tier := range cost.Tiers {
			fmt.Fprintf(output, "{InputTokensAbove:%d,Input:%s,Output:%s,CacheRead:%s,CacheWrite:%s},",
				tier.InputTokensAbove, number(tier.Input), number(tier.Output), number(tier.CacheRead), number(tier.CacheWrite))
		}
		output.WriteString("}")
	}
	output.WriteString("}")
}

func renderCompatibility(output *bytes.Buffer, compatibility *llm.ModelCompatibility) {
	output.WriteString("&llm.ModelCompatibility{")
	if value := compatibility.OpenAIChat; value != nil {
		fmt.Fprintf(output, "OpenAIChat:&llm.OpenAIChatCompatibility{MaxTokensField:llm.MaxTokensField(%q),InstructionRole:llm.InstructionRole(%q),ReasoningEffort:llm.CompatibilityToggle(%q),StreamingUsage:llm.CompatibilityToggle(%q),FinishReason:llm.CompatibilityToggle(%q),ToolResultName:llm.CompatibilityToggle(%q),AssistantAfterToolResult:llm.CompatibilityToggle(%q),ReasoningContentReplay:llm.CompatibilityToggle(%q),StrictTools:llm.CompatibilityToggle(%q),LongCacheRetention:llm.CompatibilityToggle(%q),SessionAffinity:llm.CompatibilityToggle(%q),SessionAffinityFormat:llm.SessionAffinityFormat(%q),ThinkingFormat:llm.ThinkingFormat(%q),CacheControlFormat:llm.CacheControlFormat(%q),",
			value.MaxTokensField, value.InstructionRole, value.ReasoningEffort, value.StreamingUsage, value.FinishReason,
			value.ToolResultName, value.AssistantAfterToolResult, value.ReasoningContentReplay, value.StrictTools,
			value.LongCacheRetention, value.SessionAffinity, value.SessionAffinityFormat, value.ThinkingFormat, value.CacheControlFormat)
		if value.ToolResultImageFallback != llm.CompatibilityDefault {
			fmt.Fprintf(output, "ToolResultImageFallback:llm.CompatibilityToggle(%q),", value.ToolResultImageFallback)
		}
		output.WriteString("},")
	}
	if value := compatibility.OpenAIResponses; value != nil {
		fmt.Fprintf(output, "OpenAIResponses:&llm.OpenAIResponsesCompatibility{DeveloperRole:llm.CompatibilityToggle(%q),EncryptedReasoning:llm.CompatibilityToggle(%q),StrictTools:llm.CompatibilityToggle(%q),LongCacheRetention:llm.CompatibilityToggle(%q),ExplicitPromptCacheMode:llm.CompatibilityToggle(%q),SessionAffinityFormat:llm.SessionAffinityFormat(%q)},",
			value.DeveloperRole, value.EncryptedReasoning, value.StrictTools, value.LongCacheRetention, value.ExplicitPromptCacheMode, value.SessionAffinityFormat)
	}
	if value := compatibility.Anthropic; value != nil {
		fmt.Fprintf(output, "Anthropic:&llm.AnthropicCompatibility{EagerToolInputStreaming:llm.CompatibilityToggle(%q),LongCacheRetention:llm.CompatibilityToggle(%q),SessionAffinity:llm.CompatibilityToggle(%q),CacheControlOnTools:llm.CompatibilityToggle(%q),Temperature:llm.CompatibilityToggle(%q),AdaptiveThinking:llm.CompatibilityToggle(%q),EmptyThinkingSignature:llm.CompatibilityToggle(%q),StrictTools:llm.CompatibilityToggle(%q)},",
			value.EagerToolInputStreaming, value.LongCacheRetention, value.SessionAffinity, value.CacheControlOnTools,
			value.Temperature, value.AdaptiveThinking, value.EmptyThinkingSignature, value.StrictTools)
	}
	output.WriteString("}")
}

func number(value float64) string {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		panic("validated cost contains non-finite number")
	}
	return strconv.FormatFloat(value, 'g', -1, 64)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "cataloggen:", err)
	os.Exit(1)
}
