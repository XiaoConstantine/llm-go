package llm

import (
	"fmt"
	"slices"
)

// CompatibilityToggle is a tri-state model compatibility override. Default
// leaves protocol behavior unchanged; Enabled and Disabled explicitly opt in or
// out. The zero value is Default.
type CompatibilityToggle string

const (
	CompatibilityDefault  CompatibilityToggle = ""
	CompatibilityEnabled  CompatibilityToggle = "enabled"
	CompatibilityDisabled CompatibilityToggle = "disabled"
)

// MaxTokensField identifies the Chat Completions request field used for an
// output-token limit. The zero value uses the protocol default.
type MaxTokensField string

const (
	MaxTokensFieldDefault    MaxTokensField = ""
	MaxTokensFieldCompletion MaxTokensField = "max_completion_tokens"
	MaxTokensFieldLegacy     MaxTokensField = "max_tokens"
)

// InstructionRole selects the Chat Completions role used for canonical system
// instructions. The zero value uses system.
type InstructionRole string

const (
	InstructionRoleDefault   InstructionRole = ""
	InstructionRoleSystem    InstructionRole = "system"
	InstructionRoleDeveloper InstructionRole = "developer"
)

// ThinkingFormat identifies an OpenAI-compatible reasoning request convention.
// The zero value uses reasoning_effort.
type ThinkingFormat string

const (
	ThinkingFormatDefault    ThinkingFormat = ""
	ThinkingFormatOpenAI     ThinkingFormat = "openai"
	ThinkingFormatOpenRouter ThinkingFormat = "openrouter"
	ThinkingFormatDeepSeek   ThinkingFormat = "deepseek"
	ThinkingFormatTogether   ThinkingFormat = "together"
	ThinkingFormatZAI        ThinkingFormat = "zai"
	ThinkingFormatQwen       ThinkingFormat = "qwen"
	ThinkingFormatString     ThinkingFormat = "string-thinking"
)

// SessionAffinityFormat identifies provider session-affinity header conventions.
// The zero value is resolved from the configured provider endpoint.
type SessionAffinityFormat string

const (
	SessionAffinityDefault         SessionAffinityFormat = ""
	SessionAffinityOpenAI          SessionAffinityFormat = "openai"
	SessionAffinityOpenAINoSession SessionAffinityFormat = "openai-nosession"
	SessionAffinityOpenRouter      SessionAffinityFormat = "openrouter"
)

// CacheControlFormat identifies a prompt-cache marker convention. Empty means
// the protocol default. Metadata support does not by itself add cache markers
// to a request that contains none.
type CacheControlFormat string

const (
	CacheControlDefault   CacheControlFormat = ""
	CacheControlAnthropic CacheControlFormat = "anthropic"
)

// OpenAIChatCompatibility describes model-specific differences among OpenAI
// Chat Completions-compatible APIs.
type OpenAIChatCompatibility struct {
	MaxTokensField           MaxTokensField
	InstructionRole          InstructionRole
	ThinkingFormat           ThinkingFormat
	ReasoningEffort          CompatibilityToggle
	StreamingUsage           CompatibilityToggle
	FinishReason             CompatibilityToggle
	ToolResultName           CompatibilityToggle
	ToolResultImageFallback  CompatibilityToggle
	AssistantAfterToolResult CompatibilityToggle
	ReasoningContentReplay   CompatibilityToggle
	StrictTools              CompatibilityToggle
	CacheControlFormat       CacheControlFormat
	LongCacheRetention       CompatibilityToggle
	// SessionAffinity enables SessionID headers for this compatible endpoint.
	SessionAffinity CompatibilityToggle
	// SessionAffinityFormat selects the header convention when enabled.
	SessionAffinityFormat SessionAffinityFormat
}

// OpenAIResponsesCompatibility describes model-specific differences among
// OpenAI Responses-compatible APIs.
type OpenAIResponsesCompatibility struct {
	DeveloperRole      CompatibilityToggle
	EncryptedReasoning CompatibilityToggle
	StrictTools        CompatibilityToggle
	// AdditionalTools enables replaying deferred tool definitions as
	// additional_tools input items.
	AdditionalTools CompatibilityToggle
	// ToolSearch enables client tool-search call/output replay when
	// AdditionalTools is not enabled.
	ToolSearch              CompatibilityToggle
	LongCacheRetention      CompatibilityToggle
	ExplicitPromptCacheMode CompatibilityToggle
	// SessionAffinityFormat selects the SessionID header convention. Its zero
	// value auto-detects native OpenAI or OpenRouter from provider routing.
	SessionAffinityFormat SessionAffinityFormat
}

// GeminiCompatibility describes model-specific differences among Gemini
// GenerateContent-compatible APIs.
type GeminiCompatibility struct {
	// ThinkingLevelsOnly restricts explicit reasoning controls to low, medium,
	// and high effort. Token budgets, disabled reasoning, and other effort
	// levels are rejected before provider I/O.
	ThinkingLevelsOnly CompatibilityToggle
}

// AnthropicCompatibility describes model-specific differences among Anthropic
// Messages-compatible APIs.
type AnthropicCompatibility struct {
	EagerToolInputStreaming CompatibilityToggle
	LongCacheRetention      CompatibilityToggle
	SessionAffinity         CompatibilityToggle
	CacheControlOnTools     CompatibilityToggle
	Temperature             CompatibilityToggle
	AdaptiveThinking        CompatibilityToggle
	EmptyThinkingSignature  CompatibilityToggle
	StrictTools             CompatibilityToggle
	// ToolReferences enables defer_loading definitions and tool_reference
	// result content.
	ToolReferences CompatibilityToggle
}

// ModelCompatibility contains at most one protocol-specific compatibility
// block. Catalog validation requires that block to match the model API.
type ModelCompatibility struct {
	OpenAIChat      *OpenAIChatCompatibility
	OpenAIResponses *OpenAIResponsesCompatibility
	Gemini          *GeminiCompatibility
	Anthropic       *AnthropicCompatibility
}

// Validate reports whether compatibility is well formed for api.
func (c *ModelCompatibility) Validate(api API) error {
	if c == nil {
		return nil
	}
	blocks := 0
	if c.OpenAIChat != nil {
		blocks++
	}
	if c.OpenAIResponses != nil {
		blocks++
	}
	if c.Gemini != nil {
		blocks++
	}
	if c.Anthropic != nil {
		blocks++
	}
	if blocks == 0 {
		return fmt.Errorf("compatibility contains no protocol block")
	}
	if blocks > 1 {
		return fmt.Errorf("compatibility contains more than one protocol block")
	}
	if c.OpenAIChat != nil {
		if api != APIOpenAIChatCompletions {
			return fmt.Errorf("OpenAI Chat compatibility requires API %q", APIOpenAIChatCompletions)
		}
		if err := c.OpenAIChat.validate(); err != nil {
			return fmt.Errorf("OpenAI Chat compatibility: %w", err)
		}
	}
	if c.OpenAIResponses != nil {
		if api != APIOpenAIResponses && api != APIAzureOpenAIResponses && api != APIOpenAICodexResponses {
			return fmt.Errorf("OpenAI Responses compatibility requires a Responses API")
		}
		if err := c.OpenAIResponses.validate(); err != nil {
			return fmt.Errorf("OpenAI Responses compatibility: %w", err)
		}
	}
	if c.Gemini != nil {
		if api != APIGeminiGenerateContent && api != APIGoogleVertex {
			return fmt.Errorf("gemini compatibility requires a generate-content API")
		}
		if err := c.Gemini.validate(); err != nil {
			return fmt.Errorf("gemini compatibility: %w", err)
		}
	}
	if c.Anthropic != nil {
		if api != APIAnthropicMessages {
			return fmt.Errorf("anthropic compatibility requires API %q", APIAnthropicMessages)
		}
		if err := c.Anthropic.validate(); err != nil {
			return fmt.Errorf("anthropic compatibility: %w", err)
		}
	}
	return nil
}

func (c OpenAIChatCompatibility) validate() error {
	if err := validateEnum("max tokens field", string(c.MaxTokensField), "", string(MaxTokensFieldCompletion), string(MaxTokensFieldLegacy)); err != nil {
		return err
	}
	if err := validateEnum("instruction role", string(c.InstructionRole), "", string(InstructionRoleSystem), string(InstructionRoleDeveloper)); err != nil {
		return err
	}
	if err := validateEnum("thinking format", string(c.ThinkingFormat), "", string(ThinkingFormatOpenAI), string(ThinkingFormatOpenRouter), string(ThinkingFormatDeepSeek), string(ThinkingFormatTogether), string(ThinkingFormatZAI), string(ThinkingFormatQwen), string(ThinkingFormatString)); err != nil {
		return err
	}
	if err := validateEnum("cache-control format", string(c.CacheControlFormat), "", string(CacheControlAnthropic)); err != nil {
		return err
	}
	if err := validateSessionAffinityFormat(c.SessionAffinityFormat); err != nil {
		return err
	}
	return validateToggles(c.ReasoningEffort, c.StreamingUsage, c.FinishReason, c.ToolResultName,
		c.ToolResultImageFallback, c.AssistantAfterToolResult, c.ReasoningContentReplay, c.StrictTools,
		c.LongCacheRetention, c.SessionAffinity)
}

func (c OpenAIResponsesCompatibility) validate() error {
	if err := validateSessionAffinityFormat(c.SessionAffinityFormat); err != nil {
		return err
	}
	return validateToggles(c.DeveloperRole, c.EncryptedReasoning, c.StrictTools, c.AdditionalTools,
		c.ToolSearch, c.LongCacheRetention, c.ExplicitPromptCacheMode)
}

func validateSessionAffinityFormat(value SessionAffinityFormat) error {
	return validateEnum("session-affinity format", string(value), "", string(SessionAffinityOpenAI),
		string(SessionAffinityOpenAINoSession), string(SessionAffinityOpenRouter))
}

func (c GeminiCompatibility) validate() error {
	return validateToggles(c.ThinkingLevelsOnly)
}

func (c AnthropicCompatibility) validate() error {
	return validateToggles(c.EagerToolInputStreaming, c.LongCacheRetention, c.SessionAffinity,
		c.CacheControlOnTools, c.Temperature, c.AdaptiveThinking, c.EmptyThinkingSignature, c.StrictTools,
		c.ToolReferences)
}

func validateToggles(values ...CompatibilityToggle) error {
	for _, value := range values {
		if err := validateEnum("compatibility toggle", string(value), "", string(CompatibilityEnabled), string(CompatibilityDisabled)); err != nil {
			return err
		}
	}
	return nil
}

func validateEnum(name, value string, allowed ...string) error {
	if slices.Contains(allowed, value) {
		return nil
	}
	return fmt.Errorf("%s %q is invalid", name, value)
}

func cloneModelCompatibility(compatibility *ModelCompatibility) *ModelCompatibility {
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
