package llm

import (
	"strings"
	"testing"
)

func TestModelCompatibilityValidation(t *testing.T) {
	validChat := &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{
		MaxTokensField:         MaxTokensFieldLegacy,
		InstructionRole:        InstructionRoleDeveloper,
		ThinkingFormat:         ThinkingFormatDeepSeek,
		ReasoningEffort:        CompatibilityDisabled,
		StreamingUsage:         CompatibilityEnabled,
		ReasoningContentReplay: CompatibilityEnabled,
		StrictTools:            CompatibilityDisabled,
		CacheControlFormat:     CacheControlAnthropic,
		LongCacheRetention:     CompatibilityDisabled,
	}}
	if err := validChat.Validate(APIOpenAIChatCompletions); err != nil {
		t.Fatalf("Validate(valid Chat) error = %v", err)
	}
	if err := (&ModelCompatibility{OpenAIResponses: &OpenAIResponsesCompatibility{
		EncryptedReasoning: CompatibilityEnabled,
		StrictTools:        CompatibilityDisabled,
	}}).Validate(APIOpenAIResponses); err != nil {
		t.Fatalf("Validate(valid Responses) error = %v", err)
	}
	if err := (&ModelCompatibility{OpenAIResponses: &OpenAIResponsesCompatibility{
		StrictTools: CompatibilityEnabled,
	}}).Validate(APIAzureOpenAIResponses); err != nil {
		t.Fatalf("Validate(valid Azure Responses) error = %v", err)
	}
	if err := (&ModelCompatibility{Anthropic: &AnthropicCompatibility{
		Temperature: CompatibilityDisabled,
		StrictTools: CompatibilityEnabled,
	}}).Validate(APIAnthropicMessages); err != nil {
		t.Fatalf("Validate(valid Anthropic) error = %v", err)
	}
	if err := (*ModelCompatibility)(nil).Validate(APIOpenAIChatCompletions); err != nil {
		t.Fatalf("Validate(nil) error = %v", err)
	}

	tests := []struct {
		name          string
		compatibility *ModelCompatibility
		api           API
		want          string
	}{
		{name: "empty", compatibility: &ModelCompatibility{}, api: APIOpenAIChatCompletions, want: "no protocol block"},
		{name: "multiple", compatibility: &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{}, Anthropic: &AnthropicCompatibility{}}, api: APIOpenAIChatCompletions, want: "more than one protocol block"},
		{name: "API mismatch", compatibility: validChat, api: APIAnthropicMessages, want: "requires API"},
		{name: "max tokens field", compatibility: &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{MaxTokensField: "future"}}, api: APIOpenAIChatCompletions, want: "max tokens field"},
		{name: "instruction role", compatibility: &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{InstructionRole: "future"}}, api: APIOpenAIChatCompletions, want: "instruction role"},
		{name: "thinking format", compatibility: &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{ThinkingFormat: "future"}}, api: APIOpenAIChatCompletions, want: "thinking format"},
		{name: "cache format", compatibility: &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{CacheControlFormat: "future"}}, api: APIOpenAIChatCompletions, want: "cache-control format"},
		{name: "Chat session format", compatibility: &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{SessionAffinityFormat: "future"}}, api: APIOpenAIChatCompletions, want: "session-affinity format"},
		{name: "Responses session format", compatibility: &ModelCompatibility{OpenAIResponses: &OpenAIResponsesCompatibility{SessionAffinityFormat: "future"}}, api: APIOpenAIResponses, want: "session-affinity format"},
		{name: "toggle", compatibility: &ModelCompatibility{OpenAIResponses: &OpenAIResponsesCompatibility{StrictTools: "sometimes"}}, api: APIOpenAIResponses, want: "compatibility toggle"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.compatibility.Validate(test.api)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Validate() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestModelInfoOwnsCompatibility(t *testing.T) {
	chat := &OpenAIChatCompatibility{MaxTokensField: MaxTokensFieldLegacy}
	model := Model{
		Provider:      "provider",
		ID:            "model",
		API:           APIOpenAIChatCompletions,
		Reasoning:     true,
		Compatibility: &ModelCompatibility{OpenAIChat: chat},
	}
	first := model.Info()
	if !first.Reasoning || first.Compatibility == nil || first.Compatibility.OpenAIChat == nil {
		t.Fatalf("Info().Compatibility = %#v", first.Compatibility)
	}
	first.Compatibility.OpenAIChat.MaxTokensField = MaxTokensFieldCompletion
	if chat.MaxTokensField != MaxTokensFieldLegacy {
		t.Fatalf("Info() aliased model compatibility: %#v", chat)
	}
	second := model.Info()
	if second.Compatibility.OpenAIChat.MaxTokensField != MaxTokensFieldLegacy {
		t.Fatalf("second Info().Compatibility = %#v", second.Compatibility)
	}
}
