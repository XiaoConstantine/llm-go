package llm

import (
	"strings"
	"testing"
)

func TestTransformHistoryWarningsAndOwnership(t *testing.T) {
	history := []Message{{Role: RoleUser, Content: []Part{{Kind: PartImage, Data: []byte{1}, MediaType: "image/png"}, {Kind: PartAudio, Data: []byte{2}, MediaType: "audio/wav"}}}, {Role: RoleAssistant, Content: []Part{{Text: "visible"}}, ProviderData: []byte(`{"signature":"secret"}`)}}
	source := ModelInfo{Provider: "a", Model: "one", API: "api-a", Capabilities: []Capability{CapabilityGeneration, CapabilityVision, CapabilityAudio}}
	target := ModelInfo{Provider: "b", Model: "two", API: "api-b", Capabilities: []Capability{CapabilityGeneration}}
	output, warnings, err := TransformHistory(source, target, history)
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 3 || warnings[0].Code != TransformReplacedImage || warnings[1].Code != TransformReplacedAudio || warnings[2].Code != TransformRemovedProviderData {
		t.Fatalf("warnings = %#v", warnings)
	}
	if output[0].Content[0].Kind != PartText || output[1].Text() != "visible" || len(output[1].ProviderData) != 0 {
		t.Fatalf("output = %#v", output)
	}
	output[0].Content[0].Text = "changed"
	output[1].Content[0].Text = "changed"
	if history[0].Content[0].Kind != PartImage || history[1].Content[0].Text != "visible" || len(history[1].ProviderData) == 0 {
		t.Fatal("input history was mutated")
	}

	toolHistory := []Message{{Role: RoleAssistant, ToolCalls: []ToolCall{{Name: "x", Arguments: []byte(`{}`)}}}}
	if _, _, err := TransformHistory(source, target, toolHistory); err == nil || !strings.Contains(err.Error(), "tool history") {
		t.Fatalf("tool transform error = %v", err)
	}
}

func TestTransformHistoryUnknownOrPartialIdentityDropsProviderData(t *testing.T) {
	history := []Message{{Role: RoleUser, ProviderData: []byte(`{"opaque":true}`)}}
	for _, test := range []struct {
		name           string
		source, target ModelInfo
	}{
		{name: "empty"},
		{name: "provider only", source: ModelInfo{Provider: "same"}, target: ModelInfo{Provider: "same"}},
		{name: "missing model", source: ModelInfo{Provider: "same", API: APIOpenAIResponses}, target: ModelInfo{Provider: "same", API: APIOpenAIResponses}},
		{name: "blank provider", source: ModelInfo{Provider: " ", API: APIOpenAIResponses, Model: "model"}, target: ModelInfo{Provider: " ", API: APIOpenAIResponses, Model: "model"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			output, warnings, err := TransformHistory(test.source, test.target, history)
			if err != nil {
				t.Fatal(err)
			}
			if len(output[0].ProviderData) != 0 || len(warnings) != 1 || warnings[0].Code != TransformRemovedProviderData {
				t.Fatalf("output/warnings = %#v / %#v", output, warnings)
			}
		})
	}
	identity := ModelInfo{Provider: "same", API: APIOpenAIResponses, Model: "model"}
	output, warnings, err := TransformHistory(identity, identity, history)
	if err != nil || len(warnings) != 0 || string(output[0].ProviderData) != string(history[0].ProviderData) {
		t.Fatalf("exact identity output/warnings/error = %#v / %#v / %v", output, warnings, err)
	}
}

func TestTransformHistoryChatToolResultImageCompatibility(t *testing.T) {
	history := []Message{
		{Role: RoleUser},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call", Name: "view", Arguments: []byte(`{}`)}}},
		{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call", Name: "view", Content: []Part{{Kind: PartImage, Data: []byte{1, 2}, MediaType: "image/png"}}}}},
	}
	source := ModelInfo{Provider: "openai", Model: "source", API: APIOpenAIChatCompletions, Capabilities: []Capability{CapabilityGeneration, CapabilityTools, CapabilityVision}}
	disabled := ModelInfo{Provider: "openai", Model: "target", API: APIOpenAIChatCompletions, Capabilities: []Capability{CapabilityGeneration, CapabilityTools, CapabilityVision}, Compatibility: &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{}}}
	converted, warnings, err := TransformHistory(source, disabled, history)
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 1 || warnings[0].Code != TransformReplacedImage || converted[2].ToolResults[0].Content[0].Kind != PartText {
		t.Fatalf("disabled transform = %#v, warnings %#v", converted, warnings)
	}
	enabled := disabled
	enabled.Compatibility = &ModelCompatibility{OpenAIChat: &OpenAIChatCompatibility{ToolResultImageFallback: CompatibilityEnabled}}
	converted, warnings, err = TransformHistory(source, enabled, history)
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 || converted[2].ToolResults[0].Content[0].Kind != PartImage {
		t.Fatalf("enabled transform = %#v, warnings %#v", converted, warnings)
	}
}
