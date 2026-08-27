package llm

import (
	"fmt"
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

func TestTransformHistoryRepairsMissingAndOrphanedToolResults(t *testing.T) {
	model := ModelInfo{Provider: "provider", Model: "model", API: APIOpenAIChatCompletions,
		Capabilities: []Capability{CapabilityGeneration, CapabilityTools}}
	history := []Message{
		{Role: RoleUser, Content: []Part{{Text: "start"}}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "call_one", Name: "one", Arguments: []byte(`{}`)},
			{ID: "call_two", Name: "two", Arguments: []byte(`{}`)},
		}},
		{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call_one", Name: "one", Content: []Part{{Text: "done"}}}}},
		{Role: RoleUser, Content: []Part{{Text: "continue"}}},
		{Role: RoleTool, ToolResults: []ToolResult{{CallID: "orphan", Name: "missing", Content: []Part{{Text: "late"}}}}},
	}
	output, warnings, err := TransformHistory(model, model, history)
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 5 || output[3].Role != RoleTool || len(output[3].ToolResults) != 1 ||
		output[3].ToolResults[0].CallID != "call_two" || !output[3].ToolResults[0].IsError || output[4].Role != RoleUser {
		t.Fatalf("repaired output = %#v", output)
	}
	if got := fmt.Sprint(warningCodes(warnings)); got != "[added_missing_tool_result removed_orphan_tool_result]" {
		t.Fatalf("warning codes = %s", got)
	}
	if err := (Request{Messages: output}).Validate(); err != nil {
		t.Fatalf("repaired output does not validate: %v", err)
	}
}

func TestTransformHistoryNormalizesToolCallIDsAndMatchingResults(t *testing.T) {
	source := ModelInfo{Provider: "google", Model: "source", API: APIGeminiGenerateContent,
		Capabilities: []Capability{CapabilityGeneration, CapabilityTools}}
	target := ModelInfo{Provider: "openai", Model: "target", API: APIOpenAIChatCompletions,
		Capabilities: []Capability{CapabilityGeneration, CapabilityTools}}
	longID := strings.Repeat("a", 80)
	history := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "call|unsafe", Name: "first", Arguments: []byte(`{}`)},
			{ID: longID, Name: "second", Arguments: []byte(`{}`)},
			{Name: "third", Arguments: []byte(`{}`)},
		}},
		{Role: RoleTool, ToolResults: []ToolResult{
			{CallID: "call|unsafe", Name: "wrong"},
			{CallID: longID, Name: "second"},
			{Name: "third"},
		}},
	}
	output, warnings, err := TransformHistory(source, target, history)
	if err != nil {
		t.Fatal(err)
	}
	calls := output[0].ToolCalls
	results := output[1].ToolResults
	if len(calls) != 3 || len(results) != 3 {
		t.Fatalf("calls/results = %#v / %#v", calls, results)
	}
	seen := make(map[string]struct{}, len(calls))
	for index, call := range calls {
		if call.ID == "" || len(call.ID) > 40 || strings.ContainsAny(call.ID, "|") {
			t.Errorf("calls[%d].ID = %q", index, call.ID)
		}
		if _, duplicate := seen[call.ID]; duplicate {
			t.Errorf("duplicate normalized ID %q", call.ID)
		}
		seen[call.ID] = struct{}{}
		if results[index].CallID != call.ID {
			t.Errorf("results[%d].CallID = %q, want %q", index, results[index].CallID, call.ID)
		}
	}
	if results[0].Name != "first" {
		t.Fatalf("corrected result name = %q", results[0].Name)
	}
	if got := fmt.Sprint(warningCodes(warnings)); got != "[normalized_tool_call_id normalized_tool_call_id normalized_tool_call_id corrected_tool_result_name]" {
		t.Fatalf("warning codes = %s", got)
	}
}

func TestTransformHistoryNormalizesDuplicateToolCallIDs(t *testing.T) {
	model := ModelInfo{Provider: "openai", Model: "model", API: APIOpenAIChatCompletions,
		Capabilities: []Capability{CapabilityGeneration, CapabilityTools}}
	history := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{ID: "duplicate", Name: "one", Arguments: []byte(`{}`)},
			{ID: "duplicate", Name: "two", Arguments: []byte(`{}`)},
			{ID: "duplicate", Name: "three", Arguments: []byte(`{}`)},
		}},
		{Role: RoleTool, ToolResults: []ToolResult{
			{CallID: "duplicate", Name: "one"},
			{CallID: "duplicate", Name: "two"},
			{CallID: "duplicate", Name: "three"},
		}},
	}
	output, warnings, err := TransformHistory(model, model, history)
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]struct{})
	for index, call := range output[0].ToolCalls {
		if _, duplicate := seen[call.ID]; duplicate {
			t.Fatalf("tool call %d repeats normalized ID %q", index, call.ID)
		}
		seen[call.ID] = struct{}{}
		if output[1].ToolResults[index].CallID != call.ID {
			t.Fatalf("tool result %d ID = %q, want %q", index, output[1].ToolResults[index].CallID, call.ID)
		}
	}
	if got := fmt.Sprint(warningCodes(warnings)); got != "[normalized_tool_call_id normalized_tool_call_id]" {
		t.Fatalf("warning codes = %s", got)
	}
}

func TestTransformHistoryPreservesDeferredToolMarkersWithoutDefinitions(t *testing.T) {
	model := ModelInfo{Provider: "anthropic", Model: "model", API: APIAnthropicMessages,
		Capabilities: []Capability{CapabilityGeneration, CapabilityTools}}
	history := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call", Name: "search", Arguments: []byte(`{}`)}}},
		{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call", Name: "search", AddedToolNames: []string{"loaded"}}}},
	}
	output, warnings, err := TransformHistory(model, model, history)
	if err != nil || len(warnings) != 0 {
		t.Fatalf("TransformHistory() = (%#v, %#v, %v)", output, warnings, err)
	}
	if got := fmt.Sprint(output[1].ToolResults[0].AddedToolNames); got != "[loaded]" {
		t.Fatalf("AddedToolNames = %s", got)
	}
	output[1].ToolResults[0].AddedToolNames[0] = "changed"
	if history[1].ToolResults[0].AddedToolNames[0] != "loaded" {
		t.Fatal("output aliases source AddedToolNames")
	}
}

func TestTransformHistoryDropsProviderDataFromRepairedSameModelMessage(t *testing.T) {
	model := ModelInfo{Provider: "provider", Model: "model", API: APIOpenAIChatCompletions,
		Capabilities: []Capability{CapabilityGeneration, CapabilityTools}}
	history := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call", Name: "search", Arguments: []byte(`{}`)}}},
		{Role: RoleTool, ProviderData: []byte(`{"call_id":"call","name":"wrong"}`),
			ToolResults: []ToolResult{{CallID: "call", Name: "wrong"}}},
	}
	output, warnings, err := TransformHistory(model, model, history)
	if err != nil {
		t.Fatal(err)
	}
	if output[1].ToolResults[0].Name != "search" || len(output[1].ProviderData) != 0 {
		t.Fatalf("repaired output = %#v", output)
	}
	if got := fmt.Sprint(warningCodes(warnings)); got != "[corrected_tool_result_name removed_provider_data]" {
		t.Fatalf("warning codes = %s", got)
	}
}

func warningCodes(warnings []TransformWarning) []TransformWarningCode {
	codes := make([]TransformWarningCode, len(warnings))
	for index, warning := range warnings {
		codes[index] = warning.Code
	}
	return codes
}
