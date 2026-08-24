package llm

import "testing"

func TestMessageText(t *testing.T) {
	message := Message{Content: []Part{
		{Text: "hello "},
		{Kind: PartImage, Data: []byte("image")},
		{Kind: PartText, Text: "world"},
	}}

	if got, want := message.Text(), "hello world"; got != want {
		t.Fatalf("Message.Text() = %q, want %q", got, want)
	}
}

func TestToolMessageText(t *testing.T) {
	message := Message{Role: RoleTool, ToolResults: []ToolResult{
		{Name: "lookup", Content: []Part{{Text: "first"}}},
		{Name: "lookup", Content: []Part{
			{Kind: PartImage, Data: []byte("image")},
			{Text: "second"},
		}},
	}}

	if got, want := message.Text(), "firstsecond"; got != want {
		t.Fatalf("Message.Text() = %q, want %q", got, want)
	}
}

func TestModelInfoOwnsCapabilities(t *testing.T) {
	model := Model{
		Provider:     "provider",
		ID:           "model",
		Capabilities: []Capability{CapabilityGeneration, CapabilityStreaming},
		Cost:         &ModelCost{Input: 1, Tiers: []ModelCostTier{{InputTokensAbove: 1_000, Input: 2}}},
	}
	info := model.Info()
	info.Capabilities[0] = CapabilityAudio
	info.Cost.Input = 99
	info.Cost.Tiers[0].Input = 99
	if model.Capabilities[0] != CapabilityGeneration {
		t.Fatalf("Model.Info() returned model-owned capabilities: %v", model.Capabilities)
	}
	if model.Cost.Input != 1 || model.Cost.Tiers[0].Input != 2 {
		t.Fatalf("Model.Info() returned model-owned cost: %#v", model.Cost)
	}
	if info.Provider != model.Provider || info.Model != model.ID {
		t.Fatalf("Model.Info() = %#v, want provider %q model %q", info, model.Provider, model.ID)
	}
}

func TestResponseText(t *testing.T) {
	response := Response{Message: Message{Content: []Part{{Text: "ok"}}}}

	if got, want := response.Text(), "ok"; got != want {
		t.Fatalf("Response.Text() = %q, want %q", got, want)
	}
}
