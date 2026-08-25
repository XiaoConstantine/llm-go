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

func TestRequestValidatesToolChoiceAndCacheControls(t *testing.T) {
	base := Request{
		Messages: []Message{{Role: RoleUser}},
		Tools:    []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)}},
	}
	for _, choice := range []ToolChoice{
		{}, {Mode: ToolChoiceNone}, {Mode: ToolChoiceRequired}, {Mode: ToolChoiceNamed, Name: "lookup"},
	} {
		request := base
		request.ToolChoice = choice
		if err := request.Validate(); err != nil {
			t.Errorf("Validate(%#v) error = %v", choice, err)
		}
	}
	invalid := base
	invalid.ToolChoice = ToolChoice{Mode: ToolChoiceNamed, Name: "missing"}
	if err := invalid.Validate(); err == nil {
		t.Fatal("Validate(named missing tool) succeeded")
	}
	invalid = base
	invalid.CacheRetention = CacheRetentionNone
	invalid.CacheKey = "cache"
	if err := invalid.Validate(); err == nil {
		t.Fatal("Validate(cache disabled with key) succeeded")
	}
	withoutCache := base
	withoutCache.CacheRetention = CacheRetentionNone
	withoutCache.SessionID = "session"
	if err := withoutCache.Validate(); err != nil {
		t.Fatalf("Validate(cache disabled with session affinity) error = %v", err)
	}
	valid := base
	valid.CacheRetention = CacheRetentionLong
	valid.CacheKey = "stable-key"
	valid.SessionID = "session"
	if err := valid.Validate(); err != nil {
		t.Fatalf("Validate(cache controls) error = %v", err)
	}
}

func TestModelInfoIncludesProtocolAndTokenLimits(t *testing.T) {
	model := Model{Provider: "provider", ID: "model", API: APIOpenAIResponses, ContextWindow: 128_000, MaxOutputTokens: 16_384}
	info := model.Info()
	if info.API != model.API || info.ContextWindow != model.ContextWindow || info.MaxOutputTokens != model.MaxOutputTokens {
		t.Fatalf("Model.Info() = %#v", info)
	}
}

func TestResponseText(t *testing.T) {
	response := Response{Message: Message{Content: []Part{{Text: "ok"}}}}

	if got, want := response.Text(), "ok"; got != want {
		t.Fatalf("Response.Text() = %q, want %q", got, want)
	}
}
