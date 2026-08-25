package openai

import (
	jsonv2 "encoding/json/v2"
	"fmt"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestConsecutiveToolMessagesDelaySyntheticImageFallback(t *testing.T) {
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
				{ID: "a", Name: "inspect", Arguments: []byte(`{}`)},
				{ID: "b", Name: "inspect", Arguments: []byte(`{}`)},
			}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "a", Content: []llm.Part{{Kind: llm.PartImage, Data: []byte{1}, MediaType: "image/png"}}}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "b", Content: []llm.Part{{Text: "done"}}}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "continue"}}},
		},
		Tools: []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}},
	}
	if err := request.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
	wire, err := newChatRequestFor("generate", "model", request, llm.OpenAIChatCompatibility{
		MaxTokensField: llm.MaxTokensFieldCompletion, ToolResultImageFallback: llm.CompatibilityEnabled,
		AssistantAfterToolResult: llm.CompatibilityEnabled,
	})
	if err != nil {
		t.Fatal(err)
	}
	var roles []string
	for _, message := range wire.Messages {
		roles = append(roles, message.Role)
	}
	want := []string{"assistant", "tool", "tool", "assistant", "user", "user"}
	if fmt.Sprint(roles) != fmt.Sprint(want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	if wire.Messages[1].ToolCallID != "a" || wire.Messages[2].ToolCallID != "b" || wire.Messages[3].Content != assistantAfterToolResultText {
		t.Fatalf("wire messages = %#v", wire.Messages)
	}
}

func TestToolImageFallbackBridgePrecedesSyntheticUser(t *testing.T) {
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "inspect", Arguments: []byte(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Kind: llm.PartImage, Data: []byte{1}, MediaType: "image/png"}}}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "continue"}}},
		},
		Tools: []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}},
	}
	wire, err := newChatRequestFor("generate", "model", request, llm.OpenAIChatCompatibility{MaxTokensField: llm.MaxTokensFieldCompletion, ToolResultImageFallback: llm.CompatibilityEnabled, AssistantAfterToolResult: llm.CompatibilityEnabled})
	if err != nil {
		t.Fatal(err)
	}
	var roles []string
	for _, message := range wire.Messages {
		roles = append(roles, message.Role)
	}
	want := []string{"assistant", "tool", "assistant", "user", "user"}
	if fmt.Sprint(roles) != fmt.Sprint(want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	if wire.Messages[2].Content != assistantAfterToolResultText {
		t.Fatalf("bridge content = %#v", wire.Messages[2].Content)
	}
}

func TestChatImageToolFallbackRequiresExplicitCompatibility(t *testing.T) {
	client, err := New(Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityTools, llm.CapabilityVision}})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "inspect", Arguments: []byte(`{}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Kind: llm.PartImage, Data: []byte{1}, MediaType: "image/png"}}}}},
	}, Tools: []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}}}
	if err := client.checkCapabilities("generate", request); err == nil || !strings.Contains(err.Error(), "explicit synthetic-user") {
		t.Fatalf("checkCapabilities() error = %v", err)
	}
}

func TestChatToolChoiceCacheAndImageToolFallbackWire(t *testing.T) {
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "go"}}},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "inspect", Arguments: []byte(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Kind: llm.PartImage, Data: []byte{1, 2, 3}, MediaType: "image/png"}}}}},
		},
		Tools:          []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice:     llm.ToolChoice{Mode: llm.ToolChoiceNamed, Name: "inspect"},
		CacheRetention: llm.CacheRetentionLong, SessionID: "session",
	}
	wire, err := newChatRequestFor("generate", "model", request, llm.OpenAIChatCompatibility{
		MaxTokensField: llm.MaxTokensFieldCompletion, ToolResultImageFallback: llm.CompatibilityEnabled,
		LongCacheRetention: llm.CompatibilityEnabled, CacheControlFormat: llm.CacheControlAnthropic,
	})
	if err != nil {
		t.Fatal(err)
	}
	body, _ := jsonv2.Marshal(wire)
	text := string(body)
	for _, want := range []string{
		`"tool_choice":{`, `"function":{"name":"inspect"}`,
		`"prompt_cache_key":"session"`, `"prompt_cache_retention":"24h"`,
		`"role":"tool","content":"(see attached image)"`,
		`"role":"user","content":[{"type":"text","text":"Attached image(s) from tool result:"`,
		`"image_url":{"url":"data:image/png;base64,AQID"}`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("wire = %s, want %s", text, want)
		}
	}
}
