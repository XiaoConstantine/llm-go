package responses

import (
	"context"
	jsonv2 "encoding/json/v2"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestToolChoiceCacheAndImageToolResultWire(t *testing.T) {
	client, err := New(Config{Model: "model", APIKey: "key", Capabilities: []llm.Capability{llm.CapabilityTools, llm.CapabilityVision}})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "go"}}},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "inspect", Arguments: []byte(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Text: "screen"}, {Kind: llm.PartImage, Data: []byte{1}, MediaType: "image/png"}}}}},
		},
		Tools:          []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice:     llm.ToolChoice{Mode: llm.ToolChoiceNamed, Name: "inspect"},
		CacheRetention: llm.CacheRetentionLong, CacheKey: "stable",
	}
	params, err := client.prepare(context.Background(), "generate", request, false)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := jsonv2.Marshal(params)
	text := string(body)
	for _, want := range []string{
		`"tool_choice":{"name":"inspect","type":"function"}`,
		`"prompt_cache_key":"stable"`, `"prompt_cache_retention":"24h"`,
		`"type":"function_call_output"`, `"image_url":"data:image/png;base64,AQ==","type":"input_image"`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("wire = %s, want %s", text, want)
		}
	}
}
