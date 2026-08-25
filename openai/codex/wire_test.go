package codex

import (
	jsonv2 "encoding/json/v2"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestToolChoiceSessionAndImageToolResultWire(t *testing.T) {
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "inspect", Arguments: []byte(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Kind: llm.PartImage, Data: []byte{2}, MediaType: "image/jpeg"}}}}},
		},
		Tools:          []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice:     llm.ToolChoice{Mode: llm.ToolChoiceRequired},
		CacheRetention: llm.CacheRetentionShort, SessionID: "session",
	}
	params, err := requestToWire("generate", "model", request)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := jsonv2.Marshal(params)
	text := string(body)
	for _, want := range []string{`"tool_choice":"required"`, `"prompt_cache_key":"session"`, `"image_url":"data:image/jpeg;base64,Ag==","type":"input_image"`} {
		if !strings.Contains(text, want) {
			t.Errorf("wire = %s, want %s", text, want)
		}
	}
	request.ToolChoice = llm.ToolChoice{Mode: llm.ToolChoiceNamed, Name: "inspect"}
	if _, err := requestToWire("generate", "model", request); err == nil || !strings.Contains(err.Error(), "named tool choice") {
		t.Fatalf("named Codex tool choice error = %v", err)
	}
}
