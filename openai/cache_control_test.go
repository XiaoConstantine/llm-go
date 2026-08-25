package openai

import (
	jsonv2 "encoding/json/v2"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestChatCacheControlAnthropicChoosesCacheableText(t *testing.T) {
	compatibility := llm.OpenAIChatCompatibility{MaxTokensField: llm.MaxTokensFieldCompletion, CacheControlFormat: llm.CacheControlAnthropic}
	tool := llm.Tool{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}
	call := llm.ToolCall{ID: "call", Name: "inspect", Arguments: []byte(`{}`)}
	for _, test := range []struct {
		name    string
		request llm.Request
		want    string
	}{
		{
			name: "final assistant",
			request: llm.Request{Messages: []llm.Message{
				{Role: llm.RoleSystem, Content: []llm.Part{{Text: "system"}}},
				{Role: llm.RoleUser, Content: []llm.Part{{Text: "user"}}},
				{Role: llm.RoleAssistant, Content: []llm.Part{{Text: "assistant"}}},
			}, CacheRetention: llm.CacheRetentionShort},
			want: `"role":"assistant","content":[{"type":"text","text":"assistant","cache_control":{"type":"ephemeral"}}]`,
		},
		{
			name: "final tool result",
			request: llm.Request{
				Messages: []llm.Message{
					{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{call}},
					{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Text: "result"}}}}},
				},
				Tools: []llm.Tool{tool}, CacheRetention: llm.CacheRetentionShort,
			},
			want: `"role":"tool","content":[{"type":"text","text":"result","cache_control":{"type":"ephemeral"}}],"tool_call_id":"call"`,
		},
		{
			name: "trailing image",
			request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{
				{Text: "caption"}, {Kind: llm.PartImage, Data: []byte{1}, MediaType: "image/png"},
			}}}, CacheRetention: llm.CacheRetentionShort},
			want: `"content":[{"type":"text","text":"caption","cache_control":{"type":"ephemeral"}},{"type":"image_url","image_url":{"url":"data:image/png;base64,AQ=="}}]`,
		},
		{
			name: "image only final message",
			request: llm.Request{Messages: []llm.Message{
				{Role: llm.RoleAssistant, Content: []llm.Part{{Text: "cache me"}}},
				{Role: llm.RoleUser, Content: []llm.Part{{Kind: llm.PartImage, Data: []byte{1}, MediaType: "image/png"}}},
			}, CacheRetention: llm.CacheRetentionShort},
			want: `"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AQ=="},"cache_control":{"type":"ephemeral"}}]`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			wire, err := newChatRequestFor("generate", "model", test.request, compatibility)
			if err != nil {
				t.Fatal(err)
			}
			body, err := jsonv2.Marshal(wire)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(body), test.want) {
				t.Fatalf("wire = %s, want %s", body, test.want)
			}
			if test.name == "final assistant" && !strings.Contains(string(body), `"role":"system","content":[{"type":"text","text":"system","cache_control":{"type":"ephemeral"}}]`) {
				t.Fatalf("system marker missing: %s", body)
			}
		})
	}
}

func TestChatCacheControlAnthropicSkipsNonCacheableParts(t *testing.T) {
	for _, content := range [][]contentPart{
		nil,
		{{Type: "image_url"}},
		{{Type: "image_url", ImageURL: &imageURL{}}},
		{{Type: "input_audio", InputAudio: &inputAudio{Data: "audio", Format: "wav"}}},
	} {
		message := chatMessage{Role: string(llm.RoleUser), Content: content}
		if addAnthropicCacheControlToContent(&message, &chatCacheControl{Type: "ephemeral"}) {
			t.Fatalf("marked non-cacheable content: %#v", content)
		}
	}
}
