package llm_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"reflect"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestDocumentPartLegacyWire(t *testing.T) {
	data := []byte("%PDF-fixture")
	encoded := base64.StdEncoding.EncodeToString(data)
	for _, test := range []struct {
		name, kind, mime, filename string
		data                       []byte
		want                       map[string]any
	}{
		{"chat-pdf", "chat", "application/pdf", "report.pdf", data, map[string]any{"type": "file", "file": map[string]any{"file_data": "data:application/pdf;base64," + encoded, "filename": "report.pdf"}}},
		{"chat-pdf-default-name", "chat", "application/pdf", "", data, map[string]any{"type": "file", "file": map[string]any{"file_data": "data:application/pdf;base64," + encoded, "filename": "part-1.pdf"}}},
		{"chat-pdf-id", "chat", "application/pdf", "ignored.pdf", []byte("file-fixture"), map[string]any{"type": "file", "file": map[string]any{"file_id": "file-fixture"}}},
		{"chat-text", "chat", "text/markdown", "ignored.md", []byte("text"), map[string]any{"type": "file", "file": map[string]any{"file_data": "dGV4dA=="}}},
		{"responses-pdf", "responses", "application/pdf", "report.pdf", data, map[string]any{"type": "input_file", "file_data": "data:application/pdf;base64," + encoded, "filename": "report.pdf"}},
		{"responses-pdf-default-name", "responses", "application/pdf", "", data, map[string]any{"type": "input_file", "file_data": "data:application/pdf;base64," + encoded, "filename": "part-1.pdf"}},
		{"anthropic-pdf", "anthropic", "application/pdf", "report.pdf", data, map[string]any{"type": "document", "title": "report pdf", "source": map[string]any{"type": "base64", "media_type": "application/pdf", "data": encoded}}},
		{"anthropic-text", "anthropic", "text/markdown", "notes.md", []byte("text"), map[string]any{"type": "document", "title": "notes md", "source": map[string]any{"type": "text", "media_type": "text/plain", "data": "text"}}},
		{"anthropic-empty-name", "anthropic", "text/plain", "", []byte("text"), map[string]any{"type": "document", "title": "Document", "source": map[string]any{"type": "text", "media_type": "text/plain", "data": "text"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			client, fixture := protocolClient(t, test.kind, false)
			request := protocolRequest()
			request.Messages[0].Content = append(request.Messages[0].Content, llm.Part{Kind: llm.PartFile, Data: test.data, MediaType: test.mime, Filename: test.filename})
			if _, err := client.Generate(context.Background(), request); err != nil {
				t.Fatal(err)
			}
			key := "messages"
			if test.kind == "responses" {
				key = "input"
			}
			messages := fixture.body[key].([]any)
			content := messages[0].(map[string]any)["content"].([]any)
			if !reflect.DeepEqual(content[1], test.want) {
				t.Fatalf("document = %#v; want %#v", content[1], test.want)
			}
			if string(request.Messages[0].Content[1].Data) != string(test.data) || request.Messages[0].Content[1].Filename != test.filename {
				t.Fatal("input document mutated")
			}
		})
	}
}

func TestDocumentPartRefusalBeforeIO(t *testing.T) {
	for _, test := range []struct {
		name, kind string
		change     func(*llm.Request)
	}{
		{"unsupported-mime", "chat", func(r *llm.Request) { r.Messages[0].Content[0].MediaType = "application/octet-stream" }},
		{"responses-text", "responses", func(r *llm.Request) { r.Messages[0].Content[0].MediaType = "text/plain" }},
		{"anthropic-unsupported", "anthropic", func(r *llm.Request) { r.Messages[0].Content[0].MediaType = "application/zip" }},
		{"codex-files", "codex", func(r *llm.Request) {}},
		{"empty-data", "chat", func(r *llm.Request) { r.Messages[0].Content[0].Data = nil }},
		{"invalid-filename", "responses", func(r *llm.Request) { r.Messages[0].Content[0].Filename = string([]byte{0xff}) }},
		{"invalid-text", "anthropic", func(r *llm.Request) {
			r.Messages[0].Content[0].MediaType = "text/plain"
			r.Messages[0].Content[0].Data = []byte{0xff}
		}},
		{"assistant-file", "chat", func(r *llm.Request) { r.Messages[0].Role = llm.RoleAssistant }},
		{"system-file", "anthropic", func(r *llm.Request) { r.Messages[0].Role = llm.RoleSystem }},
		{"tool-result-file", "responses", func(r *llm.Request) {
			file := r.Messages[0].Content[0]
			r.Tools = []llm.Tool{{Name: "tool", InputSchema: json.RawMessage(`{"type":"object"}`)}}
			r.Messages = []llm.Message{{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "tool", Arguments: json.RawMessage(`{}`)}}}, {Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Name: "tool", Content: []llm.Part{file}}}}}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			client, fixture := protocolClient(t, test.kind, false)
			request := protocolRequest()
			request.Messages[0].Content = []llm.Part{{Kind: llm.PartFile, Data: []byte("%PDF"), MediaType: "application/pdf"}}
			test.change(&request)
			if _, err := client.Generate(context.Background(), request); err == nil || fixture.calls != 0 {
				t.Fatalf("Generate error %v; HTTP calls %d", err, fixture.calls)
			}
			if _, err := client.Stream(context.Background(), request); err == nil || fixture.calls != 0 {
				t.Fatalf("Stream error %v; HTTP calls %d", err, fixture.calls)
			}
		})
	}
}
