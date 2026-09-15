package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestStreamDeliversSchemaInvalidCompletedToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(writer,
			`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"count\":0}"}}]},"finish_reason":null}]}`+"\n\n"+
				`data: {"id":"chat","model":"model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n"+
				"data: [DONE]\n\n")
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","properties":{"count":{"type":"integer","minimum":1}}}`)}}}
	returned, err := client.Stream(t.Context(), request)
	if err != nil {
		t.Fatal(err)
	}
	response, err := llm.CollectStructural(returned)
	if err != nil || len(response.Message.ToolCalls) != 1 || string(response.Message.ToolCalls[0].Arguments) != `{"count":0}` {
		t.Fatalf("returned tool call = %#v, %v", response, err)
	}
	if err := llm.ValidateToolCalls(request.Tools, response.Message.ToolCalls); err == nil {
		t.Fatal("explicit validation accepted count=0")
	}
}

func TestToolStrictnessPreferAndRequire(t *testing.T) {
	var calls atomic.Int32
	bodies := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		calls.Add(1)
		body, _ := io.ReadAll(request.Body)
		bodies <- string(body)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"id","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	config := Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityTools}, BaseURL: server.URL, HTTPClient: server.Client()}
	unsupported, err := NewWithCompatibility(config, &llm.OpenAIChatCompatibility{StrictTools: llm.CompatibilityDisabled})
	if err != nil {
		t.Fatal(err)
	}
	base := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "tool", InputSchema: []byte(`{"type":"object"}`), Strictness: llm.ToolStrictRequire}}}
	if response, err := unsupported.Generate(context.Background(), base); response != nil || err == nil {
		t.Fatalf("required Generate() = %#v, %v", response, err)
	}
	if calls.Load() != 0 {
		t.Fatalf("required HTTP calls = %d", calls.Load())
	}
	base.Tools[0].Strictness = llm.ToolStrictPrefer
	if _, err := unsupported.Generate(context.Background(), base); err != nil {
		t.Fatal(err)
	}
	if body := <-bodies; strings.Contains(body, `"strict":true`) {
		t.Fatalf("unsupported prefer body = %s", body)
	}

	supported, err := NewWithCompatibility(config, &llm.OpenAIChatCompatibility{StrictTools: llm.CompatibilityEnabled})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := supported.Generate(context.Background(), base); err != nil {
		t.Fatal(err)
	}
	if body := <-bodies; !strings.Contains(body, `"strict":true`) {
		t.Fatalf("supported prefer body = %s", body)
	}
}

func TestGenerateLeavesArgumentSchemasToExecutionOwner(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"id","model":"model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call","type":"function","function":{"name":"lookup","arguments":"{\"count\":0}"}}]},"finish_reason":"tool_calls"}]}`)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityTools}, BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","properties":{"count":{"type":"integer","minimum":1}},"required":["count"]}`)}}}
	response, err := client.Generate(t.Context(), request)
	if err != nil || response == nil || len(response.Message.ToolCalls) != 1 || string(response.Message.ToolCalls[0].Arguments) != `{"count":0}` {
		t.Fatalf("returned tool call = %#v, %v", response, err)
	}
	if err := llm.ValidateToolCalls(request.Tools, response.Message.ToolCalls); err == nil {
		t.Fatal("explicit validation accepted count=0")
	}
}
