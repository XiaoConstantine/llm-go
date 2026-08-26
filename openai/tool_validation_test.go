package openai

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestStreamRejectsInvalidCompletedToolArgumentsBeforeDelivery(t *testing.T) {
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
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","properties":{"count":{"type":"integer","minimum":1}}}`)}}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	var terminal error
	for {
		chunk, err := stream.Recv()
		if len(chunk.ToolCalls) != 0 {
			t.Fatalf("invalid ToolCalls were delivered: %#v", chunk.ToolCalls)
		}
		if err != nil {
			terminal = err
			break
		}
	}
	var modelErr *llm.Error
	if !errors.As(terminal, &modelErr) || modelErr.Kind != llm.KindMalformedResponse || modelErr.Op != "stream" {
		t.Fatalf("terminal = %v (%#v)", terminal, modelErr)
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

func TestGenerateValidatesCompletedToolArgumentsAgainstSchema(t *testing.T) {
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
	response, err := client.Generate(context.Background(), request)
	if response != nil || err == nil || !strings.Contains(err.Error(), `$.count`) {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse {
		t.Fatalf("error classification = %#v", modelErr)
	}
}
