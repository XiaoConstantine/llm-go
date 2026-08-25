package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
)

const validToolResponse = `{
	"id":"msg_tools",
	"type":"message",
	"role":"assistant",
	"content":[
		{"type":"text","text":"checking"},
		{"type":"tool_use","id":"toolu_one","name":"lookup","input":{"city":"Paris"}}
	],
	"model":"claude-served",
	"stop_reason":"tool_use",
	"stop_sequence":null,
	"usage":{"input_tokens":8,"output_tokens":5}
}`

func TestNewConfiguresToolCapability(t *testing.T) {
	client, err := New(Config{
		Model: "model",
		Capabilities: []llm.Capability{
			llm.CapabilityGeneration,
			llm.CapabilityTools,
			llm.CapabilityTools,
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	first := client.Info()
	second := client.Info()
	if len(first.Capabilities) != 2 || first.Capabilities[0] != llm.CapabilityGeneration ||
		first.Capabilities[1] != llm.CapabilityTools {
		t.Fatalf("Info() = %#v", first)
	}
	first.Capabilities[1] = llm.CapabilityStreaming
	if second.Capabilities[1] != llm.CapabilityTools {
		t.Fatal("Info returned aliased capabilities")
	}

	for _, capability := range []llm.Capability{
		llm.CapabilityJSON,
		llm.Capability("future"),
	} {
		t.Run(string(capability), func(t *testing.T) {
			client, err := New(Config{Model: "model", Capabilities: []llm.Capability{capability}})
			if client != nil {
				t.Fatalf("New() client = %#v", client)
			}
			requireModelError(t, err, llm.KindInvalidRequest, "configure")
		})
	}
}

func TestGenerateTranslatesToolConversationAndResponse(t *testing.T) {
	payloads := make(chan []byte, 1)
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			panic(err)
		}
		payloads <- body
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{
			"id":"msg_tools",
			"type":"message",
			"role":"assistant",
			"content":[
				{"type":"text","text":"I need two more lookups."},
				{"type":"tool_use","id":"toolu_response_1","name":"lookup","input":{"city":"Berlin"}},
				{"type":"tool_use","id":"toolu_response_2","name":"lookup","input":{"city":"Rome"}}
			],
			"model":"claude-served",
			"stop_reason":"tool_use",
			"usage":{"input_tokens":12,"output_tokens":7}
		}`)
	}))

	client, err := New(Config{
		Model:        "claude-request",
		Capabilities: []llm.Capability{llm.CapabilityTools},
		BaseURL:      server.URL + "/v1",
		HTTPClient:   server.Client(),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "Compare cities."}}},
			{
				Role:    llm.RoleAssistant,
				Content: []llm.Part{{Text: "I will look them up."}},
				ToolCalls: []llm.ToolCall{
					{Name: "lookup", Arguments: []byte(`{"city":"Paris"}`)},
					{Name: "lookup", Arguments: []byte(`{"city":"London"}`)},
				},
			},
			{
				Role: llm.RoleTool,
				ToolResults: []llm.ToolResult{
					{Name: "lookup", Content: []llm.Part{{Text: "15 C"}}},
					{Name: "lookup", Content: []llm.Part{{Text: ""}}, IsError: true},
				},
			},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "Continue."}}},
		},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a city",
			InputSchema: []byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
			Strict:      true,
		}},
	}
	response, err := client.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.FinishReason != llm.FinishReasonToolCall || response.Text() != "I need two more lookups." ||
		len(response.Message.ToolCalls) != 2 {
		t.Fatalf("Generate() response = %#v", response)
	}
	for index, want := range []struct{ id, city string }{{"toolu_response_1", "Berlin"}, {"toolu_response_2", "Rome"}} {
		call := response.Message.ToolCalls[index]
		if call.ID != want.id || call.Name != "lookup" {
			t.Fatalf("ToolCalls[%d] = %#v", index, call)
		}
		var arguments map[string]any
		if err := jsonv2.Unmarshal(call.Arguments, &arguments); err != nil || arguments["city"] != want.city {
			t.Fatalf("ToolCalls[%d].Arguments = %s, error = %v", index, call.Arguments, err)
		}
	}

	var payload map[string]any
	if err := jsonv2.Unmarshal(<-payloads, &payload); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	tools, ok := payload["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v", payload["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["name"] != "lookup" || tool["description"] != "Look up a city" || tool["strict"] != true {
		t.Fatalf("tool = %#v", tool)
	}
	schema, ok := tool["input_schema"].(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("input_schema = %#v", tool["input_schema"])
	}

	messages, ok := payload["messages"].([]any)
	if !ok || len(messages) != 4 {
		t.Fatalf("messages = %#v", payload["messages"])
	}
	assistant := messages[1].(map[string]any)
	assistantBlocks := assistant["content"].([]any)
	if assistant["role"] != "assistant" || len(assistantBlocks) != 3 {
		t.Fatalf("assistant message = %#v", assistant)
	}
	if assistantBlocks[0].(map[string]any)["text"] != "I will look them up." {
		t.Fatalf("assistant text = %#v", assistantBlocks[0])
	}
	for index, wantID := range []string{"toolu_llm_go_1", "toolu_llm_go_2"} {
		block := assistantBlocks[index+1].(map[string]any)
		if block["type"] != "tool_use" || block["id"] != wantID || block["name"] != "lookup" {
			t.Fatalf("assistant tool block %d = %#v", index, block)
		}
	}

	results := messages[2].(map[string]any)
	resultBlocks := results["content"].([]any)
	if results["role"] != "user" || len(resultBlocks) != 2 {
		t.Fatalf("tool result message = %#v", results)
	}
	firstResult := resultBlocks[0].(map[string]any)
	if firstResult["type"] != "tool_result" || firstResult["tool_use_id"] != "toolu_llm_go_1" || firstResult["content"] != "15 C" {
		t.Fatalf("first tool result = %#v", firstResult)
	}
	secondResult := resultBlocks[1].(map[string]any)
	if secondResult["tool_use_id"] != "toolu_llm_go_2" || secondResult["content"] != nil || secondResult["is_error"] != true {
		t.Fatalf("second tool result = %#v", secondResult)
	}
}

func TestGenerateToolPreflightAndNoIO(t *testing.T) {
	var calls atomic.Int64
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityTools},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, errors.New("unexpected request")
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	validTool := llm.Tool{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)}
	toolCall := func(id, name string) llm.ToolCall {
		return llm.ToolCall{ID: id, Name: name, Arguments: []byte(`{}`)}
	}
	toolResult := func(id, name string) llm.ToolResult {
		return llm.ToolResult{CallID: id, Name: name}
	}
	repeatedID := []llm.Message{
		{Role: llm.RoleUser},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{toolCall("same", "lookup")}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{toolResult("same", "lookup")}},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{toolCall("same", "lookup")}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{toolResult("same", "lookup")}},
	}
	for _, test := range []struct {
		name     string
		request  llm.Request
		contains string
	}{
		{name: "invalid definition name", request: llm.Request{Messages: textRequest("hello").Messages, Tools: []llm.Tool{{Name: "bad name", InputSchema: []byte(`{}`)}}}, contains: "must contain 1-64"},
		{name: "long definition name", request: llm.Request{Messages: textRequest("hello").Messages, Tools: []llm.Tool{{Name: strings.Repeat("a", 65), InputSchema: []byte(`{}`)}}}, contains: "must contain 1-64"},
		{name: "boolean schema", request: llm.Request{Messages: textRequest("hello").Messages, Tools: []llm.Tool{{Name: "lookup", InputSchema: []byte(`true`)}}}, contains: "must be a JSON object"},
		{name: "missing schema type", request: llm.Request{Messages: textRequest("hello").Messages, Tools: []llm.Tool{{Name: "lookup", InputSchema: []byte(`{}`)}}}, contains: `must declare top-level type "object"`},
		{name: "null schema type", request: llm.Request{Messages: textRequest("hello").Messages, Tools: []llm.Tool{{Name: "lookup", InputSchema: []byte(`{"type":null}`)}}}, contains: `must declare top-level type "object"`},
		{name: "non-object schema type", request: llm.Request{Messages: textRequest("hello").Messages, Tools: []llm.Tool{{Name: "lookup", InputSchema: []byte(`{"type":"array"}`)}}}, contains: `must declare top-level type "object"`},
		{name: "undeclared history", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{toolCall("call", "missing")}}, {Role: llm.RoleTool, ToolResults: []llm.ToolResult{toolResult("call", "missing")}}}, Tools: []llm.Tool{validTool}}, contains: "undeclared tool"},
		{name: "non-object arguments", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "lookup", Arguments: []byte(`[]`)}}}, {Role: llm.RoleTool, ToolResults: []llm.ToolResult{toolResult("call", "lookup")}}}, Tools: []llm.Tool{validTool}}, contains: "arguments must be a JSON object"},
		{name: "repeated history ID", request: llm.Request{Messages: repeatedID, Tools: []llm.Tool{validTool}}, contains: "repeats ID"},
		{name: "interrupted history", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{toolCall("call", "lookup")}}, {Role: llm.RoleUser}}, Tools: []llm.Tool{validTool}}, contains: "must contain tool results"},
		{name: "unfinished history", request: llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, {Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{toolCall("call", "lookup")}}}, Tools: []llm.Tool{validTool}}, contains: "messages end with 1 pending"},
	} {
		t.Run(test.name, func(t *testing.T) {
			response, err := client.Generate(context.Background(), test.request)
			if response != nil {
				t.Fatalf("Generate() response = %#v", response)
			}
			requireModelError(t, err, llm.KindInvalidRequest, "generate")
			if !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("Generate() error = %v, want %q", err, test.contains)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls.Load())
	}
}

func TestGenerateRejectsMalformedToolResponse(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "missing ID", body: strings.Replace(validToolResponse, `"id":"toolu_one",`, "", 1), want: "has no ID"},
		{name: "null ID", body: strings.Replace(validToolResponse, `"id":"toolu_one"`, `"id":null`, 1), want: "has no ID"},
		{name: "empty ID", body: strings.Replace(validToolResponse, `"id":"toolu_one"`, `"id":""`, 1), want: "has no ID"},
		{name: "duplicate ID", body: strings.Replace(validToolResponse, `{"type":"tool_use","id":"toolu_one","name":"lookup","input":{"city":"Paris"}}`, `{"type":"tool_use","id":"toolu_one","name":"lookup","input":{}},{"type":"tool_use","id":"toolu_one","name":"lookup","input":{}}`, 1), want: "repeats ID"},
		{name: "missing name", body: strings.Replace(validToolResponse, `"name":"lookup",`, "", 1), want: "invalid name"},
		{name: "invalid name", body: strings.Replace(validToolResponse, `"name":"lookup"`, `"name":"bad name"`, 1), want: "invalid name"},
		{name: "undeclared name", body: strings.Replace(validToolResponse, `"name":"lookup"`, `"name":"other"`, 1), want: "undeclared tool"},
		{name: "missing input", body: strings.Replace(validToolResponse, `,"input":{"city":"Paris"}`, "", 1), want: "JSON object"},
		{name: "null input", body: strings.Replace(validToolResponse, `"input":{"city":"Paris"}`, `"input":null`, 1), want: "JSON object"},
		{name: "array input", body: strings.Replace(validToolResponse, `"input":{"city":"Paris"}`, `"input":[]`, 1), want: "strict JSON object"},
		{name: "duplicate input field", body: strings.Replace(validToolResponse, `"input":{"city":"Paris"}`, `"input":{"city":"Paris","city":"Rome"}`, 1), want: "duplicate object member"},
		{name: "tool stop without call", body: strings.Replace(validToolResponse, `{"type":"text","text":"checking"},
		{"type":"tool_use","id":"toolu_one","name":"lookup","input":{"city":"Paris"}}`, `{"type":"text","text":"checking"}`, 1), want: "has no tool calls"},
		{name: "stop inconsistent with call", body: strings.Replace(validToolResponse, `"stop_reason":"tool_use"`, `"stop_reason":"end_turn"`, 1), want: "inconsistent with tool use"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := newStaticToolClient(t, test.body)
			response, err := client.Generate(context.Background(), toolRequest())
			if response != nil {
				t.Fatalf("Generate() response = %#v", response)
			}
			requireModelError(t, err, llm.KindMalformedResponse, "generate")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Generate() error = %q, want substring %q", err, test.want)
			}
		})
	}
}

func TestGenerateRejectsReusedHistoricalToolID(t *testing.T) {
	client := newStaticToolClient(t, validToolResponse)
	request := toolRequest()
	request.Messages = append([]llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "toolu_one", Name: "lookup", Arguments: []byte(`{}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "toolu_one", Name: "lookup"}}},
	}, request.Messages...)
	response, err := client.Generate(context.Background(), request)
	if response != nil {
		t.Fatalf("Generate() response = %#v", response)
	}
	requireModelError(t, err, llm.KindMalformedResponse, "generate")
	if !strings.Contains(err.Error(), "repeats ID") {
		t.Fatalf("Generate() error = %v", err)
	}
}

func TestGenerateRejectsReusedSynthesizedHistoricalToolID(t *testing.T) {
	body := strings.Replace(validToolResponse, `"id":"toolu_one"`, `"id":"toolu_llm_go_1"`, 1)
	client := newStaticToolClient(t, body)
	request := toolRequest()
	request.Messages = append([]llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{Name: "lookup", Arguments: []byte(`{}`)}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{Name: "lookup"}}},
	}, request.Messages...)
	response, err := client.Generate(context.Background(), request)
	if response != nil {
		t.Fatalf("Generate() response = %#v", response)
	}
	requireModelError(t, err, llm.KindMalformedResponse, "generate")
	if !strings.Contains(err.Error(), "repeats ID") {
		t.Fatalf("Generate() error = %v", err)
	}
}

func TestGenerateClassifiesTruncatedToolResponse(t *testing.T) {
	for _, reason := range []string{"max_tokens", "model_context_window_exceeded"} {
		t.Run(reason, func(t *testing.T) {
			body := strings.Replace(validToolResponse, `"stop_reason":"tool_use"`, `"stop_reason":"`+reason+`"`, 1)
			client := newStaticToolClient(t, body)
			response, err := client.Generate(context.Background(), toolRequest())
			if response != nil {
				t.Fatalf("Generate() response = %#v", response)
			}
			requireModelError(t, err, llm.KindUnsupported, "generate")
			if !strings.Contains(err.Error(), "truncated tool use") {
				t.Fatalf("Generate() error = %v", err)
			}
		})
	}
}

func toolRequest() llm.Request {
	return llm.Request{
		Messages: textRequest("use a tool").Messages,
		Tools: []llm.Tool{{
			Name:        "lookup",
			InputSchema: []byte(`{"type":"object"}`),
		}},
	}
}

func newStaticToolClient(t *testing.T, body string) *Client {
	t.Helper()
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityTools},
		BaseURL:      "http://example.com/v1",
		HTTPClient:   &http.Client{Transport: staticTransport{status: http.StatusOK, body: body}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return client
}
