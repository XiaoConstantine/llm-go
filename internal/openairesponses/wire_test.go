package openairesponses

import (
	jsonv2 "encoding/json/v2"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestRequestOnlySerializesCallerInstructions(t *testing.T) {
	codec := Codec{Provider: "openai"}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
	params, err := codec.Request("generate", "model", request, RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	body, err := jsonv2.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `"instructions"`) {
		t.Fatalf("user-only wire contains instructions: %s", body)
	}

	request.Messages = append([]llm.Message{{Role: llm.RoleSystem, Content: []llm.Part{{Text: "Follow the caller."}}}}, request.Messages...)
	params, err = codec.Request("generate", "model", request, RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	body, err = jsonv2.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"instructions":"Follow the caller."`) {
		t.Fatalf("system instructions missing from wire: %s", body)
	}
}

func TestExplicitPromptCacheWireSemantics(t *testing.T) {
	codec := Codec{Provider: "openai", MaxProviderDataBytes: 1 << 20}
	base := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
	for _, test := range []struct {
		name      string
		retention llm.CacheRetention
		key       string
		explicit  bool
		want      []string
		doNotWant []string
	}{
		{name: "default leaves policy unchanged", explicit: true, want: []string{`"content":"hello"`}, doNotWant: []string{"prompt_cache_options", "prompt_cache_breakpoint", "prompt_cache_key"}},
		{name: "none disables implicit caching", retention: llm.CacheRetentionNone, key: "ignored", explicit: true, want: []string{`"prompt_cache_options":{"mode":"explicit"}`}, doNotWant: []string{"prompt_cache_breakpoint", "prompt_cache_key"}},
		{name: "short uses key and retention only", retention: llm.CacheRetentionShort, key: "stable", explicit: true, want: []string{`"prompt_cache_key":"stable"`, `"prompt_cache_retention":"in_memory"`}, doNotWant: []string{"prompt_cache_options", "prompt_cache_breakpoint"}},
		{name: "explicit long uses minimum TTL", retention: llm.CacheRetentionLong, key: "stable", explicit: true, want: []string{`"prompt_cache_key":"stable"`, `"prompt_cache_options":{"ttl":"30m"}`}, doNotWant: []string{`"prompt_cache_retention"`, "prompt_cache_breakpoint"}},
		{name: "legacy long uses 24 hour retention", retention: llm.CacheRetentionLong, key: "stable", want: []string{`"prompt_cache_key":"stable"`, `"prompt_cache_retention":"24h"`}, doNotWant: []string{"prompt_cache_options", "prompt_cache_breakpoint"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := base
			request.CacheRetention = test.retention
			request.CacheKey = test.key
			params, err := codec.Request("generate", "model", request, RequestOptions{ExplicitPromptCache: test.explicit})
			if err != nil {
				t.Fatal(err)
			}
			body, err := jsonv2.Marshal(params)
			if err != nil {
				t.Fatal(err)
			}
			wire := string(body)
			for _, want := range test.want {
				if !strings.Contains(wire, want) {
					t.Errorf("wire = %s, want %s", wire, want)
				}
			}
			for _, unwanted := range test.doNotWant {
				if strings.Contains(wire, unwanted) {
					t.Errorf("wire = %s, unexpectedly contains %s", wire, unwanted)
				}
			}
		})
	}
}

func TestExplicitPromptCacheBreakpoints(t *testing.T) {
	codec := Codec{Provider: "openai", MaxProviderDataBytes: 1 << 20}
	request := llm.Request{
		CacheRetention: llm.CacheRetentionShort,
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.Part{{Text: "stable instructions", CacheBreakpoint: true}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "changing input"}}},
		},
	}
	params, err := codec.Request("generate", "model", request, RequestOptions{ExplicitPromptCache: true})
	if err != nil {
		t.Fatal(err)
	}
	body, err := jsonv2.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	wire := string(body)
	for _, want := range []string{`"role":"system"`, `"text":"stable instructions"`, `"prompt_cache_breakpoint":{"mode":"explicit"}`, `"prompt_cache_options":{"mode":"explicit"}`} {
		if !strings.Contains(wire, want) {
			t.Errorf("wire = %s, want %s", wire, want)
		}
	}
	if strings.Contains(wire, `"instructions"`) {
		t.Errorf("wire = %s, structured system prompt must not also use instructions", wire)
	}

	if _, err := codec.Request("generate", "model", request, RequestOptions{}); err == nil {
		t.Fatal("breakpoint succeeded without explicit-cache compatibility")
	}
}

func TestPreferredToolStrictnessFollowsRequestOptions(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "tool", InputSchema: []byte(`{"type":"object"}`), Strictness: llm.ToolStrictPrefer}}}
	codec := Codec{Provider: "provider"}
	unsupported, err := codec.Request("generate", "model", request, RequestOptions{StrictTools: false})
	if err != nil {
		t.Fatal(err)
	}
	if unsupported.Tools[0].OfFunction.Strict.Value {
		t.Fatal("unsupported prefer emitted strict")
	}
	supported, err := codec.Request("generate", "model", request, RequestOptions{StrictTools: true})
	if err != nil {
		t.Fatal(err)
	}
	if !supported.Tools[0].OfFunction.Strict.Value {
		t.Fatal("supported prefer omitted strict")
	}
}

func TestDeferredToolsUseAdditionalToolsBeforeToolSearch(t *testing.T) {
	request := deferredToolRequest()
	codec := Codec{Provider: "openai"}
	params, err := codec.Request("generate", "model", request, RequestOptions{AdditionalTools: true, ToolSearch: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(params.Tools) != 1 || params.Tools[0].OfFunction == nil || params.Tools[0].OfFunction.Name != "lookup" {
		t.Fatalf("top-level tools = %#v", params.Tools)
	}
	wire := marshalWire(t, params)
	for _, want := range []string{`"type":"additional_tools"`, `"role":"developer"`, `"name":"loaded"`} {
		if !strings.Contains(wire, want) {
			t.Errorf("wire = %s, want %s", wire, want)
		}
	}
	if strings.Contains(wire, `"type":"tool_search_call"`) || strings.Contains(wire, `"defer_loading":true`) {
		t.Fatalf("additional_tools wire contains tool-search fields: %s", wire)
	}
}

func TestDeferredToolsUseClientToolSearchFallback(t *testing.T) {
	params, err := (Codec{Provider: "openai"}).Request("generate", "model", deferredToolRequest(), RequestOptions{ToolSearch: true})
	if err != nil {
		t.Fatal(err)
	}
	wire := marshalWire(t, params)
	for _, want := range []string{`"type":"tool_search_call"`, `"type":"tool_search_output"`, `"execution":"client"`,
		`"status":"completed"`, `"name":"loaded"`, `"defer_loading":true`} {
		if !strings.Contains(wire, want) {
			t.Errorf("wire = %s, want %s", wire, want)
		}
	}
	if strings.Contains(wire, `"type":"additional_tools"`) {
		t.Fatalf("tool-search wire contains additional_tools: %s", wire)
	}
}

func TestDeferredToolMarkersFallBackToImmediateDefinitions(t *testing.T) {
	params, err := (Codec{Provider: "compatible"}).Request("generate", "model", deferredToolRequest(), RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(params.Tools) != 2 {
		t.Fatalf("top-level tools = %#v", params.Tools)
	}
	wire := marshalWire(t, params)
	if strings.Contains(wire, `"type":"additional_tools"`) || strings.Contains(wire, `"type":"tool_search_call"`) {
		t.Fatalf("unsupported wire contains deferred items: %s", wire)
	}
}

func deferredToolRequest() llm.Request {
	return llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "start"}}},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "lookup", Arguments: []byte(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Name: "lookup", AddedToolNames: []string{"loaded"}}}},
		},
		Tools: []llm.Tool{
			{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)},
			{Name: "loaded", InputSchema: []byte(`{"type":"object"}`)},
		},
	}
}

func marshalWire(t *testing.T, value any) string {
	t.Helper()
	body, err := jsonv2.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(body)
}
