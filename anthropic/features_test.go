package anthropic

import (
	"context"
	jsonv2 "encoding/json/v2"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestRequestToWireMapsImagesThinkingToolChoiceAndCache(t *testing.T) {
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.Part{{Text: "system"}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "look"}, {Kind: llm.PartImage, Data: []byte{1, 2, 3}, MediaType: "image/png; name=x.png"}}},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "inspect", Arguments: []byte(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Text: "result"}, {Kind: llm.PartImage, Data: []byte{4}, MediaType: "image/jpeg"}}}}},
		},
		Tools:           []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice:      llm.ToolChoice{Mode: llm.ToolChoiceNamed, Name: "inspect"},
		ReasoningEffort: llm.ReasoningEffortHigh,
		CacheRetention:  llm.CacheRetentionLong,
	}
	wire, _, err := requestToWireWithCompatibility("generate", "claude", 4096, request, llm.AnthropicCompatibility{
		AdaptiveThinking: llm.CompatibilityEnabled, LongCacheRetention: llm.CompatibilityEnabled,
		CacheControlOnTools: llm.CompatibilityEnabled,
	})
	if err != nil {
		t.Fatalf("requestToWireWithCompatibility() error = %v", err)
	}
	body, err := jsonv2.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	text := string(body)
	for _, want := range []string{
		`"tool_choice":{"type":"tool","name":"inspect"}`,
		`"thinking":{"type":"adaptive","display":"summarized"}`,
		`"output_config":{"effort":"high"}`,
		`"source":{"type":"base64","media_type":"image/png","data":"AQID"}`,
		`"source":{"type":"base64","media_type":"image/jpeg","data":"BA=="}`,
		`"cache_control":{"type":"ephemeral","ttl":"1h"}`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("request = %s, want %s", text, want)
		}
	}
}

func TestAnthropicThinkingResponseReplay(t *testing.T) {
	response, err := responseFromWire("claude", llm.Request{}, nil, messageResponse{
		ID: "msg", Type: "message", Role: "assistant", Model: "claude", StopReason: stringPointer("end_turn"),
		Content: &[]responseBlock{
			{Type: "thinking", Thinking: stringPointer("considering"), Signature: stringPointer("signed")},
			{Type: "redacted_thinking", Data: stringPointer("opaque")},
			{Type: "text", Text: stringPointer("answer")},
		},
		Usage: &responseUsage{InputTokens: intPointer(1), OutputTokens: intPointer(2)},
	})
	if err != nil {
		t.Fatalf("responseFromWire() error = %v", err)
	}
	if response.ReasoningSummary != "considering\n\n[Reasoning redacted]" || response.Text() != "answer" || len(response.Message.ProviderData) == 0 {
		t.Fatalf("response = %#v", response)
	}
	wire, _, err := requestToWire("generate", "claude", 4096, llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{{Text: "before"}}}, response.Message,
	}})
	if err != nil {
		t.Fatalf("replay request error = %v", err)
	}
	blocks := wire.Messages[1].Content.([]inputContentBlock)
	if len(blocks) != 3 || blocks[0].Type != "thinking" || blocks[0].Signature == nil || *blocks[0].Signature != "signed" || blocks[1].Type != "redacted_thinking" || blocks[1].Data != "opaque" {
		t.Fatalf("replay blocks = %#v", blocks)
	}
	foreign, _, err := requestToWire("generate", "other", 4096, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, response.Message}})
	if err != nil {
		t.Fatalf("foreign replay error = %v", err)
	}
	if content, ok := foreign.Messages[1].Content.(string); !ok || content != "answer" {
		t.Fatalf("foreign model replay = %#v", foreign.Messages[1].Content)
	}
}

func TestSessionAffinityHeaderIsCompatibilityGated(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if got := request.Header.Get("X-Session-Affinity"); got != "session" {
			t.Errorf("X-Session-Affinity = %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, validResponse)
	}))
	defer server.Close()
	client, err := NewWithOptions(Config{Model: "claude", BaseURL: server.URL, HTTPClient: server.Client()}, Options{ModelCompatibility: &llm.AnthropicCompatibility{SessionAffinity: llm.CompatibilityEnabled}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "go"}}}}, CacheRetention: llm.CacheRetentionNone, SessionID: "session"}); err != nil {
		t.Fatal(err)
	}
}

func TestStreamEmitsSignedThinkingEventsAndReplayData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w,
			streamEvent("message_start", `{"type":"message_start","message":{"id":"msg","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`)+
				streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`)+
				streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}`)+
				streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}`)+
				streamEvent("content_block_stop", `{"type":"content_block_stop","index":0}`)+
				streamEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`)+
				streamEvent("message_stop", `{"type":"message_stop"}`))
	}))
	defer server.Close()
	client, err := New(Config{Model: "claude", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming}})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "go"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var kinds []llm.StreamEventKind
	var reasoning string
	var providerData []byte
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		reasoning += chunk.ReasoningSummary
		providerData = append(providerData[:0], chunk.ProviderData...)
		for _, event := range chunk.Events {
			kinds = append(kinds, event.Kind)
		}
	}
	if reasoning != "think" || !strings.Contains(string(providerData), `"signature":"signed"`) {
		t.Fatalf("reasoning/providerData = %q/%s", reasoning, providerData)
	}
	want := []llm.StreamEventKind{llm.StreamEventStart, llm.StreamEventReasoningStart, llm.StreamEventReasoningDelta, llm.StreamEventReasoningEnd, llm.StreamEventDone}
	if len(kinds) != len(want) {
		t.Fatalf("event kinds = %v", kinds)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("event kinds = %v", kinds)
		}
	}
}

func TestThinkingReplayUsesConfiguredAliasGenerateAndStream(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg","type":"message","role":"assistant","content":[{"type":"thinking","thinking":"think","signature":"signed"},{"type":"text","text":"ok"}],"model":"canonical","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w,
			streamEvent("message_start", `{"type":"message_start","message":{"id":"msg","type":"message","role":"assistant","content":[],"model":"canonical","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`)+
				streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`)+
				streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}`)+
				streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}`)+
				streamEvent("content_block_stop", `{"type":"content_block_stop","index":0}`)+
				streamEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`)+
				streamEvent("message_stop", `{"type":"message_stop"}`))
	}))
	defer server.Close()
	client, err := New(Config{Model: "alias", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming}})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	response, err := client.Generate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if response.Model != "canonical" {
		t.Fatalf("response model = %q", response.Model)
	}
	assertAnthropicAliasReplay(t, response.Message.ProviderData)
	stream, err := client.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var providerData []byte
	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if len(chunk.ProviderData) != 0 {
			providerData = append([]byte(nil), chunk.ProviderData...)
		}
	}
	assertAnthropicAliasReplay(t, providerData)
}

func assertAnthropicAliasReplay(t *testing.T, providerData []byte) {
	t.Helper()
	wire, _, err := requestToWire("generate", "alias", 4096, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, {Role: llm.RoleAssistant, ProviderData: providerData}}})
	if err != nil {
		t.Fatal(err)
	}
	blocks, ok := wire.Messages[1].Content.([]inputContentBlock)
	if !ok || len(blocks) != 1 || blocks[0].Type != "thinking" || blocks[0].Signature == nil || *blocks[0].Signature != "signed" {
		t.Fatalf("alias replay = %#v", wire.Messages[1].Content)
	}
}

func TestEmptyThinkingEventsAreOrderedStartEndDone(t *testing.T) {
	body := streamEvent("message_start", `{"type":"message_start","message":{"id":"msg","type":"message","role":"assistant","content":[],"model":"canonical","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`) +
		streamEvent("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`) +
		streamEvent("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}`) +
		streamEvent("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		streamEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`) +
		streamEvent("message_stop", `{"type":"message_stop"}`)
	var kinds []llm.StreamEventKind
	err := readEventStream(context.Background(), strings.NewReader(body), newStreamDecoder("", nil, nil), func(chunk llm.Chunk) bool {
		for _, event := range chunk.Events {
			kinds = append(kinds, event.Kind)
		}
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.StreamEventKind{llm.StreamEventStart, llm.StreamEventReasoningStart, llm.StreamEventReasoningEnd, llm.StreamEventDone}
	if len(kinds) != len(want) {
		t.Fatalf("events = %v", kinds)
	}
	for index := range want {
		if kinds[index] != want[index] {
			t.Fatalf("events = %v", kinds)
		}
	}
}

func TestAnthropicCompatibilityDefaultsForToolsAndCache(t *testing.T) {
	request := llm.Request{
		Messages:       []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "go"}}}},
		Tools:          []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}},
		CacheRetention: llm.CacheRetentionLong,
	}
	for _, test := range []struct {
		name                                               string
		compatibility                                      llm.AnthropicCompatibility
		wantEager, wantToolCache, wantLegacyBeta, wantLong bool
	}{
		{name: "default", wantEager: true, wantToolCache: true, wantLong: true},
		{name: "enabled", compatibility: llm.AnthropicCompatibility{EagerToolInputStreaming: llm.CompatibilityEnabled, LongCacheRetention: llm.CompatibilityEnabled, CacheControlOnTools: llm.CompatibilityEnabled}, wantEager: true, wantToolCache: true, wantLong: true},
		{name: "disabled", compatibility: llm.AnthropicCompatibility{EagerToolInputStreaming: llm.CompatibilityDisabled, LongCacheRetention: llm.CompatibilityDisabled, CacheControlOnTools: llm.CompatibilityDisabled}, wantLegacyBeta: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			wire, _, err := requestToWireWithCompatibility("generate", "model", 4096, request, test.compatibility)
			if err != nil {
				t.Fatal(err)
			}
			if wire.Tools[0].EagerInputStreaming != test.wantEager || (wire.Tools[0].CacheControl != nil) != test.wantToolCache {
				t.Fatalf("tool wire = %#v", wire.Tools[0])
			}
			client, err := NewWithOptions(Config{Model: "model"}, Options{ModelCompatibility: &test.compatibility})
			if err != nil {
				t.Fatal(err)
			}
			compatErr := client.checkCompatibility("generate", request)
			if (compatErr == nil) != test.wantLong {
				t.Fatalf("long-cache error = %v", compatErr)
			}
			hasBeta := strings.Contains(client.requestHeaders(request).Get("Anthropic-Beta"), "fine-grained-tool-streaming-2025-05-14")
			if hasBeta != test.wantLegacyBeta {
				t.Fatalf("legacy beta = %v", hasBeta)
			}
			requestWithoutTools := request
			requestWithoutTools.Tools = nil
			if strings.Contains(client.requestHeaders(requestWithoutTools).Get("Anthropic-Beta"), "fine-grained-tool-streaming-2025-05-14") {
				t.Fatal("legacy beta sent without tools")
			}
		})
	}
}

func TestAnthropicThinkingBudgetContracts(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, ReasoningBudgetTokens: 2048}
	if _, _, err := requestToWireWithCompatibility("generate", "model", 4096, request, llm.AnthropicCompatibility{AdaptiveThinking: llm.CompatibilityEnabled}); err == nil || !strings.Contains(err.Error(), "adaptive") {
		t.Fatalf("adaptive explicit budget error = %v", err)
	}
	request.ReasoningBudgetTokens = 0
	request.ReasoningEffort = llm.ReasoningEffortXHigh
	adaptive, _, err := requestToWireWithCompatibility("generate", "model", 4096, request, llm.AnthropicCompatibility{AdaptiveThinking: llm.CompatibilityEnabled})
	if err != nil {
		t.Fatal(err)
	}
	if adaptive.OutputConfig == nil || adaptive.OutputConfig.Effort != llm.ReasoningEffortXHigh {
		t.Fatalf("adaptive xhigh = %#v", adaptive.OutputConfig)
	}
	for _, effort := range []llm.ReasoningEffort{llm.ReasoningEffortMedium, llm.ReasoningEffortHigh} {
		request.ReasoningBudgetTokens = 0
		request.ReasoningEffort = effort
		wire, _, err := requestToWire("generate", "model", 4096, request)
		if err != nil {
			t.Fatalf("effort %s: %v", effort, err)
		}
		if wire.Thinking == nil || wire.Thinking.BudgetTokens != 3072 {
			t.Fatalf("effort %s thinking = %#v", effort, wire.Thinking)
		}
	}
}

func TestBudgetOnlyThinkingBetaHeaderGenerateAndStream(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		requests++
		if !strings.Contains(request.Header.Get("Anthropic-Beta"), "interleaved-thinking-2025-05-14") {
			t.Errorf("Anthropic-Beta = %q", request.Header.Get("Anthropic-Beta"))
		}
		if strings.Contains(request.Header.Get("Accept"), "text/event-stream") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, streamEvent("message_start", `{"type":"message_start","message":{"id":"msg","type":"message","role":"assistant","content":[],"model":"model","stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`)+streamEvent("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`)+streamEvent("message_stop", `{"type":"message_stop"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, validResponse)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming}})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, ReasoningBudgetTokens: 1024, MaxOutputTokens: 2048}
	if _, err := client.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	stream, err := client.Stream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if requests != 2 {
		t.Fatalf("requests = %d", requests)
	}
}

func TestEmptyThinkingReplayPreservesRequiredWireField(t *testing.T) {
	response := messageResponse{ID: "msg", Type: "message", Role: "assistant", Model: "served", StopReason: stringPointer("end_turn"), Content: &[]responseBlock{{Type: "thinking", Thinking: stringPointer(""), Signature: stringPointer("signed")}}, Usage: &responseUsage{InputTokens: intPointer(1), OutputTokens: intPointer(1)}}
	converted, err := responseFromWire("alias", llm.Request{}, nil, response)
	if err != nil {
		t.Fatal(err)
	}
	wire, _, err := requestToWire("generate", "alias", 4096, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, converted.Message}})
	if err != nil {
		t.Fatal(err)
	}
	body, err := jsonv2.Marshal(wire)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), `"type":"thinking","thinking":"","signature":"signed"`) {
		t.Fatalf("empty thinking replay wire = %s", body)
	}
}

func TestNonstreamEmptyThinkingSignatureCompatibility(t *testing.T) {
	response := messageResponse{ID: "msg", Type: "message", Role: "assistant", Model: "served", StopReason: stringPointer("end_turn"), Content: &[]responseBlock{{Type: "thinking", Thinking: stringPointer("think"), Signature: stringPointer("")}}, Usage: &responseUsage{InputTokens: intPointer(1), OutputTokens: intPointer(1)}}
	if _, err := responseFromWire("alias", llm.Request{}, nil, response); err == nil || !strings.Contains(err.Error(), "empty signature") {
		t.Fatalf("strict empty signature error = %v", err)
	}
	converted, err := responseFromWire("alias", llm.Request{}, nil, response, true)
	if err != nil || len(converted.Message.ProviderData) == 0 {
		t.Fatalf("compatible response = %#v, %v", converted, err)
	}
	wire, _, err := requestToWireWithCompatibility("generate", "alias", 4096, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}, converted.Message}}, llm.AnthropicCompatibility{EmptyThinkingSignature: llm.CompatibilityEnabled})
	if err != nil {
		t.Fatal(err)
	}
	body, _ := jsonv2.Marshal(wire)
	if !strings.Contains(string(body), `"signature":""`) {
		t.Fatalf("empty signature replay wire = %s", body)
	}
}

func TestAnthropicOAuthIdentityWire(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		betas := request.Header.Get("Anthropic-Beta")
		if request.Header.Get("Authorization") != "Bearer token" || request.Header.Get("User-Agent") != "claude-cli/2.1.75" || request.Header.Get("X-App") != "cli" ||
			!strings.Contains(betas, "claude-code-20250219") || !strings.Contains(betas, "oauth-2025-04-20") {
			t.Errorf("OAuth headers = %#v", request.Header)
		}
		body, _ := io.ReadAll(request.Body)
		if !strings.Contains(string(body), `"system":[{"type":"text","text":"You are Claude Code, Anthropic's official CLI for Claude."}`) {
			t.Errorf("OAuth body = %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, validResponse)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", AccessToken: "token", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}); err != nil {
		t.Fatal(err)
	}
}

func stringPointer(value string) *string { return &value }
func intPointer(value int) *int          { return &value }
