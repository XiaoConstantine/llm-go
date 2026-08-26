package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestNewRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		want   string
	}{
		{name: "empty model", config: Config{}, want: "model must not be empty"},
		{name: "relative base URL", config: Config{Model: "model", BaseURL: "/v1"}, want: "scheme must be http or https"},
		{name: "unsupported scheme", config: Config{Model: "model", BaseURL: "file:///v1"}, want: "scheme must be http or https"},
		{name: "query", config: Config{Model: "model", BaseURL: "https://example.com/v1?q=1"}, want: "must not contain a query"},
		{name: "unsupported capability", config: Config{Model: "model", Capabilities: []llm.Capability{"future"}}, want: "is not implemented"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(test.config)
			if client != nil {
				t.Fatalf("New() client = %#v, want nil", client)
			}
			requireModelError(t, err, llm.KindInvalidRequest, "configure")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("New() error = %q, want substring %q", err, test.want)
			}
		})
	}

	client, err := NewWithOptions(Config{Model: "model"}, Options{MaxTokensField: "future_tokens"})
	if client != nil || err == nil || !strings.Contains(err.Error(), "max tokens field") {
		t.Fatalf("NewWithOptions(unsupported max tokens field) = (%#v, %v)", client, err)
	}
}

func TestClientInfoReturnsIndependentCapabilities(t *testing.T) {
	client, err := New(Config{
		Model:        " test-model ",
		Capabilities: []llm.Capability{llm.CapabilityVision, llm.CapabilityAudio, llm.CapabilityGeneration, llm.CapabilityVision},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	first := client.Info()
	if first.Provider != "openai" || first.Model != "test-model" {
		t.Fatalf("Info() = %#v", first)
	}
	if got, want := first.Capabilities, []llm.Capability{llm.CapabilityGeneration, llm.CapabilityVision, llm.CapabilityAudio}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Capabilities = %#v, want %#v", got, want)
	}
	first.Capabilities[0] = llm.CapabilityAudio
	second := client.Info()
	if second.Capabilities[0] != llm.CapabilityGeneration {
		t.Fatalf("Info() retained caller mutation: %#v", second.Capabilities)
	}
}

func TestRequestTranslatesReasoningEffort(t *testing.T) {
	request, err := newChatRequest("model", llm.Request{
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
		ReasoningEffort: llm.ReasoningEffortHigh,
	})
	if err != nil {
		t.Fatalf("newChatRequest() error = %v", err)
	}
	if request.ReasoningEffort != llm.ReasoningEffortHigh {
		t.Fatalf("ReasoningEffort = %q", request.ReasoningEffort)
	}
}

func TestRequestTranslatesAudioInput(t *testing.T) {
	request, err := newChatRequest("model", llm.Request{Messages: []llm.Message{{
		Role: llm.RoleUser,
		Content: []llm.Part{
			{Text: "listen"},
			{Kind: llm.PartAudio, Data: []byte("wav"), MediaType: "audio/wav"},
			{Kind: llm.PartAudio, Data: []byte("mp3"), MediaType: "audio/mpeg; codecs=mp3"},
		},
	}}})
	if err != nil {
		t.Fatalf("newChatRequest() error = %v", err)
	}
	content, ok := request.Messages[0].Content.([]contentPart)
	if !ok || len(content) != 3 {
		t.Fatalf("message content = %#v", request.Messages[0].Content)
	}
	if content[0].Type != "text" || content[0].Text == nil || *content[0].Text != "listen" {
		t.Fatalf("text content = %#v", content[0])
	}
	if content[1].Type != "input_audio" || content[1].InputAudio == nil ||
		content[1].InputAudio.Data != "d2F2" || content[1].InputAudio.Format != "wav" {
		t.Fatalf("WAV content = %#v", content[1])
	}
	if content[2].Type != "input_audio" || content[2].InputAudio == nil ||
		content[2].InputAudio.Data != "bXAz" || content[2].InputAudio.Format != "mp3" {
		t.Fatalf("MP3 content = %#v", content[2])
	}
	encoded, err := jsonv2.Marshal(&request)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	if !strings.Contains(string(encoded), `"type":"input_audio","input_audio":{"data":"d2F2","format":"wav"}`) ||
		!strings.Contains(string(encoded), `"type":"input_audio","input_audio":{"data":"bXAz","format":"mp3"}`) {
		t.Fatalf("encoded request = %s", encoded)
	}
}

func TestAudioInputValidationBeforeIO(t *testing.T) {
	var calls atomic.Int64
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityAudio},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, errors.New("unexpected request")
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	for _, test := range []struct {
		name    string
		message llm.Message
		kind    llm.ErrorKind
		want    string
	}{
		{
			name: "assistant audio",
			message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{{
				Kind: llm.PartAudio, Data: []byte("audio"), MediaType: "audio/wav",
			}}},
			kind: llm.KindUnsupported,
			want: "only in user messages",
		},
		{
			name: "unsupported format",
			message: llm.Message{Role: llm.RoleUser, Content: []llm.Part{{
				Kind: llm.PartAudio, Data: []byte("audio"), MediaType: "audio/flac",
			}}},
			kind: llm.KindInvalidRequest,
			want: "use WAV or MP3",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{test.message}})
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			requireModelError(t, err, test.kind, "generate")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Generate() error = %v, want %q", err, test.want)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("HTTP calls = %d, want zero", calls.Load())
	}
}

func TestRequestAppliesModelCompatibility(t *testing.T) {
	t.Run("reasoning formats", func(t *testing.T) {
		tests := []struct {
			name          string
			format        llm.ThinkingFormat
			effort        llm.ReasoningEffort
			support       llm.CompatibilityToggle
			wantEffort    llm.ReasoningEffort
			wantReasoning *reasoningOptions
			wantThinking  any
			wantEnabled   *bool
		}{
			{name: "OpenAI", format: llm.ThinkingFormatOpenAI, effort: llm.ReasoningEffortHigh, wantEffort: llm.ReasoningEffortHigh},
			{name: "OpenRouter", format: llm.ThinkingFormatOpenRouter, effort: llm.ReasoningEffortHigh, wantReasoning: &reasoningOptions{Effort: llm.ReasoningEffortHigh}},
			{name: "DeepSeek", format: llm.ThinkingFormatDeepSeek, effort: llm.ReasoningEffortHigh, support: llm.CompatibilityDisabled, wantThinking: thinkingOptions{Type: "enabled"}},
			{name: "DeepSeek off", format: llm.ThinkingFormatDeepSeek, effort: llm.ReasoningEffortNone, wantThinking: thinkingOptions{Type: "disabled"}},
			{name: "Together", format: llm.ThinkingFormatTogether, effort: llm.ReasoningEffortHigh, support: llm.CompatibilityDisabled, wantReasoning: &reasoningOptions{Enabled: new(true)}},
			{name: "ZAI", format: llm.ThinkingFormatZAI, effort: llm.ReasoningEffortNone, wantThinking: thinkingOptions{Type: "disabled"}},
			{name: "Qwen", format: llm.ThinkingFormatQwen, effort: llm.ReasoningEffortHigh, wantEnabled: new(true)},
			{name: "string", format: llm.ThinkingFormatString, effort: llm.ReasoningEffortHigh, wantThinking: "high"},
		}
		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				compatibility := llm.OpenAIChatCompatibility{MaxTokensField: llm.MaxTokensFieldCompletion, ThinkingFormat: test.format, ReasoningEffort: test.support}
				request, err := newChatRequestFor("generate", "model", llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, ReasoningEffort: test.effort}, compatibility)
				if err != nil {
					t.Fatalf("newChatRequestFor() error = %v", err)
				}
				if request.ReasoningEffort != test.wantEffort || !reflect.DeepEqual(request.Reasoning, test.wantReasoning) ||
					!reflect.DeepEqual(request.Thinking, test.wantThinking) || !reflect.DeepEqual(request.EnableThinking, test.wantEnabled) {
					t.Fatalf("reasoning fields = effort %q, reasoning %#v, thinking %#v, enabled %#v", request.ReasoningEffort, request.Reasoning, request.Thinking, request.EnableThinking)
				}
			})
		}
	})

	t.Run("message conversion", func(t *testing.T) {
		compatibility := llm.OpenAIChatCompatibility{
			MaxTokensField:           llm.MaxTokensFieldLegacy,
			InstructionRole:          llm.InstructionRoleDeveloper,
			ToolResultName:           llm.CompatibilityEnabled,
			AssistantAfterToolResult: llm.CompatibilityEnabled,
			ReasoningContentReplay:   llm.CompatibilityEnabled,
		}
		canonical := llm.Request{
			Messages: []llm.Message{
				{Role: llm.RoleSystem, Content: []llm.Part{{Text: "instructions"}}},
				{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "tool", Arguments: json.RawMessage(`{}`)}}},
				{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Text: "result"}}}}},
				{Role: llm.RoleUser, Content: []llm.Part{{Text: "next"}}},
			},
			MaxOutputTokens: 10,
		}
		request, err := newChatRequestFor("generate", "model", canonical, compatibility)
		if err != nil {
			t.Fatalf("newChatRequestFor() error = %v", err)
		}
		if request.MaxTokens == nil || *request.MaxTokens != 10 || request.MaxCompletionTokens != nil {
			t.Fatalf("token limit fields = %#v", request)
		}
		if len(request.Messages) != 5 || request.Messages[0].Role != "developer" || request.Messages[2].Name != "tool" ||
			request.Messages[3].Role != "assistant" || request.Messages[3].Content != assistantAfterToolResultText || request.Messages[4].Role != "user" {
			t.Fatalf("messages = %#v", request.Messages)
		}
		if request.Messages[1].ReasoningContent == nil || *request.Messages[1].ReasoningContent != "" {
			t.Fatalf("assistant reasoning replay = %#v", request.Messages[1])
		}
	})

	t.Run("tool result names follow reused IDs sequentially", func(t *testing.T) {
		request, err := newChatRequestFor("generate", "model", llm.Request{Messages: []llm.Message{
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "reused", Name: "first", Arguments: json.RawMessage(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "reused", Content: []llm.Part{{Text: "one"}}}}},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "reused", Name: "second", Arguments: json.RawMessage(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "reused", Content: []llm.Part{{Text: "two"}}}}},
		}}, llm.OpenAIChatCompatibility{MaxTokensField: llm.MaxTokensFieldCompletion, ToolResultName: llm.CompatibilityEnabled})
		if err != nil {
			t.Fatalf("newChatRequestFor() error = %v", err)
		}
		if request.Messages[1].Name != "first" || request.Messages[3].Name != "second" {
			t.Fatalf("tool result messages = %#v, %#v", request.Messages[1], request.Messages[3])
		}
	})
}

func TestClientEnforcesModelCompatibilityBeforeIO(t *testing.T) {
	var calls atomic.Int32
	client, err := NewWithCompatibility(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityTools},
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, errors.New("unexpected I/O")
		})},
	}, &llm.OpenAIChatCompatibility{
		StrictTools:     llm.CompatibilityDisabled,
		ReasoningEffort: llm.CompatibilityDisabled,
	})
	if err != nil {
		t.Fatalf("NewWithOptions() error = %v", err)
	}
	requests := []llm.Request{
		{Messages: []llm.Message{{Role: llm.RoleUser}}, Tools: []llm.Tool{{Name: "tool", InputSchema: json.RawMessage(`{"type":"object"}`), Strict: true}}},
		{Messages: []llm.Message{{Role: llm.RoleUser}}, ReasoningEffort: llm.ReasoningEffortHigh},
	}
	for _, request := range requests {
		response, err := client.Generate(context.Background(), request)
		if response != nil {
			t.Fatalf("Generate() response = %#v, want nil", response)
		}
		requireModelError(t, err, llm.KindUnsupported, "generate")
	}
	if calls.Load() != 0 {
		t.Fatalf("provider calls = %d, want zero", calls.Load())
	}
}

func TestClientUsesConfiguredProviderIdentity(t *testing.T) {
	client, err := New(Config{Provider: " ollama ", Model: "model"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if got := client.Info().Provider; got != "ollama" {
		t.Fatalf("Info().Provider = %q, want %q", got, "ollama")
	}

	temperature := 3.0
	response, err := client.Generate(context.Background(), llm.Request{
		Messages:    []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}},
		Temperature: &temperature,
	})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Provider != "ollama" {
		t.Fatalf("Generate() error = %#v, want provider %q", modelErr, "ollama")
	}

	_, err = New(Config{Provider: " ollama "})
	if !errors.As(err, &modelErr) || modelErr.Provider != "ollama" {
		t.Fatalf("New(invalid) error = %#v, want provider %q", modelErr, "ollama")
	}
}

func TestGenerateTranslatesCompatibleReasoningFields(t *testing.T) {
	tests := []struct {
		name      string
		message   string
		want      string
		wantError string
	}{
		{name: "reasoning content", message: `{"role":"assistant","content":"answer","reasoning_content":"thinking"}`, want: "thinking"},
		{name: "reasoning alias", message: `{"role":"assistant","content":"answer","reasoning":"thinking"}`, want: "thinking"},
		{name: "ambiguous", message: `{"role":"assistant","content":"answer","reasoning_content":"one","reasoning":"two"}`, wantError: "both reasoning_content and reasoning"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, `{"choices":[{"index":0,"message":`+test.message+`,"finish_reason":"stop"}]}`)
			}))
			defer server.Close()
			client, err := New(Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client()})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if test.wantError != "" {
				if response != nil || err == nil || !strings.Contains(err.Error(), test.wantError) {
					t.Fatalf("Generate() = (%#v, %v), want %q", response, err, test.wantError)
				}
				return
			}
			if err != nil || response == nil || response.ReasoningSummary != "" {
				t.Fatalf("Generate() = (%#v, %v)", response, err)
			}
			data, err := ParseReasoningData(response.Message)
			if err != nil {
				t.Fatalf("ParseReasoningData() error = %v", err)
			}
			got := data.ReasoningContent
			if test.name == "reasoning alias" {
				got = data.Reasoning
			}
			if got == nil || *got != test.want {
				t.Fatalf("ParseReasoningData() = %#v", data)
			}
		})
	}
}

func TestCompatibleReasoningStateRoundTripsOriginalFields(t *testing.T) {
	index := 0
	stop := "stop"
	content := "answer"
	thinking := "thinking"
	details := []json.RawMessage{
		json.RawMessage(`{"type":"reasoning.encrypted","data":"secret"}`),
		json.RawMessage(`{"type":"reasoning.text","text":"thinking"}`),
	}
	emptyDetails := []json.RawMessage{}
	tests := []struct {
		name        string
		message     responseMessage
		field       string
		detailCount int
	}{
		{name: "reasoning_content", message: responseMessage{Role: "assistant", Content: &content, ReasoningContent: &thinking}, field: "reasoning_content"},
		{name: "reasoning", message: responseMessage{Role: "assistant", Content: &content, Reasoning: &thinking}, field: "reasoning"},
		{name: "reasoning_details", message: responseMessage{Role: "assistant", Content: &content, ReasoningDetails: &details}, field: "reasoning_details", detailCount: 2},
		{name: "empty reasoning_details", message: responseMessage{Role: "assistant", Content: &content, ReasoningDetails: &emptyDetails}, field: "reasoning_details"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, err := responseFromWire("model", llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}, chatResponse{
				Choices: []chatChoice{{Index: &index, Message: &test.message, FinishReason: &stop}},
			})
			if err != nil {
				t.Fatalf("responseFromWire() error = %v", err)
			}
			data, err := ParseReasoningData(response.Message)
			if err != nil {
				t.Fatalf("ParseReasoningData() error = %v", err)
			}
			if test.field == "reasoning_content" && (data.ReasoningContent == nil || *data.ReasoningContent != thinking || data.Reasoning != nil) {
				t.Fatalf("ReasoningData = %#v", data)
			}
			if test.field == "reasoning" && (data.Reasoning == nil || *data.Reasoning != thinking || data.ReasoningContent != nil) {
				t.Fatalf("ReasoningData = %#v", data)
			}
			if test.field == "reasoning_details" {
				if !data.HasReasoningDetails || len(data.ReasoningDetails) != test.detailCount {
					t.Fatalf("ReasoningDetails = %s (present %t)", data.ReasoningDetails, data.HasReasoningDetails)
				}
				if test.detailCount != 0 {
					if string(data.ReasoningDetails[0]) != string(details[0]) || string(data.ReasoningDetails[1]) != string(details[1]) {
						t.Fatalf("ReasoningDetails = %s", data.ReasoningDetails)
					}
					data.ReasoningDetails[0][0] = '['
					again, err := ParseReasoningData(response.Message)
					if err != nil || string(again.ReasoningDetails[0]) != string(details[0]) {
						t.Fatalf("second ParseReasoningData() = (%s, %v)", again.ReasoningDetails, err)
					}
				}
			}

			wire, err := newChatRequest("model", llm.Request{Messages: []llm.Message{
				{Role: llm.RoleUser}, response.Message, {Role: llm.RoleUser},
			}})
			if err != nil {
				t.Fatalf("newChatRequest() error = %v", err)
			}
			assistant := wire.Messages[1]
			switch test.field {
			case "reasoning_content":
				if assistant.ReasoningContent == nil || *assistant.ReasoningContent != thinking || assistant.Reasoning != nil {
					t.Fatalf("assistant = %#v", assistant)
				}
			case "reasoning":
				if assistant.Reasoning == nil || *assistant.Reasoning != thinking || assistant.ReasoningContent != nil {
					t.Fatalf("assistant = %#v", assistant)
				}
			case "reasoning_details":
				if assistant.ReasoningDetails == nil || len(*assistant.ReasoningDetails) != test.detailCount {
					t.Fatalf("assistant reasoning details = %v", assistant.ReasoningDetails)
				}
				if test.detailCount != 0 && (string((*assistant.ReasoningDetails)[0]) != string(details[0]) || string((*assistant.ReasoningDetails)[1]) != string(details[1])) {
					t.Fatalf("assistant reasoning details = %v", assistant.ReasoningDetails)
				}
				if test.detailCount == 0 {
					encoded, err := jsonv2.Marshal(wire)
					if err != nil || !strings.Contains(string(encoded), `"reasoning_details":[]`) {
						t.Fatalf("encoded continuation = (%s, %v)", encoded, err)
					}
				}
			}
		})
	}
}

func TestRelabelProviderErrorPreservesJoinedCauses(t *testing.T) {
	closeErr := errors.New("close failed")
	err := relabelProviderError(errors.Join(malformedStream("bad event"), closeErr), "ollama")
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Provider != "ollama" {
		t.Fatalf("relabelProviderError() model error = %#v, want provider %q", modelErr, "ollama")
	}
	if !errors.Is(err, closeErr) {
		t.Fatalf("relabelProviderError() lost joined cause: %v", err)
	}
}

func TestGenerateClassifiesProviderInterruption(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"insufficient_system_resource"}]}`)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindProvider, "generate")
}

func TestGenerateTranslatesChatCompletion(t *testing.T) {
	type record struct {
		path   string
		header http.Header
		body   []byte
	}
	records := make(chan record, 1)
	server := httptest.NewTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			panic(err)
		}
		records <- record{path: r.URL.Path, header: r.Header.Clone(), body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"chat-1",
			"model":"served-model",
			"choices":[{
				"index":0,
				"message":{"role":"assistant","content":"{\"status\":\"checking\"}","tool_calls":[{
					"id":"call-out","type":"function","function":{"name":"weather","arguments":"{\"city\":\"NYC\"}"}
				}]},
				"finish_reason":"tool_calls"
			}],
			"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2},"completion_tokens_details":{"reasoning_tokens":4}}
		}`)
	}))

	headers := http.Header{
		"authorization": {"Basic custom"},
		"x-trace":       {"original"},
	}
	client, err := New(Config{
		Model: "request-model",
		Capabilities: []llm.Capability{
			llm.CapabilityTools,
			llm.CapabilityJSON,
			llm.CapabilityVision,
		},
		APIKey:     "ignored-key",
		BaseURL:    server.URL + "/proxy/v1/",
		HTTPClient: server.Client(),
		Headers:    headers,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	headers.Set("Authorization", "mutated")
	headers.Set("X-Trace", "mutated")

	temperature := 0.25
	topP := 0.8
	presencePenalty := 0.1
	frequencyPenalty := -0.2
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.Part{{Text: "be concise"}}},
			{Role: llm.RoleUser, Content: []llm.Part{
				{Text: ""},
				{Kind: llm.PartImage, Data: []byte("png"), MediaType: "image/png"},
			}},
			{Role: llm.RoleAssistant, Content: []llm.Part{{Text: "using tools"}}, ToolCalls: []llm.ToolCall{
				{ID: "call_llm_go_1", Name: "weather", Arguments: []byte(`{"city":"NYC"}`)},
				{Name: "clock", Arguments: []byte(`{"zone":"UTC"}`)},
			}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{
				{CallID: "call_llm_go_1", Name: "weather", Content: []llm.Part{{Text: "sunny"}}, IsError: true},
				{Name: "clock", Content: []llm.Part{{Text: "12:00"}}},
			}},
		},
		Tools: []llm.Tool{
			{Name: "weather", Description: "get weather", InputSchema: []byte(`{"type":"object"}`), Strict: true},
			{Name: "clock", InputSchema: []byte(`{"type":"object"}`)},
		},
		ResponseFormat:   llm.ResponseFormatJSON,
		MaxOutputTokens:  123,
		Temperature:      &temperature,
		TopP:             &topP,
		PresencePenalty:  &presencePenalty,
		FrequencyPenalty: &frequencyPenalty,
	}

	response, err := client.Generate(context.Background(), request)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.ID != "chat-1" || response.Model != "served-model" || response.Text() != `{"status":"checking"}` {
		t.Fatalf("Generate() response = %#v", response)
	}
	if response.FinishReason != llm.FinishReasonToolCall {
		t.Fatalf("FinishReason = %q, want %q", response.FinishReason, llm.FinishReasonToolCall)
	}
	if response.Usage == nil || *response.Usage != (llm.Usage{InputTokens: 6, OutputTokens: 7, CacheReadTokens: 3, CacheWriteTokens: 2, ReasoningTokens: 4, TotalTokens: 18}) {
		t.Fatalf("Usage = %#v", response.Usage)
	}
	if got := response.Message.ToolCalls; len(got) != 1 || got[0].ID != "call-out" ||
		got[0].Name != "weather" || string(got[0].Arguments) != `{"city":"NYC"}` {
		t.Fatalf("ToolCalls = %#v", got)
	}

	got := <-records
	if got.path != "/proxy/v1/chat/completions" {
		t.Fatalf("request path = %q", got.path)
	}
	if value := got.header.Get("Authorization"); value != "Basic custom" {
		t.Fatalf("Authorization = %q, want custom header", value)
	}
	if value := got.header.Get("X-Trace"); value != "original" {
		t.Fatalf("X-Trace = %q, want original", value)
	}
	if value := got.header.Get("Content-Type"); value != "application/json" {
		t.Fatalf("Content-Type = %q", value)
	}
	if got.header.Get("User-Agent") != "OpenAI/Go 3.52.0" || got.header.Get("X-Stainless-Retry-Count") != "0" {
		t.Fatalf("SDK headers = %#v", got.header)
	}

	var payload map[string]any
	if err := jsonv2.Unmarshal(got.body, &payload); err != nil {
		t.Fatalf("decode request: %v\n%s", err, got.body)
	}
	if payload["model"] != "request-model" || payload["max_completion_tokens"] != float64(123) {
		t.Fatalf("request scalar fields = %#v", payload)
	}
	if payload["temperature"] != temperature || payload["top_p"] != topP ||
		payload["presence_penalty"] != presencePenalty || payload["frequency_penalty"] != frequencyPenalty {
		t.Fatalf("request sampling fields = %#v", payload)
	}
	format, ok := payload["response_format"].(map[string]any)
	if !ok || format["type"] != "json_object" {
		t.Fatalf("response_format = %#v", payload["response_format"])
	}
	messages, ok := payload["messages"].([]any)
	if !ok || len(messages) != 5 {
		t.Fatalf("messages = %#v, want five wire messages", payload["messages"])
	}
	user := messages[1].(map[string]any)
	content := user["content"].([]any)
	if text, exists := content[0].(map[string]any)["text"]; !exists || text != "" {
		t.Fatalf("empty multimodal text part = %#v", content[0])
	}
	image := content[1].(map[string]any)["image_url"].(map[string]any)
	if image["url"] != "data:image/png;base64,cG5n" {
		t.Fatalf("image URL = %#v", image["url"])
	}
	assistantCalls := messages[2].(map[string]any)["tool_calls"].([]any)
	synthesizedID := assistantCalls[1].(map[string]any)["id"]
	if synthesizedID != "call_llm_go_2" || messages[3].(map[string]any)["tool_call_id"] != "call_llm_go_1" ||
		messages[4].(map[string]any)["tool_call_id"] != synthesizedID {
		t.Fatalf("expanded tool messages = %#v", messages[3:])
	}
	if content := messages[3].(map[string]any)["content"]; content != "Error: sunny" {
		t.Fatalf("error tool result content = %#v, want %q", content, "Error: sunny")
	}
	tools := payload["tools"].([]any)
	firstFunction := tools[0].(map[string]any)["function"].(map[string]any)
	if firstFunction["strict"] != true {
		t.Fatalf("tool strict = %#v", firstFunction["strict"])
	}
	secondFunction := tools[1].(map[string]any)["function"].(map[string]any)
	if _, exists := secondFunction["strict"]; exists {
		t.Fatalf("false tool strict was encoded: %#v", secondFunction)
	}
}

func TestNonStreamingSDKIgnoresAmbientConfiguration(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "ambient-key")
	t.Setenv("OPENAI_ADMIN_KEY", "ambient-admin-key")
	t.Setenv("OPENAI_ORG_ID", "ambient-org")
	t.Setenv("OPENAI_PROJECT_ID", "ambient-project")
	t.Setenv("OPENAI_BASE_URL", "http://ambient.invalid/")
	t.Setenv("OPENAI_CUSTOM_HEADERS", "X-Ambient: leaked")

	requests := make(chan *http.Request, 1)
	client, err := New(Config{
		Model:   "model",
		APIKey:  "ignored-key",
		BaseURL: "http://configured.example/proxy/v1?",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests <- request.Clone(request.Context())
			body := `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": {"application/json"}},
				Body:       io.NopCloser(strings.NewReader(body)),
			}, nil
		})},
		Headers: http.Header{
			"authorization": {"Basic header-owned"},
			"x-trace":       {"trace"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}); err != nil || response == nil {
		t.Fatalf("Generate() = (%#v, %v)", response, err)
	}

	request := <-requests
	if request.URL.Host != "configured.example" || request.URL.Path != "/proxy/v1/chat/completions" ||
		request.URL.RawQuery != "" || !request.URL.ForceQuery {
		t.Fatalf("request URL = %#v, want chat completions path with empty query marker", request.URL)
	}
	if request.Header.Get("Authorization") != "Basic header-owned" || request.Header.Get("X-Trace") != "trace" ||
		request.Header.Get("OpenAI-Organization") != "" || request.Header.Get("OpenAI-Project") != "" ||
		request.Header.Get("X-Ambient") != "" {
		t.Fatalf("request headers = %#v", request.Header)
	}
}

func TestGeneratePreflightOrderAndNoIO(t *testing.T) {
	var calls atomic.Int64
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityVision},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, errors.New("unexpected request")
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	cause := errors.New("stop now")
	canceled, cancel := context.WithCancelCause(context.Background())
	cancel(cause)

	response, err := client.Generate(canceled, llm.Request{})
	if response != nil {
		t.Fatalf("Generate(invalid) response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "validate")

	valid := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	response, err = client.Generate(canceled, valid)
	if response != nil {
		t.Fatalf("Generate(canceled) response = %#v, want nil", response)
	}
	if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Generate(canceled) error = %v, want cancellation and cause", err)
	}

	audio := llm.Request{Messages: []llm.Message{{
		Role:    llm.RoleUser,
		Content: []llm.Part{{Kind: llm.PartAudio, Data: []byte("audio"), MediaType: "audio/wav"}},
	}}}
	response, err = client.Generate(context.Background(), audio)
	if response != nil {
		t.Fatalf("Generate(audio) response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindUnsupported, "generate")

	mixedUnsupported := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.Part{{Kind: llm.PartImage, Data: []byte("image"), MediaType: "image/png"}}},
		{Role: llm.RoleUser, Content: []llm.Part{{Kind: llm.PartAudio, Data: []byte("audio"), MediaType: "audio/wav"}}},
	}}
	response, err = client.Generate(context.Background(), mixedUnsupported)
	if response != nil {
		t.Fatalf("Generate(mixed unsupported) response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindUnsupported, "generate")
	if !strings.Contains(err.Error(), "audio") {
		t.Fatalf("Generate(mixed unsupported) error = %v, want capability error first", err)
	}

	badImage := llm.Request{Messages: []llm.Message{{
		Role:    llm.RoleUser,
		Content: []llm.Part{{Kind: llm.PartImage, Data: []byte("data"), MediaType: "text/plain"}},
	}}}
	response, err = client.Generate(context.Background(), badImage)
	if response != nil {
		t.Fatalf("Generate(bad image) response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "generate")
	if got := calls.Load(); got != 0 {
		t.Fatalf("HTTP calls = %d, want zero", got)
	}
}

func TestGenerateRequiresConfiguredCapabilities(t *testing.T) {
	var calls atomic.Int64
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, errors.New("unexpected request")
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	tests := []struct {
		name    string
		request llm.Request
		want    string
	}{
		{
			name: "tools",
			request: llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
				Tools:    []llm.Tool{{Name: "lookup", InputSchema: []byte(`{}`)}},
			},
			want: "tool capability",
		},
		{
			name: "JSON",
			request: llm.Request{
				Messages:       []llm.Message{{Role: llm.RoleUser}},
				ResponseFormat: llm.ResponseFormatJSON,
			},
			want: "JSON capability",
		},
		{
			name: "vision",
			request: llm.Request{Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: []llm.Part{{Kind: llm.PartImage, Data: []byte("image"), MediaType: "image/png"}},
			}}},
			want: "vision capability",
		},
		{
			name: "audio",
			request: llm.Request{Messages: []llm.Message{{
				Role:    llm.RoleUser,
				Content: []llm.Part{{Kind: llm.PartAudio, Data: []byte("audio"), MediaType: "audio/wav"}},
			}}},
			want: "audio capability",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, err := client.Generate(context.Background(), test.request)
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			requireModelError(t, err, llm.KindUnsupported, "generate")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Generate() error = %q, want substring %q", err, test.want)
			}
		})
	}

	if got := calls.Load(); got != 0 {
		t.Fatalf("HTTP calls = %d, want zero", got)
	}
}

func TestGenerateRejectsProviderConstraintsBeforeIO(t *testing.T) {
	var calls atomic.Int64
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityTools, llm.CapabilityJSON},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			calls.Add(1)
			return nil, errors.New("unexpected request")
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	temperature := 2.1
	presencePenalty := 2.1
	frequencyPenalty := -2.1
	tooManyTools := make([]llm.Tool, maxTools+1)
	for i := range tooManyTools {
		tooManyTools[i] = llm.Tool{
			Name:        fmt.Sprintf("tool_%d", i),
			InputSchema: []byte(`{}`),
		}
	}
	tests := []struct {
		name    string
		request llm.Request
		want    string
	}{
		{
			name: "temperature",
			request: llm.Request{
				Messages:    []llm.Message{{Role: llm.RoleUser}},
				Temperature: &temperature,
			},
			want: "temperature",
		},
		{
			name: "presence penalty",
			request: llm.Request{
				Messages:        []llm.Message{{Role: llm.RoleUser}},
				PresencePenalty: &presencePenalty,
			},
			want: "presence penalty",
		},
		{
			name: "frequency penalty",
			request: llm.Request{
				Messages:         []llm.Message{{Role: llm.RoleUser}},
				FrequencyPenalty: &frequencyPenalty,
			},
			want: "frequency penalty",
		},
		{
			name: "stop count",
			request: llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
				Stop:     []string{"1", "2", "3", "4", "5"},
			},
			want: "more than four",
		},
		{
			name: "JSON with stop",
			request: llm.Request{
				Messages:       []llm.Message{{Role: llm.RoleUser}},
				ResponseFormat: llm.ResponseFormatJSON,
				Stop:           []string{"END"},
			},
			want: "cannot be combined",
		},
		{
			name: "tool name",
			request: llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
				Tools:    []llm.Tool{{Name: "bad name", InputSchema: []byte(`{}`)}},
			},
			want: "1-64",
		},
		{
			name: "boolean tool schema",
			request: llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
				Tools:    []llm.Tool{{Name: "lookup", InputSchema: []byte(`true`)}},
			},
			want: "must be a JSON object",
		},
		{
			name: "tool count",
			request: llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
				Tools:    tooManyTools,
			},
			want: "more than 128",
		},
		{
			name: "historical call name",
			request: llm.Request{Messages: []llm.Message{
				{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{Name: "bad name", Arguments: []byte(`{}`)}}},
				{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{Name: "bad name"}}},
			}},
			want: "tool call 0 name",
		},
		{
			name: "unresolved tool call",
			request: llm.Request{Messages: []llm.Message{{
				Role:      llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "lookup", Arguments: []byte(`{}`)}},
			}}},
			want: "pending tool calls",
		},
		{
			name: "interleaved tool result",
			request: llm.Request{Messages: []llm.Message{
				{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call-1", Name: "lookup", Arguments: []byte(`{}`)}}},
				{Role: llm.RoleUser, Content: []llm.Part{{Text: "wait"}}},
				{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call-1"}}},
			}},
			want: "must contain tool results",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, err := client.Generate(context.Background(), test.request)
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			requireModelError(t, err, llm.KindInvalidRequest, "generate")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Generate() error = %q, want substring %q", err, test.want)
			}
		})
	}

	if got := calls.Load(); got != 0 {
		t.Fatalf("HTTP calls = %d, want zero", got)
	}
}

func TestStreamPreflightOrder(t *testing.T) {
	client, err := New(Config{Model: "model"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	stream, err := client.Stream(canceled, llm.Request{})
	if stream != nil {
		t.Fatalf("Stream(invalid) = %#v, want nil", stream)
	}
	requireModelError(t, err, llm.KindInvalidRequest, "validate")

	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	stream, err = client.Stream(canceled, request)
	if stream != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("Stream(canceled) = (%#v, %v)", stream, err)
	}

	stream, err = client.Stream(context.Background(), request)
	if stream != nil {
		t.Fatalf("Stream() = %#v, want nil", stream)
	}
	requireModelError(t, err, llm.KindUnsupported, "stream")
}

func TestGenerateClassifiesHTTPError(t *testing.T) {
	tests := []struct {
		name       string
		status     int
		body       string
		retryAfter string
		wantKind   llm.ErrorKind
	}{
		{name: "authentication", status: 401, body: `{"error":{"message":"bad key","type":"auth","code":"invalid_api_key"}}`, wantKind: llm.KindAuthentication},
		{name: "authentication beats text", status: 401, body: `{"error":{"message":"invalid context window and key"}}`, wantKind: llm.KindAuthentication},
		{name: "permission", status: 403, body: `{"error":{"message":"denied"}}`, wantKind: llm.KindPermission},
		{name: "rate limit", status: 429, body: `{"error":{"message":"slow down","type":"rate_limit"}}`, retryAfter: "3", wantKind: llm.KindRateLimit},
		{name: "context limit", status: 400, body: `{"error":{"message":"too long","code":"context_length_exceeded"}}`, wantKind: llm.KindContextLimit},
		{name: "not found beats text", status: 404, body: `{"error":{"message":"context window route missing"}}`, wantKind: llm.KindInvalidRequest},
		{name: "invalid request", status: 422, body: `{"message":"invalid"}`, wantKind: llm.KindInvalidRequest},
		{name: "provider", status: 503, body: `not json`, wantKind: llm.KindProvider},
		{name: "provider beats text", status: 503, body: `{"error":{"message":"context window unavailable"}}`, wantKind: llm.KindProvider},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if test.retryAfter != "" {
					w.Header().Set("Retry-After", test.retryAfter)
				}
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			}))
			client, err := New(Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client()})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			response, err := client.Generate(context.Background(), llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
			})
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			modelErr := requireModelError(t, err, test.wantKind, "generate")
			if modelErr.HTTPStatus != test.status {
				t.Fatalf("HTTPStatus = %d, want %d", modelErr.HTTPStatus, test.status)
			}
			if test.retryAfter != "" && modelErr.RetryAfter != 3*time.Second {
				t.Fatalf("RetryAfter = %v, want 3s", modelErr.RetryAfter)
			}
			if _, ok := errors.AsType[*APIError](err); !ok {
				t.Fatalf("errors.As(%v, *APIError) = false", err)
			}
		})
	}
}

func TestGenerateRejectsMalformedResponse(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "duplicate name", body: `{"choices":[],"choices":[]}`, want: "duplicate object member"},
		{name: "no choices", body: `{"choices":[]}`, want: "no choices"},
		{name: "multiple choices", body: `{"choices":[{"index":0,"message":{"role":"assistant","content":"first"},"finish_reason":"stop"},{"index":1,"message":{"role":"assistant","content":"second"},"finish_reason":"stop"}]}`, want: "want exactly one"},
		{name: "missing choice index", body: `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, want: "no index"},
		{name: "null choice index", body: `{"choices":[{"index":null,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, want: "no index"},
		{name: "wrong choice index", body: `{"choices":[{"index":1,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, want: "index 1"},
		{name: "missing message", body: `{"choices":[{"index":0,"finish_reason":"stop"}]}`, want: "no message"},
		{name: "missing role", body: `{"choices":[{"index":0,"message":{"content":"ok"},"finish_reason":"stop"}]}`, want: `role ""`},
		{name: "missing finish reason", body: `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`, want: "no finish reason"},
		{name: "wrong role", body: `{"choices":[{"index":0,"message":{"role":"tool","content":"ok"},"finish_reason":"stop"}]}`, want: `role "tool"`},
		{name: "unknown finish reason", body: `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"other"}]}`, want: "unknown finish reason"},
		{name: "tool finish without calls", body: `{"choices":[{"index":0,"message":{"role":"assistant"},"finish_reason":"tool_calls"}]}`, want: "has no tool calls"},
		{name: "legacy finish without call", body: `{"choices":[{"index":0,"message":{"role":"assistant"},"finish_reason":"function_call"}]}`, want: "has no function call"},
		{name: "malformed arguments", body: `{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"one","type":"function","function":{"name":"tool","arguments":"{"}}]},"finish_reason":"tool_calls"}]}`, want: "not strict JSON"},
		{name: "missing tool type", body: `{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"one","function":{"name":"tool","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, want: "unsupported type"},
		{name: "missing tool ID", body: `{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"tool","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, want: "has no ID"},
		{name: "duplicate tool ID", body: `{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"one","type":"function","function":{"name":"a","arguments":"{}"}},{"id":"one","type":"function","function":{"name":"b","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, want: "repeats ID"},
		{name: "undeclared tool", body: `{"choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"one","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, want: "undeclared function"},
		{name: "missing usage field", body: `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`, want: "incomplete token usage"},
		{name: "null usage field", body: `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":null}}`, want: "incomplete token usage"},
		{name: "negative usage", body: `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":-1,"completion_tokens":2,"total_tokens":1}}`, want: "negative token usage"},
		{name: "inconsistent usage", body: `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":99}}`, want: "inconsistent token usage"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(Config{
				Model:   "model",
				BaseURL: "http://example.com/v1",
				HTTPClient: &http.Client{Transport: staticResponse{
					status: http.StatusOK,
					body:   test.body,
				}},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			response, err := client.Generate(context.Background(), llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
			})
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			requireModelError(t, err, llm.KindMalformedResponse, "generate")
			if !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Generate() error = %q, want substring %q", err, test.want)
			}
		})
	}
}

func TestGeneratePreservesMalformedResponseCause(t *testing.T) {
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticResponse{
			status: http.StatusOK,
			body:   `{`,
		}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
	})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindMalformedResponse, "generate")
	if _, ok := errors.AsType[*jsontext.SyntacticError](err); !ok {
		t.Fatalf("errors.As(%v, *jsontext.SyntacticError) = false", err)
	}
}

func TestGenerateValidatesJSONModeResponse(t *testing.T) {
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityJSON},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticResponse{
			status: http.StatusOK,
			body:   `{"choices":[{"index":0,"message":{"role":"assistant","content":"not JSON"},"finish_reason":"stop"}]}`,
		}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	response, err := client.Generate(context.Background(), llm.Request{
		Messages:       []llm.Message{{Role: llm.RoleUser}},
		ResponseFormat: llm.ResponseFormatJSON,
	})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindMalformedResponse, "generate")
	if !strings.Contains(err.Error(), "not strict JSON") {
		t.Fatalf("Generate() error = %v", err)
	}

	filteredClient, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityJSON},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticResponse{
			status: http.StatusOK,
			body:   `{"choices":[{"index":0,"message":{"role":"assistant","content":"filtered partial"},"finish_reason":"content_filter"}]}`,
		}},
	})
	if err != nil {
		t.Fatalf("New(filtered) error = %v", err)
	}
	response, err = filteredClient.Generate(context.Background(), llm.Request{
		Messages:       []llm.Message{{Role: llm.RoleUser}},
		ResponseFormat: llm.ResponseFormatJSON,
	})
	if err != nil {
		t.Fatalf("Generate(filtered) error = %v", err)
	}
	if response.FinishReason != llm.FinishReasonContentFilter || response.Text() != "filtered partial" {
		t.Fatalf("Generate(filtered) response = %#v", response)
	}

	truncatedClient, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityJSON},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticResponse{
			status: http.StatusOK,
			body:   `{"choices":[{"index":0,"message":{"role":"assistant","content":"{"},"finish_reason":"length"}]}`,
		}},
	})
	if err != nil {
		t.Fatalf("New(truncated) error = %v", err)
	}
	response, err = truncatedClient.Generate(context.Background(), llm.Request{
		Messages:       []llm.Message{{Role: llm.RoleUser}},
		ResponseFormat: llm.ResponseFormatJSON,
	})
	if err != nil {
		t.Fatalf("Generate(truncated) error = %v", err)
	}
	if response.FinishReason != llm.FinishReasonLength || response.Text() != "{" {
		t.Fatalf("Generate(truncated) response = %#v", response)
	}
}

func TestGenerateDecodesLegacyFunctionCall(t *testing.T) {
	client, err := New(Config{
		Model:        "model",
		Capabilities: []llm.Capability{llm.CapabilityTools},
		BaseURL:      "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticResponse{
			status: http.StatusOK,
			body: `{"choices":[{"index":0,
				"message":{"role":"assistant","content":null,"function_call":{"name":"lookup","arguments":"{\"id\":1}"}},
				"finish_reason":"function_call"
			}]}`,
		}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
		Tools:    []llm.Tool{{Name: "lookup", InputSchema: []byte(`{}`)}},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.FinishReason != llm.FinishReasonToolCall || len(response.Message.ToolCalls) != 1 {
		t.Fatalf("Generate() response = %#v", response)
	}
	call := response.Message.ToolCalls[0]
	if call.ID != "" || call.Name != "lookup" || string(call.Arguments) != `{"id":1}` {
		t.Fatalf("legacy ToolCall = %#v", call)
	}
}

func TestGeneratePreservesRefusalData(t *testing.T) {
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticResponse{
			status: http.StatusOK,
			body:   `{"choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":"cannot comply"},"finish_reason":"stop"}]}`,
		}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	data, err := ParseMessageData(response.Message)
	if err != nil {
		t.Fatalf("ParseMessageData() error = %v", err)
	}
	if data.Refusal == nil || *data.Refusal != "cannot comply" {
		t.Fatalf("MessageData = %#v", data)
	}

	wrequest, err := newChatRequest("model", llm.Request{Messages: []llm.Message{response.Message}})
	if err != nil {
		t.Fatalf("newChatRequest() error = %v", err)
	}
	if got := wrequest.Messages[0].Refusal; got == nil || *got != "cannot comply" {
		t.Fatalf("round-tripped refusal = %#v", got)
	}
	if wrequest.Messages[0].Content != nil {
		t.Fatalf("round-tripped refusal content = %#v, want nil", wrequest.Messages[0].Content)
	}
}

func TestGeneratePreservesEmptyRefusalData(t *testing.T) {
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: staticResponse{
			status: http.StatusOK,
			body:   `{"choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":""},"finish_reason":"stop"}]}`,
		}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	data, err := ParseMessageData(response.Message)
	if err != nil {
		t.Fatalf("ParseMessageData() error = %v", err)
	}
	if data.Refusal == nil || *data.Refusal != "" {
		t.Fatalf("MessageData = %#v, want present empty refusal", data)
	}

	wrequest, err := newChatRequest("model", llm.Request{Messages: []llm.Message{response.Message}})
	if err != nil {
		t.Fatalf("newChatRequest() error = %v", err)
	}
	if got := wrequest.Messages[0].Refusal; got == nil || *got != "" {
		t.Fatalf("round-tripped refusal = %#v, want present empty string", got)
	}
	if wrequest.Messages[0].Content != nil {
		t.Fatalf("round-tripped refusal content = %#v, want nil", wrequest.Messages[0].Content)
	}
}

func TestNewChatRequestIgnoresUnrecognizedProviderData(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{
			name: "foreign provider",
			data: []byte(`{"provider":"other","version":1,"data":{"refusal":"foreign"}}`),
		},
		{
			name: "future OpenAI version",
			data: []byte(`{"provider":"openai","version":2,"data":{"refusal":{"future":true}}}`),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			wrequest, err := newChatRequest("model", llm.Request{Messages: []llm.Message{{
				Role:         llm.RoleAssistant,
				ProviderData: test.data,
			}}})
			if err != nil {
				t.Fatalf("newChatRequest() error = %v", err)
			}
			if got := wrequest.Messages[0].Refusal; got != nil {
				t.Fatalf("unrecognized provider refusal = %#v, want nil", got)
			}
		})
	}
}

func TestParseMessageDataRejectsMalformedData(t *testing.T) {
	tests := []llm.Message{
		{ProviderData: []byte(`{`)},
		{ProviderData: []byte(`{"provider":"openai","version":1}`)},
		{ProviderData: []byte(`{"provider":"openai","version":1,"data":null}`)},
		{ProviderData: []byte(`{"provider":"openai","version":1,"data":{"refusal":{}}}`)},
	}
	for _, message := range tests {
		if data, err := ParseMessageData(message); err == nil {
			t.Fatalf("ParseMessageData(%s) = (%#v, nil), want error", message.ProviderData, data)
		}
	}
}

func TestGeneratePreservesCancellationDuringHTTP(t *testing.T) {
	cause := errors.New("cancel request")
	ctx, cancel := context.WithCancelCause(context.Background())
	var calls atomic.Int64
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
			calls.Add(1)
			cancel(cause)
			<-request.Context().Done()
			return nil, request.Context().Err()
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	response, err := client.Generate(ctx, llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
	})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Generate() error = %v, want cancellation and custom cause", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("HTTP calls = %d, want one", got)
	}
}

func TestGeneratePrefersCancellationDuringResponseProcessing(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		body     string
		cancelAt int64
	}{
		{
			name:     "JSON decoding",
			status:   http.StatusOK,
			body:     `{`,
			cancelAt: 4,
		},
		{
			name:     "response validation",
			status:   http.StatusOK,
			body:     `{"choices":[]}`,
			cancelAt: 4,
		},
		{
			name:     "provider error decoding",
			status:   http.StatusBadRequest,
			body:     `{"error":{"message":"invalid request"}}`,
			cancelAt: 3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cause := errors.New("cancel during " + test.name)
			ctx := newCancelOnCheckContext(test.cancelAt, cause)
			client, err := New(Config{
				Model:   "model",
				BaseURL: "http://example.com/v1",
				HTTPClient: &http.Client{Transport: staticResponse{
					status: test.status,
					body:   test.body,
				}},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			response, err := client.Generate(ctx, llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
			})
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
				t.Fatalf("Generate() error = %v, want cancellation and custom cause", err)
			}
		})
	}
}

func TestGeneratePreservesTransportError(t *testing.T) {
	sentinel := errors.New("network down")
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, sentinel
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	response, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser}},
	})
	if response != nil {
		t.Fatalf("Generate() response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindTransport, "generate")
	if !errors.Is(err, sentinel) {
		t.Fatalf("Generate() error = %v, want sentinel identity", err)
	}
}

func TestGeneratePreservesResponseCloseError(t *testing.T) {
	closeErr := errors.New("close failed")
	tests := []struct {
		name   string
		status int
		body   string
		kind   llm.ErrorKind
	}{
		{
			name:   "success response",
			status: http.StatusOK,
			body:   `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
			kind:   llm.KindTransport,
		},
		{
			name:   "provider response",
			status: http.StatusTooManyRequests,
			body:   `{"error":{"message":"slow down","type":"rate_limit_error"}}`,
			kind:   llm.KindRateLimit,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(Config{
				Model:   "model",
				BaseURL: "http://example.com/v1",
				HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode: test.status,
						Status:     http.StatusText(test.status),
						Header:     make(http.Header),
						Body:       &closeErrorBody{Reader: strings.NewReader(test.body), err: closeErr},
					}, nil
				})},
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			response, err := client.Generate(context.Background(), llm.Request{
				Messages: []llm.Message{{Role: llm.RoleUser}},
			})
			if response != nil {
				t.Fatalf("Generate() response = %#v, want nil", response)
			}
			requireModelError(t, err, test.kind, "generate")
			if !errors.Is(err, closeErr) {
				t.Fatalf("Generate() error = %v, want close error", err)
			}
		})
	}
}

func TestNonStreamingSDKSingleAttemptAndBodyBounds(t *testing.T) {
	t.Run("error", func(t *testing.T) {
		var calls atomic.Int64
		var closed atomic.Bool
		client, err := New(Config{
			Model:   "model",
			BaseURL: "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				calls.Add(1)
				return &http.Response{
					StatusCode: http.StatusInternalServerError,
					Status:     "500 Internal Server Error",
					Header:     make(http.Header),
					Body: &observedCloseBody{
						Reader: strings.NewReader(strings.Repeat("x", maxErrorBodyBytes+1)),
						closed: &closed,
					},
				}, nil
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		_, callErr := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
		modelErr := requireModelError(t, callErr, llm.KindProvider, "generate")
		var apiErr *APIError
		if modelErr.HTTPStatus != http.StatusInternalServerError || !errors.As(callErr, &apiErr) ||
			!strings.Contains(apiErr.Message, "response body exceeds") {
			t.Fatalf("call error = %#v, API error = %#v", modelErr, apiErr)
		}
		if calls.Load() != 1 {
			t.Fatalf("HTTP calls = %d, want one", calls.Load())
		}
		if !closed.Load() {
			t.Fatal("provider error body was not closed")
		}
	})

	t.Run("success", func(t *testing.T) {
		var closed atomic.Bool
		client, err := New(Config{
			Model:   "model",
			BaseURL: "http://example.com/v1",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Status:     "200 OK",
					Header:     make(http.Header),
					Body: &observedCloseBody{
						Reader: strings.NewReader(strings.Repeat("x", maxResponseBodyBytes+1)),
						closed: &closed,
					},
				}, nil
			})},
		})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
		if response != nil {
			t.Fatalf("Generate() response = %#v", response)
		}
		requireModelError(t, err, llm.KindMalformedResponse, "generate")
		if !closed.Load() {
			t.Fatal("success response body was not closed")
		}
	})
}

func TestNonStreamingSDKRedirectCancellation(t *testing.T) {
	server := httptest.NewTestServer(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		http.Redirect(writer, request, "/redirected", http.StatusTemporaryRedirect)
	}))
	cause := errors.New("cancel during redirect")
	redirectErr := errors.New("reject redirect")
	ctx, cancel := context.WithCancelCause(context.Background())
	httpClient := server.Client()
	httpClient.CheckRedirect = func(*http.Request, []*http.Request) error {
		cancel(cause)
		return redirectErr
	}
	client, err := New(Config{
		Model:      "model",
		BaseURL:    server.URL,
		HTTPClient: httpClient,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	_, callErr := client.Generate(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if !errors.Is(callErr, context.Canceled) || !errors.Is(callErr, cause) || errors.Is(callErr, redirectErr) {
		t.Fatalf("call error = %v", callErr)
	}
	if modelErr, ok := errors.AsType[*llm.Error](callErr); ok {
		t.Fatalf("call error contains model error: %#v", modelErr)
	}
}

func TestNonStreamingSDKCancellationAfterResponseClosesBody(t *testing.T) {
	cause := errors.New("cancel after response")
	ctx, cancel := context.WithCancelCause(context.Background())
	body := newBlockingCloseBody()
	client, err := New(Config{
		Model:   "model",
		BaseURL: "http://example.com/v1",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			cancel(cause)
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Header:     http.Header{"Content-Type": {"application/json"}},
				Body:       body,
			}, nil
		})},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	result := make(chan error, 1)
	go func() {
		_, callErr := client.Generate(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
		result <- callErr
	}()
	select {
	case callErr := <-result:
		if !errors.Is(callErr, context.Canceled) || !errors.Is(callErr, cause) {
			t.Fatalf("call error = %v", callErr)
		}
	case <-time.After(5 * time.Second):
		_ = body.Close()
		t.Fatal("call blocked reading a response after cancellation")
	}
	select {
	case <-body.closed:
	default:
		t.Fatal("canceled response body was not closed")
	}
}

func TestReadLimited(t *testing.T) {
	body, tooLarge, err := readLimited(strings.NewReader("abcde"), 4)
	if err != nil {
		t.Fatalf("readLimited() error = %v", err)
	}
	if got := string(body); got != "abcd" || !tooLarge {
		t.Fatalf("readLimited() = (%q, %v), want (abcd, true)", got, tooLarge)
	}
}

func TestRetryAfterClampsOverflow(t *testing.T) {
	if delay := retryAfter("9223372036854775807"); delay <= 0 {
		t.Fatalf("retryAfter(max int64) = %v, want a positive clamped duration", delay)
	}
}

func requireModelError(t *testing.T, err error, kind llm.ErrorKind, op string) *llm.Error {
	t.Helper()
	if err == nil {
		t.Fatal("error = nil")
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) {
		t.Fatalf("errors.As(%v, *llm.Error) = false", err)
	}
	wantProvider := "openai"
	if op == "validate" {
		wantProvider = ""
	}
	if modelErr.Kind != kind || modelErr.Op != op || modelErr.Provider != wantProvider {
		t.Fatalf("model error = %#v, want kind %v op %q provider %q", modelErr, kind, op, wantProvider)
	}
	return modelErr
}

type observedCloseBody struct {
	io.Reader
	closed *atomic.Bool
}

func (body *observedCloseBody) Close() error {
	body.closed.Store(true)
	return nil
}

type blockingCloseBody struct {
	closed chan struct{}
	once   sync.Once
}

func newBlockingCloseBody() *blockingCloseBody {
	return &blockingCloseBody{closed: make(chan struct{})}
}

func (body *blockingCloseBody) Read([]byte) (int, error) {
	<-body.closed
	return 0, errors.New("body closed")
}

func (body *blockingCloseBody) Close() error {
	body.once.Do(func() { close(body.closed) })
	return nil
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

type cancelOnCheckContext struct {
	context.Context
	cancel   context.CancelCauseFunc
	cause    error
	cancelAt int64
	checks   atomic.Int64
}

func newCancelOnCheckContext(cancelAt int64, cause error) *cancelOnCheckContext {
	ctx, cancel := context.WithCancelCause(context.Background())
	return &cancelOnCheckContext{
		Context:  ctx,
		cancel:   cancel,
		cause:    cause,
		cancelAt: cancelAt,
	}
}

func (ctx *cancelOnCheckContext) Err() error {
	if ctx.checks.Add(1) == ctx.cancelAt {
		ctx.cancel(ctx.cause)
	}
	return ctx.Context.Err()
}

type staticResponse struct {
	status int
	body   string
}

func (response staticResponse) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: response.status,
		Status:     http.StatusText(response.status),
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(response.body)),
	}, nil
}
