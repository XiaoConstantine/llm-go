package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"reflect"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/anthropic"
	"github.com/XiaoConstantine/llm-go/openai"
	"github.com/XiaoConstantine/llm-go/openai/codex"
	"github.com/XiaoConstantine/llm-go/openai/responses"
)

type protocolFixture struct {
	body  map[string]any
	calls int
	kind  string
}

func (f *protocolFixture) RoundTrip(req *http.Request) (*http.Response, error) {
	f.calls++
	if req.URL.Host != "fixture.invalid" {
		return nil, fmt.Errorf("non-fixture HTTP request refused")
	}
	if err := json.NewDecoder(req.Body).Decode(&f.body); err != nil {
		return nil, err
	}
	body := `{"id":"chat-fixture","model":"arbitrary-id","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	contentType := "application/json"
	switch f.kind {
	case "anthropic":
		body = `{"id":"msg-fixture","type":"message","role":"assistant","model":"arbitrary-id","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	case "responses", "codex":
		contentType = "text/event-stream"
		body = "event: response.completed\ndata: " + `{"type":"response.completed","response":{"id":"resp-fixture","model":"arbitrary-id","status":"completed","output":[]}}` + "\n\n"
	}
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {contentType}}, Body: io.NopCloser(strings.NewReader(body)), Request: req}, nil
}

func protocolClient(t *testing.T, kind string, adaptive bool) (llm.Generator, *protocolFixture) {
	t.Helper()
	fixture := &protocolFixture{kind: kind}
	client := &http.Client{Transport: fixture}
	caps := []llm.Capability{llm.CapabilityGeneration, llm.CapabilityTools, llm.CapabilityStreaming}
	var model llm.Generator
	var err error
	switch kind {
	case "chat":
		model, err = openai.New(openai.Config{Model: "arbitrary-id", APIKey: "fixture", BaseURL: "https://fixture.invalid/v1", HTTPClient: client, Capabilities: caps})
	case "responses":
		model, err = responses.New(responses.Config{Model: "arbitrary-id", APIKey: "fixture", BaseURL: "https://fixture.invalid/v1", HTTPClient: client, Capabilities: caps})
	case "codex":
		model, err = codex.New(codex.Config{Model: "arbitrary-id", AccessToken: "fixture", AccountID: "fixture-account", BaseURL: "https://fixture.invalid", HTTPClient: client, Capabilities: caps})
	case "anthropic":
		mode := llm.CompatibilityDisabled
		if adaptive {
			mode = llm.CompatibilityEnabled
		}
		model, err = anthropic.NewWithOptions(anthropic.Config{Model: "arbitrary-id", APIKey: "fixture", BaseURL: "https://fixture.invalid/v1", HTTPClient: client, Capabilities: caps}, anthropic.Options{ModelCompatibility: &llm.AnthropicCompatibility{AdaptiveThinking: mode}})
	}
	if err != nil {
		t.Fatal(err)
	}
	return model, fixture
}

func protocolRequest() llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
}

func protocolPtr[T any](value T) *T { return &value }

func protocolValue[T any](value *T) string {
	if value == nil {
		return "absent"
	}
	return fmt.Sprint(*value)
}

func TestProtocolOptionsParallelPresence(t *testing.T) {
	for _, kind := range []string{"chat", "responses", "codex", "anthropic"} {
		for _, parallel := range []*bool{nil, protocolPtr(false), protocolPtr(true)} {
			name := "absent"
			if parallel != nil {
				name = fmt.Sprint(*parallel)
			}
			t.Run(kind+"/"+name, func(t *testing.T) {
				client, fixture := protocolClient(t, kind, false)
				request := protocolRequest()
				request.ParallelToolCalls = parallel
				request.Tools = []llm.Tool{{Name: "probe", InputSchema: json.RawMessage(`{"type":"object","properties":{}}`)}}
				if _, err := client.Generate(context.Background(), request); err != nil {
					t.Fatal(err)
				}
				var actual any
				var present bool
				if kind == "anthropic" {
					choice, _ := fixture.body["tool_choice"].(map[string]any)
					actual, present = choice["disable_parallel_tool_use"]
					if parallel != nil && actual != !*parallel {
						t.Fatalf("wire = %#v", fixture.body)
					}
				} else {
					actual, present = fixture.body["parallel_tool_calls"]
					if parallel != nil && actual != *parallel {
						t.Fatalf("wire = %#v", fixture.body)
					}
				}
				if present != (parallel != nil) {
					t.Fatalf("presence = %v, wire = %#v", present, fixture.body)
				}
			})
		}
	}
}

func TestProtocolOptionsChatWire(t *testing.T) {
	client, fixture := protocolClient(t, "chat", false)
	request := protocolRequest()
	extra := map[string]any{"repetition_penalty": 1.2, "custom.literal": map[string]any{"nested": []any{"original"}}}
	envelope, err := llm.NewChatExtraFields(extra)
	if err != nil {
		t.Fatal(err)
	}
	extra["repetition_penalty"] = 99
	extra["custom.literal"].(map[string]any)["nested"].([]any)[0] = "mutated"
	request.OpenAIChat = &llm.OpenAIChatOptions{LogitBias: map[string]int64{"12": -3}, LogProbs: protocolPtr(true), TopLogProbs: protocolPtr(3), User: protocolPtr("person"), Verbosity: protocolPtr("high"), Prediction: &llm.ChatPrediction{Content: "predicted"}, Store: protocolPtr(false), Metadata: map[string]string{"task": "test"}, SafetyIdentifier: protocolPtr("safety"), ServiceTier: protocolPtr("priority"), ExtraFields: envelope}
	if _, err := client.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]any{"logprobs": true, "top_logprobs": float64(3), "user": "person", "verbosity": "high", "store": false, "safety_identifier": "safety", "service_tier": "priority", "repetition_penalty": 1.2, "prediction": map[string]any{"type": "content", "content": "predicted"}, "metadata": map[string]any{"task": "test"}, "logit_bias": map[string]any{"12": float64(-3)}, "custom.literal": map[string]any{"nested": []any{"original"}}} {
		if !reflect.DeepEqual(fixture.body[key], want) {
			t.Errorf("%s = %#v, want %#v", key, fixture.body[key], want)
		}
	}
}

func TestProtocolOptionsAnthropicToolChoice(t *testing.T) {
	for _, tools := range []bool{false, true} {
		for _, mode := range []llm.ToolChoiceMode{llm.ToolChoiceAuto, llm.ToolChoiceRequired, llm.ToolChoiceNamed, llm.ToolChoiceNone} {
			if !tools && (mode == llm.ToolChoiceRequired || mode == llm.ToolChoiceNamed) {
				continue
			}
			for _, parallel := range []bool{false, true} {
				t.Run(fmt.Sprintf("tools=%v/mode=%s/parallel=%v", tools, mode, parallel), func(t *testing.T) {
					client, fixture := protocolClient(t, "anthropic", false)
					request := protocolRequest()
					request.ParallelToolCalls = &parallel
					request.ToolChoice.Mode = mode
					if tools {
						request.Tools = []llm.Tool{{Name: "probe", InputSchema: json.RawMessage(`{"type":"object","properties":{}}`)}}
					}
					if mode == llm.ToolChoiceNamed {
						request.ToolChoice.Name = "probe"
					}
					if _, err := client.Generate(context.Background(), request); err != nil {
						t.Fatal(err)
					}
					choice, _ := fixture.body["tool_choice"].(map[string]any)
					actual, present := choice["disable_parallel_tool_use"]
					if present != (tools && mode != llm.ToolChoiceNone) {
						t.Fatalf("wire = %#v", fixture.body)
					}
					if present && actual != !parallel {
						t.Fatalf("wire = %#v", fixture.body)
					}
				})
			}
		}
	}
}

func TestProtocolOptionsLogprobs(t *testing.T) {
	for _, logprobs := range []*bool{nil, protocolPtr(false), protocolPtr(true)} {
		for _, top := range []*int{nil, protocolPtr(0), protocolPtr(20)} {
			name := "logprobs=" + protocolValue(logprobs) + "/top=" + protocolValue(top)
			t.Run(name, func(t *testing.T) {
				client, fixture := protocolClient(t, "chat", false)
				request := protocolRequest()
				request.OpenAIChat = &llm.OpenAIChatOptions{LogProbs: logprobs, TopLogProbs: top}
				_, err := client.Generate(context.Background(), request)
				if top != nil && logprobs != nil && !*logprobs {
					if err == nil || fixture.calls != 0 {
						t.Fatalf("error %v; HTTP %d", err, fixture.calls)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if top != nil && fixture.body["logprobs"] != true {
					t.Fatalf("wire = %#v", fixture.body)
				}
				if top == nil && logprobs != nil && fixture.body["logprobs"] != *logprobs {
					t.Fatalf("wire = %#v", fixture.body)
				}
			})
		}
	}
}

func TestProtocolOptionsAnthropicWireAndExactBudget(t *testing.T) {
	client, fixture := protocolClient(t, "anthropic", false)
	request := protocolRequest()
	request.TopK = protocolPtr(0)
	extra, err := llm.NewAnthropicExtraFields(map[string]any{"metadata": map[string]any{"user_id": "person"}, "betas": []any{"fixture-beta"}})
	if err != nil {
		t.Fatal(err)
	}
	request.Anthropic = &llm.AnthropicOptions{ExtraFields: extra}
	if _, err := client.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if fixture.body["top_k"] != float64(0) || !reflect.DeepEqual(fixture.body["metadata"], map[string]any{"user_id": "person"}) || !reflect.DeepEqual(fixture.body["betas"], []any{"fixture-beta"}) {
		t.Fatalf("wire = %#v", fixture.body)
	}
	request.TopK = nil
	request.ReasoningPolicy = llm.ReasoningPolicyExact
	request.ReasoningBudgetTokens = 1536
	request.MaxOutputTokens = 2048
	request.Anthropic.ThinkingDisplay = protocolPtr("omitted")
	if _, err := client.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(fixture.body["thinking"], map[string]any{"type": "enabled", "budget_tokens": float64(1536), "display": "omitted"}) {
		t.Fatalf("wire = %#v", fixture.body)
	}
	request.ReasoningBudgetTokens = 2048
	before := fixture.calls
	if _, err := client.Generate(context.Background(), request); err == nil || fixture.calls != before {
		t.Fatalf("invalid exact budget: %v, calls %d", err, fixture.calls)
	}
}

func TestProtocolOptionsExactEffortAndVerbosity(t *testing.T) {
	for _, kind := range []string{"chat", "responses", "codex", "anthropic"} {
		for _, effort := range []llm.ReasoningEffort{llm.ReasoningEffortDefault, llm.ReasoningEffortNone, llm.ReasoningEffortMinimal, llm.ReasoningEffortLow, llm.ReasoningEffortMedium, llm.ReasoningEffortHigh, llm.ReasoningEffortXHigh, llm.ReasoningEffortMax} {
			t.Run(kind+"/"+string(effort), func(t *testing.T) {
				client, fixture := protocolClient(t, kind, true)
				request := protocolRequest()
				request.ReasoningPolicy = llm.ReasoningPolicyExact
				request.ReasoningEffort = effort
				if kind == "responses" || kind == "codex" {
					request.OpenAIResponses = &llm.OpenAIResponsesOptions{Verbosity: protocolPtr("high"), ServiceTier: protocolPtr("priority")}
				}
				if _, err := client.Generate(context.Background(), request); err != nil {
					t.Fatal(err)
				}
				var actual any
				switch kind {
				case "chat":
					actual = fixture.body["reasoning_effort"]
				case "responses", "codex":
					reasoning, _ := fixture.body["reasoning"].(map[string]any)
					actual = reasoning["effort"]
					text, _ := fixture.body["text"].(map[string]any)
					if text["verbosity"] != "high" || fixture.body["service_tier"] != "priority" {
						t.Fatalf("wire = %#v", fixture.body)
					}
				case "anthropic":
					output, _ := fixture.body["output_config"].(map[string]any)
					actual = output["effort"]
					if effort == llm.ReasoningEffortNone {
						if !reflect.DeepEqual(fixture.body["thinking"], map[string]any{"type": "disabled"}) {
							t.Fatalf("wire = %#v", fixture.body)
						}
						return
					}
				}
				if effort == llm.ReasoningEffortDefault {
					if actual != nil {
						t.Fatalf("default produced override %v", actual)
					}
				} else if actual != string(effort) {
					t.Fatalf("effort = %v, want %s", actual, effort)
				}
			})
		}
	}
}

func TestProtocolOptionsValidationPrecedesIO(t *testing.T) {
	for _, test := range []struct {
		name, kind string
		change     func(*llm.Request)
	}{
		{"negative-top-k", "anthropic", func(r *llm.Request) { r.TopK = protocolPtr(-1) }},
		{"chat-top-k", "chat", func(r *llm.Request) { r.TopK = protocolPtr(1) }},
		{"responses-top-k", "responses", func(r *llm.Request) { r.TopK = protocolPtr(1) }},
		{"codex-top-k", "codex", func(r *llm.Request) { r.TopK = protocolPtr(1) }},
		{"wrong-chat-options", "responses", func(r *llm.Request) { r.OpenAIChat = &llm.OpenAIChatOptions{} }},
		{"wrong-responses-options", "anthropic", func(r *llm.Request) { r.OpenAIResponses = &llm.OpenAIResponsesOptions{} }},
		{"wrong-anthropic-options", "chat", func(r *llm.Request) { r.Anthropic = &llm.AnthropicOptions{} }},
		{"codex-chat-options", "codex", func(r *llm.Request) { r.OpenAIChat = &llm.OpenAIChatOptions{} }},
		{"unknown-policy", "chat", func(r *llm.Request) { r.ReasoningPolicy = "unknown" }},
		{"invalid-logit-bias", "chat", func(r *llm.Request) { r.OpenAIChat = &llm.OpenAIChatOptions{LogitBias: map[string]int64{"token": 1}} }},
		{"out-of-range-logit-bias", "chat", func(r *llm.Request) { r.OpenAIChat = &llm.OpenAIChatOptions{LogitBias: map[string]int64{"1": 101}} }},
		{"out-of-range-top-logprobs", "chat", func(r *llm.Request) { r.OpenAIChat = &llm.OpenAIChatOptions{TopLogProbs: protocolPtr(21)} }},
		{"invalid-verbosity", "responses", func(r *llm.Request) {
			r.OpenAIResponses = &llm.OpenAIResponsesOptions{Verbosity: protocolPtr("automatic")}
		}},
		{"invalid-tier", "codex", func(r *llm.Request) {
			r.OpenAIResponses = &llm.OpenAIResponsesOptions{ServiceTier: protocolPtr("unknown")}
		}},
		{"invalid-display", "anthropic", func(r *llm.Request) { r.Anthropic = &llm.AnthropicOptions{ThinkingDisplay: protocolPtr("unknown")} }},
		{"display-without-thinking", "anthropic", func(r *llm.Request) { r.Anthropic = &llm.AnthropicOptions{ThinkingDisplay: protocolPtr("omitted")} }},
		{"exact-effort-budget", "anthropic", func(r *llm.Request) {
			r.ReasoningPolicy = llm.ReasoningPolicyExact
			r.ReasoningEffort = llm.ReasoningEffortHigh
			r.ReasoningBudgetTokens = 1024
		}},
		{"exact-implicit-budget", "anthropic", func(r *llm.Request) {
			r.ReasoningPolicy = llm.ReasoningPolicyExact
			r.ReasoningEffort = llm.ReasoningEffortHigh
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			client, fixture := protocolClient(t, test.kind, false)
			request := protocolRequest()
			test.change(&request)
			if _, err := client.Generate(context.Background(), request); err == nil || fixture.calls != 0 {
				t.Fatalf("error %v; HTTP calls %d", err, fixture.calls)
			}
			if _, err := client.Stream(context.Background(), request); err == nil || fixture.calls != 0 {
				t.Fatalf("stream error %v; HTTP calls %d", err, fixture.calls)
			}
		})
	}
}

func TestProtocolExtrasOwnership(t *testing.T) {
	for _, key := range []string{"model", "messages", "tools", "tool_choice", "input", "reasoning", "thinking", "stream", "max_tokens", "max_output_tokens", "max_completion_tokens", "parallel_tool_calls", "headers", "base_url", "authorization"} {
		for _, value := range []any{nil, "changed", map[string]any{"nested": true}} {
			if _, err := llm.NewChatExtraFields(map[string]any{key: value}); err == nil {
				t.Errorf("Chat accepted owned %s", key)
			}
			if _, err := llm.NewAnthropicExtraFields(map[string]any{key: value}); err == nil {
				t.Errorf("Anthropic accepted owned %s", key)
			}
		}
	}
	for _, key := range []string{"messages.0", "*.model", "tools.#", "metadata.user_id"} {
		if _, err := llm.NewAnthropicExtraFields(map[string]any{key: nil}); err == nil {
			t.Errorf("Anthropic accepted path %s", key)
		}
	}
	for _, value := range []any{math.NaN(), make(chan int), json.RawMessage(`{"x":1,"x":2}`)} {
		if _, err := llm.NewChatExtraFields(map[string]any{"extension": value}); err == nil {
			t.Errorf("accepted invalid JSON %T", value)
		}
	}
	envelope, err := llm.NewChatExtraFields(map[string]any{"extension": map[string]any{"key": "original"}})
	if err != nil {
		t.Fatal(err)
	}
	first := envelope.Fields()
	first["extension"][0] = '['
	if string(envelope.Fields()["extension"]) != `{"key":"original"}` {
		t.Fatal("envelope leaked its storage")
	}
}
