package llm

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
)

func TestBudgetRequestBoundariesAndOwnership(t *testing.T) {
	request := validGenerationRequest()
	request.MaxOutputTokens = 500
	request.Messages[0].Content[0].Text = "owned"
	estimator := TokenEstimatorFunc(func(context.Context, Request) (int, error) { return 700, nil })
	budgeted, budget, err := BudgetRequest(context.Background(), ModelInfo{ContextWindow: 1000, MaxOutputTokens: 400}, request, estimator, 100)
	if err != nil {
		t.Fatal(err)
	}
	if budgeted.MaxOutputTokens != 200 || !budget.Clamped || budget.AvailableOutputTokens != 200 {
		t.Fatalf("budget = %#v, request = %#v", budget, budgeted)
	}
	budgeted.Messages[0].Content[0].Text = "changed"
	if request.Messages[0].Content[0].Text != "owned" {
		t.Fatal("BudgetRequest aliased input")
	}
	unknownLimit, _, err := BudgetRequest(context.Background(), ModelInfo{ContextWindow: 1000}, Request{Messages: request.Messages}, estimator, 100)
	if err != nil || unknownLimit.MaxOutputTokens != 200 {
		t.Fatalf("unknown output limit = %#v, %v", unknownLimit, err)
	}
	var nilEstimator TokenEstimatorFunc
	if _, _, err := BudgetRequest(context.Background(), ModelInfo{ContextWindow: 100}, validGenerationRequest(), nilEstimator, 0); err == nil {
		t.Fatal("typed-nil estimator succeeded")
	}
	maxInt := int(^uint(0) >> 1)
	for _, test := range []struct {
		name     string
		info     ModelInfo
		reserve  int
		estimate int
	}{
		{"unknown", ModelInfo{}, 0, 0}, {"negative reserve", ModelInfo{ContextWindow: 10}, -1, 0},
		{"negative estimate", ModelInfo{ContextWindow: 10}, 0, -1}, {"impossible", ModelInfo{ContextWindow: 10}, 5, 6},
		{"invalid model", ModelInfo{ContextWindow: 10, MaxOutputTokens: 11}, 0, 1},
		{"overflow safe", ModelInfo{ContextWindow: maxInt}, 2, maxInt - 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, _, err := BudgetRequest(context.Background(), test.info, validGenerationRequest(), TokenEstimatorFunc(func(context.Context, Request) (int, error) { return test.estimate, nil }), test.reserve)
			if err == nil {
				t.Fatal("BudgetRequest succeeded")
			}
		})
	}
}

func TestBudgetRequestOwnsSamplingAndProtocolOptions(t *testing.T) {
	type mutation struct {
		name   string
		change func(*Request)
	}
	for _, protocol := range []struct {
		name      string
		request   func() Request
		mutations []mutation
	}{
		{
			name: "sampling",
			request: func() Request {
				r := validGenerationRequest()
				r.TopK, r.ParallelToolCalls = new(0), new(false)
				return r
			},
			mutations: []mutation{
				{"top-k", func(r *Request) { *r.TopK = 9 }},
				{"parallel-tools", func(r *Request) { *r.ParallelToolCalls = true }},
			},
		},
		{
			name: "chat",
			request: func() Request {
				r := validGenerationRequest()
				r.OpenAIChat = &OpenAIChatOptions{
					LogitBias: map[string]int64{"42": 1}, LogProbs: new(true), TopLogProbs: new(1),
					User: new("original"), Verbosity: new("low"), Prediction: &ChatPrediction{Content: "original"},
					Store: new(false), Metadata: map[string]string{"task": "original"},
					SafetyIdentifier: new("original"), ServiceTier: new("default"),
					ExtraFields: ChatExtraFields{fields: map[string]json.RawMessage{"custom": json.RawMessage(`{"nested":[{"value":"original"}]}`)}},
				}
				return r
			},
			mutations: []mutation{
				{"options-container", func(r *Request) { r.OpenAIChat.User = new("changed") }},
				{"logit-bias-map", func(r *Request) { r.OpenAIChat.LogitBias["42"] = 2 }},
				{"log-probs", func(r *Request) { *r.OpenAIChat.LogProbs = false }},
				{"top-log-probs", func(r *Request) { *r.OpenAIChat.TopLogProbs = 2 }},
				{"user", func(r *Request) { *r.OpenAIChat.User = "changed" }},
				{"verbosity", func(r *Request) { *r.OpenAIChat.Verbosity = "high" }},
				{"prediction", func(r *Request) { r.OpenAIChat.Prediction.Content = "changed" }},
				{"store", func(r *Request) { *r.OpenAIChat.Store = true }},
				{"metadata-map", func(r *Request) { r.OpenAIChat.Metadata["task"] = "changed" }},
				{"safety-identifier", func(r *Request) { *r.OpenAIChat.SafetyIdentifier = "changed" }},
				{"service-tier", func(r *Request) { *r.OpenAIChat.ServiceTier = "priority" }},
				{"extra-fields-map", func(r *Request) { delete(r.OpenAIChat.ExtraFields.fields, "custom") }},
				{"extra-fields-nested-json", func(r *Request) { r.OpenAIChat.ExtraFields.fields["custom"][22] = 'X' }},
			},
		},
		{
			name: "responses",
			request: func() Request {
				r := validGenerationRequest()
				r.OpenAIResponses = &OpenAIResponsesOptions{Verbosity: new("low"), ServiceTier: new("default")}
				return r
			},
			mutations: []mutation{
				{"options-container", func(r *Request) { r.OpenAIResponses.Verbosity = new("high") }},
				{"verbosity", func(r *Request) { *r.OpenAIResponses.Verbosity = "high" }},
				{"service-tier", func(r *Request) { *r.OpenAIResponses.ServiceTier = "priority" }},
			},
		},
		{
			name: "anthropic",
			request: func() Request {
				r := validGenerationRequest()
				r.ReasoningBudgetTokens = 1024
				r.Anthropic = &AnthropicOptions{
					ThinkingDisplay: new("summarized"),
					ExtraFields:     AnthropicExtraFields{fields: map[string]json.RawMessage{"custom": json.RawMessage(`{"nested":[{"value":"original"}]}`)}},
				}
				return r
			},
			mutations: []mutation{
				{"options-container", func(r *Request) { r.Anthropic.ThinkingDisplay = new("omitted") }},
				{"thinking-display", func(r *Request) { *r.Anthropic.ThinkingDisplay = "omitted" }},
				{"extra-fields-map", func(r *Request) { delete(r.Anthropic.ExtraFields.fields, "custom") }},
				{"extra-fields-nested-json", func(r *Request) { r.Anthropic.ExtraFields.fields["custom"][22] = 'X' }},
			},
		},
	} {
		for _, change := range protocol.mutations {
			t.Run(protocol.name+"/"+change.name, func(t *testing.T) {
				original, expected := protocol.request(), protocol.request()
				copy, _, err := BudgetRequest(t.Context(), ModelInfo{ContextWindow: 4096}, original,
					TokenEstimatorFunc(func(context.Context, Request) (int, error) { return 1, nil }), 0)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(copy.OpenAIChat, original.OpenAIChat) || !reflect.DeepEqual(copy.OpenAIResponses, original.OpenAIResponses) || !reflect.DeepEqual(copy.Anthropic, original.Anthropic) || !reflect.DeepEqual(copy.TopK, original.TopK) || !reflect.DeepEqual(copy.ParallelToolCalls, original.ParallelToolCalls) {
					t.Fatal("budgeting changed sampling or protocol settings")
				}
				change.change(&copy)
				if !reflect.DeepEqual(original, expected) {
					t.Fatal("mutating the budgeted request changed its input")
				}
			})
		}
	}
}

func TestBudgetRequestPreservesAbsentProtocolOptions(t *testing.T) {
	for _, request := range []Request{
		validGenerationRequest(),
		{Messages: validGenerationRequest().Messages, OpenAIChat: &OpenAIChatOptions{}},
		{Messages: validGenerationRequest().Messages, OpenAIResponses: &OpenAIResponsesOptions{}},
		{Messages: validGenerationRequest().Messages, Anthropic: &AnthropicOptions{}},
	} {
		copy, _, err := BudgetRequest(t.Context(), ModelInfo{ContextWindow: 100}, request,
			TokenEstimatorFunc(func(context.Context, Request) (int, error) { return 1, nil }), 0)
		if err != nil {
			t.Fatal(err)
		}
		if copy.TopK != nil || copy.ParallelToolCalls != nil || !reflect.DeepEqual(copy.OpenAIChat, request.OpenAIChat) || !reflect.DeepEqual(copy.OpenAIResponses, request.OpenAIResponses) || !reflect.DeepEqual(copy.Anthropic, request.Anthropic) {
			t.Fatal("budgeting invented an option or changed nil storage")
		}
	}
}
