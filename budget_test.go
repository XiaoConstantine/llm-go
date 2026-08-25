package llm

import (
	"context"
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
