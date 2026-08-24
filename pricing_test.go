package llm

import (
	"math"
	"strings"
	"testing"
)

func TestCalculateCostUsesDisjointTokenCategoriesAndTiers(t *testing.T) {
	cost := ModelCost{
		Input: 2, Output: 8, CacheRead: 0.2, CacheWrite: 2.5,
		Tiers: []ModelCostTier{
			{InputTokensAbove: 10_000, Input: 4, Output: 12, CacheRead: 0.4, CacheWrite: 5},
			{InputTokensAbove: 1_000, Input: 3, Output: 10, CacheRead: 0.3, CacheWrite: 4},
		},
	}
	usage := Usage{
		InputTokens: 800, OutputTokens: 200, CacheReadTokens: 300,
		CacheWriteTokens: 100, CacheWrite1hTokens: 40, ReasoningTokens: 50,
		TotalTokens: 1_400,
	}
	got, err := CalculateCost(cost, usage)
	if err != nil {
		t.Fatalf("CalculateCost() error = %v", err)
	}
	// Total input usage is 1,200, so the 1,000-token tier prices the full request.
	want := UsageCost{
		Input:      0.0024,
		Output:     0.002,
		CacheRead:  0.00009,
		CacheWrite: 0.00048, // 60 at $4/M + 40 one-hour writes at 2*$3/M.
		Total:      0.00497,
	}
	assertUsageCost(t, got, want)
}

func TestCalculateCostUsesStrictHighestTierThreshold(t *testing.T) {
	cost := ModelCost{Input: 1, Tiers: []ModelCostTier{
		{InputTokensAbove: 100, Input: 2},
		{InputTokensAbove: 200, Input: 3},
	}}
	for _, test := range []struct {
		input int
		want  float64
	}{
		{input: 100, want: 0.0001},
		{input: 101, want: 0.000202},
		{input: 201, want: 0.000603},
	} {
		usage := Usage{InputTokens: test.input, TotalTokens: test.input}
		got, err := CalculateCost(cost, usage)
		if err != nil {
			t.Fatalf("CalculateCost(%d) error = %v", test.input, err)
		}
		if math.Abs(got.Total-test.want) > 1e-12 {
			t.Fatalf("CalculateCost(%d).Total = %.12f, want %.12f", test.input, got.Total, test.want)
		}
	}
}

func TestCalculateCostAvoidsRepresentableIntermediateOverflow(t *testing.T) {
	ordinary, err := CalculateCost(ModelCost{Input: math.MaxFloat64}, Usage{InputTokens: 2, TotalTokens: 2})
	if err != nil || math.IsInf(ordinary.Total, 0) || ordinary.Total == 0 {
		t.Fatalf("ordinary CalculateCost() = (%#v, %v)", ordinary, err)
	}
	oneHour, err := CalculateCost(ModelCost{Input: math.MaxFloat64}, Usage{CacheWriteTokens: 2, CacheWrite1hTokens: 2, TotalTokens: 2})
	if err != nil || math.IsInf(oneHour.Total, 0) || oneHour.Total == 0 {
		t.Fatalf("one-hour CalculateCost() = (%#v, %v)", oneHour, err)
	}
}

func TestPricingValidation(t *testing.T) {
	validUsage := Usage{InputTokens: 1, TotalTokens: 1}
	tests := []struct {
		name  string
		cost  ModelCost
		usage Usage
		want  string
	}{
		{name: "negative rate", cost: ModelCost{Input: -1}, usage: validUsage, want: "finite and nonnegative"},
		{name: "NaN rate", cost: ModelCost{Output: math.NaN()}, usage: validUsage, want: "finite and nonnegative"},
		{name: "negative threshold", cost: ModelCost{Tiers: []ModelCostTier{{InputTokensAbove: -1}}}, usage: validUsage, want: "threshold"},
		{name: "duplicate threshold", cost: ModelCost{Tiers: []ModelCostTier{{InputTokensAbove: 1}, {InputTokensAbove: 1}}}, usage: validUsage, want: "more than once"},
		{name: "negative usage", usage: Usage{InputTokens: -1}, want: "must not be negative"},
		{name: "cache split", usage: Usage{CacheWriteTokens: 1, CacheWrite1hTokens: 2, TotalTokens: 1}, want: "must not exceed"},
		{name: "reasoning subset", usage: Usage{OutputTokens: 1, ReasoningTokens: 2, TotalTokens: 1}, want: "must not exceed"},
		{name: "inconsistent total", usage: Usage{InputTokens: 1, TotalTokens: 2}, want: "do not equal"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := CalculateCost(test.cost, test.usage)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("CalculateCost() error = %v, want %q", err, test.want)
			}
		})
	}
}

func assertUsageCost(t *testing.T, got, want UsageCost) {
	t.Helper()
	gotValues := [...]float64{got.Input, got.Output, got.CacheRead, got.CacheWrite, got.Total}
	wantValues := [...]float64{want.Input, want.Output, want.CacheRead, want.CacheWrite, want.Total}
	for index := range gotValues {
		if math.Abs(gotValues[index]-wantValues[index]) > 1e-12 {
			t.Fatalf("UsageCost = %#v, want %#v", got, want)
		}
	}
}
