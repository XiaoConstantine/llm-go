package llm

import (
	"fmt"
	"math"
)

const tokensPerMillion = 1_000_000

// Validate reports whether the pricing table is finite, nonnegative, and has
// valid, unique tier thresholds.
func (c ModelCost) Validate() error {
	if err := validateCostRates(c.Input, c.Output, c.CacheRead, c.CacheWrite); err != nil {
		return err
	}
	thresholds := make(map[int]struct{}, len(c.Tiers))
	for index, tier := range c.Tiers {
		if tier.InputTokensAbove < 0 {
			return fmt.Errorf("cost tier %d input threshold must not be negative", index)
		}
		if _, exists := thresholds[tier.InputTokensAbove]; exists {
			return fmt.Errorf("cost tier input threshold %d is configured more than once", tier.InputTokensAbove)
		}
		thresholds[tier.InputTokensAbove] = struct{}{}
		if err := validateCostRates(tier.Input, tier.Output, tier.CacheRead, tier.CacheWrite); err != nil {
			return fmt.Errorf("cost tier %d: %w", index, err)
		}
	}
	return nil
}

func validateCostRates(rates ...float64) error {
	names := [...]string{"input", "output", "cache-read", "cache-write"}
	for index, rate := range rates {
		if math.IsNaN(rate) || math.IsInf(rate, 0) || rate < 0 {
			return fmt.Errorf("%s cost rate must be finite and nonnegative", names[index])
		}
	}
	return nil
}

// CalculateCost calculates request cost in US dollars. CacheWrite1hTokens are
// charged at twice the selected input rate, matching Anthropic's one-hour cache
// retention pricing; remaining cache writes use the selected cache-write rate.
func CalculateCost(cost ModelCost, usage Usage) (UsageCost, error) {
	if err := cost.Validate(); err != nil {
		return UsageCost{}, err
	}
	if err := validateUsageForCost(usage); err != nil {
		return UsageCost{}, err
	}

	rates := cost
	matchedThreshold := -1
	inputUsage := usage.InputTokens + usage.CacheReadTokens + usage.CacheWriteTokens
	for _, tier := range cost.Tiers {
		if inputUsage > tier.InputTokensAbove && tier.InputTokensAbove > matchedThreshold {
			rates.Input = tier.Input
			rates.Output = tier.Output
			rates.CacheRead = tier.CacheRead
			rates.CacheWrite = tier.CacheWrite
			matchedThreshold = tier.InputTokensAbove
		}
	}
	shortWrite := usage.CacheWriteTokens - usage.CacheWrite1hTokens
	calculated := UsageCost{
		Input:      tokenCost(rates.Input, usage.InputTokens),
		Output:     tokenCost(rates.Output, usage.OutputTokens),
		CacheRead:  tokenCost(rates.CacheRead, usage.CacheReadTokens),
		CacheWrite: tokenCost(rates.CacheWrite, shortWrite) + 2*tokenCost(rates.Input, usage.CacheWrite1hTokens),
	}
	calculated.Total = calculated.Input + calculated.Output + calculated.CacheRead + calculated.CacheWrite
	for _, value := range [...]float64{calculated.Input, calculated.Output, calculated.CacheRead, calculated.CacheWrite, calculated.Total} {
		if math.IsInf(value, 0) || math.IsNaN(value) {
			return UsageCost{}, fmt.Errorf("calculated cost is not finite")
		}
	}
	return calculated, nil
}

func tokenCost(rate float64, tokens int) float64 {
	return rate * (float64(tokens) / tokensPerMillion)
}

func validateUsageForCost(usage Usage) error {
	values := [...]struct {
		name  string
		value int
	}{
		{name: "input tokens", value: usage.InputTokens},
		{name: "output tokens", value: usage.OutputTokens},
		{name: "cache-read tokens", value: usage.CacheReadTokens},
		{name: "cache-write tokens", value: usage.CacheWriteTokens},
		{name: "one-hour cache-write tokens", value: usage.CacheWrite1hTokens},
		{name: "reasoning tokens", value: usage.ReasoningTokens},
		{name: "total tokens", value: usage.TotalTokens},
	}
	for _, field := range values {
		if field.value < 0 {
			return fmt.Errorf("%s must not be negative", field.name)
		}
	}
	if usage.CacheWrite1hTokens > usage.CacheWriteTokens {
		return fmt.Errorf("one-hour cache-write tokens must not exceed cache-write tokens")
	}
	if usage.ReasoningTokens > usage.OutputTokens {
		return fmt.Errorf("reasoning tokens must not exceed output tokens")
	}
	input, ok := addTokenCounts(usage.InputTokens, usage.CacheReadTokens, usage.CacheWriteTokens)
	if !ok {
		return fmt.Errorf("input token count overflows int")
	}
	total, ok := addTokenCounts(input, usage.OutputTokens)
	if !ok {
		return fmt.Errorf("total token count overflows int")
	}
	if usage.TotalTokens != total {
		return fmt.Errorf("total tokens %d do not equal token categories %d", usage.TotalTokens, total)
	}
	return nil
}

func addTokenCounts(values ...int) (int, bool) {
	total := 0
	for _, value := range values {
		if value > int(^uint(0)>>1)-total {
			return 0, false
		}
		total += value
	}
	return total, true
}

func cloneModelCost(cost *ModelCost) *ModelCost {
	if cost == nil {
		return nil
	}
	clone := *cost
	clone.Tiers = append([]ModelCostTier(nil), cost.Tiers...)
	return &clone
}
