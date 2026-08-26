package llm

import (
	"context"
	"fmt"
)

// TokenEstimator estimates input tokens for a request. Implementations must
// treat Request as read-only, be safe for the caller's intended concurrency,
// and honor ctx. Estimates are explicitly approximate unless the implementation
// uses the target model's exact tokenizer.
type TokenEstimator interface {
	EstimateInputTokens(context.Context, Request) (int, error)
}

// TokenEstimatorFunc adapts a function to TokenEstimator.
type TokenEstimatorFunc func(context.Context, Request) (int, error)

func (f TokenEstimatorFunc) EstimateInputTokens(ctx context.Context, request Request) (int, error) {
	return f(ctx, request)
}

// TokenBudget reports how BudgetRequest derived the output limit.
type TokenBudget struct {
	EstimatedInputTokens  int
	ReserveTokens         int
	AvailableOutputTokens int
	MaxOutputTokens       int
	Clamped               bool
}

// BudgetRequest returns an owned Request copy whose MaxOutputTokens fits the
// known context window, known model output limit, estimated input, and reserve.
// A zero requested output limit selects the largest computed limit. A zero model
// MaxOutputTokens means that limit is unknown, so only the context window bounds
// output. A zero ContextWindow is rejected because no safe model-aware budget
// can be computed. Low-level generators never call this helper implicitly.
func BudgetRequest(ctx context.Context, info ModelInfo, request Request, estimator TokenEstimator, reserve int) (Request, TokenBudget, error) {
	if err := request.Validate(); err != nil {
		return Request{}, TokenBudget{}, err
	}
	if err := contextError(ctx); err != nil {
		return Request{}, TokenBudget{}, err
	}
	if info.ContextWindow <= 0 {
		return Request{}, TokenBudget{}, fmt.Errorf("model context window is unknown")
	}
	if info.MaxOutputTokens < 0 || info.MaxOutputTokens > info.ContextWindow {
		return Request{}, TokenBudget{}, fmt.Errorf("model max output tokens are invalid")
	}
	if reserve < 0 {
		return Request{}, TokenBudget{}, fmt.Errorf("token reserve must not be negative")
	}
	if estimator == nil {
		return Request{}, TokenBudget{}, fmt.Errorf("token estimator must not be nil")
	}
	if function, ok := estimator.(TokenEstimatorFunc); ok && function == nil {
		return Request{}, TokenBudget{}, fmt.Errorf("token estimator function must not be nil")
	}
	estimated, err := estimator.EstimateInputTokens(ctx, request)
	if err != nil {
		return Request{}, TokenBudget{}, fmt.Errorf("estimate input tokens: %w", err)
	}
	if err := contextError(ctx); err != nil {
		return Request{}, TokenBudget{}, err
	}
	if estimated < 0 {
		return Request{}, TokenBudget{}, fmt.Errorf("estimated input tokens must not be negative")
	}
	if estimated > info.ContextWindow || reserve > info.ContextWindow-estimated {
		return Request{}, TokenBudget{}, fmt.Errorf("estimated input tokens and reserve do not leave output capacity")
	}
	available := info.ContextWindow - estimated - reserve
	limit := available
	if info.MaxOutputTokens != 0 && info.MaxOutputTokens < limit {
		limit = info.MaxOutputTokens
	}
	if limit <= 0 {
		return Request{}, TokenBudget{}, fmt.Errorf("token budget does not leave output capacity")
	}
	requested := request.MaxOutputTokens
	selected := requested
	clamped := false
	if selected == 0 {
		selected = limit
	} else if selected > limit {
		selected = limit
		clamped = true
	}
	result := cloneRequest(request)
	result.MaxOutputTokens = selected
	return result, TokenBudget{EstimatedInputTokens: estimated, ReserveTokens: reserve,
		AvailableOutputTokens: available, MaxOutputTokens: selected, Clamped: clamped}, nil
}

func cloneRequest(request Request) Request {
	clone := request
	clone.Messages = make([]Message, len(request.Messages))
	for index, message := range request.Messages {
		converted := Message{Role: message.Role, ProviderData: append([]byte(nil), message.ProviderData...),
			Content: make([]Part, len(message.Content))}
		for partIndex, part := range message.Content {
			converted.Content[partIndex] = clonePart(part)
		}
		converted.ToolCalls = make([]ToolCall, len(message.ToolCalls))
		for callIndex, call := range message.ToolCalls {
			converted.ToolCalls[callIndex] = cloneToolCall(call)
		}
		converted.ToolResults = make([]ToolResult, len(message.ToolResults))
		for resultIndex, result := range message.ToolResults {
			convertedResult := ToolResult{CallID: result.CallID, Name: result.Name, IsError: result.IsError, Content: make([]Part, len(result.Content))}
			for partIndex, part := range result.Content {
				convertedResult.Content[partIndex] = clonePart(part)
			}
			converted.ToolResults[resultIndex] = convertedResult
		}
		clone.Messages[index] = converted
	}
	clone.Tools = make([]Tool, len(request.Tools))
	for index, tool := range request.Tools {
		clone.Tools[index] = tool
		clone.Tools[index].InputSchema = append([]byte(nil), tool.InputSchema...)
	}
	clone.Stop = append([]string(nil), request.Stop...)
	if request.Temperature != nil {
		value := *request.Temperature
		clone.Temperature = &value
	}
	if request.TopP != nil {
		value := *request.TopP
		clone.TopP = &value
	}
	if request.PresencePenalty != nil {
		value := *request.PresencePenalty
		clone.PresencePenalty = &value
	}
	if request.FrequencyPenalty != nil {
		value := *request.FrequencyPenalty
		clone.FrequencyPenalty = &value
	}
	return clone
}
