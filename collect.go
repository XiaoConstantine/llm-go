package llm

import (
	"errors"
	"fmt"
	"io"
)

// Collect receives and assembles all chunks from stream according to the Chunk
// contract. It always closes stream before returning. The returned Response is
// non-nil once collection begins and contains every valid chunk committed before
// a failure. A non-EOF receive, invariant, schema-validation, cancellation, or
// close failure is returned with that partial response. A close failure is
// joined after an earlier failure and never replaces it.
//
// tools declares schemas for completed tool calls. Passing nil is valid only
// when the stream emits no tool calls.
func Collect(stream Stream, tools []Tool) (response *Response, err error) {
	response = &Response{Message: Message{Role: RoleAssistant}}
	if stream == nil {
		return response, collectError("stream must not be nil")
	}
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			if err != nil {
				err = errors.Join(err, closeErr)
			} else {
				err = closeErr
			}
		}
	}()
	validator, validationErr := newToolCallValidator(tools)
	if validationErr != nil {
		return response, collectError("tools: %v", validationErr)
	}

	var providerDataSeen bool
	for chunkIndex := 0; ; chunkIndex++ {
		chunk, recvErr := stream.Recv()
		if recvErr != nil {
			if errors.Is(recvErr, io.EOF) {
				return response, nil
			}
			return response, recvErr
		}
		if schemaErr := validator.validate(completedToolCalls(chunk)); schemaErr != nil {
			return response, collectError("chunk[%d] completed tool calls: %v", chunkIndex, schemaErr)
		}
		if invariantErr := collectChunk(response, chunk, chunkIndex, &providerDataSeen); invariantErr != nil {
			return response, invariantErr
		}
	}
}

func collectChunk(response *Response, chunk Chunk, index int, providerDataSeen *bool) error {
	if chunk.ID != "" {
		if response.ID != "" && response.ID != chunk.ID {
			return collectError("chunk[%d].ID changed from %q to %q", index, response.ID, chunk.ID)
		}
		response.ID = chunk.ID
	}
	if chunk.Model != "" {
		if response.Model != "" && response.Model != chunk.Model {
			return collectError("chunk[%d].Model changed from %q to %q", index, response.Model, chunk.Model)
		}
		response.Model = chunk.Model
	}
	if chunk.FinishReason != "" {
		if !validFinishReason(chunk.FinishReason) {
			return collectError("chunk[%d].FinishReason %q is invalid", index, chunk.FinishReason)
		}
		if response.FinishReason != "" && response.FinishReason != chunk.FinishReason {
			return collectError("chunk[%d].FinishReason changed from %q to %q", index, response.FinishReason, chunk.FinishReason)
		}
		response.FinishReason = chunk.FinishReason
	}
	for partIndex, part := range chunk.Content {
		if err := validatePart(part); err != nil {
			return collectError("chunk[%d].content[%d]: %v", index, partIndex, err)
		}
		response.Message.Content = append(response.Message.Content, clonePart(part))
	}
	for callIndex, call := range chunk.ToolCalls {
		if err := validateToolCall(call); err != nil {
			return collectError("chunk[%d].tool calls[%d]: %v", index, callIndex, err)
		}
		response.Message.ToolCalls = append(response.Message.ToolCalls, cloneToolCall(call))
	}
	response.ReasoningSummary += chunk.ReasoningSummary
	if len(chunk.ProviderData) != 0 {
		if *providerDataSeen {
			return collectError("chunk[%d].ProviderData appears more than once", index)
		}
		if err := validateJSON(chunk.ProviderData); err != nil {
			return collectError("chunk[%d].ProviderData: %v", index, err)
		}
		*providerDataSeen = true
		response.Message.ProviderData = append(response.Message.ProviderData[:0], chunk.ProviderData...)
	}
	if chunk.Usage != nil {
		if err := validateUsageForCost(*chunk.Usage); err != nil {
			return collectError("chunk[%d].Usage: %v", index, err)
		}
		if response.Usage != nil && !usageIsCumulative(*response.Usage, *chunk.Usage) {
			return collectError("chunk[%d].Usage decreased from a prior cumulative value", index)
		}
		usage := *chunk.Usage
		if chunk.Usage.Cost != nil {
			cost := *chunk.Usage.Cost
			usage.Cost = &cost
		}
		response.Usage = &usage
	}
	return nil
}

func usageIsCumulative(previous, next Usage) bool {
	return next.InputTokens >= previous.InputTokens && next.OutputTokens >= previous.OutputTokens &&
		next.CacheReadTokens >= previous.CacheReadTokens && next.CacheWriteTokens >= previous.CacheWriteTokens &&
		next.CacheWrite1hTokens >= previous.CacheWrite1hTokens && next.ReasoningTokens >= previous.ReasoningTokens &&
		next.TotalTokens >= previous.TotalTokens
}

func validFinishReason(reason FinishReason) bool {
	switch reason {
	case FinishReasonStop, FinishReasonLength, FinishReasonToolCall, FinishReasonContentFilter:
		return true
	default:
		return false
	}
}

func clonePart(part Part) Part {
	part.Data = append([]byte(nil), part.Data...)
	return part
}

func cloneToolCall(call ToolCall) ToolCall {
	call.Arguments = append([]byte(nil), call.Arguments...)
	return call
}

func collectError(format string, args ...any) error {
	return &Error{Kind: KindMalformedResponse, Op: "collect", Err: fmt.Errorf(format, args...)}
}
