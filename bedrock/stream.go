package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"encoding/json/jsontext"

	llm "github.com/XiaoConstantine/llm-go"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

const maxBufferedBytes = 16 << 20

type blockKind uint8

const (
	blockText blockKind = iota + 1
	blockReasoning
	blockTool
)

type blockState struct {
	kind           blockKind
	output         int
	text           strings.Builder
	signature      strings.Builder
	id             string
	name           string
	arguments      strings.Builder
	stopped        bool
	contentIndices []int
	reasoningIndex int
	toolIndex      int
}

type decoder struct {
	op           string
	model        string
	declared     map[string]struct{}
	seenIDs      map[string]struct{}
	blocks       map[int32]*blockState
	nextOutput   int
	started      bool
	stopped      bool
	finish       llm.FinishReason
	usage        *llm.Usage
	toolCount    int
	contentCount int
	reasoning    []reasoningData
	buffered     int
}

func (c *Client) produce(ctx context.Context, op string, request llm.Request, input *bedrockruntime.ConverseStreamInput, emit internalstream.Emit) (err error) {
	defer func() { err = relabelProviderError(err, c.provider) }()
	eventStream, err := c.runtime.ConverseStream(ctx, input)
	if err != nil {
		if contextErr := contextErr(ctx); contextErr != nil {
			return contextErr
		}
		return sdkError(op, err)
	}
	if eventStream == nil {
		return malformed(op, "SDK returned no event stream")
	}
	defer func() {
		if closeErr := eventStream.Close(); closeErr != nil {
			err = errors.Join(err, modelError(llm.KindTransport, op, "close event stream: %w", closeErr))
		}
	}()
	declared := make(map[string]struct{}, len(request.Tools))
	for _, tool := range request.Tools {
		declared[tool.Name] = struct{}{}
	}
	seen := make(map[string]struct{})
	for _, message := range request.Messages {
		for _, call := range message.ToolCalls {
			seen[normalizeToolID(call.ID)] = struct{}{}
		}
	}
	decoder := &decoder{op: op, model: c.model, declared: declared, seenIDs: seen, blocks: make(map[int32]*blockState)}
	for event := range eventStream.Events() {
		if contextErr := contextErr(ctx); contextErr != nil {
			return contextErr
		}
		chunk, eventErr := decoder.consume(event)
		if eventErr != nil {
			return eventErr
		}
		if chunk != nil && !emit(*chunk) {
			return nil
		}
	}
	if contextErr := contextErr(ctx); contextErr != nil {
		return contextErr
	}
	if streamErr := eventStream.Err(); streamErr != nil {
		return sdkError(op, streamErr)
	}
	final, err := decoder.final()
	if err != nil {
		return err
	}
	if !emit(final) {
		return nil
	}
	return nil
}

func (d *decoder) consume(event types.ConverseStreamOutput) (*llm.Chunk, error) {
	switch value := event.(type) {
	case *types.ConverseStreamOutputMemberMessageStart:
		if d.started || value.Value.Role != types.ConversationRoleAssistant {
			return nil, malformed(d.op, "invalid or repeated assistant message start")
		}
		d.started = true
		return &llm.Chunk{Events: []llm.StreamEvent{{Kind: llm.StreamEventStart}}}, nil
	case *types.ConverseStreamOutputMemberContentBlockStart:
		if d.stopped {
			return nil, malformed(d.op, "content block started after message stop")
		}
		if !d.started || value.Value.ContentBlockIndex == nil {
			return nil, malformed(d.op, "content block start is missing message start or index")
		}
		index := *value.Value.ContentBlockIndex
		if index < 0 {
			return nil, malformed(d.op, "content block index %d is negative", index)
		}
		if _, exists := d.blocks[index]; exists {
			return nil, malformed(d.op, "content block %d started more than once", index)
		}
		start, ok := value.Value.Start.(*types.ContentBlockStartMemberToolUse)
		if !ok || start.Value.ToolUseId == nil || start.Value.Name == nil {
			return nil, malformed(d.op, "content block %d has unsupported start", index)
		}
		block := &blockState{kind: blockTool, output: d.nextOutput, id: *start.Value.ToolUseId, name: *start.Value.Name,
			reasoningIndex: -1, toolIndex: -1}
		d.nextOutput++
		d.blocks[index] = block
		return &llm.Chunk{Events: []llm.StreamEvent{{Kind: llm.StreamEventToolCallStart, Index: block.output, ToolCallID: block.id, ToolName: block.name}}}, nil
	case *types.ConverseStreamOutputMemberContentBlockDelta:
		if d.stopped {
			return nil, malformed(d.op, "content delta appeared after message stop")
		}
		return d.delta(value.Value)
	case *types.ConverseStreamOutputMemberContentBlockStop:
		if d.stopped {
			return nil, malformed(d.op, "content block stopped after message stop")
		}
		return d.stopBlock(value.Value)
	case *types.ConverseStreamOutputMemberMessageStop:
		if d.stopped {
			return nil, malformed(d.op, "message stop appeared more than once")
		}
		finish, err := stopReason(d.op, value.Value.StopReason)
		if err != nil {
			return nil, err
		}
		d.stopped, d.finish = true, finish
		return nil, nil
	case *types.ConverseStreamOutputMemberMetadata:
		if d.usage != nil {
			return nil, malformed(d.op, "event stream contains repeated usage metadata")
		}
		usage, err := usageFromSDK(value.Value.Usage)
		if err != nil {
			return nil, malformed(d.op, "usage: %v", err)
		}
		d.usage = usage
		return nil, nil
	default:
		return nil, malformed(d.op, "event stream contains unsupported event %T", event)
	}
}

func (d *decoder) delta(event types.ContentBlockDeltaEvent) (*llm.Chunk, error) {
	if event.ContentBlockIndex == nil || event.Delta == nil {
		return nil, malformed(d.op, "content delta is missing index or value")
	}
	index := *event.ContentBlockIndex
	block := d.blocks[index]
	chunk := &llm.Chunk{}
	switch value := event.Delta.(type) {
	case *types.ContentBlockDeltaMemberText:
		if block == nil {
			block = &blockState{kind: blockText, output: d.nextOutput, reasoningIndex: -1, toolIndex: -1}
			d.nextOutput++
			d.blocks[index] = block
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventTextStart, Index: block.output})
		}
		if block.kind != blockText || block.stopped {
			return nil, malformed(d.op, "text delta conflicts with content block %d", index)
		}
		if err := d.addBytes(len(value.Value)); err != nil {
			return nil, err
		}
		block.text.WriteString(value.Value)
		block.contentIndices = append(block.contentIndices, d.contentCount)
		d.contentCount++
		chunk.Content = []llm.Part{{Kind: llm.PartText, Text: value.Value}}
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventTextDelta, Index: block.output, Delta: value.Value})
	case *types.ContentBlockDeltaMemberToolUse:
		if block == nil || block.kind != blockTool || block.stopped || value.Value.Input == nil {
			return nil, malformed(d.op, "tool delta conflicts with content block %d", index)
		}
		if err := d.addBytes(len(*value.Value.Input)); err != nil {
			return nil, err
		}
		block.arguments.WriteString(*value.Value.Input)
		chunk.Events = []llm.StreamEvent{{Kind: llm.StreamEventToolCallDelta, Index: block.output,
			Delta: *value.Value.Input, ToolCallID: block.id, ToolName: block.name}}
	case *types.ContentBlockDeltaMemberReasoningContent:
		if block == nil {
			block = &blockState{kind: blockReasoning, output: d.nextOutput, reasoningIndex: -1, toolIndex: -1}
			d.nextOutput++
			d.blocks[index] = block
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventReasoningStart, Index: block.output})
		}
		if block.kind != blockReasoning || block.stopped {
			return nil, malformed(d.op, "reasoning delta conflicts with content block %d", index)
		}
		switch reasoning := value.Value.(type) {
		case *types.ReasoningContentBlockDeltaMemberText:
			if err := d.addBytes(len(reasoning.Value)); err != nil {
				return nil, err
			}
			block.text.WriteString(reasoning.Value)
			chunk.ReasoningSummary = reasoning.Value
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventReasoningDelta, Index: block.output, Delta: reasoning.Value})
		case *types.ReasoningContentBlockDeltaMemberSignature:
			block.signature.WriteString(reasoning.Value)
		default:
			return nil, malformed(d.op, "unsupported reasoning delta %T", value.Value)
		}
	default:
		return nil, malformed(d.op, "unsupported content delta %T", event.Delta)
	}
	return chunk, nil
}

func (d *decoder) stopBlock(event types.ContentBlockStopEvent) (*llm.Chunk, error) {
	if event.ContentBlockIndex == nil {
		return nil, malformed(d.op, "content block stop has no index")
	}
	block := d.blocks[*event.ContentBlockIndex]
	if block == nil || block.stopped {
		return nil, malformed(d.op, "content block %d stopped without one active block", *event.ContentBlockIndex)
	}
	block.stopped = true
	chunk := &llm.Chunk{}
	switch block.kind {
	case blockText:
		chunk.Events = []llm.StreamEvent{{Kind: llm.StreamEventTextEnd, Index: block.output, Content: block.text.String()}}
	case blockReasoning:
		block.reasoningIndex = len(d.reasoning)
		d.reasoning = append(d.reasoning, reasoningData{Index: block.output, Text: block.text.String(), Signature: block.signature.String()})
		chunk.Events = []llm.StreamEvent{{Kind: llm.StreamEventReasoningEnd, Index: block.output, Content: block.text.String()}}
	case blockTool:
		arguments := json.RawMessage(block.arguments.String())
		if len(arguments) == 0 || !jsontext.Value(arguments).IsValid() {
			return nil, malformed(d.op, "tool call %d arguments are incomplete JSON", block.output)
		}
		if block.id == "" || block.name == "" {
			return nil, malformed(d.op, "tool call %d has incomplete identity", block.output)
		}
		if _, ok := d.declared[block.name]; !ok {
			return nil, malformed(d.op, "tool call %d names undeclared tool %q", block.output, block.name)
		}
		if _, duplicate := d.seenIDs[block.id]; duplicate {
			return nil, malformed(d.op, "tool call %d repeats ID %q", block.output, block.id)
		}
		d.seenIDs[block.id] = struct{}{}
		block.toolIndex = d.toolCount
		d.toolCount++
		call := llm.ToolCall{ID: block.id, Name: block.name, Arguments: append(json.RawMessage(nil), arguments...)}
		chunk.ToolCalls = []llm.ToolCall{call}
		chunk.Events = []llm.StreamEvent{{Kind: llm.StreamEventToolCallEnd, Index: block.output,
			ToolCallID: block.id, ToolName: block.name, ToolCall: &call}}
	}
	return chunk, nil
}

func (d *decoder) addBytes(count int) error {
	if count > maxBufferedBytes-d.buffered {
		return malformed(d.op, "buffered response exceeds %d bytes", maxBufferedBytes)
	}
	d.buffered += count
	return nil
}

func (d *decoder) final() (llm.Chunk, error) {
	if !d.started || !d.stopped {
		return llm.Chunk{}, malformed(d.op, "event stream ended before message start and stop")
	}
	for index, block := range d.blocks {
		if !block.stopped {
			return llm.Chunk{}, malformed(d.op, "event stream ended before content block %d stopped", index)
		}
	}
	if d.finish == llm.FinishReasonToolCall && d.toolCount == 0 || d.finish != llm.FinishReasonToolCall && d.toolCount != 0 {
		return llm.Chunk{}, malformed(d.op, "finish reason is inconsistent with streamed tool calls")
	}
	blocks := make([]*blockState, 0, len(d.blocks))
	for _, block := range d.blocks {
		blocks = append(blocks, block)
	}
	slices.SortFunc(blocks, func(left, right *blockState) int { return left.output - right.output })
	var order []orderData
	for _, block := range blocks {
		switch block.kind {
		case blockText:
			for _, index := range block.contentIndices {
				order = append(order, orderData{Kind: "content", Index: index})
			}
		case blockReasoning:
			order = append(order, orderData{Kind: "reasoning", Index: block.reasoningIndex})
		case blockTool:
			order = append(order, orderData{Kind: "tool_call", Index: block.toolIndex})
		}
	}
	providerData, err := marshalMessageData(d.model, d.reasoning, order)
	if err != nil {
		return llm.Chunk{}, malformed(d.op, "encode reasoning replay: %v", err)
	}
	return llm.Chunk{Model: d.model, FinishReason: d.finish, Usage: d.usage, ProviderData: providerData,
		Events: []llm.StreamEvent{{Kind: llm.StreamEventDone, FinishReason: d.finish}}}, nil
}

func stopReason(op string, reason types.StopReason) (llm.FinishReason, error) {
	switch reason {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return llm.FinishReasonStop, nil
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return llm.FinishReasonLength, nil
	case types.StopReasonToolUse:
		return llm.FinishReasonToolCall, nil
	default:
		return "", &llm.Error{Kind: llm.KindProvider, Op: op, Provider: defaultProvider, Err: fmt.Errorf("bedrock stopped with reason %q", reason)}
	}
}

func usageFromSDK(usage *types.TokenUsage) (*llm.Usage, error) {
	if usage == nil {
		return nil, nil
	}
	values := []*int32{usage.InputTokens, usage.OutputTokens, usage.CacheReadInputTokens, usage.CacheWriteInputTokens, usage.TotalTokens}
	for _, value := range values {
		if value != nil && *value < 0 {
			return nil, errors.New("token count is negative")
		}
	}
	result := &llm.Usage{InputTokens: intValue(usage.InputTokens), OutputTokens: intValue(usage.OutputTokens),
		CacheReadTokens: intValue(usage.CacheReadInputTokens), CacheWriteTokens: intValue(usage.CacheWriteInputTokens), TotalTokens: intValue(usage.TotalTokens)}
	if result.TotalTokens == 0 {
		result.TotalTokens = result.InputTokens + result.OutputTokens + result.CacheReadTokens + result.CacheWriteTokens
	}
	if result.TotalTokens < result.InputTokens+result.OutputTokens+result.CacheReadTokens+result.CacheWriteTokens {
		return nil, errors.New("total token count is smaller than its components")
	}
	return result, nil
}

func intValue(value *int32) int {
	if value == nil {
		return 0
	}
	return int(*value)
}

func sdkError(op string, err error) error {
	kind := llm.KindTransport
	status := 0
	retryAfter := time.Duration(0)
	var responseErr *smithyhttp.ResponseError
	if errors.As(err, &responseErr) {
		status = responseErr.HTTPStatusCode()
		if responseErr.Response != nil && responseErr.Response.Response != nil {
			retryAfter = parseRetryAfter(responseErr.Response.Header.Get("Retry-After"))
		}
	}
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) {
		kind = llm.KindProvider
		code, detail := strings.ToLower(apiErr.ErrorCode()), strings.ToLower(apiErr.ErrorMessage())
		switch {
		case strings.Contains(code, "unrecognizedclient") || strings.Contains(code, "invalidsignature") ||
			strings.Contains(code, "incompletesignature") || strings.Contains(code, "expiredtoken") ||
			strings.Contains(code, "invalidclienttoken") || status == 401:
			kind = llm.KindAuthentication
		case strings.Contains(code, "accessdenied") || status == 403:
			kind = llm.KindPermission
		case strings.Contains(code, "throttl") || status == 429:
			kind = llm.KindRateLimit
		case strings.Contains(code, "validation"):
			kind = llm.KindInvalidRequest
		}
		if strings.Contains(detail, "context") && (strings.Contains(detail, "length") || strings.Contains(detail, "token")) {
			kind = llm.KindContextLimit
		}
	}
	return &llm.Error{Kind: kind, Op: op, Provider: defaultProvider, HTTPStatus: status, RetryAfter: retryAfter, Err: err}
}

func parseRetryAfter(value string) time.Duration {
	if seconds, err := strconv.ParseUint(strings.TrimSpace(value), 10, 31); err == nil {
		return time.Duration(seconds) * time.Second
	}
	if when, err := http.ParseTime(value); err == nil {
		if delay := time.Until(when); delay > 0 {
			return delay
		}
	}
	return 0
}
