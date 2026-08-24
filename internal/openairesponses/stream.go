package openairesponses

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	"github.com/openai/openai-go/v3/packages/ssestream"
	openairesponses "github.com/openai/openai-go/v3/responses"
)

// APIError is an error reported by an OpenAI Responses endpoint.
type APIError struct {
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

func (e *APIError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Message != "" {
		return e.Message
	}
	if e.Code != "" {
		return e.Code
	}
	if e.Type != "" {
		return e.Type
	}
	return "OpenAI provider error"
}

// Stream is the OpenAI SDK stream consumed by Codec.
type Stream = ssestream.Stream[openairesponses.ResponseStreamEventUnion]

// OpenStream starts one Responses API event stream.
type OpenStream func(context.Context, string, openairesponses.ResponseNewParams) (*Stream, error)

// Produce decodes one Responses API event stream into neutral chunks.
func (c Codec) Produce(ctx context.Context, op, model string, params openairesponses.ResponseNewParams, open OpenStream, emit internalstream.Emit) error {
	stream, err := open(ctx, op, params)
	if err != nil {
		return err
	}
	state := streamState{
		codec:         c,
		op:            op,
		defaultModel:  model,
		declaredTools: declaredToolNames(params),
		outputItems:   make(map[int]jsontext.Value),
		pendingItems:  make(map[int]outputItemIdentity),
		toolCalls:     make(map[int]llm.ToolCall),
	}

	for stream.Next() {
		if err := ContextError(ctx); err != nil {
			return c.closeStream(op, stream, err)
		}
		if err := state.consume(stream.Current(), emit); err != nil {
			return c.closeStream(op, stream, err)
		}
		if state.stopped {
			return c.closeStream(op, stream, nil)
		}
		if state.terminal {
			return c.closeStream(op, stream, nil)
		}
	}
	err = stream.Err()
	if contextErr := ContextError(ctx); contextErr != nil {
		err = contextErr
	} else if err != nil {
		err = c.streamReadError(op, err)
	} else if !state.terminal {
		err = c.malformedResponse(op, "event stream ended before a terminal response")
	}
	return c.closeStream(op, stream, err)
}

func (c Codec) streamReadError(op string, err error) error {
	var providerEvent *ssestream.StreamError
	if errors.As(err, &providerEvent) {
		return c.eventError(op, providerEvent.Event.Data)
	}
	var syntaxErr *json.SyntaxError
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &syntaxErr) || errors.As(err, &typeErr) || errors.Is(err, bufio.ErrTooLong) {
		return c.malformedResponse(op, "decode event stream: %v", err)
	}
	return c.transportError(op, fmt.Errorf("read event stream: %w", err))
}

func (c Codec) closeStream(op string, stream *Stream, err error) error {
	if closeErr := stream.Close(); closeErr != nil {
		closeErr = c.transportError(op, fmt.Errorf("close event stream: %w", closeErr))
		if err != nil {
			return errors.Join(err, closeErr)
		}
		return closeErr
	}
	return err
}

func (e *APIError) empty() bool {
	return e == nil || e.Type == "" && e.Code == "" && e.Message == ""
}

func classifyAPIError(apiErr *APIError) llm.ErrorKind {
	if apiErr == nil {
		return llm.KindProvider
	}
	if isContextLimitError(apiErr) {
		return llm.KindContextLimit
	}
	detail := strings.ToLower(apiErr.Type + " " + apiErr.Code)
	switch {
	case strings.Contains(detail, "auth"), strings.Contains(detail, "invalid_api_key"):
		return llm.KindAuthentication
	case strings.Contains(detail, "permission"), strings.Contains(detail, "forbidden"):
		return llm.KindPermission
	case strings.Contains(detail, "rate_limit"), strings.Contains(detail, "usage_limit"), strings.Contains(detail, "quota"):
		return llm.KindRateLimit
	case strings.Contains(detail, "invalid_request"):
		return llm.KindInvalidRequest
	default:
		return llm.KindProvider
	}
}

func isContextLimitError(apiErr *APIError) bool {
	if apiErr == nil {
		return false
	}
	detail := strings.ToLower(apiErr.Type + " " + apiErr.Code + " " + apiErr.Message)
	return strings.Contains(detail, "context_length_exceeded") ||
		strings.Contains(detail, "maximum context length") ||
		strings.Contains(detail, "context window")
}

func declaredToolNames(params openairesponses.ResponseNewParams) map[string]struct{} {
	names := make(map[string]struct{}, len(params.Tools))
	for _, tool := range params.Tools {
		if tool.OfFunction != nil {
			names[tool.OfFunction.Name] = struct{}{}
		}
	}
	return names
}

type streamState struct {
	codec         Codec
	op            string
	defaultModel  string
	declaredTools map[string]struct{}
	outputItems   map[int]jsontext.Value
	pendingItems  map[int]outputItemIdentity
	toolCalls     map[int]llm.ToolCall
	providerBytes int
	stopped       bool
	terminal      bool
}

func (s *streamState) consume(event openairesponses.ResponseStreamEventUnion, emit internalstream.Emit) error {
	raw := []byte(event.RawJSON())
	if len(raw) == 0 || !jsontext.Value(raw).IsValid() {
		return s.codec.malformedResponse(s.op, "event contains invalid JSON")
	}

	switch event.Type {
	case "response.output_text.delta":
		if event.Delta != "" && !emit(llm.Chunk{Content: []llm.Part{{Kind: llm.PartText, Text: event.Delta}}}) {
			s.stopped = true
		}
	case "response.reasoning_summary_text.delta":
		if event.Delta != "" && !emit(llm.Chunk{ReasoningSummary: event.Delta}) {
			s.stopped = true
		}
	case "response.output_item.added":
		return s.consumeAddedOutputItem(event)
	case "response.output_item.done", "response.output_item.completed":
		if err := s.consumeOutputItem(event); err != nil {
			return err
		}
	case "error":
		return s.codec.eventError(s.op, raw)
	case "response.failed":
		return s.codec.failedResponseError(s.op, raw)
	case "response.completed", "response.done":
		if !event.JSON.Response.Valid() {
			return s.codec.malformedResponse(s.op, "terminal event is missing response")
		}
		return s.finish(event.Response, llm.FinishReasonStop, emit)
	case "response.incomplete":
		if !event.JSON.Response.Valid() {
			return s.codec.malformedResponse(s.op, "incomplete event is missing response")
		}
		switch event.Response.IncompleteDetails.Reason {
		case "max_output_tokens":
			return s.finish(event.Response, llm.FinishReasonLength, emit)
		case "content_filter":
			return s.finish(event.Response, llm.FinishReasonContentFilter, emit)
		default:
			return &llm.Error{Kind: llm.KindProvider, Op: s.op, Provider: s.codec.Provider, Err: fmt.Errorf("response incomplete: %s", event.Response.IncompleteDetails.Reason)}
		}
	}
	return nil
}

func (s *streamState) consumeOutputItem(event openairesponses.ResponseStreamEventUnion) error {
	index, err := s.codec.outputIndex(s.op, event, "completed")
	if err != nil {
		return err
	}
	raw := jsontext.Value(event.Item.RawJSON())
	if len(raw) == 0 || raw.Kind() != jsontext.KindBeginObject {
		return s.codec.malformedResponse(s.op, "completed output item %d is missing an object", index)
	}
	raw = append(jsontext.Value(nil), raw...)
	identity, err := decodeOutputItemIdentity(raw)
	if err != nil {
		return s.codec.malformedResponse(s.op, "decode output item %d identity: %v", index, err)
	}
	if pending, exists := s.pendingItems[index]; exists {
		if !pending.matches(identity) {
			return s.codec.malformedResponse(s.op, "output item %d changed identity before completion", index)
		}
		delete(s.pendingItems, index)
	}
	if previous, exists := s.outputItems[index]; exists {
		if !bytes.Equal(previous, raw) {
			return s.codec.malformedResponse(s.op, "output index %d was reused for a different item", index)
		}
	} else {
		if s.providerBytes > s.codec.MaxProviderDataBytes-len(raw) {
			return s.codec.malformedResponse(s.op, "provider data exceeds %d bytes", s.codec.MaxProviderDataBytes)
		}
		s.outputItems[index] = raw
		s.providerBytes += len(raw)
	}

	var item struct {
		CallID    string         `json:"call_id"`
		Name      string         `json:"name"`
		Arguments jsontext.Value `json:"arguments"`
	}
	if err := jsonv2.Unmarshal(raw, &item); err != nil {
		return s.codec.malformedResponse(s.op, "decode output item %d: %v", index, err)
	}
	if identity.Type != "function_call" {
		return nil
	}
	if _, exists := s.toolCalls[index]; exists {
		return nil
	}
	if item.CallID == "" || item.Name == "" {
		return s.codec.malformedResponse(s.op, "function call at output index %d is missing call_id or name", index)
	}
	if _, declared := s.declaredTools[item.Name]; !declared {
		return s.codec.malformedResponse(s.op, "function call at output index %d names undeclared tool %q", index, item.Name)
	}
	arguments, err := functionArguments(item.Arguments)
	if err != nil {
		return s.codec.malformedResponse(s.op, "function call at output index %d arguments: %v", index, err)
	}
	s.toolCalls[index] = llm.ToolCall{ID: item.CallID, Name: item.Name, Arguments: arguments}
	return nil
}

type outputItemIdentity struct {
	Type   string `json:"type"`
	ID     string `json:"id"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
}

func (s *streamState) consumeAddedOutputItem(event openairesponses.ResponseStreamEventUnion) error {
	index, err := s.codec.outputIndex(s.op, event, "added")
	if err != nil {
		return err
	}
	raw := jsontext.Value(event.Item.RawJSON())
	if len(raw) == 0 || raw.Kind() != jsontext.KindBeginObject {
		return s.codec.malformedResponse(s.op, "added output item %d is missing an object", index)
	}
	identity, err := decodeOutputItemIdentity(raw)
	if err != nil {
		return s.codec.malformedResponse(s.op, "decode added output item %d identity: %v", index, err)
	}
	if _, exists := s.pendingItems[index]; exists {
		return s.codec.malformedResponse(s.op, "output item %d was added more than once", index)
	}
	if _, completed := s.outputItems[index]; completed {
		return s.codec.malformedResponse(s.op, "output item %d was added after completion", index)
	}
	s.pendingItems[index] = identity
	return nil
}

func (c Codec) outputIndex(op string, event openairesponses.ResponseStreamEventUnion, state string) (int, error) {
	if !event.JSON.OutputIndex.Valid() {
		return 0, c.malformedResponse(op, "%s output item is missing output_index", state)
	}
	if event.OutputIndex < 0 || event.OutputIndex > int64(^uint(0)>>1) {
		return 0, c.malformedResponse(op, "output item has invalid index %d", event.OutputIndex)
	}
	return int(event.OutputIndex), nil
}

func decodeOutputItemIdentity(raw jsontext.Value) (outputItemIdentity, error) {
	var identity outputItemIdentity
	if err := jsonv2.Unmarshal(raw, &identity); err != nil {
		return outputItemIdentity{}, err
	}
	if identity.Type == "" {
		return outputItemIdentity{}, errors.New("type must not be empty")
	}
	return identity, nil
}

func (i outputItemIdentity) matches(completed outputItemIdentity) bool {
	return i.Type == completed.Type &&
		(i.ID == "" || i.ID == completed.ID) &&
		(i.CallID == "" || i.CallID == completed.CallID) &&
		(i.Name == "" || i.Name == completed.Name)
}

func functionArguments(raw jsontext.Value) (jsontext.Value, error) {
	if raw.Kind() == jsontext.KindString {
		var encoded string
		if err := jsonv2.Unmarshal(raw, &encoded); err != nil {
			return nil, err
		}
		raw = jsontext.Value(encoded)
	}
	if len(raw) == 0 || !raw.IsValid() {
		return nil, errors.New("must contain strict JSON")
	}
	return append(jsontext.Value(nil), raw...), nil
}

func (s *streamState) finish(response openairesponses.Response, reason llm.FinishReason, emit internalstream.Emit) error {
	if s.terminal {
		return s.codec.malformedResponse(s.op, "received more than one terminal response")
	}
	hasPendingItems := len(s.pendingItems) != 0
	if reason == llm.FinishReasonStop && hasPendingItems {
		return s.codec.malformedResponse(s.op, "terminal response has %d unfinished output items", len(s.pendingItems))
	}
	status := string(response.Status)
	wantStatus := "completed"
	if reason != llm.FinishReasonStop {
		wantStatus = "incomplete"
	}
	if status != wantStatus {
		return s.codec.malformedResponse(s.op, "terminal response has status %q", status)
	}
	var providerData jsontext.Value
	if !hasPendingItems {
		if err := s.backfillReasoning(response.Output); err != nil {
			return s.codec.malformedResponse(s.op, "merge terminal response output: %v", err)
		}
		var err error
		providerData, err = s.codec.marshalProviderData(s.defaultModel, s.outputItems)
		if err != nil {
			return s.codec.malformedResponse(s.op, "encode provider data: %v", err)
		}
	}
	usage, err := responseUsage(response)
	if err != nil {
		return s.codec.malformedResponse(s.op, "usage: %v", err)
	}
	toolCalls := orderedToolCalls(s.toolCalls)
	if len(toolCalls) != 0 && reason == llm.FinishReasonStop {
		reason = llm.FinishReasonToolCall
	}
	model := string(response.Model)
	if model == "" {
		model = s.defaultModel
	}
	s.terminal = true
	emit(llm.Chunk{
		ID:           response.ID,
		Model:        model,
		ToolCalls:    toolCalls,
		ProviderData: providerData,
		FinishReason: reason,
		Usage:        usage,
	})
	return nil
}

func (s *streamState) backfillReasoning(output []openairesponses.ResponseOutputItemUnion) error {
	encryptedByID := make(map[string]string)
	for _, item := range output {
		if item.Type != "reasoning" || item.ID == "" || item.EncryptedContent == "" {
			continue
		}
		if previous, exists := encryptedByID[item.ID]; exists && previous != item.EncryptedContent {
			return fmt.Errorf("reasoning item %q has conflicting encrypted content", item.ID)
		}
		encryptedByID[item.ID] = item.EncryptedContent
	}
	if len(encryptedByID) == 0 {
		return nil
	}

	for index, raw := range s.outputItems {
		var item struct {
			Type             string `json:"type"`
			ID               string `json:"id"`
			EncryptedContent string `json:"encrypted_content"`
		}
		if err := jsonv2.Unmarshal(raw, &item); err != nil {
			return fmt.Errorf("decode output item %d: %w", index, err)
		}
		encrypted := encryptedByID[item.ID]
		if item.Type != "reasoning" || encrypted == "" || item.EncryptedContent != "" {
			continue
		}

		var fields map[string]jsontext.Value
		if err := jsonv2.Unmarshal(raw, &fields); err != nil {
			return fmt.Errorf("decode reasoning item %d: %w", index, err)
		}
		encoded, err := jsonv2.Marshal(encrypted)
		if err != nil {
			return fmt.Errorf("encode reasoning item %d encrypted content: %w", index, err)
		}
		fields["encrypted_content"] = encoded
		enriched, err := jsonv2.Marshal(fields)
		if err != nil {
			return fmt.Errorf("encode reasoning item %d: %w", index, err)
		}
		remaining := s.providerBytes - len(raw)
		if remaining > s.codec.MaxProviderDataBytes-len(enriched) {
			return fmt.Errorf("provider data exceeds %d bytes", s.codec.MaxProviderDataBytes)
		}
		s.outputItems[index] = enriched
		s.providerBytes = remaining + len(enriched)
	}
	return nil
}

func orderedToolCalls(calls map[int]llm.ToolCall) []llm.ToolCall {
	if len(calls) == 0 {
		return nil
	}
	ordered := make([]llm.ToolCall, 0, len(calls))
	for _, index := range slices.Sorted(maps.Keys(calls)) {
		ordered = append(ordered, calls[index])
	}
	return ordered
}

func (c Codec) marshalProviderData(model string, items map[int]jsontext.Value) (jsontext.Value, error) {
	if len(items) == 0 {
		return nil, nil
	}
	ordered := make([]jsontext.Value, len(items))
	for index, item := range items {
		if index < 0 || index >= len(ordered) || ordered[index] != nil {
			return nil, errors.New("output item indexes must be contiguous from zero")
		}
		ordered[index] = item
	}
	for index, item := range ordered {
		if item == nil {
			return nil, fmt.Errorf("output item index %d is missing", index)
		}
	}
	data, err := jsonv2.Marshal(providerDataEnvelope{
		API:     c.ProviderDataAPI,
		Version: providerDataVersion,
		Model:   model,
		Output:  ordered,
	})
	if err != nil {
		return nil, err
	}
	if len(data) > c.MaxProviderDataBytes {
		return nil, fmt.Errorf("exceeds %d bytes", c.MaxProviderDataBytes)
	}
	return data, nil
}

func responseUsage(response openairesponses.Response) (*llm.Usage, error) {
	if !response.JSON.Usage.Valid() {
		return nil, nil
	}
	input := response.Usage.InputTokens
	output := response.Usage.OutputTokens
	total := response.Usage.TotalTokens
	if input < 0 || output < 0 || total < 0 {
		return nil, errors.New("token counts must not be negative")
	}
	if input > int64(^uint(0)>>1) || output > int64(^uint(0)>>1) || total > int64(^uint(0)>>1) {
		return nil, errors.New("token count exceeds int range")
	}
	if total != input+output {
		return nil, fmt.Errorf("total tokens %d do not equal input %d plus output %d", total, input, output)
	}
	return &llm.Usage{InputTokens: int(input), OutputTokens: int(output), TotalTokens: int(total)}, nil
}

func (c Codec) eventError(op string, raw []byte) error {
	var event struct {
		Type    string   `json:"type"`
		Code    string   `json:"code"`
		Message string   `json:"message"`
		Error   APIError `json:"error"`
	}
	if err := jsonv2.Unmarshal(raw, &event); err != nil {
		return c.malformedResponse(op, "decode error event: %v", err)
	}
	apiErr := &event.Error
	if apiErr.empty() {
		apiErr = &APIError{Type: event.Type, Code: event.Code, Message: event.Message}
	}
	return &llm.Error{Kind: classifyAPIError(apiErr), Op: op, Provider: c.Provider, Err: apiErr}
}

func (c Codec) failedResponseError(op string, raw []byte) error {
	var event struct {
		Response struct {
			Error APIError `json:"error"`
		} `json:"response"`
	}
	if err := jsonv2.Unmarshal(raw, &event); err != nil {
		return c.malformedResponse(op, "decode failed response: %v", err)
	}
	apiErr := &event.Response.Error
	if apiErr.empty() {
		return c.malformedResponse(op, "failed response is missing error details")
	}
	return &llm.Error{Kind: classifyAPIError(apiErr), Op: op, Provider: c.Provider, Err: apiErr}
}

func (c Codec) malformedResponse(op, format string, args ...any) *llm.Error {
	return &llm.Error{Kind: llm.KindMalformedResponse, Op: op, Provider: c.Provider, Err: fmt.Errorf(format, args...)}
}

func (c Codec) transportError(op string, err error) *llm.Error {
	return &llm.Error{Kind: llm.KindTransport, Op: op, Provider: c.Provider, Err: err}
}

// ContextError preserves both cancellation identity and a distinct cause.
func ContextError(ctx context.Context) error {
	err := ctx.Err()
	if err == nil {
		return nil
	}
	cause := context.Cause(ctx)
	if cause == nil || errors.Is(cause, err) {
		if cause != nil {
			return cause
		}
		return err
	}
	return errors.Join(err, cause)
}

// MergeChunk assembles a streamed chunk into response.
func MergeChunk(response *llm.Response, chunk llm.Chunk) {
	if chunk.ID != "" {
		response.ID = chunk.ID
	}
	if chunk.Model != "" {
		response.Model = chunk.Model
	}
	response.Message.Content = append(response.Message.Content, chunk.Content...)
	response.Message.ToolCalls = append(response.Message.ToolCalls, chunk.ToolCalls...)
	response.ReasoningSummary += chunk.ReasoningSummary
	if len(chunk.ProviderData) != 0 {
		response.Message.ProviderData = append(response.Message.ProviderData[:0], chunk.ProviderData...)
	}
	if chunk.FinishReason != "" {
		response.FinishReason = chunk.FinishReason
	}
	if chunk.Usage != nil {
		usage := *chunk.Usage
		response.Usage = &usage
	}
}
