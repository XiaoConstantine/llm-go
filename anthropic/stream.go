package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"sync"

	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
)

const (
	streamReadBufferBytes  = 32 << 10
	maxStreamEventBytes    = 4 << 20
	maxStreamContentBlocks = 1024
	streamUTF8BOM          = "\xef\xbb\xbf"
)

var errStreamEmitStopped = errors.New("stream emission stopped")

func (c *Client) produceStream(ctx context.Context, payload []byte, declared, priorToolIDs map[string]struct{}, emit internalstream.Emit) (err error) {
	defer func() {
		err = relabelProviderError(err, c.provider)
	}()

	response, err := c.openStream(ctx, payload)
	if err != nil {
		return err
	}

	var closeOnce sync.Once
	var closeErr error
	closeBody := func() {
		closeOnce.Do(func() {
			closeErr = response.Body.Close()
		})
	}
	closingDone := make(chan struct{})
	stopClosing := context.AfterFunc(ctx, func() {
		closeBody()
		close(closingDone)
	})

	streamErr := readEventStream(ctx, response.Body, newStreamDecoder(response.Header.Get("Request-Id"), declared, priorToolIDs), emit)
	if !stopClosing() {
		<-closingDone
	}
	closeBody()

	if contextErr := contextErr(ctx); contextErr != nil {
		return errors.Join(contextErr, streamBodyCloseError(closeErr))
	}
	if streamErr != nil {
		return errors.Join(streamErr, streamBodyCloseError(closeErr))
	}
	if closeErr != nil {
		return streamTransportError("close response body: %w", closeErr)
	}
	return nil
}

func (c *Client) openStream(ctx context.Context, payload []byte) (*http.Response, error) {
	headers := c.headers.Clone()
	headers.Set("Accept", "text/event-stream")
	response, err, downstreamErr := c.executeRaw(ctx, payload, headers)
	if err != nil && !errors.Is(err, errSDKRawErrorResponse) {
		contextErr := contextErr(ctx)
		if downstreamErr != nil || contextErr == nil || response == nil || response.Body == nil {
			if contextErr != nil {
				return nil, contextErr
			}
			return nil, streamTransportError("send request: %w", err)
		}
	}
	if response == nil || response.Body == nil {
		return nil, streamTransportError("SDK returned no HTTP response")
	}
	if contextErr := contextErr(ctx); contextErr != nil {
		return nil, errors.Join(contextErr, streamBodyCloseError(response.Body.Close()))
	}

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, tooLarge, readErr, closeErr := readAndCloseStreamBody(ctx, response.Body, maxErrorBodyBytes)
		closeFailure := streamBodyCloseError(closeErr)
		if readErr != nil {
			if contextErr := contextErr(ctx); contextErr != nil {
				return nil, errors.Join(contextErr, closeFailure)
			}
			return nil, errors.Join(streamTransportError("read error response: %w", readErr), closeFailure)
		}
		responseErr := responseError("stream", response, body, tooLarge)
		if contextErr := contextErr(ctx); contextErr != nil {
			return nil, errors.Join(contextErr, closeFailure)
		}
		return nil, errors.Join(responseErr, closeFailure)
	}

	mediaType, _, mediaErr := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if mediaErr != nil || !strings.EqualFold(mediaType, "text/event-stream") {
		closeFailure := streamBodyCloseError(response.Body.Close())
		if contextErr := contextErr(ctx); contextErr != nil {
			return nil, errors.Join(contextErr, closeFailure)
		}
		return nil, errors.Join(
			malformedStream("response content type %q is not text/event-stream", response.Header.Get("Content-Type")),
			closeFailure,
		)
	}
	return response, nil
}

func readAndCloseStreamBody(ctx context.Context, body io.ReadCloser, limit int64) ([]byte, bool, error, error) {
	var closeOnce sync.Once
	var closeErr error
	closeBody := func() {
		closeOnce.Do(func() {
			closeErr = body.Close()
		})
	}
	closingDone := make(chan struct{})
	stopClosing := context.AfterFunc(ctx, func() {
		closeBody()
		close(closingDone)
	})

	data, tooLarge, readErr := readLimited(body, limit)
	if !stopClosing() {
		<-closingDone
	}
	closeBody()
	return data, tooLarge, readErr, closeErr
}

func streamBodyCloseError(err error) error {
	if err == nil {
		return nil
	}
	return streamTransportError("close response body: %w", err)
}

func readEventStream(ctx context.Context, body io.Reader, decoder *streamDecoder, emit internalstream.Emit) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, streamReadBufferBytes), maxStreamEventBytes+len(streamUTF8BOM)+2)
	scanner.Split(splitStreamLines)

	var eventName string
	var data strings.Builder
	hasData := false
	eventBytes := 0
	firstLine := true

	dispatch := func() (bool, error) {
		defer func() {
			eventName = ""
			data.Reset()
			hasData = false
			eventBytes = 0
		}()
		if !hasData {
			return false, nil
		}
		return decoder.consume(eventName, data.String(), emit)
	}

	for scanner.Scan() {
		line := scanner.Bytes()
		if firstLine {
			line = bytes.TrimPrefix(line, []byte(streamUTF8BOM))
			firstLine = false
		}
		if contextErr := contextErr(ctx); contextErr != nil {
			return contextErr
		}
		if len(line) > maxStreamEventBytes {
			return malformedStream("event stream line exceeds %d bytes", maxStreamEventBytes)
		}
		eventBytes += len(line) + 1
		if eventBytes > maxStreamEventBytes {
			return malformedStream("event stream event exceeds %d bytes", maxStreamEventBytes)
		}
		if len(line) == 0 {
			done, dispatchErr := dispatch()
			if dispatchErr != nil {
				if errors.Is(dispatchErr, errStreamEmitStopped) {
					return nil
				}
				return dispatchErr
			}
			if done {
				return nil
			}
			continue
		}
		if line[0] == ':' {
			continue
		}

		field, value, found := bytes.Cut(line, []byte{':'})
		if !found {
			value = nil
		}
		if len(value) != 0 && value[0] == ' ' {
			value = value[1:]
		}
		switch string(field) {
		case "event":
			eventName = string(value)
		case "data":
			if hasData {
				data.WriteByte('\n')
			}
			data.Write(value)
			hasData = true
		}
	}

	if contextErr := contextErr(ctx); contextErr != nil {
		return contextErr
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return malformedStream("event stream line exceeds %d bytes", maxStreamEventBytes)
		}
		return streamTransportError("read event stream: %w", err)
	}
	if hasData {
		done, err := dispatch()
		if err != nil {
			if errors.Is(err, errStreamEmitStopped) {
				return nil
			}
			return err
		}
		if done {
			return nil
		}
	}
	return malformedStream("event stream ended before message_stop")
}

func splitStreamLines(data []byte, atEOF bool) (advance int, token []byte, err error) {
	for index, character := range data {
		switch character {
		case '\n':
			return index + 1, data[:index], nil
		case '\r':
			if index+1 == len(data) && !atEOF {
				return 0, nil, nil
			}
			advance := index + 1
			if index+1 < len(data) && data[index+1] == '\n' {
				advance++
			}
			return advance, data[:index], nil
		}
	}
	if atEOF && len(data) != 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

type streamDecoder struct {
	started           bool
	messageDeltaSeen  bool
	finished          bool
	requestID         string
	id                string
	model             string
	nextBlock         int
	activeBlock       *streamBlock
	usage             usageAccumulator
	declared          map[string]struct{}
	seenToolIDs       map[string]struct{}
	toolCalls         []llm.ToolCall
	toolArgumentBytes int
	pendingEvents     []llm.StreamEvent
}

type streamBlock struct {
	index        int
	kind         string
	toolID       string
	toolName     string
	hasToolDelta bool
	content      strings.Builder
}

type streamEnvelope struct {
	Type string `json:"type"`
}

type streamMessageStart struct {
	Message *streamMessage `json:"message"`
}

type streamMessage struct {
	ID         string             `json:"id"`
	Type       string             `json:"type"`
	Role       string             `json:"role"`
	Content    *[]json.RawMessage `json:"content"`
	Model      string             `json:"model"`
	StopReason json.RawMessage    `json:"stop_reason"`
	Usage      *streamUsage       `json:"usage"`
}

type streamContentBlockEvent struct {
	Index        *int                `json:"index"`
	ContentBlock *streamContentBlock `json:"content_block"`
}

type streamContentBlock struct {
	Type  string           `json:"type"`
	Text  *string          `json:"text"`
	ID    *string          `json:"id"`
	Name  *string          `json:"name"`
	Input *json.RawMessage `json:"input"`
}

type streamContentDeltaEvent struct {
	Index *int         `json:"index"`
	Delta *streamDelta `json:"delta"`
}

type streamDelta struct {
	Type        string  `json:"type"`
	Text        *string `json:"text"`
	PartialJSON *string `json:"partial_json"`
}

type streamContentStopEvent struct {
	Index *int `json:"index"`
}

type streamMessageDeltaEvent struct {
	Delta *streamMessageDelta `json:"delta"`
	Usage *streamUsage        `json:"usage"`
}

type streamMessageDelta struct {
	StopReason *string `json:"stop_reason"`
}

type streamUsage struct {
	InputTokens              *int `json:"input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
}

type streamErrorEvent struct {
	Error     *APIError `json:"error"`
	RequestID string    `json:"request_id"`
}

func newStreamDecoder(requestID string, declared, priorToolIDs map[string]struct{}) *streamDecoder {
	seen := make(map[string]struct{}, len(priorToolIDs))
	for id := range priorToolIDs {
		seen[id] = struct{}{}
	}
	return &streamDecoder{requestID: requestID, declared: declared, seenToolIDs: seen}
}

func (decoder *streamDecoder) consume(eventName, data string, emit internalstream.Emit) (bool, error) {
	var envelope streamEnvelope
	if err := jsonv2.Unmarshal([]byte(data), &envelope); err != nil {
		return false, malformedStream("decode %q event: %w", eventName, err)
	}
	if eventName == "" {
		return false, malformedStream("event has no name")
	}
	if envelope.Type == "" {
		return false, malformedStream("event %q has no type", eventName)
	}
	if envelope.Type != eventName {
		return false, malformedStream("event name %q does not match type %q", eventName, envelope.Type)
	}

	switch eventName {
	case "ping":
		return false, nil
	case "error":
		return false, decodeStreamError(data, decoder.requestID)
	case "message_start":
		return false, decoder.consumeMessageStart(data)
	case "content_block_start":
		return false, decoder.consumeContentStart(data, emit)
	case "content_block_delta":
		return false, decoder.consumeContentDelta(data, emit)
	case "content_block_stop":
		return false, decoder.consumeContentStop(data, emit)
	case "message_delta":
		return false, decoder.consumeMessageDelta(data, emit)
	case "message_stop":
		return decoder.consumeMessageStop()
	default:
		return false, nil
	}
}

func (decoder *streamDecoder) consumeMessageStart(data string) error {
	if decoder.started {
		return malformedStream("event stream contains repeated message_start")
	}
	var event streamMessageStart
	if err := jsonv2.Unmarshal([]byte(data), &event); err != nil {
		return malformedStream("decode message_start event: %w", err)
	}
	message := event.Message
	if message == nil {
		return malformedStream("message_start has no message")
	}
	if message.ID == "" {
		return malformedStream("message_start has no message ID")
	}
	if message.Type != "message" {
		return malformedStream("message_start has message type %q, want message", message.Type)
	}
	if message.Role != string(llm.RoleAssistant) {
		return malformedStream("message_start has role %q, want assistant", message.Role)
	}
	if message.Content == nil || len(*message.Content) != 0 {
		return malformedStream("message_start content must be an empty array")
	}
	if message.Model == "" {
		return malformedStream("message_start has no model")
	}
	if len(message.StopReason) == 0 || !bytes.Equal(bytes.TrimSpace(message.StopReason), []byte("null")) {
		return malformedStream("message_start stop reason must be null")
	}
	if message.Usage == nil {
		return malformedStream("message_start has no usage")
	}
	if err := decoder.usage.merge(message.Usage); err != nil {
		return malformedStream("message_start usage: %w", err)
	}
	if !decoder.usage.hasInput || !decoder.usage.hasOutput {
		return malformedStream("message_start usage is incomplete")
	}

	decoder.started = true
	decoder.id = message.ID
	decoder.model = message.Model
	decoder.pendingEvents = append(decoder.pendingEvents, llm.StreamEvent{Kind: llm.StreamEventStart})
	return nil
}

func (decoder *streamDecoder) consumeContentStart(data string, emit internalstream.Emit) error {
	if err := decoder.requireContentEvent("content_block_start"); err != nil {
		return err
	}
	if decoder.activeBlock != nil {
		return malformedStream("content_block_start arrived before block %d stopped", decoder.activeBlock.index)
	}
	var event streamContentBlockEvent
	if err := jsonv2.Unmarshal([]byte(data), &event); err != nil {
		return malformedStream("decode content_block_start event: %w", err)
	}
	if event.Index == nil || *event.Index != decoder.nextBlock {
		return malformedStream("content_block_start index is %v, want %d", optionalIndex(event.Index), decoder.nextBlock)
	}
	if *event.Index >= maxStreamContentBlocks {
		return malformedStream("content block index %d exceeds limit %d", *event.Index, maxStreamContentBlocks)
	}
	if event.ContentBlock == nil {
		return malformedStream("content_block_start %d has no content block", *event.Index)
	}
	if event.ContentBlock.Type == "" {
		return malformedStream("content_block_start %d has no content block type", *event.Index)
	}

	block := &streamBlock{index: *event.Index, kind: event.ContentBlock.Type}
	chunk := llm.Chunk{ID: decoder.id, Model: decoder.model}
	switch block.kind {
	case "text":
		if event.ContentBlock.Text == nil {
			return malformedStream("text content block %d has no text", block.index)
		}
		decoder.pendingEvents = append(decoder.pendingEvents, llm.StreamEvent{Kind: llm.StreamEventTextStart, Index: block.index})
		if *event.ContentBlock.Text != "" {
			if err := appendStreamContent(&block.content, *event.ContentBlock.Text); err != nil {
				return err
			}
			chunk.Content = []llm.Part{{Kind: llm.PartText, Text: *event.ContentBlock.Text}}
			chunk.Events = append(decoder.takePendingEvents(), llm.StreamEvent{Kind: llm.StreamEventTextDelta, Index: block.index, Delta: *event.ContentBlock.Text})
		}
	case "tool_use":
		if event.ContentBlock.ID == nil || *event.ContentBlock.ID == "" {
			return malformedStream("tool use %d has no ID", block.index)
		}
		if _, exists := decoder.seenToolIDs[*event.ContentBlock.ID]; exists {
			return malformedStream("tool use %d repeats ID %q", block.index, *event.ContentBlock.ID)
		}
		if event.ContentBlock.Name == nil || !validToolName(*event.ContentBlock.Name) {
			return malformedStream("tool use %d has invalid name", block.index)
		}
		if _, exists := decoder.declared[*event.ContentBlock.Name]; !exists {
			return malformedStream("tool use %d names undeclared tool %q", block.index, *event.ContentBlock.Name)
		}
		if event.ContentBlock.Input == nil || !jsontext.Value(*event.ContentBlock.Input).IsValid() || jsontext.Value(*event.ContentBlock.Input).Kind() != jsontext.KindBeginObject {
			return malformedStream("tool use %d input must be a strict JSON object", block.index)
		}
		decoder.seenToolIDs[*event.ContentBlock.ID] = struct{}{}
		block.toolID = *event.ContentBlock.ID
		block.toolName = *event.ContentBlock.Name
		if err := appendStreamContent(&block.content, string(*event.ContentBlock.Input)); err != nil {
			return err
		}
		chunk.Events = append(decoder.takePendingEvents(), llm.StreamEvent{Kind: llm.StreamEventToolCallStart, Index: block.index, ToolCallID: block.toolID, ToolName: block.toolName})
	default:
		return unsupported("stream", fmt.Sprintf("response content block type %q is not implemented", block.kind))
	}
	decoder.activeBlock = block
	if len(chunk.Content) != 0 || len(chunk.Events) != 0 {
		if !emit(chunk) {
			return errStreamEmitStopped
		}
	}
	return nil
}

func (decoder *streamDecoder) consumeContentDelta(data string, emit internalstream.Emit) error {
	if err := decoder.requireContentEvent("content_block_delta"); err != nil {
		return err
	}
	var event streamContentDeltaEvent
	if err := jsonv2.Unmarshal([]byte(data), &event); err != nil {
		return malformedStream("decode content_block_delta event: %w", err)
	}
	block := decoder.activeBlock
	if block == nil {
		return malformedStream("content_block_delta arrived without an active block")
	}
	if event.Index == nil || *event.Index != block.index {
		return malformedStream("content_block_delta index is %v, want %d", optionalIndex(event.Index), block.index)
	}
	if event.Delta == nil {
		return malformedStream("content_block_delta %d has no delta", *event.Index)
	}
	if event.Delta.Type == "" {
		return malformedStream("content_block_delta %d has no delta type", *event.Index)
	}
	chunk := llm.Chunk{ID: decoder.id, Model: decoder.model}
	switch {
	case block.kind == "text" && event.Delta.Type == "text_delta":
		if event.Delta.Text == nil {
			return malformedStream("text delta %d has no text", block.index)
		}
		if *event.Delta.Text == "" {
			return nil
		}
		if err := appendStreamContent(&block.content, *event.Delta.Text); err != nil {
			return err
		}
		chunk.Content = []llm.Part{{Kind: llm.PartText, Text: *event.Delta.Text}}
		chunk.Events = append(decoder.takePendingEvents(), llm.StreamEvent{Kind: llm.StreamEventTextDelta, Index: block.index, Delta: *event.Delta.Text})
	case block.kind == "tool_use" && event.Delta.Type == "input_json_delta":
		if event.Delta.PartialJSON == nil {
			return malformedStream("tool argument delta %d has no partial_json", block.index)
		}
		if *event.Delta.PartialJSON == "" {
			return nil
		}
		if !block.hasToolDelta {
			block.hasToolDelta = true
			block.content.Reset()
		}
		if err := appendStreamContent(&block.content, *event.Delta.PartialJSON); err != nil {
			return err
		}
		chunk.Events = append(decoder.takePendingEvents(), llm.StreamEvent{Kind: llm.StreamEventToolCallDelta, Index: block.index, Delta: *event.Delta.PartialJSON, ToolCallID: block.toolID, ToolName: block.toolName})
	default:
		return unsupported("stream", fmt.Sprintf("response content delta type %q is not implemented for %q", event.Delta.Type, block.kind))
	}
	if !emit(chunk) {
		return errStreamEmitStopped
	}
	return nil
}

func (decoder *streamDecoder) consumeContentStop(data string, emit internalstream.Emit) error {
	if err := decoder.requireContentEvent("content_block_stop"); err != nil {
		return err
	}
	var event streamContentStopEvent
	if err := jsonv2.Unmarshal([]byte(data), &event); err != nil {
		return malformedStream("decode content_block_stop event: %w", err)
	}
	block := decoder.activeBlock
	if block == nil {
		return malformedStream("content_block_stop arrived without an active block")
	}
	if event.Index == nil || *event.Index != block.index {
		return malformedStream("content_block_stop index is %v, want %d", optionalIndex(event.Index), block.index)
	}
	chunk := llm.Chunk{ID: decoder.id, Model: decoder.model}
	switch block.kind {
	case "text":
		decoder.pendingEvents = append(decoder.pendingEvents, llm.StreamEvent{Kind: llm.StreamEventTextEnd, Index: block.index, Content: block.content.String()})
	case "tool_use":
		arguments := json.RawMessage(block.content.String())
		if !jsontext.Value(arguments).IsValid() || jsontext.Value(arguments).Kind() != jsontext.KindBeginObject {
			return malformedStream("tool use %d arguments are not a strict JSON object", block.index)
		}
		if len(arguments) > maxResponseBodyBytes-decoder.toolArgumentBytes {
			return malformedStream("streamed tool arguments exceed %d bytes", maxResponseBodyBytes)
		}
		decoder.toolArgumentBytes += len(arguments)
		call := llm.ToolCall{ID: block.toolID, Name: block.toolName, Arguments: append(json.RawMessage(nil), arguments...)}
		decoder.toolCalls = append(decoder.toolCalls, call)
		eventCall := call
		eventCall.Arguments = append(json.RawMessage(nil), call.Arguments...)
		chunk.Events = []llm.StreamEvent{{Kind: llm.StreamEventToolCallEnd, Index: block.index, ToolCallID: call.ID, ToolName: call.Name, ToolCall: &eventCall}}
	}
	decoder.activeBlock = nil
	decoder.nextBlock++
	if len(chunk.Events) != 0 && !emit(chunk) {
		return errStreamEmitStopped
	}
	return nil
}

func (decoder *streamDecoder) takePendingEvents() []llm.StreamEvent {
	events := decoder.pendingEvents
	decoder.pendingEvents = nil
	return events
}

func appendStreamContent(builder *strings.Builder, fragment string) error {
	if builder.Len() > maxResponseBodyBytes-len(fragment) {
		return malformedStream("buffered stream output exceeds %d bytes", maxResponseBodyBytes)
	}
	builder.WriteString(fragment)
	return nil
}

func (decoder *streamDecoder) consumeMessageDelta(data string, emit internalstream.Emit) error {
	if !decoder.started {
		return malformedStream("message_delta arrived before message_start")
	}
	if decoder.finished {
		return malformedStream("message_delta arrived after the finish reason")
	}
	if decoder.activeBlock != nil {
		return malformedStream("message_delta arrived before content block %d stopped", decoder.activeBlock.index)
	}
	var event streamMessageDeltaEvent
	if err := jsonv2.Unmarshal([]byte(data), &event); err != nil {
		return malformedStream("decode message_delta event: %w", err)
	}
	if event.Delta == nil {
		return malformedStream("message_delta has no delta")
	}
	decoder.messageDeltaSeen = true
	if event.Usage != nil {
		if err := decoder.usage.merge(event.Usage); err != nil {
			return malformedStream("message_delta usage: %w", err)
		}
	}
	if event.Delta.StopReason == nil {
		return nil
	}
	if event.Usage == nil || event.Usage.OutputTokens == nil {
		return malformedStream("finish reason arrived without cumulative output usage")
	}
	finish, err := finishReason("stream", *event.Delta.StopReason)
	if err != nil {
		return err
	}
	if finish == llm.FinishReasonToolCall {
		if len(decoder.toolCalls) == 0 {
			return malformedStream("stop reason tool_use has no tool calls")
		}
	} else if len(decoder.toolCalls) != 0 {
		if finish == llm.FinishReasonLength {
			return unsupported("stream", "truncated tool use is not representable")
		}
		return malformedStream("stop reason %q is inconsistent with tool use", *event.Delta.StopReason)
	}
	usage, err := decoder.usage.value()
	if err != nil {
		return malformedStream("final usage: %w", err)
	}
	decoder.finished = true
	toolCalls := make([]llm.ToolCall, len(decoder.toolCalls))
	for index, call := range decoder.toolCalls {
		toolCalls[index] = call
		toolCalls[index].Arguments = append(json.RawMessage(nil), call.Arguments...)
	}
	if !emit(llm.Chunk{
		ID:           decoder.id,
		Model:        decoder.model,
		ToolCalls:    toolCalls,
		FinishReason: finish,
		Usage:        usage,
		Events:       append(decoder.takePendingEvents(), llm.StreamEvent{Kind: llm.StreamEventDone, FinishReason: finish}),
	}) {
		return errStreamEmitStopped
	}
	return nil
}

func (decoder *streamDecoder) consumeMessageStop() (bool, error) {
	if !decoder.started {
		return false, malformedStream("message_stop arrived before message_start")
	}
	if decoder.activeBlock != nil {
		return false, malformedStream("message_stop arrived before content block %d stopped", decoder.activeBlock.index)
	}
	if !decoder.finished {
		return false, malformedStream("message_stop arrived before a finish reason")
	}
	return true, nil
}

func (decoder *streamDecoder) requireContentEvent(name string) error {
	if !decoder.started {
		return malformedStream("%s arrived before message_start", name)
	}
	if decoder.finished {
		return malformedStream("%s arrived after the finish reason", name)
	}
	if decoder.messageDeltaSeen {
		return malformedStream("%s arrived after message_delta", name)
	}
	return nil
}

type usageAccumulator struct {
	inputTokens              int
	outputTokens             int
	cacheCreationInputTokens int
	cacheReadInputTokens     int
	hasInput                 bool
	hasOutput                bool
	hasCacheCreation         bool
	hasCacheRead             bool
}

func (usage *usageAccumulator) merge(update *streamUsage) error {
	if err := mergeTokenCount("input tokens", update.InputTokens, &usage.inputTokens, &usage.hasInput); err != nil {
		return err
	}
	if err := mergeTokenCount("output tokens", update.OutputTokens, &usage.outputTokens, &usage.hasOutput); err != nil {
		return err
	}
	if err := mergeTokenCount("cache creation input tokens", update.CacheCreationInputTokens,
		&usage.cacheCreationInputTokens, &usage.hasCacheCreation); err != nil {
		return err
	}
	return mergeTokenCount("cache read input tokens", update.CacheReadInputTokens,
		&usage.cacheReadInputTokens, &usage.hasCacheRead)
}

func mergeTokenCount(name string, update *int, current *int, seen *bool) error {
	if update == nil {
		return nil
	}
	if *update < 0 {
		return fmt.Errorf("%s must not be negative", name)
	}
	if *seen && *update < *current {
		return fmt.Errorf("%s decreased from %d to %d", name, *current, *update)
	}
	*current = *update
	*seen = true
	return nil
}

func (usage *usageAccumulator) value() (*llm.Usage, error) {
	if !usage.hasInput || !usage.hasOutput {
		return nil, errors.New("token counts are incomplete")
	}
	inputTokens, ok := addInts(usage.inputTokens, usage.cacheCreationInputTokens, usage.cacheReadInputTokens)
	if !ok {
		return nil, errors.New("input token count overflows int")
	}
	totalTokens, ok := addInts(inputTokens, usage.outputTokens)
	if !ok {
		return nil, errors.New("total token count overflows int")
	}
	return &llm.Usage{
		InputTokens:  inputTokens,
		OutputTokens: usage.outputTokens,
		TotalTokens:  totalTokens,
	}, nil
}

func decodeStreamError(data, requestID string) error {
	var event streamErrorEvent
	if err := jsonv2.Unmarshal([]byte(data), &event); err != nil {
		return malformedStream("decode error event: %w", err)
	}
	if event.Error == nil || event.Error.empty() {
		return malformedStream("event contains an empty provider error")
	}
	if event.Error.RequestID == "" {
		event.Error.RequestID = event.RequestID
	}
	if event.Error.RequestID == "" {
		event.Error.RequestID = requestID
	}
	return &llm.Error{
		Kind:     classifyAPIError(event.Error),
		Op:       "stream",
		Provider: "anthropic",
		Err:      event.Error,
	}
}

func optionalIndex(index *int) any {
	if index == nil {
		return "missing"
	}
	return *index
}

func malformedStream(format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindMalformedResponse,
		Op:       "stream",
		Provider: "anthropic",
		Err:      fmt.Errorf(format, args...),
	}
}

func streamTransportError(format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindTransport,
		Op:       "stream",
		Provider: "anthropic",
		Err:      fmt.Errorf(format, args...),
	}
}
