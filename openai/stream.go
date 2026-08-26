package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"sync"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
)

const (
	streamReadBufferBytes = 32 << 10
	maxStreamEventBytes   = 4 << 20
	maxStreamToolCalls    = 1024
	streamUTF8BOM         = "\xef\xbb\xbf"
)

var (
	errStreamEmitStopped = errors.New("stream emission stopped")
)

type streamResponse struct {
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Choices *[]streamChoice `json:"choices"`
	Usage   *responseUsage  `json:"usage"`
	Error   *APIError       `json:"error"`
}

type streamChoice struct {
	Index        *int            `json:"index"`
	Delta        *streamDelta    `json:"delta"`
	FinishReason json.RawMessage `json:"finish_reason"`
}

type streamDelta struct {
	Role             *string                  `json:"role"`
	Content          *string                  `json:"content"`
	Refusal          *string                  `json:"refusal"`
	ReasoningContent *string                  `json:"reasoning_content"`
	Reasoning        *string                  `json:"reasoning"`
	ReasoningDetails *[]json.RawMessage       `json:"reasoning_details"`
	ToolCalls        []streamToolCallDelta    `json:"tool_calls"`
	FunctionCall     *streamFunctionCallDelta `json:"function_call"`
}

type streamToolCallDelta struct {
	Index    *int                     `json:"index"`
	ID       *string                  `json:"id"`
	Type     *string                  `json:"type"`
	Function *streamFunctionCallDelta `json:"function"`
}

type streamFunctionCallDelta struct {
	Name      *string `json:"name"`
	Arguments *string `json:"arguments"`
}

func (c *Client) produceStream(ctx context.Context, format llm.ResponseFormat, declaredTools map[string]struct{}, payload []byte, sessionID string, emit internalstream.Emit) (err error) {
	defer func() {
		err = relabelProviderError(err, c.provider)
	}()

	response, err := c.openStream(ctx, payload, sessionID)
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

	streamErr := readEventStream(ctx, response.Body, newStreamDecoderWithCompatibility(c.model, format, declaredTools,
		c.compatibility.FinishReason != llm.CompatibilityDisabled), emit)
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

func (c *Client) openStream(ctx context.Context, payload []byte, sessionID string) (*http.Response, error) {
	headers := c.requestHeaders(sessionID)
	headers.Set("Accept", "text/event-stream")
	response, err, downstreamErr := c.executeRaw(ctx, c.sdkChatPath, payload, headers)
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

	var data strings.Builder
	hasData := false
	eventBytes := 0
	firstLine := true

	dispatch := func() (bool, error) {
		defer func() {
			data.Reset()
			hasData = false
			eventBytes = 0
		}()
		if !hasData || data.Len() == 0 {
			return false, nil
		}
		return decoder.consume(data.String(), emit)
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
		if !bytes.Equal(field, []byte("data")) {
			continue
		}
		if len(value) != 0 && value[0] == ' ' {
			value = value[1:]
		}
		if hasData {
			data.WriteByte('\n')
		}
		data.Write(value)
		hasData = true
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
	return malformedStream("event stream ended without [DONE]")
}

func splitStreamLines(data []byte, atEOF bool) (advance int, token []byte, err error) {
	for i, character := range data {
		switch character {
		case '\n':
			return i + 1, data[:i], nil
		case '\r':
			if i+1 == len(data) && !atEOF {
				return 0, nil, nil
			}
			advance := i + 1
			if i+1 < len(data) && data[i+1] == '\n' {
				advance++
			}
			return advance, data[:i], nil
		}
	}
	if atEOF && len(data) != 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

type streamDecoder struct {
	configuredModel      string
	format               llm.ResponseFormat
	declaredTools        map[string]struct{}
	id                   string
	model                string
	emittedModel         string
	content              strings.Builder
	reasoning            strings.Builder
	reasoningField       string
	reasoningDetails     []json.RawMessage
	reasoningDetailsSeen bool
	refusal              strings.Builder
	toolMode             streamToolMode
	toolCalls            []*streamToolCallBuilder
	legacyCall           streamFunctionCallBuilder
	legacyStarted        bool
	bufferedBytes        int
	refusalSeen          bool
	started              bool
	textStarted          bool
	reasoningStarted     bool
	reasoningEnded       bool
	finished             bool
	usageSeen            bool
	supportsFinishReason bool
}

type streamToolMode uint8

const (
	streamToolModeNone streamToolMode = iota
	streamToolModeModern
	streamToolModeLegacy
)

type streamToolCallBuilder struct {
	id       strings.Builder
	callType strings.Builder
	function streamFunctionCallBuilder
	started  bool
}

type streamFunctionCallBuilder struct {
	name      strings.Builder
	arguments strings.Builder
}

func newStreamDecoder(configuredModel string, format llm.ResponseFormat, declaredTools map[string]struct{}) *streamDecoder {
	return newStreamDecoderWithCompatibility(configuredModel, format, declaredTools, true)
}

func newStreamDecoderWithCompatibility(configuredModel string, format llm.ResponseFormat, declaredTools map[string]struct{}, supportsFinishReason bool) *streamDecoder {
	return &streamDecoder{
		configuredModel:      configuredModel,
		format:               format,
		declaredTools:        declaredTools,
		supportsFinishReason: supportsFinishReason,
	}
}

func (d *streamDecoder) consume(data string, emit internalstream.Emit) (bool, error) {
	if data == "[DONE]" {
		if !d.finished {
			if d.supportsFinishReason {
				return false, malformedStream("event stream ended before a finish reason")
			}
			finish := llm.FinishReasonStop
			providerFinish := "stop"
			switch d.toolMode {
			case streamToolModeModern:
				finish = llm.FinishReasonToolCall
				providerFinish = "tool_calls"
			case streamToolModeLegacy:
				finish = llm.FinishReasonToolCall
				providerFinish = "function_call"
			}
			chunk := llm.Chunk{ID: d.id, Model: d.modelForChunk(true)}
			if !d.started {
				d.started = true
				chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventStart})
			}
			if err := d.finishChunk(&chunk, finish, providerFinish); err != nil {
				return false, err
			}
			if !emit(chunk) {
				return false, errStreamEmitStopped
			}
		}
		return true, nil
	}

	var event streamResponse
	if err := jsonv2.Unmarshal([]byte(data), &event); err != nil {
		return false, malformedStream("decode event: %w", err)
	}
	if event.Error != nil {
		if event.Error.empty() {
			return false, malformedStream("event contains an empty provider error")
		}
		return false, &llm.Error{
			Kind:     classifyAPIError(event.Error),
			Op:       "stream",
			Provider: "openai",
			Err:      event.Error,
		}
	}
	if err := d.observeMetadata(event.ID, event.Model); err != nil {
		return false, err
	}
	usage, err := usageFromWire(event.Usage)
	if err != nil {
		return false, malformedStream("event contains %w", err)
	}

	if event.Choices == nil {
		return false, malformedStream("event has no choices field")
	}
	choices := *event.Choices
	if len(choices) == 0 {
		if usage == nil {
			return false, malformedStream("event contains neither a choice nor usage")
		}
		if !d.finished && d.supportsFinishReason {
			return false, malformedStream("usage arrived before the finish reason")
		}
		if d.usageSeen {
			return false, malformedStream("event stream contains repeated usage")
		}
		d.usageSeen = true
		chunk := llm.Chunk{ID: d.id, Model: d.modelForChunk(true), Usage: usage}
		if !d.started {
			d.started = true
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventStart})
		}
		if !emit(chunk) {
			return false, errStreamEmitStopped
		}
		return false, nil
	}
	if len(choices) != 1 {
		return false, malformedStream("event contains %d choices, want exactly one", len(choices))
	}
	if d.finished {
		return false, malformedStream("event contains a choice after the finish reason")
	}

	choice := choices[0]
	if choice.Index == nil {
		return false, malformedStream("event choice has no index")
	}
	if *choice.Index != 0 {
		return false, malformedStream("event choice has index %d, want 0", *choice.Index)
	}
	if choice.Delta == nil {
		return false, malformedStream("event choice has no delta")
	}
	if len(choice.FinishReason) == 0 && d.supportsFinishReason {
		return false, malformedStream("event choice has no finish reason field")
	}
	if choice.Delta.Role != nil && *choice.Delta.Role != string(llm.RoleAssistant) {
		return false, malformedStream("event delta has role %q, want assistant", *choice.Delta.Role)
	}
	chunk := llm.Chunk{ID: d.id, Model: d.modelForChunk(false)}
	if !d.started {
		d.started = true
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventStart})
	}
	reasoning, err := reasoningFromWire(choice.Delta.ReasoningContent, choice.Delta.Reasoning)
	if err != nil {
		return false, malformedStream("event delta %v", err)
	}
	if reasoning != nil && *reasoning != "" {
		if d.reasoningEnded || d.textStarted || d.toolMode != streamToolModeNone {
			return false, malformedStream("reasoning delta arrived after text or tool output started")
		}
		field := "reasoning"
		if choice.Delta.ReasoningContent != nil {
			field = "reasoning_content"
		}
		if d.reasoningField != "" && d.reasoningField != field {
			return false, malformedStream("reasoning field changed from %s to %s", d.reasoningField, field)
		}
		d.reasoningField = field
		if err := d.buffer(&d.reasoning, *reasoning); err != nil {
			return false, err
		}
		if !d.reasoningStarted {
			d.reasoningStarted = true
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventReasoningStart, Index: 0})
		}
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventReasoningDelta, Index: 0, Delta: *reasoning})
	}
	if err := d.appendReasoningDetails(choice.Delta.ReasoningDetails); err != nil {
		return false, err
	}
	if len(choice.Delta.ToolCalls) != 0 || choice.Delta.FunctionCall != nil {
		if d.reasoningStarted && !d.reasoningEnded {
			d.reasoningEnded = true
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventReasoningEnd, Index: 0, Content: d.reasoning.String()})
		}
		events, err := d.consumeToolDelta(*choice.Delta)
		if err != nil {
			return false, err
		}
		chunk.Events = append(chunk.Events, events...)
	}
	if choice.Delta.Content != nil && *choice.Delta.Content != "" {
		if d.reasoningStarted && !d.reasoningEnded {
			d.reasoningEnded = true
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventReasoningEnd, Index: 0, Content: d.reasoning.String()})
		}
		if err := d.buffer(&d.content, *choice.Delta.Content); err != nil {
			return false, err
		}
		if !d.textStarted {
			d.textStarted = true
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventTextStart, Index: 0})
		}
		chunk.Content = []llm.Part{{Text: *choice.Delta.Content}}
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventTextDelta, Index: 0, Delta: *choice.Delta.Content})
	}
	if choice.Delta.Refusal != nil {
		d.refusalSeen = true
		if err := d.buffer(&d.refusal, *choice.Delta.Refusal); err != nil {
			return false, err
		}
	}

	var finish llm.FinishReason
	var providerFinish string
	hasFinish := false
	if len(choice.FinishReason) != 0 {
		finish, providerFinish, hasFinish, err = decodeStreamFinishReason(choice.FinishReason)
		if err != nil {
			return false, err
		}
	}
	if hasFinish {
		if err := d.finishChunk(&chunk, finish, providerFinish); err != nil {
			return false, err
		}
	}
	if usage != nil {
		if !hasFinish && d.supportsFinishReason {
			return false, malformedStream("usage arrived before the finish reason")
		}
		if d.usageSeen {
			return false, malformedStream("event stream contains repeated usage")
		}
		d.usageSeen = true
		chunk.Usage = usage
	}

	if len(chunk.Content) != 0 || len(chunk.ToolCalls) != 0 || chunk.FinishReason != "" ||
		len(chunk.ProviderData) != 0 || chunk.Usage != nil || len(chunk.Events) != 0 {
		if !emit(chunk) {
			return false, errStreamEmitStopped
		}
	}
	return false, nil
}

func (d *streamDecoder) finishChunk(chunk *llm.Chunk, finish llm.FinishReason, providerFinish string) error {
	toolCalls, err := d.finishToolCalls(providerFinish)
	if err != nil {
		return err
	}
	d.finished = true
	chunk.Model = d.modelForChunk(true)
	chunk.ToolCalls = toolCalls
	chunk.FinishReason = finish
	if d.reasoningStarted && !d.reasoningEnded {
		d.reasoningEnded = true
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventReasoningEnd, Index: 0, Content: d.reasoning.String()})
	}
	if d.textStarted {
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventTextEnd, Index: 0, Content: d.content.String()})
	}
	for index := range toolCalls {
		call := cloneToolCallForEvent(toolCalls[index])
		chunk.Events = append(chunk.Events, llm.StreamEvent{
			Kind:       llm.StreamEventToolCallEnd,
			Index:      index,
			ToolCallID: call.ID,
			ToolName:   call.Name,
			ToolCall:   &call,
		})
	}
	chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventDone, FinishReason: finish})
	if d.format == llm.ResponseFormatJSON && finish == llm.FinishReasonStop && !d.refusalSeen &&
		!jsontext.Value(d.content.String()).IsValid() {
		return malformedStream("completed JSON response is not strict JSON")
	}
	if d.refusalSeen || d.reasoningStarted || d.reasoningDetailsSeen {
		data := wireMessageData{}
		if d.reasoningDetailsSeen {
			details := cloneRawMessages(d.reasoningDetails)
			data.ReasoningDetails = &details
		}
		if d.refusalSeen {
			refusal := d.refusal.String()
			data.Refusal = &refusal
		}
		if d.reasoningStarted {
			reasoning := d.reasoning.String()
			if d.reasoningField == "reasoning_content" {
				data.ReasoningContent = &reasoning
			} else {
				data.Reasoning = &reasoning
			}
		}
		providerData, marshalErr := marshalMessageData(data)
		if marshalErr != nil {
			return malformedStream("encode provider message state: %w", marshalErr)
		}
		chunk.ProviderData = providerData
	}
	return nil
}

func (d *streamDecoder) consumeToolDelta(delta streamDelta) ([]llm.StreamEvent, error) {
	var events []llm.StreamEvent
	modern := len(delta.ToolCalls) != 0
	legacy := delta.FunctionCall != nil
	if modern && legacy {
		return nil, malformedStream("event contains both tool_calls and function_call")
	}
	if modern {
		if d.toolMode == streamToolModeLegacy {
			return nil, malformedStream("event stream mixes tool_calls and function_call")
		}
		d.toolMode = streamToolModeModern
		for _, fragment := range delta.ToolCalls {
			if fragment.Index == nil {
				return nil, malformedStream("streamed tool call has no index")
			}
			index := *fragment.Index
			if index < 0 || index >= maxStreamToolCalls {
				return nil, malformedStream("streamed tool call index %d is outside [0, %d)", index, maxStreamToolCalls)
			}
			if len(d.toolCalls) <= index {
				d.toolCalls = append(d.toolCalls, make([]*streamToolCallBuilder, index-len(d.toolCalls)+1)...)
			}
			call := d.toolCalls[index]
			if call == nil {
				call = new(streamToolCallBuilder)
				d.toolCalls[index] = call
			}
			if err := d.appendFragment(&call.id, fragment.ID); err != nil {
				return nil, err
			}
			if err := d.appendFragment(&call.callType, fragment.Type); err != nil {
				return nil, err
			}
			if fragment.Function != nil {
				if err := d.consumeFunctionDelta(&call.function, *fragment.Function); err != nil {
					return nil, err
				}
			}
			if !call.started {
				call.started = true
				events = append(events, llm.StreamEvent{Kind: llm.StreamEventToolCallStart, Index: index, ToolCallID: call.id.String(), ToolName: call.function.name.String()})
			}
			if fragment.Function != nil && fragment.Function.Arguments != nil && *fragment.Function.Arguments != "" {
				events = append(events, llm.StreamEvent{Kind: llm.StreamEventToolCallDelta, Index: index, Delta: *fragment.Function.Arguments, ToolCallID: call.id.String(), ToolName: call.function.name.String()})
			}
		}
	}
	if legacy {
		if d.toolMode == streamToolModeModern {
			return nil, malformedStream("event stream mixes tool_calls and function_call")
		}
		d.toolMode = streamToolModeLegacy
		if err := d.consumeFunctionDelta(&d.legacyCall, *delta.FunctionCall); err != nil {
			return nil, err
		}
		if !d.legacyStarted {
			d.legacyStarted = true
			events = append(events, llm.StreamEvent{Kind: llm.StreamEventToolCallStart, Index: 0, ToolName: d.legacyCall.name.String()})
		}
		if delta.FunctionCall.Arguments != nil && *delta.FunctionCall.Arguments != "" {
			events = append(events, llm.StreamEvent{Kind: llm.StreamEventToolCallDelta, Index: 0, Delta: *delta.FunctionCall.Arguments, ToolName: d.legacyCall.name.String()})
		}
	}
	return events, nil
}

func (d *streamDecoder) consumeFunctionDelta(call *streamFunctionCallBuilder, delta streamFunctionCallDelta) error {
	if err := d.appendFragment(&call.name, delta.Name); err != nil {
		return err
	}
	return d.appendFragment(&call.arguments, delta.Arguments)
}

func cloneToolCallForEvent(call llm.ToolCall) llm.ToolCall {
	call.Arguments = append(json.RawMessage(nil), call.Arguments...)
	return call
}

func (d *streamDecoder) appendFragment(builder *strings.Builder, fragment *string) error {
	if fragment == nil {
		return nil
	}
	return d.buffer(builder, *fragment)
}

func (d *streamDecoder) finishToolCalls(providerFinish string) ([]llm.ToolCall, error) {
	switch d.toolMode {
	case streamToolModeNone:
		switch providerFinish {
		case "tool_calls":
			return nil, malformedStream("finish reason tool_calls has no tool calls")
		case "function_call":
			return nil, malformedStream("finish reason function_call has no function call")
		default:
			return nil, nil
		}
	case streamToolModeModern:
		if providerFinish != "tool_calls" {
			return nil, malformedStream("finish reason %q is inconsistent with streamed tool_calls", providerFinish)
		}
		return d.finishModernToolCalls()
	case streamToolModeLegacy:
		if providerFinish != "function_call" {
			return nil, malformedStream("finish reason %q is inconsistent with streamed function_call", providerFinish)
		}
		call, err := d.finishFunctionCall("", &d.legacyCall, "function call")
		if err != nil {
			return nil, err
		}
		return []llm.ToolCall{call}, nil
	default:
		panic("openai: invalid stream tool mode")
	}
}

func (d *streamDecoder) finishModernToolCalls() ([]llm.ToolCall, error) {
	if len(d.toolCalls) == 0 {
		return nil, malformedStream("finish reason tool_calls has no tool calls")
	}
	calls := make([]llm.ToolCall, len(d.toolCalls))
	ids := make(map[string]struct{}, len(d.toolCalls))
	for index, buffered := range d.toolCalls {
		if buffered == nil {
			return nil, malformedStream("streamed tool calls have no call at index %d", index)
		}
		id := buffered.id.String()
		if id == "" {
			return nil, malformedStream("tool call %d has no ID", index)
		}
		if _, exists := ids[id]; exists {
			return nil, malformedStream("tool call %d repeats ID %q", index, id)
		}
		ids[id] = struct{}{}
		if callType := buffered.callType.String(); callType != "function" {
			return nil, malformedStream("tool call %d has unsupported type %q", index, callType)
		}
		call, err := d.finishFunctionCall(id, &buffered.function, fmt.Sprintf("tool call %d", index))
		if err != nil {
			return nil, err
		}
		calls[index] = call
	}
	return calls, nil
}

func (d *streamDecoder) finishFunctionCall(id string, buffered *streamFunctionCallBuilder, label string) (llm.ToolCall, error) {
	call, err := functionCallValue(id, functionCall{
		Name:      buffered.name.String(),
		Arguments: buffered.arguments.String(),
	}, label)
	if err != nil {
		return llm.ToolCall{}, malformedStream("%w", err)
	}
	if _, declared := d.declaredTools[call.Name]; !declared {
		return llm.ToolCall{}, malformedStream("%s names undeclared function %q", label, call.Name)
	}
	return call, nil
}

func (d *streamDecoder) observeMetadata(id, model string) error {
	if id != "" {
		if d.id != "" && d.id != id {
			return malformedStream("event ID changed from %q to %q", d.id, id)
		}
		d.id = id
	}
	if model != "" {
		if d.model != "" && d.model != model {
			return malformedStream("event model changed from %q to %q", d.model, model)
		}
		d.model = model
	}
	return nil
}

func (d *streamDecoder) modelForChunk(fallback bool) string {
	if d.emittedModel != "" {
		return d.emittedModel
	}
	if d.model != "" {
		d.emittedModel = d.model
		return d.emittedModel
	}
	if fallback {
		d.emittedModel = d.configuredModel
	}
	return d.emittedModel
}

func (d *streamDecoder) appendReasoningDetails(details *[]json.RawMessage) error {
	if details == nil {
		return nil
	}
	d.reasoningDetailsSeen = true
	for _, detail := range *details {
		if len(detail) > maxResponseBodyBytes-d.bufferedBytes {
			return malformedStream("buffered stream output exceeds %d bytes", maxResponseBodyBytes)
		}
		d.reasoningDetails = append(d.reasoningDetails, append(json.RawMessage(nil), detail...))
		d.bufferedBytes += len(detail)
	}
	return nil
}

func (d *streamDecoder) buffer(builder *strings.Builder, fragment string) error {
	if len(fragment) > maxResponseBodyBytes-d.bufferedBytes {
		return malformedStream("buffered stream output exceeds %d bytes", maxResponseBodyBytes)
	}
	builder.WriteString(fragment)
	d.bufferedBytes += len(fragment)
	return nil
}

func decodeStreamFinishReason(raw json.RawMessage) (llm.FinishReason, string, bool, error) {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return "", "", false, nil
	}
	var reason string
	if err := jsonv2.Unmarshal(raw, &reason); err != nil {
		return "", "", false, malformedStream("decode finish reason: %w", err)
	}
	if reason == "insufficient_system_resource" {
		return "", "", false, providerInterruption("stream", reason)
	}
	finish, ok := finishReasonValue(reason)
	if !ok {
		return "", "", false, malformedStream("event has unknown finish reason %q", reason)
	}
	return finish, reason, true, nil
}

func malformedStream(format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindMalformedResponse,
		Op:       "stream",
		Provider: "openai",
		Err:      fmt.Errorf(format, args...),
	}
}

func streamTransportError(format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindTransport,
		Op:       "stream",
		Provider: "openai",
		Err:      fmt.Errorf(format, args...),
	}
}
