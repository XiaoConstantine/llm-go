package mistral

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
)

var errEmitStopped = errors.New("stream emission stopped")

type streamResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
	Usage   *streamUsage   `json:"usage"`
}

type streamChoice struct {
	Index        *int         `json:"index"`
	Delta        *streamDelta `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type streamDelta struct {
	Role      *string               `json:"role"`
	Content   json.RawMessage       `json:"content"`
	ToolCalls []streamToolCallDelta `json:"tool_calls"`
}

type streamToolCallDelta struct {
	Index    *int                 `json:"index"`
	ID       *string              `json:"id"`
	Type     *string              `json:"type"`
	Function *streamFunctionDelta `json:"function"`
}

type streamFunctionDelta struct {
	Name      *string         `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type streamUsage struct {
	PromptTokens     *int `json:"prompt_tokens"`
	CompletionTokens *int `json:"completion_tokens"`
	TotalTokens      *int `json:"total_tokens"`
}

type streamContentItem struct {
	Type     string         `json:"type"`
	Text     string         `json:"text"`
	Thinking []thinkingPart `json:"thinking"`
}

func (c *Client) produce(ctx context.Context, op string, request llm.Request, payload []byte, priorIDs map[string]struct{}, emit internalstream.Emit) (err error) {
	defer func() { err = relabelProviderError(err, c.provider) }()
	response, err := c.open(ctx, op, payload, request.SessionID)
	if err != nil {
		return err
	}
	var closeOnce sync.Once
	var closeErr error
	closeBody := func() { closeOnce.Do(func() { closeErr = response.Body.Close() }) }
	closingDone := make(chan struct{})
	stopClosing := context.AfterFunc(ctx, func() {
		closeBody()
		close(closingDone)
	})
	declared := make(map[string]struct{}, len(request.Tools))
	for _, tool := range request.Tools {
		declared[tool.Name] = struct{}{}
	}
	decoder := newStreamDecoder(op, c.model, declared, priorIDs)
	streamErr := readEventStream(ctx, response.Body, decoder, emit)
	if !stopClosing() {
		<-closingDone
	}
	closeBody()
	if contextErr := contextErr(ctx); contextErr != nil {
		return errors.Join(contextErr, closeError(op, closeErr))
	}
	if streamErr != nil {
		return errors.Join(streamErr, closeError(op, closeErr))
	}
	if closeErr != nil {
		return transportError(op, "close response body: %w", closeErr)
	}
	return nil
}

func closeError(op string, err error) error {
	if err == nil {
		return nil
	}
	return transportError(op, "close response body: %w", err)
}

func readEventStream(ctx context.Context, reader io.Reader, decoder *streamDecoder, emit internalstream.Emit) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, streamReadBufferBytes), maxStreamEventBytes+3)
	scanner.Split(splitLines)
	var data strings.Builder
	eventBytes := 0
	hasData := false
	firstLine := true
	dispatch := func() error {
		defer func() {
			data.Reset()
			eventBytes = 0
			hasData = false
		}()
		if !hasData {
			return nil
		}
		return decoder.consume(data.String(), emit)
	}
	for scanner.Scan() {
		line := scanner.Bytes()
		if firstLine {
			line = bytes.TrimPrefix(line, []byte("\xef\xbb\xbf"))
			firstLine = false
		}
		if err := contextErr(ctx); err != nil {
			return err
		}
		if len(line) == 0 {
			if err := dispatch(); err != nil {
				if errors.Is(err, errEmitStopped) {
					return nil
				}
				return err
			}
			continue
		}
		eventBytes += len(line) + 1
		if eventBytes > maxStreamEventBytes {
			return malformedStream(decoder.op, "event exceeds %d bytes", maxStreamEventBytes)
		}
		if line[0] == ':' {
			continue
		}
		field, value, found := bytes.Cut(line, []byte{':'})
		if !found || !bytes.Equal(field, []byte("data")) {
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
	if err := contextErr(ctx); err != nil {
		return err
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return malformedStream(decoder.op, "event exceeds %d bytes", maxStreamEventBytes)
		}
		return transportError(decoder.op, "read event stream: %w", err)
	}
	if hasData {
		if err := dispatch(); err != nil {
			if errors.Is(err, errEmitStopped) {
				return nil
			}
			return err
		}
	}
	if !decoder.finished {
		return malformedStream(decoder.op, "event stream ended before a finish reason")
	}
	return nil
}

func splitLines(data []byte, atEOF bool) (advance int, token []byte, err error) {
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

type outputKind uint8

const (
	outputNone outputKind = iota
	outputText
	outputReasoning
)

type streamDecoder struct {
	op              string
	configuredModel string
	declared        map[string]struct{}
	seenIDs         map[string]struct{}
	id              string
	model           string
	emittedModel    string
	started         bool
	finished        bool
	usageSeen       bool
	currentKind     outputKind
	currentIndex    int
	currentContent  strings.Builder
	bufferedBytes   int
	thinking        []string
	toolCalls       map[int]*toolCallBuilder
}

type toolCallBuilder struct {
	id        strings.Builder
	callType  strings.Builder
	name      strings.Builder
	arguments strings.Builder
	started   bool
}

func newStreamDecoder(op, model string, declared, priorIDs map[string]struct{}) *streamDecoder {
	seen := make(map[string]struct{}, len(priorIDs))
	for id := range priorIDs {
		seen[id] = struct{}{}
	}
	return &streamDecoder{op: op, configuredModel: model, declared: declared, seenIDs: seen, currentIndex: -1,
		toolCalls: make(map[int]*toolCallBuilder)}
}

func (d *streamDecoder) consume(data string, emit internalstream.Emit) error {
	if data == "[DONE]" {
		return nil
	}
	var event streamResponse
	if err := jsonv2.Unmarshal([]byte(data), &event); err != nil {
		return malformedStream(d.op, "decode event: %v", err)
	}
	if err := d.observeMetadata(event.ID, event.Model); err != nil {
		return err
	}
	usage, err := usageFromWire(event.Usage)
	if err != nil {
		return malformedStream(d.op, "decode usage: %v", err)
	}
	if len(event.Choices) == 0 {
		if usage == nil {
			return malformedStream(d.op, "event contains neither a choice nor usage")
		}
		if d.usageSeen {
			return malformedStream(d.op, "event stream contains repeated usage")
		}
		d.usageSeen = true
		chunk := llm.Chunk{ID: d.id, Model: d.modelForChunk(d.finished), Usage: usage}
		d.start(&chunk)
		if !emit(chunk) {
			return errEmitStopped
		}
		return nil
	}
	if len(event.Choices) != 1 {
		return malformedStream(d.op, "event contains %d choices, want one", len(event.Choices))
	}
	if d.finished {
		return malformedStream(d.op, "event contains a choice after the finish reason")
	}
	choice := event.Choices[0]
	if choice.Index != nil && *choice.Index != 0 {
		return malformedStream(d.op, "event choice has index %d, want 0", *choice.Index)
	}
	if choice.Delta == nil {
		return malformedStream(d.op, "event choice has no delta")
	}
	if choice.Delta.Role != nil && *choice.Delta.Role != string(llm.RoleAssistant) {
		return malformedStream(d.op, "event delta role is %q, want assistant", *choice.Delta.Role)
	}
	chunk := llm.Chunk{ID: d.id, Model: d.modelForChunk(false)}
	d.start(&chunk)
	items, err := contentItems(choice.Delta.Content)
	if err != nil {
		return malformedStream(d.op, "decode content delta: %v", err)
	}
	for _, item := range items {
		switch item.kind {
		case outputText:
			if err := d.contentDelta(&chunk, outputText, item.text); err != nil {
				return err
			}
		case outputReasoning:
			if err := d.contentDelta(&chunk, outputReasoning, item.text); err != nil {
				return err
			}
		}
	}
	if len(choice.Delta.ToolCalls) != 0 {
		d.finishCurrent(&chunk)
		if err := d.toolDeltas(&chunk, choice.Delta.ToolCalls); err != nil {
			return err
		}
	}
	if choice.FinishReason != nil && *choice.FinishReason != "" {
		if err := d.finish(&chunk, *choice.FinishReason, usage); err != nil {
			return err
		}
	} else if usage != nil {
		if d.usageSeen {
			return malformedStream(d.op, "event stream contains repeated usage")
		}
		d.usageSeen = true
		chunk.Usage = usage
	}
	if len(chunk.Content) != 0 || len(chunk.ToolCalls) != 0 || len(chunk.Events) != 0 || chunk.FinishReason != "" || chunk.Usage != nil || len(chunk.ProviderData) != 0 {
		if !emit(chunk) {
			return errEmitStopped
		}
	}
	return nil
}

type decodedContent struct {
	kind outputKind
	text string
}

func contentItems(raw json.RawMessage) ([]decodedContent, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	if jsontext.Value(trimmed).Kind() == jsontext.KindString {
		var text string
		if err := jsonv2.Unmarshal(trimmed, &text); err != nil {
			return nil, err
		}
		if text == "" {
			return nil, nil
		}
		return []decodedContent{{kind: outputText, text: text}}, nil
	}
	var items []streamContentItem
	if err := jsonv2.Unmarshal(trimmed, &items); err != nil {
		return nil, err
	}
	output := make([]decodedContent, 0, len(items))
	for index, item := range items {
		switch item.Type {
		case "text":
			if item.Text != "" {
				output = append(output, decodedContent{kind: outputText, text: item.Text})
			}
		case "thinking":
			var text strings.Builder
			for _, part := range item.Thinking {
				if part.Type != "text" {
					return nil, fmt.Errorf("thinking item %d has part type %q", index, part.Type)
				}
				text.WriteString(part.Text)
			}
			if text.Len() != 0 {
				output = append(output, decodedContent{kind: outputReasoning, text: text.String()})
			}
		default:
			return nil, fmt.Errorf("content item %d has unsupported type %q", index, item.Type)
		}
	}
	return output, nil
}

func (d *streamDecoder) start(chunk *llm.Chunk) {
	if !d.started {
		d.started = true
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventStart})
	}
}

func (d *streamDecoder) contentDelta(chunk *llm.Chunk, kind outputKind, text string) error {
	if text == "" {
		return nil
	}
	if d.currentKind != kind {
		d.finishCurrent(chunk)
		d.currentKind = kind
		d.currentIndex++
		startKind := llm.StreamEventTextStart
		if kind == outputReasoning {
			startKind = llm.StreamEventReasoningStart
		}
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: startKind, Index: d.currentIndex})
	}
	if len(text) > maxResponseBodyBytes-d.bufferedBytes {
		return malformedStream(d.op, "buffered content exceeds %d bytes", maxResponseBodyBytes)
	}
	d.bufferedBytes += len(text)
	d.currentContent.WriteString(text)
	if kind == outputText {
		chunk.Content = append(chunk.Content, llm.Part{Kind: llm.PartText, Text: text})
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventTextDelta, Index: d.currentIndex, Delta: text})
	} else {
		chunk.ReasoningSummary += text
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventReasoningDelta, Index: d.currentIndex, Delta: text})
	}
	return nil
}

func (d *streamDecoder) finishCurrent(chunk *llm.Chunk) {
	if d.currentKind == outputNone {
		return
	}
	content := d.currentContent.String()
	endKind := llm.StreamEventTextEnd
	if d.currentKind == outputReasoning {
		endKind = llm.StreamEventReasoningEnd
		d.thinking = append(d.thinking, content)
	}
	chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: endKind, Index: d.currentIndex, Content: content})
	d.currentContent.Reset()
	d.currentKind = outputNone
}

func (d *streamDecoder) toolDeltas(chunk *llm.Chunk, deltas []streamToolCallDelta) error {
	for _, delta := range deltas {
		index := 0
		if delta.Index != nil {
			index = *delta.Index
		}
		if index < 0 || index >= maxStreamToolCalls {
			return malformedStream(d.op, "tool call index %d is outside [0,%d)", index, maxStreamToolCalls)
		}
		builder := d.toolCalls[index]
		if builder == nil {
			builder = new(toolCallBuilder)
			d.toolCalls[index] = builder
		}
		if delta.ID != nil && *delta.ID != "" && *delta.ID != "null" {
			builder.id.WriteString(*delta.ID)
		}
		if delta.Type != nil {
			builder.callType.WriteString(*delta.Type)
		}
		var arguments string
		if delta.Function != nil {
			if delta.Function.Name != nil {
				builder.name.WriteString(*delta.Function.Name)
			}
			var err error
			arguments, err = argumentFragment(delta.Function.Arguments)
			if err != nil {
				return malformedStream(d.op, "tool call %d arguments: %v", index, err)
			}
			if len(arguments) > maxResponseBodyBytes-d.bufferedBytes {
				return malformedStream(d.op, "tool call %d arguments exceed %d bytes", index, maxResponseBodyBytes)
			}
			d.bufferedBytes += len(arguments)
			builder.arguments.WriteString(arguments)
		}
		if !builder.started {
			builder.started = true
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventToolCallStart, Index: index,
				ToolCallID: builder.id.String(), ToolName: builder.name.String()})
		}
		if arguments != "" {
			chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventToolCallDelta, Index: index,
				Delta: arguments, ToolCallID: builder.id.String(), ToolName: builder.name.String()})
		}
	}
	return nil
}

func argumentFragment(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return "", nil
	}
	if jsontext.Value(trimmed).Kind() == jsontext.KindString {
		var value string
		if err := jsonv2.Unmarshal(trimmed, &value); err != nil {
			return "", err
		}
		return value, nil
	}
	if !jsontext.Value(trimmed).IsValid() {
		return "", errors.New("value is not valid JSON")
	}
	return string(trimmed), nil
}

func (d *streamDecoder) finish(chunk *llm.Chunk, providerReason string, usage *llm.Usage) error {
	d.finishCurrent(chunk)
	finish, err := finishReason(d.op, providerReason)
	if err != nil {
		return err
	}
	toolCalls, err := d.completeToolCalls(finish)
	if err != nil {
		return err
	}
	chunk.ToolCalls = toolCalls
	chunk.FinishReason = finish
	chunk.ID = d.id
	chunk.Model = d.modelForChunk(true)
	if usage != nil {
		if d.usageSeen {
			return malformedStream(d.op, "event stream contains repeated usage")
		}
		d.usageSeen = true
		chunk.Usage = usage
	}
	providerData, err := marshalMessageData(d.configuredModel, d.thinking)
	if err != nil {
		return malformedStream(d.op, "encode thinking replay: %v", err)
	}
	chunk.ProviderData = providerData
	for index := range toolCalls {
		call := toolCalls[index]
		call.Arguments = append(json.RawMessage(nil), call.Arguments...)
		chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventToolCallEnd, Index: index,
			ToolCallID: call.ID, ToolName: call.Name, ToolCall: &call})
	}
	chunk.Events = append(chunk.Events, llm.StreamEvent{Kind: llm.StreamEventDone, FinishReason: finish})
	d.finished = true
	return nil
}

func (d *streamDecoder) completeToolCalls(finish llm.FinishReason) ([]llm.ToolCall, error) {
	if len(d.toolCalls) == 0 {
		if finish == llm.FinishReasonToolCall {
			return nil, malformedStream(d.op, "finish reason tool_calls has no tool calls")
		}
		return nil, nil
	}
	if finish != llm.FinishReasonToolCall {
		if finish == llm.FinishReasonLength {
			return nil, unsupported(d.op, "truncated Mistral tool calls are not representable")
		}
		return nil, malformedStream(d.op, "finish reason is inconsistent with streamed tool calls")
	}
	maxIndex := -1
	for index := range d.toolCalls {
		maxIndex = max(maxIndex, index)
	}
	calls := make([]llm.ToolCall, maxIndex+1)
	for index := range calls {
		builder := d.toolCalls[index]
		if builder == nil {
			return nil, malformedStream(d.op, "tool calls have no call at index %d", index)
		}
		id := builder.id.String()
		if id == "" || id == "null" {
			id = newToolCallIDNormalizer().normalize(fmt.Sprintf("toolcall:%d", index))
		}
		if _, duplicate := d.seenIDs[id]; duplicate {
			return nil, malformedStream(d.op, "tool call %d repeats ID %q", index, id)
		}
		d.seenIDs[id] = struct{}{}
		if callType := builder.callType.String(); callType != "" && callType != "function" {
			return nil, malformedStream(d.op, "tool call %d has type %q", index, callType)
		}
		name := builder.name.String()
		if name == "" {
			return nil, malformedStream(d.op, "tool call %d has no function name", index)
		}
		if _, declared := d.declared[name]; !declared {
			return nil, malformedStream(d.op, "tool call %d names undeclared function %q", index, name)
		}
		arguments := json.RawMessage(builder.arguments.String())
		if len(arguments) == 0 || !jsontext.Value(arguments).IsValid() {
			return nil, malformedStream(d.op, "tool call %d arguments are not complete JSON", index)
		}
		calls[index] = llm.ToolCall{ID: id, Name: name, Arguments: append(json.RawMessage(nil), arguments...)}
	}
	return calls, nil
}

func finishReason(op, reason string) (llm.FinishReason, error) {
	switch reason {
	case "stop", "":
		return llm.FinishReasonStop, nil
	case "length", "model_length":
		return llm.FinishReasonLength, nil
	case "tool_calls":
		return llm.FinishReasonToolCall, nil
	case "error":
		return "", &llm.Error{Kind: llm.KindProvider, Op: op, Provider: defaultProvider, Err: errors.New("Mistral generation stopped with an error")}
	default:
		return llm.FinishReasonStop, nil
	}
}

func usageFromWire(wire *streamUsage) (*llm.Usage, error) {
	if wire == nil {
		return nil, nil
	}
	input, output := 0, 0
	if wire.PromptTokens != nil {
		input = *wire.PromptTokens
	}
	if wire.CompletionTokens != nil {
		output = *wire.CompletionTokens
	}
	total := input + output
	if wire.TotalTokens != nil {
		total = *wire.TotalTokens
	}
	if input < 0 || output < 0 || total < 0 || total < input+output {
		return nil, errors.New("token counts are inconsistent")
	}
	return &llm.Usage{InputTokens: input, OutputTokens: output, TotalTokens: total}, nil
}

func (d *streamDecoder) observeMetadata(id, model string) error {
	if id != "" {
		if d.id != "" && d.id != id {
			return malformedStream(d.op, "response ID changed from %q to %q", d.id, id)
		}
		d.id = id
	}
	if model != "" {
		if d.model != "" && d.model != model {
			return malformedStream(d.op, "response model changed from %q to %q", d.model, model)
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
		return d.model
	}
	if fallback {
		d.emittedModel = d.configuredModel
	}
	return d.emittedModel
}
