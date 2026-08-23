package codex

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	openaioption "github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	openairesponses "github.com/openai/openai-go/v3/responses"
)

var errRawErrorResponse = errors.New("Codex SDK returned a raw error response")

// APIError is a provider error reported by the Codex Responses endpoint.
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
	return "Codex provider error"
}

type sdkResponseStream = ssestream.Stream[openairesponses.ResponseStreamEventUnion]

func (c *Client) produce(ctx context.Context, op string, params openairesponses.ResponseNewParams, emit internalstream.Emit) error {
	stream, err := c.openStream(ctx, op, params)
	if err != nil {
		return err
	}
	state := streamState{
		op:            op,
		defaultModel:  c.model,
		declaredTools: declaredToolNames(params),
		outputItems:   make(map[int]jsontext.Value),
		pendingItems:  make(map[int]outputItemIdentity),
		toolCalls:     make(map[int]llm.ToolCall),
	}

	for stream.Next() {
		if err := contextErr(ctx); err != nil {
			return closeSDKStream(op, stream, err)
		}
		if err := state.consume(stream.Current(), emit); err != nil {
			return closeSDKStream(op, stream, err)
		}
		if state.stopped {
			return closeSDKStream(op, stream, nil)
		}
		if state.terminal {
			return closeSDKStream(op, stream, nil)
		}
	}
	err = stream.Err()
	if contextErr := contextErr(ctx); contextErr != nil {
		err = contextErr
	} else if err != nil {
		err = streamReadError(op, err)
	} else if !state.terminal {
		err = malformedResponse(op, "event stream ended before a terminal response")
	}
	return closeSDKStream(op, stream, err)
}

func streamReadError(op string, err error) error {
	var providerEvent *ssestream.StreamError
	if errors.As(err, &providerEvent) {
		return eventError(op, providerEvent.Event.Data)
	}
	var syntaxErr *json.SyntaxError
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &syntaxErr) || errors.As(err, &typeErr) || errors.Is(err, bufio.ErrTooLong) {
		return malformedResponse(op, "decode event stream: %v", err)
	}
	return transportError(op, fmt.Errorf("read event stream: %w", err))
}

func (c *Client) openStream(ctx context.Context, op string, params openairesponses.ResponseNewParams) (*sdkResponseStream, error) {
	rejected := ""
	var rejectedErr error
	for attempt := 0; attempt < 2; attempt++ {
		credentials, err := c.resolveCredentials(ctx, rejected)
		if err != nil {
			if contextErr := contextErr(ctx); contextErr != nil {
				return nil, contextErr
			}
			credentialErr := authenticationError(op, fmt.Errorf("resolve credentials: %w", err))
			if rejectedErr != nil {
				return nil, errors.Join(rejectedErr, credentialErr)
			}
			return nil, credentialErr
		}
		credentials, err = normalizeCredentials(credentials)
		if err != nil {
			credentialErr := authenticationError(op, err)
			if rejectedErr != nil {
				return nil, errors.Join(rejectedErr, credentialErr)
			}
			return nil, credentialErr
		}
		if rejected != "" && credentials.AccessToken == rejected {
			return nil, rejectedErr
		}

		var captured *http.Response
		var downstreamErr error
		stream := c.responses.NewStreaming(ctx, params, openaioption.WithMiddleware(
			c.requestMiddleware(credentials, &captured, &downstreamErr),
		))
		if stream.Err() == nil {
			return stream, nil
		}

		status := 0
		if captured != nil {
			status = captured.StatusCode
		}
		requestErr := c.initialStreamError(ctx, op, stream, captured, downstreamErr)
		if status != http.StatusUnauthorized || attempt != 0 {
			return nil, requestErr
		}
		rejected = credentials.AccessToken
		rejectedErr = requestErr
	}
	return nil, authenticationError(op, errors.New("credentials were rejected"))
}

func normalizeCredentials(credentials Credentials) (Credentials, error) {
	credentials.AccessToken = strings.TrimSpace(credentials.AccessToken)
	credentials.AccountID = strings.TrimSpace(credentials.AccountID)
	if credentials.AccessToken == "" {
		return Credentials{}, errors.New("credential resolver returned an empty access token")
	}
	if credentials.AccountID == "" {
		accountID, err := AccountIDFromToken(credentials.AccessToken)
		if err != nil {
			return Credentials{}, fmt.Errorf("resolve account ID from access token: %w", err)
		}
		credentials.AccountID = accountID
	}
	return credentials, nil
}

func (c *Client) requestMiddleware(credentials Credentials, captured **http.Response, downstreamErr *error) openaioption.Middleware {
	return func(request *http.Request, next openaioption.MiddlewareNext) (*http.Response, error) {
		for key := range request.Header {
			delete(request.Header, key)
		}
		for key, values := range c.headers {
			for _, value := range values {
				request.Header.Add(key, value)
			}
		}
		request.Header.Set("Authorization", "Bearer "+credentials.AccessToken)
		request.Header.Set("ChatGPT-Account-ID", credentials.AccountID)
		request.Header.Set("OpenAI-Beta", "responses=experimental")
		request.Header.Set("Originator", c.originator)
		request.Header.Set("User-Agent", "llm-go")
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("Accept", "text/event-stream")

		response, err := next(request)
		*captured = response
		*downstreamErr = err
		if err == nil && response != nil && response.StatusCode >= http.StatusBadRequest {
			return response, errRawErrorResponse
		}
		return response, err
	}
}

func (c *Client) initialStreamError(ctx context.Context, op string, stream *sdkResponseStream, response *http.Response, downstreamErr error) error {
	err := stream.Err()
	if contextErr := contextErr(ctx); contextErr != nil {
		return closeSDKStream(op, stream, contextErr)
	}
	if response == nil || response.Body == nil || !errors.Is(err, errRawErrorResponse) {
		if downstreamErr != nil {
			err = downstreamErr
		}
		return closeSDKStream(op, stream, transportError(op, err))
	}

	body, tooLarge, readErr := readLimited(response.Body, maxErrorBodyBytes)
	if readErr != nil {
		return closeSDKStream(op, stream, transportError(op, fmt.Errorf("read error response: %w", readErr)))
	}
	providerErr := responseError(op, response, body, tooLarge)
	return closeSDKStream(op, stream, providerErr)
}

func closeSDKStream(op string, stream *sdkResponseStream, err error) error {
	if closeErr := stream.Close(); closeErr != nil {
		closeErr = transportError(op, fmt.Errorf("close event stream: %w", closeErr))
		if err != nil {
			return errors.Join(err, closeErr)
		}
		return closeErr
	}
	return err
}

func readLimited(reader io.Reader, limit int64) ([]byte, bool, error) {
	limited := &io.LimitedReader{R: reader, N: limit + 1}
	body, err := io.ReadAll(limited)
	if err != nil {
		return nil, false, err
	}
	if int64(len(body)) <= limit {
		return body, false, nil
	}
	return body[:limit], true, nil
}

func responseError(op string, response *http.Response, body []byte, tooLarge bool) *llm.Error {
	apiErr := decodeAPIError(body)
	if apiErr == nil {
		message := strings.TrimSpace(string(body))
		if tooLarge {
			message = fmt.Sprintf("response body exceeds %d bytes", maxErrorBodyBytes)
		} else if message == "" {
			message = response.Status
		} else if len(message) > 4096 {
			message = message[:4096] + "..."
		}
		apiErr = &APIError{Message: message}
	}
	return &llm.Error{
		Kind:       classifyResponse(response.StatusCode, apiErr),
		Op:         op,
		Provider:   defaultProvider,
		HTTPStatus: response.StatusCode,
		RetryAfter: retryAfter(response.Header.Get("Retry-After")),
		Err:        apiErr,
	}
}

func decodeAPIError(body []byte) *APIError {
	var envelope struct {
		Error APIError `json:"error"`
	}
	if err := jsonv2.Unmarshal(body, &envelope); err == nil && !envelope.Error.empty() {
		return &envelope.Error
	}
	var apiErr APIError
	if err := jsonv2.Unmarshal(body, &apiErr); err == nil && !apiErr.empty() {
		return &apiErr
	}
	return nil
}

func (e *APIError) empty() bool {
	return e == nil || e.Type == "" && e.Code == "" && e.Message == ""
}

func classifyResponse(status int, apiErr *APIError) llm.ErrorKind {
	switch status {
	case http.StatusUnauthorized:
		return llm.KindAuthentication
	case http.StatusForbidden:
		return llm.KindPermission
	case http.StatusTooManyRequests:
		return llm.KindRateLimit
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		if isContextLimitError(apiErr) {
			return llm.KindContextLimit
		}
		return llm.KindInvalidRequest
	case http.StatusNotFound, http.StatusMethodNotAllowed:
		return llm.KindInvalidRequest
	default:
		return llm.KindProvider
	}
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

func retryAfter(value string) time.Duration {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0
	}
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil {
		if seconds <= 0 {
			return 0
		}
		const maxSeconds = int64(^uint64(0)>>1) / int64(time.Second)
		if seconds > maxSeconds {
			return time.Duration(^uint64(0) >> 1)
		}
		return time.Duration(seconds) * time.Second
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0
	}
	delay := time.Until(when)
	if delay <= 0 {
		return 0
	}
	return delay
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
		return malformedResponse(s.op, "event contains invalid JSON")
	}

	switch event.Type {
	case "response.output_text.delta":
		if event.Delta != "" && !emit(llm.Chunk{Content: []llm.Part{{Kind: llm.PartText, Text: event.Delta}}}) {
			s.stopped = true
		}
	case "response.output_item.added":
		return s.consumeAddedOutputItem(event)
	case "response.output_item.done", "response.output_item.completed":
		if err := s.consumeOutputItem(event); err != nil {
			return err
		}
	case "error":
		return eventError(s.op, raw)
	case "response.failed":
		return failedResponseError(s.op, raw)
	case "response.completed", "response.done":
		if !event.JSON.Response.Valid() {
			return malformedResponse(s.op, "terminal event is missing response")
		}
		return s.finish(event.Response, llm.FinishReasonStop, emit)
	case "response.incomplete":
		if !event.JSON.Response.Valid() {
			return malformedResponse(s.op, "incomplete event is missing response")
		}
		switch event.Response.IncompleteDetails.Reason {
		case "max_output_tokens":
			return s.finish(event.Response, llm.FinishReasonLength, emit)
		case "content_filter":
			return s.finish(event.Response, llm.FinishReasonContentFilter, emit)
		default:
			return &llm.Error{Kind: llm.KindProvider, Op: s.op, Provider: defaultProvider, Err: fmt.Errorf("response incomplete: %s", event.Response.IncompleteDetails.Reason)}
		}
	}
	return nil
}

func (s *streamState) consumeOutputItem(event openairesponses.ResponseStreamEventUnion) error {
	index, err := outputIndex(s.op, event, "completed")
	if err != nil {
		return err
	}
	raw := jsontext.Value(event.Item.RawJSON())
	if len(raw) == 0 || raw.Kind() != jsontext.KindBeginObject {
		return malformedResponse(s.op, "completed output item %d is missing an object", index)
	}
	raw = append(jsontext.Value(nil), raw...)
	identity, err := decodeOutputItemIdentity(raw)
	if err != nil {
		return malformedResponse(s.op, "decode output item %d identity: %v", index, err)
	}
	if pending, exists := s.pendingItems[index]; exists {
		if !pending.matches(identity) {
			return malformedResponse(s.op, "output item %d changed identity before completion", index)
		}
		delete(s.pendingItems, index)
	}
	if previous, exists := s.outputItems[index]; exists {
		if !bytes.Equal(previous, raw) {
			return malformedResponse(s.op, "output index %d was reused for a different item", index)
		}
	} else {
		if s.providerBytes > maxProviderDataBytes-len(raw) {
			return malformedResponse(s.op, "provider data exceeds %d bytes", maxProviderDataBytes)
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
		return malformedResponse(s.op, "decode output item %d: %v", index, err)
	}
	if identity.Type != "function_call" {
		return nil
	}
	if _, exists := s.toolCalls[index]; exists {
		return nil
	}
	if item.CallID == "" || item.Name == "" {
		return malformedResponse(s.op, "function call at output index %d is missing call_id or name", index)
	}
	if _, declared := s.declaredTools[item.Name]; !declared {
		return malformedResponse(s.op, "function call at output index %d names undeclared tool %q", index, item.Name)
	}
	arguments, err := functionArguments(item.Arguments)
	if err != nil {
		return malformedResponse(s.op, "function call at output index %d arguments: %v", index, err)
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
	index, err := outputIndex(s.op, event, "added")
	if err != nil {
		return err
	}
	raw := jsontext.Value(event.Item.RawJSON())
	if len(raw) == 0 || raw.Kind() != jsontext.KindBeginObject {
		return malformedResponse(s.op, "added output item %d is missing an object", index)
	}
	identity, err := decodeOutputItemIdentity(raw)
	if err != nil {
		return malformedResponse(s.op, "decode added output item %d identity: %v", index, err)
	}
	if _, exists := s.pendingItems[index]; exists {
		return malformedResponse(s.op, "output item %d was added more than once", index)
	}
	if _, completed := s.outputItems[index]; completed {
		return malformedResponse(s.op, "output item %d was added after completion", index)
	}
	s.pendingItems[index] = identity
	return nil
}

func outputIndex(op string, event openairesponses.ResponseStreamEventUnion, state string) (int, error) {
	if !event.JSON.OutputIndex.Valid() {
		return 0, malformedResponse(op, "%s output item is missing output_index", state)
	}
	if event.OutputIndex < 0 || event.OutputIndex > int64(^uint(0)>>1) {
		return 0, malformedResponse(op, "output item has invalid index %d", event.OutputIndex)
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
		return malformedResponse(s.op, "received more than one terminal response")
	}
	hasPendingItems := len(s.pendingItems) != 0
	if reason == llm.FinishReasonStop && hasPendingItems {
		return malformedResponse(s.op, "terminal response has %d unfinished output items", len(s.pendingItems))
	}
	status := string(response.Status)
	wantStatus := "completed"
	if reason != llm.FinishReasonStop {
		wantStatus = "incomplete"
	}
	if status != wantStatus {
		return malformedResponse(s.op, "terminal response has status %q", status)
	}
	var providerData jsontext.Value
	if !hasPendingItems {
		if err := s.backfillReasoning(response.Output); err != nil {
			return malformedResponse(s.op, "merge terminal response output: %v", err)
		}
		var err error
		providerData, err = marshalProviderData(s.defaultModel, s.outputItems)
		if err != nil {
			return malformedResponse(s.op, "encode provider data: %v", err)
		}
	}
	usage, err := responseUsage(response)
	if err != nil {
		return malformedResponse(s.op, "usage: %v", err)
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
		if remaining > maxProviderDataBytes-len(enriched) {
			return fmt.Errorf("provider data exceeds %d bytes", maxProviderDataBytes)
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

func marshalProviderData(model string, items map[int]jsontext.Value) (jsontext.Value, error) {
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
		API:     providerDataAPI,
		Version: providerDataVersion,
		Model:   model,
		Output:  ordered,
	})
	if err != nil {
		return nil, err
	}
	if len(data) > maxProviderDataBytes {
		return nil, fmt.Errorf("exceeds %d bytes", maxProviderDataBytes)
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

func eventError(op string, raw []byte) error {
	var event struct {
		Type    string   `json:"type"`
		Code    string   `json:"code"`
		Message string   `json:"message"`
		Error   APIError `json:"error"`
	}
	if err := jsonv2.Unmarshal(raw, &event); err != nil {
		return malformedResponse(op, "decode error event: %v", err)
	}
	apiErr := &event.Error
	if apiErr.empty() {
		apiErr = &APIError{Type: event.Type, Code: event.Code, Message: event.Message}
	}
	return &llm.Error{Kind: classifyAPIError(apiErr), Op: op, Provider: defaultProvider, Err: apiErr}
}

func failedResponseError(op string, raw []byte) error {
	var event struct {
		Response struct {
			Error APIError `json:"error"`
		} `json:"response"`
	}
	if err := jsonv2.Unmarshal(raw, &event); err != nil {
		return malformedResponse(op, "decode failed response: %v", err)
	}
	apiErr := &event.Response.Error
	if apiErr.empty() {
		return malformedResponse(op, "failed response is missing error details")
	}
	return &llm.Error{Kind: classifyAPIError(apiErr), Op: op, Provider: defaultProvider, Err: apiErr}
}
