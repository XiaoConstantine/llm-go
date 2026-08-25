package codex

import (
	"context"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	internalresponses "github.com/XiaoConstantine/llm-go/internal/openairesponses"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	openaioption "github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/ssestream"
	openairesponses "github.com/openai/openai-go/v3/responses"
)

var errRawErrorResponse = errors.New("Codex SDK returned a raw error response")

// APIError is a provider error reported by the Codex Responses endpoint.
type APIError = internalresponses.APIError

type sdkResponseStream = ssestream.Stream[openairesponses.ResponseStreamEventUnion]

func (c *Client) produce(ctx context.Context, op string, params openairesponses.ResponseNewParams, sessionID string, emit internalstream.Emit) error {
	if c.transport == TransportSSE {
		return c.produceSSE(ctx, op, params, sessionID, emit)
	}
	err := c.produceWebSocket(ctx, op, params, sessionID, emit)
	if c.transport != TransportAuto || err == nil {
		return err
	}
	var connectFailure *webSocketConnectFailure
	if !errors.As(err, &connectFailure) || contextErr(ctx) != nil {
		return err
	}
	return c.produceSSE(ctx, op, params, sessionID, emit)
}

func (c *Client) produceSSE(ctx context.Context, op string, params openairesponses.ResponseNewParams, sessionID string, emit internalstream.Emit) error {
	open := func(ctx context.Context, op string, params openairesponses.ResponseNewParams) (*sdkResponseStream, error) {
		return c.openStream(ctx, op, params, sessionID)
	}
	return responseCodec().Produce(ctx, op, c.model, params, open, emit)
}

func (c *Client) openStream(ctx context.Context, op string, params openairesponses.ResponseNewParams, sessionID string) (*sdkResponseStream, error) {
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
			c.requestMiddleware(credentials, sessionID, &captured, &downstreamErr),
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

func (c *Client) requestMiddleware(credentials Credentials, sessionID string, captured **http.Response, downstreamErr *error) openaioption.Middleware {
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
		if sessionID != "" {
			request.Header.Set("Session-Id", sessionID)
			request.Header.Set("X-Client-Request-Id", sessionID)
		}

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
		Retryable:  retryableHint(response.Header.Get("X-Should-Retry")),
		Err:        apiErr,
	}
}

func decodeAPIError(body []byte) *APIError {
	var envelope struct {
		Error APIError `json:"error"`
	}
	if err := jsonv2.Unmarshal(body, &envelope); err == nil && !apiErrorEmpty(&envelope.Error) {
		return &envelope.Error
	}
	var apiErr APIError
	if err := jsonv2.Unmarshal(body, &apiErr); err == nil && !apiErrorEmpty(&apiErr) {
		return &apiErr
	}
	return nil
}

func apiErrorEmpty(err *APIError) bool {
	return err == nil || err.Type == "" && err.Code == "" && err.Message == ""
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

func retryableHint(value string) *bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "true":
		value := true
		return &value
	case "false":
		value := false
		return &value
	default:
		return nil
	}
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
