package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	openaisdk "github.com/openai/openai-go/v3"
	openaioption "github.com/openai/openai-go/v3/option"
)

const (
	defaultProvider      = "openai"
	defaultBaseURL       = "https://api.openai.com/v1"
	maxTools             = 128
	maxResponseBodyBytes = 16 << 20
	maxErrorBodyBytes    = 1 << 20
)

// MaxTokensField identifies the request field used for an output-token limit.
type MaxTokensField string

const (
	// MaxTokensFieldCompletion uses max_completion_tokens, the default for
	// current OpenAI-compatible APIs.
	MaxTokensFieldCompletion MaxTokensField = "max_completion_tokens"
	// MaxTokensFieldLegacy uses max_tokens for providers such as DeepSeek.
	MaxTokensFieldLegacy MaxTokensField = "max_tokens"
)

// Config configures an OpenAI-compatible client. Model is required. Provider
// identifies the service in ModelInfo and errors; it defaults to "openai".
// BaseURL defaults to https://api.openai.com/v1. APIKey is optional so the
// client can be used with compatible servers that do not authenticate.
// Capabilities opts the configured model into optional protocol features;
// generation is always enabled, while streaming, tools, JSON mode, and vision
// must be listed explicitly. A non-nil HTTPClient and custom Headers are used
// as supplied, without being mutated by Client. A nil HTTPClient uses
// [http.DefaultClient], which has no overall request timeout; callers should use
// context deadlines or configure a client timeout. Content-Type is always
// application/json.
type Config struct {
	Provider     string
	Model        string
	Capabilities []llm.Capability
	APIKey       string
	BaseURL      string
	HTTPClient   *http.Client
	Headers      http.Header
}

// Options configures protocol compatibility for NewWithOptions. Its zero value
// uses current OpenAI request fields.
type Options struct {
	MaxTokensField MaxTokensField
}

// Client is an immutable OpenAI-compatible client. It is safe for concurrent
// use when its configured HTTP client is safe for concurrent use.
type Client struct {
	provider       string
	model          string
	capabilities   []llm.Capability
	headers        http.Header
	maxTokensField MaxTokensField
	sdkChatPath    string
	sdkClient      openaisdk.Client
}

// New constructs a Client from config using current OpenAI request fields.
func New(config Config) (*Client, error) {
	return NewWithOptions(config, Options{})
}

// NewWithOptions constructs a Client with explicit protocol compatibility.
func NewWithOptions(config Config, options Options) (_ *Client, err error) {
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = defaultProvider
	}
	defer func() {
		err = relabelProviderError(err, provider)
	}()

	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, configError("model must not be empty")
	}
	capabilities, err := configureCapabilities(config.Capabilities)
	if err != nil {
		return nil, err
	}
	maxTokensField := options.MaxTokensField
	if maxTokensField == "" {
		maxTokensField = MaxTokensFieldCompletion
	}
	switch maxTokensField {
	case MaxTokensFieldCompletion, MaxTokensFieldLegacy:
	default:
		return nil, configError("max tokens field %q is not supported", maxTokensField)
	}

	baseURL := strings.TrimSpace(config.BaseURL)
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return nil, configError("invalid base URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, configError("base URL scheme must be http or https")
	}
	if parsed.Host == "" {
		return nil, configError("base URL must be absolute")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, configError("base URL must not contain a query or fragment")
	}
	chatEndpoint, err := url.JoinPath(baseURL, "chat/completions")
	if err != nil {
		return nil, configError("invalid base URL: %w", err)
	}
	chatURL, err := url.Parse(chatEndpoint)
	if err != nil {
		return nil, configError("invalid chat completions URL: %w", err)
	}
	apiRootReference, _ := url.Parse("../")
	sdkBaseURL := chatURL.ResolveReference(apiRootReference).String()
	sdkChatPath := "chat/completions"
	if chatURL.ForceQuery {
		sdkChatPath += "?"
	}

	headers := cloneHeader(config.Headers)
	if key := strings.TrimSpace(config.APIKey); key != "" && headers.Get("Authorization") == "" {
		headers.Set("Authorization", "Bearer "+key)
	}
	headers.Set("Content-Type", "application/json")

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	// NewClient reads OPENAI_* variables and v3.52.0 exposes no public option
	// for disabling those defaults. Execute only needs Options, so construct an
	// explicitly configured Client to keep ambient credentials and routing out.
	sdkClient := openaisdk.Client{Options: []openaioption.RequestOption{
		openaioption.WithMaxRetries(0),
		openaioption.WithHTTPClient(httpClient),
		openaioption.WithBaseURL(sdkBaseURL),
	}}

	return &Client{
		provider:       provider,
		model:          model,
		capabilities:   capabilities,
		headers:        headers,
		maxTokensField: maxTokensField,
		sdkChatPath:    sdkChatPath,
		sdkClient:      sdkClient,
	}, nil
}

func configureCapabilities(configured []llm.Capability) ([]llm.Capability, error) {
	capabilities := []llm.Capability{llm.CapabilityGeneration}
	seen := map[llm.Capability]struct{}{llm.CapabilityGeneration: {}}
	for _, capability := range configured {
		switch capability {
		case llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools,
			llm.CapabilityJSON, llm.CapabilityVision:
		default:
			return nil, configError("capability %q is not implemented", capability)
		}
		if _, exists := seen[capability]; exists {
			continue
		}
		seen[capability] = struct{}{}
		capabilities = append(capabilities, capability)
	}
	return capabilities, nil
}

func cloneHeader(header http.Header) http.Header {
	clone := make(http.Header, len(header)+2)
	for key, values := range header {
		for _, value := range values {
			clone.Add(key, value)
		}
	}
	return clone
}

func restoreSDKHeaders(destination, source http.Header) {
	for sourceKey := range source {
		for destinationKey := range destination {
			if strings.EqualFold(destinationKey, sourceKey) {
				delete(destination, destinationKey)
			}
		}
	}
	for key, values := range source {
		destination[key] = slices.Clone(values)
	}
}

var errSDKRawErrorResponse = errors.New("OpenAI SDK returned a raw error response")

func captureSDKResponse(headers http.Header, captured **http.Response, downstreamErr *error) openaioption.Middleware {
	return func(request *http.Request, next openaioption.MiddlewareNext) (*http.Response, error) {
		restoreSDKHeaders(request.Header, headers)
		response, err := next(request)
		*captured = response
		*downstreamErr = err
		if err == nil && response != nil && response.StatusCode >= http.StatusBadRequest {
			return response, errSDKRawErrorResponse
		}
		return response, err
	}
}

// Info describes the configured model and its explicitly declared optional
// capabilities.
func (c *Client) Info() llm.ModelInfo {
	return llm.ModelInfo{
		Provider:     c.provider,
		Model:        c.model,
		Capabilities: append([]llm.Capability(nil), c.capabilities...),
	}
}

// Generate performs one non-streaming Chat Completions request.
func (c *Client) Generate(ctx context.Context, request llm.Request) (_ *llm.Response, err error) {
	defer func() {
		err = relabelProviderError(err, c.provider)
	}()

	if err := request.Validate(); err != nil {
		return nil, err
	}
	if err := contextErr(ctx); err != nil {
		return nil, err
	}
	if err := c.checkCapabilities("generate", request); err != nil {
		return nil, err
	}
	if err := checkRequest("generate", request); err != nil {
		return nil, err
	}

	wrequest, err := newChatRequestFor("generate", c.model, request, c.maxTokensField)
	if err != nil {
		return nil, err
	}
	body, err := jsonv2.Marshal(&wrequest)
	if err != nil {
		return nil, &llm.Error{
			Kind:     llm.KindInvalidRequest,
			Op:       "generate",
			Provider: "openai",
			Err:      fmt.Errorf("encode request: %w", err),
		}
	}

	body, err = c.post(ctx, "generate", c.sdkChatPath, body)
	if err != nil {
		return nil, err
	}
	if err := contextErr(ctx); err != nil {
		return nil, err
	}

	var wresponse chatResponse
	if err := jsonv2.Unmarshal(body, &wresponse); err != nil {
		if contextErr := contextErr(ctx); contextErr != nil {
			return nil, contextErr
		}
		return nil, malformedResponse("decode response: %w", err)
	}
	response, err := responseFromWire(c.model, request, wresponse)
	if err != nil {
		if contextErr := contextErr(ctx); contextErr != nil {
			return nil, contextErr
		}
		return nil, err
	}
	if err := contextErr(ctx); err != nil {
		return nil, err
	}
	return response, nil
}

// Stream starts one streaming Chat Completions request. Provider and transport
// failures after validation are delivered by the returned stream.
func (c *Client) Stream(ctx context.Context, request llm.Request) (_ llm.Stream, err error) {
	defer func() {
		err = relabelProviderError(err, c.provider)
	}()

	if err := request.Validate(); err != nil {
		return nil, err
	}
	if err := contextErr(ctx); err != nil {
		return nil, err
	}
	if !c.hasCapability(llm.CapabilityStreaming) {
		return nil, unsupported("stream", "configured model does not declare streaming capability")
	}
	if err := c.checkCapabilities("stream", request); err != nil {
		return nil, err
	}
	if err := checkRequest("stream", request); err != nil {
		return nil, err
	}

	wrequest, err := newChatRequestFor("stream", c.model, request, c.maxTokensField)
	if err != nil {
		return nil, err
	}
	wrequest.Stream = true
	wrequest.StreamOptions = &streamOptions{IncludeUsage: true}
	payload, err := jsonv2.Marshal(&wrequest)
	if err != nil {
		return nil, requestError("stream", "encode request: %w", err)
	}
	format := request.ResponseFormat
	declaredTools := declaredToolNames(request.Tools)
	return internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return c.produceStream(producerCtx, format, declaredTools, payload, emit)
	}), nil
}

func (c *Client) executeRaw(ctx context.Context, path string, payload []byte, headers http.Header) (*http.Response, error, error) {
	var response, captured *http.Response
	var downstreamErr error
	err := c.sdkClient.Execute(ctx, http.MethodPost, path, payload, &response,
		openaioption.WithMiddleware(captureSDKResponse(headers, &captured, &downstreamErr)))
	if response == nil {
		response = captured
	}
	return response, err, downstreamErr
}

func (c *Client) post(ctx context.Context, op, path string, payload []byte) ([]byte, error) {
	response, err, downstreamErr := c.executeRaw(ctx, path, payload, c.headers)
	if err != nil && !errors.Is(err, errSDKRawErrorResponse) {
		contextErr := contextErr(ctx)
		if downstreamErr != nil || contextErr == nil || response == nil || response.Body == nil {
			if contextErr != nil {
				return nil, contextErr
			}
			return nil, transportError(op, err)
		}
	}
	if response == nil || response.Body == nil {
		return nil, transportError(op, errors.New("SDK returned no HTTP response"))
	}
	if contextErr := contextErr(ctx); contextErr != nil {
		if closeErr := response.Body.Close(); closeErr != nil {
			return nil, errors.Join(contextErr, transportError(op, fmt.Errorf("close response body: %w", closeErr)))
		}
		return nil, contextErr
	}

	limit := int64(maxResponseBodyBytes)
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		limit = maxErrorBodyBytes
	}
	body, tooLarge, readErr := readLimited(response.Body, limit)
	closeErr := response.Body.Close()
	if readErr != nil {
		if contextErr := contextErr(ctx); contextErr != nil {
			bodyErr := fmt.Errorf("read response: %w", readErr)
			if closeErr != nil {
				bodyErr = errors.Join(bodyErr, fmt.Errorf("close response body: %w", closeErr))
			}
			return nil, errors.Join(contextErr, transportError(op, bodyErr))
		}
		if closeErr != nil {
			readErr = errors.Join(readErr, fmt.Errorf("close response body: %w", closeErr))
		}
		return nil, transportError(op, fmt.Errorf("read response: %w", readErr))
	}
	if contextErr := contextErr(ctx); contextErr != nil {
		if closeErr != nil {
			return nil, errors.Join(contextErr, transportError(op, fmt.Errorf("close response body: %w", closeErr)))
		}
		return nil, contextErr
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		responseErr := responseError(op, response, body, tooLarge)
		if contextErr := contextErr(ctx); contextErr != nil {
			if closeErr != nil {
				return nil, errors.Join(contextErr, transportError(op, fmt.Errorf("close response body: %w", closeErr)))
			}
			return nil, contextErr
		}
		if closeErr != nil {
			responseErr.Err = errors.Join(responseErr.Err, fmt.Errorf("close response body: %w", closeErr))
		}
		return nil, responseErr
	}
	if tooLarge {
		responseErr := malformedResponseFor(op, "response body exceeds %d bytes", maxResponseBodyBytes)
		if closeErr != nil {
			responseErr.Err = errors.Join(responseErr.Err, fmt.Errorf("close response body: %w", closeErr))
		}
		return nil, responseErr
	}
	if closeErr != nil {
		return nil, transportError(op, fmt.Errorf("close response body: %w", closeErr))
	}
	return body, nil
}

func readLimited(r io.Reader, limit int64) ([]byte, bool, error) {
	reader := &io.LimitedReader{R: r, N: limit + 1}
	body, err := io.ReadAll(reader)
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
		Provider:   "openai",
		HTTPStatus: response.StatusCode,
		RetryAfter: retryAfter(response.Header.Get("Retry-After")),
		Err:        apiErr,
	}
}

func decodeAPIError(body []byte) *APIError {
	var envelope errorEnvelope
	if err := jsonv2.Unmarshal(body, &envelope); err == nil && !envelope.Error.empty() {
		return &envelope.Error
	}

	var apiErr APIError
	if err := jsonv2.Unmarshal(body, &apiErr); err == nil && !apiErr.empty() {
		return &apiErr
	}
	return nil
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

	switch strings.ToLower(strings.TrimSpace(apiErr.Code)) {
	case "invalid_api_key":
		return llm.KindAuthentication
	case "insufficient_permissions":
		return llm.KindPermission
	case "rate_limit_exceeded", "insufficient_quota":
		return llm.KindRateLimit
	}
	switch strings.ToLower(strings.TrimSpace(apiErr.Type)) {
	case "authentication_error":
		return llm.KindAuthentication
	case "permission_error":
		return llm.KindPermission
	case "rate_limit_error":
		return llm.KindRateLimit
	case "invalid_request_error":
		return llm.KindInvalidRequest
	}
	return llm.KindProvider
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
		if seconds > 0 {
			if seconds > math.MaxInt64/int64(time.Second) {
				return time.Duration(math.MaxInt64)
			}
			return time.Duration(seconds) * time.Second
		}
		return 0
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

func (c *Client) checkCapabilities(op string, request llm.Request) error {
	usesTools := len(request.Tools) != 0
	hasImage := false
	hasAudio := false
	for _, message := range request.Messages {
		usesTools = usesTools || len(message.ToolCalls) != 0 || len(message.ToolResults) != 0
		for _, part := range message.Content {
			switch part.Kind {
			case llm.PartImage:
				hasImage = true
			case llm.PartAudio:
				hasAudio = true
			}
		}
		for _, result := range message.ToolResults {
			for _, part := range result.Content {
				switch part.Kind {
				case llm.PartImage:
					hasImage = true
				case llm.PartAudio:
					hasAudio = true
				}
			}
		}
	}

	if usesTools && !c.hasCapability(llm.CapabilityTools) {
		return unsupported(op, "configured model does not declare tool capability")
	}
	if request.ResponseFormat == llm.ResponseFormatJSON && !c.hasCapability(llm.CapabilityJSON) {
		return unsupported(op, "configured model does not declare JSON capability")
	}
	if hasImage && !c.hasCapability(llm.CapabilityVision) {
		return unsupported(op, "configured model does not declare vision capability")
	}
	if hasAudio {
		return unsupported(op, "audio content is not supported")
	}
	return nil
}

func (c *Client) hasCapability(target llm.Capability) bool {
	for _, capability := range c.capabilities {
		if capability == target {
			return true
		}
	}
	return false
}

func checkRequest(op string, request llm.Request) error {
	if request.Temperature != nil && *request.Temperature > 2 {
		return requestError(op, "temperature must not exceed 2")
	}
	if request.PresencePenalty != nil && (*request.PresencePenalty < -2 || *request.PresencePenalty > 2) {
		return requestError(op, "presence penalty must be between -2 and 2")
	}
	if request.FrequencyPenalty != nil && (*request.FrequencyPenalty < -2 || *request.FrequencyPenalty > 2) {
		return requestError(op, "frequency penalty must be between -2 and 2")
	}
	if len(request.Stop) > 4 {
		return requestError(op, "stop must not contain more than four sequences")
	}
	if request.ResponseFormat == llm.ResponseFormatJSON && len(request.Stop) != 0 {
		return requestError(op, "JSON response format cannot be combined with stop sequences")
	}
	if len(request.Tools) > maxTools {
		return requestError(op, "tools must not contain more than %d definitions", maxTools)
	}
	for i, tool := range request.Tools {
		if !validFunctionName(tool.Name) {
			return requestError(op, "tools[%d].name %q must contain 1-64 letters, digits, underscores, or dashes", i, tool.Name)
		}
		if jsontext.Value(tool.InputSchema).Kind() != jsontext.KindBeginObject {
			return requestError(op, "tools[%d].input schema must be a JSON object", i)
		}
	}

	for _, message := range request.Messages {
		for _, part := range message.Content {
			if part.Kind == llm.PartImage && message.Role != llm.RoleUser {
				return unsupported(op, "image content is supported only in user messages")
			}
		}
		for _, result := range message.ToolResults {
			for _, part := range result.Content {
				if part.Kind != llm.PartText {
					return unsupported(op, "binary tool results are not supported")
				}
			}
		}
		for i, call := range message.ToolCalls {
			if !validFunctionName(call.Name) {
				return requestError(op, "tool call %d name %q must contain 1-64 letters, digits, underscores, or dashes", i, call.Name)
			}
		}
	}
	return checkToolHistory(op, request.Messages)
}

func checkToolHistory(op string, messages []llm.Message) error {
	pending := 0
	for i, message := range messages {
		if pending != 0 && message.Role != llm.RoleTool {
			return requestError(op, "messages[%d] must contain tool results for %d pending calls", i, pending)
		}
		switch message.Role {
		case llm.RoleAssistant:
			pending = len(message.ToolCalls)
		case llm.RoleTool:
			pending -= len(message.ToolResults)
		}
	}
	if pending != 0 {
		return requestError(op, "messages end with %d pending tool calls", pending)
	}
	return nil
}

func validFunctionName(name string) bool {
	if len(name) == 0 || len(name) > 64 {
		return false
	}
	for i := 0; i < len(name); i++ {
		character := name[i]
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			character == '_' || character == '-' {
			continue
		}
		return false
	}
	return true
}

func contextErr(ctx context.Context) error {
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

// relabelProviderError keeps the OpenAI wire implementation independent of a
// compatible service's identity while presenting that identity at the public
// client boundary. Provider data remains tagged "openai" because it describes
// the wire format, not the service that carried it.
func relabelProviderError(err error, provider string) error {
	if err == nil || provider == defaultProvider {
		return err
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		relabeled := make([]error, len(causes))
		changed := false
		for i, cause := range causes {
			relabeled[i] = relabelProviderError(cause, provider)
			changed = changed || relabeled[i] != cause
		}
		if changed {
			return errors.Join(relabeled...)
		}
		return err
	}
	providerErr, ok := err.(*llm.Error)
	if !ok || providerErr.Provider != defaultProvider {
		return err
	}
	clone := *providerErr
	clone.Provider = provider
	return &clone
}

func configError(format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindInvalidRequest,
		Op:       "configure",
		Provider: "openai",
		Err:      fmt.Errorf(format, args...),
	}
}

func malformedResponse(format string, args ...any) error {
	return malformedResponseFor("generate", format, args...)
}

func malformedResponseFor(op, format string, args ...any) *llm.Error {
	return &llm.Error{
		Kind:     llm.KindMalformedResponse,
		Op:       op,
		Provider: "openai",
		Err:      fmt.Errorf(format, args...),
	}
}

func requestError(op, format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindInvalidRequest,
		Op:       op,
		Provider: "openai",
		Err:      fmt.Errorf(format, args...),
	}
}

func transportError(op string, err error) *llm.Error {
	return &llm.Error{
		Kind:     llm.KindTransport,
		Op:       op,
		Provider: "openai",
		Err:      err,
	}
}

func unsupported(op, message string) error {
	return &llm.Error{
		Kind:     llm.KindUnsupported,
		Op:       op,
		Provider: "openai",
		Err:      errors.New(message),
	}
}

// APIError is an error returned by an OpenAI-compatible endpoint.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitzero"`
	Param   string `json:"param,omitzero"`
	Code    string `json:"code,omitzero"`
}

// Error returns the provider's message, or its most specific available code.
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
	return "OpenAI API error"
}

func (e *APIError) empty() bool {
	return e == nil || (e.Message == "" && e.Type == "" && e.Param == "" && e.Code == "")
}

var _ llm.Generator = (*Client)(nil)
