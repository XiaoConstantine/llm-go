package anthropic

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

	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"
)

const (
	defaultProvider              = "anthropic"
	defaultBaseURL               = "https://api.anthropic.com/v1"
	messagesPath                 = "messages"
	defaultAPIVersion            = "2023-06-01"
	defaultMaxOutputTokens       = 4096
	maxRequestBodyBytes          = 32 << 20
	maxResponseBodyBytes         = 16 << 20
	maxErrorBodyBytes      int64 = 1 << 20
)

// Config configures an Anthropic-compatible client. Model is required.
// Provider identifies the service in ModelInfo and errors; it defaults to
// "anthropic". BaseURL defaults to https://api.anthropic.com/v1, APIVersion
// defaults to 2023-06-01, and DefaultMaxOutputTokens defaults to 4096.
// Capabilities opts the configured model into optional protocol features;
// generation is always enabled, while streaming and client tool use must be
// listed explicitly. APIKey is optional to support gateways that authenticate
// through custom Headers. A non-nil HTTPClient and custom Headers are used as
// supplied, without being mutated by Client. A nil HTTPClient uses
// [http.DefaultClient], which has no overall request timeout; callers should use
// context deadlines or configure a client timeout.
type Config struct {
	Provider               string
	Model                  string
	Capabilities           []llm.Capability
	APIKey                 string
	BaseURL                string
	APIVersion             string
	DefaultMaxOutputTokens int
	HTTPClient             *http.Client
	Headers                http.Header
}

// Client is an immutable Anthropic client. It is safe for concurrent use when
// its configured HTTP client is safe for concurrent use.
type Client struct {
	provider               string
	model                  string
	capabilities           []llm.Capability
	defaultMaxOutputTokens int
	headers                http.Header
	sdkPath                string
	sdkClient              anthropicsdk.Client
}

// New constructs a Client from config.
func New(config Config) (_ *Client, err error) {
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
	endpoint, err := url.JoinPath(baseURL, messagesPath)
	if err != nil {
		return nil, configError("invalid base URL: %w", err)
	}
	endpointURL, err := url.Parse(endpoint)
	if err != nil {
		return nil, configError("invalid messages URL: %w", err)
	}
	directoryReference, _ := url.Parse(".")
	sdkBaseURL := endpointURL.ResolveReference(directoryReference).String()
	sdkPath := messagesPath
	if endpointURL.ForceQuery {
		sdkPath += "?"
	}

	apiVersion := strings.TrimSpace(config.APIVersion)
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}
	if _, err := time.Parse("2006-01-02", apiVersion); err != nil {
		return nil, configError("API version must use YYYY-MM-DD: %w", err)
	}

	maxOutputTokens := config.DefaultMaxOutputTokens
	if maxOutputTokens < 0 {
		return nil, configError("default max output tokens must not be negative")
	}
	if maxOutputTokens == 0 {
		maxOutputTokens = defaultMaxOutputTokens
	}

	headers := cloneHeader(config.Headers)
	if key := strings.TrimSpace(config.APIKey); key != "" {
		setHeaderDefault(headers, "X-Api-Key", key)
	}
	headers.Set("Anthropic-Version", apiVersion)
	headers.Set("Content-Type", "application/json")

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	sdkOptions := []anthropicoption.RequestOption{
		anthropicoption.WithoutEnvironmentDefaults(),
		anthropicoption.WithMaxRetries(0),
		anthropicoption.WithHTTPClient(httpClient),
		anthropicoption.WithBaseURL(sdkBaseURL),
	}
	if key := headerValue(headers, "X-Api-Key"); key != "" {
		sdkOptions = append(sdkOptions, anthropicoption.WithAPIKey(key))
	}
	sdkClient := anthropicsdk.NewClient(sdkOptions...)

	return &Client{
		provider:               provider,
		model:                  model,
		capabilities:           capabilities,
		defaultMaxOutputTokens: maxOutputTokens,
		headers:                headers,
		sdkPath:                sdkPath,
		sdkClient:              sdkClient,
	}, nil
}

func configureCapabilities(configured []llm.Capability) ([]llm.Capability, error) {
	capabilities := []llm.Capability{llm.CapabilityGeneration}
	seen := map[llm.Capability]struct{}{llm.CapabilityGeneration: {}}
	for _, capability := range configured {
		switch capability {
		case llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools:
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
	clone := header.Clone()
	if clone == nil {
		clone = make(http.Header)
	}
	return clone
}

func headerValue(header http.Header, name string) string {
	if values := header[name]; len(values) != 0 && values[0] != "" {
		return values[0]
	}
	for key, values := range header {
		if strings.EqualFold(key, name) && len(values) != 0 && values[0] != "" {
			return values[0]
		}
	}
	return ""
}

func setHeaderDefault(header http.Header, name, value string) {
	if headerValue(header, name) != "" {
		return
	}
	for key := range header {
		if strings.EqualFold(key, name) {
			delete(header, key)
		}
	}
	header.Set(name, value)
}

func restoreHeaders(destination, source http.Header) {
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

var errSDKRawErrorResponse = errors.New("anthropic SDK returned a raw error response")

// captureSDKResponse restores llm-go's headers after the SDK has applied its
// defaults, then prevents the SDK from consuming and closing provider error
// bodies. The per-call capture also retains a response when cancellation wins
// inside the SDK after the configured HTTP client has returned.
func captureSDKResponse(headers http.Header, captured **http.Response, downstreamErr *error) anthropicoption.Middleware {
	return func(request *http.Request, next anthropicoption.MiddlewareNext) (*http.Response, error) {
		restoreHeaders(request.Header, headers)
		response, err := next(request)
		*captured = response
		*downstreamErr = err
		if err == nil && response != nil && response.StatusCode >= http.StatusBadRequest {
			return response, errSDKRawErrorResponse
		}
		return response, err
	}
}

// Info describes the configured model.
func (c *Client) Info() llm.ModelInfo {
	return llm.ModelInfo{
		Provider:     c.provider,
		Model:        c.model,
		Capabilities: append([]llm.Capability(nil), c.capabilities...),
	}
}

// Generate performs one non-streaming Messages API request.
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

	wrequest, priorToolIDs, err := requestToWire("generate", c.model, c.defaultMaxOutputTokens, request)
	if err != nil {
		return nil, err
	}
	body, err := jsonv2.Marshal(&wrequest)
	if err != nil {
		return nil, requestError("generate", "encode request: %w", err)
	}
	if len(body) > maxRequestBodyBytes {
		return nil, requestError("generate", "request body exceeds %d bytes", maxRequestBodyBytes)
	}
	if err := contextErr(ctx); err != nil {
		return nil, err
	}
	body, err = c.post(ctx, body)
	if err != nil {
		return nil, err
	}

	var wresponse messageResponse
	if err := jsonv2.Unmarshal(body, &wresponse); err != nil {
		if contextErr := contextErr(ctx); contextErr != nil {
			return nil, contextErr
		}
		return nil, malformedResponse("decode response: %w", err)
	}
	response, err := responseFromWire(request, priorToolIDs, wresponse)
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

// Stream starts one streaming Messages API request. Provider and transport
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

	wrequest, priorToolIDs, err := requestToWire("stream", c.model, c.defaultMaxOutputTokens, request)
	if err != nil {
		return nil, err
	}
	wrequest.Stream = true
	payload, err := jsonv2.Marshal(&wrequest)
	if err != nil {
		return nil, requestError("stream", "encode request: %w", err)
	}
	if len(payload) > maxRequestBodyBytes {
		return nil, requestError("stream", "request body exceeds %d bytes", maxRequestBodyBytes)
	}
	declared := make(map[string]struct{}, len(request.Tools))
	for _, tool := range request.Tools {
		declared[tool.Name] = struct{}{}
	}
	return internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return c.produceStream(producerCtx, payload, declared, priorToolIDs, emit)
	}), nil
}

func (c *Client) checkCapabilities(op string, request llm.Request) error {
	usesTools := requestUsesTools(request)
	hasBinary := false
	for _, message := range request.Messages {
		for _, part := range message.Content {
			hasBinary = hasBinary || part.Kind != llm.PartText
		}
		for _, result := range message.ToolResults {
			for _, part := range result.Content {
				hasBinary = hasBinary || part.Kind != llm.PartText
			}
		}
	}
	if usesTools && !c.hasCapability(llm.CapabilityTools) {
		return unsupported(op, "configured model does not declare tool capability")
	}
	if request.ResponseFormat == llm.ResponseFormatJSON {
		return unsupported(op, "JSON response format is not implemented")
	}
	if hasBinary {
		return unsupported(op, "binary content is not implemented")
	}
	return nil
}

func requestUsesTools(request llm.Request) bool {
	if len(request.Tools) != 0 {
		return true
	}
	for _, message := range request.Messages {
		if len(message.ToolCalls) != 0 || len(message.ToolResults) != 0 {
			return true
		}
	}
	return false
}

func (c *Client) hasCapability(target llm.Capability) bool {
	for _, capability := range c.capabilities {
		if capability == target {
			return true
		}
	}
	return false
}

func (c *Client) executeRaw(ctx context.Context, payload []byte, headers http.Header) (*http.Response, error, error) {
	var response, captured *http.Response
	var downstreamErr error
	err := c.sdkClient.Execute(ctx, http.MethodPost, c.sdkPath, payload, &response,
		anthropicoption.WithMiddleware(captureSDKResponse(headers, &captured, &downstreamErr)))
	if response == nil {
		response = captured
	}
	return response, err, downstreamErr
}

func (c *Client) post(ctx context.Context, payload []byte) ([]byte, error) {
	response, err, downstreamErr := c.executeRaw(ctx, payload, c.headers)
	if err != nil && !errors.Is(err, errSDKRawErrorResponse) {
		contextErr := contextErr(ctx)
		if downstreamErr != nil || contextErr == nil || response == nil || response.Body == nil {
			if contextErr != nil {
				return nil, contextErr
			}
			return nil, transportError("generate", err)
		}
	}
	if response == nil || response.Body == nil {
		return nil, transportError("generate", errors.New("SDK returned no HTTP response"))
	}

	limit := int64(maxResponseBodyBytes)
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		limit = maxErrorBodyBytes
	}
	body, tooLarge, readErr := readLimited(response.Body, limit)
	closeErr := response.Body.Close()
	if contextErr := contextErr(ctx); contextErr != nil {
		var bodyErr error
		if readErr != nil {
			bodyErr = fmt.Errorf("read response: %w", readErr)
		}
		if closeErr != nil {
			bodyErr = errors.Join(bodyErr, fmt.Errorf("close response body: %w", closeErr))
		}
		if bodyErr != nil {
			return nil, errors.Join(contextErr, transportError("generate", bodyErr))
		}
		return nil, contextErr
	}
	if readErr != nil {
		if closeErr != nil {
			readErr = errors.Join(readErr, fmt.Errorf("close response body: %w", closeErr))
		}
		return nil, transportError("generate", fmt.Errorf("read response: %w", readErr))
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		responseErr := responseError("generate", response, body, tooLarge)
		if closeErr != nil {
			responseErr.Err = errors.Join(responseErr.Err, fmt.Errorf("close response body: %w", closeErr))
		}
		return nil, responseErr
	}
	if tooLarge {
		responseErr := malformedResponse("response body exceeds %d bytes", maxResponseBodyBytes)
		if closeErr != nil {
			responseErr.Err = errors.Join(responseErr.Err, fmt.Errorf("close response body: %w", closeErr))
		}
		return nil, responseErr
	}
	if closeErr != nil {
		return nil, transportError("generate", fmt.Errorf("close response body: %w", closeErr))
	}
	return body, nil
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
	if apiErr.RequestID == "" {
		apiErr.RequestID = response.Header.Get("Request-Id")
	}
	return &llm.Error{
		Kind:       classifyResponse(response.StatusCode, apiErr),
		Op:         op,
		Provider:   "anthropic",
		HTTPStatus: response.StatusCode,
		RetryAfter: retryAfter(response.Header.Get("Retry-After")),
		Err:        apiErr,
	}
}

func decodeAPIError(body []byte) *APIError {
	var envelope errorEnvelope
	if err := jsonv2.Unmarshal(body, &envelope); err != nil || envelope.Error == nil || envelope.Error.empty() {
		return nil
	}
	if envelope.Error.RequestID == "" {
		envelope.Error.RequestID = envelope.RequestID
	}
	return envelope.Error
}

func classifyResponse(status int, apiErr *APIError) llm.ErrorKind {
	switch status {
	case http.StatusUnauthorized:
		return llm.KindAuthentication
	case http.StatusPaymentRequired, http.StatusForbidden:
		return llm.KindPermission
	case http.StatusTooManyRequests:
		return llm.KindRateLimit
	case http.StatusBadRequest, http.StatusNotFound,
		http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		if isContextLimitError(apiErr) {
			return llm.KindContextLimit
		}
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

	switch strings.ToLower(strings.TrimSpace(apiErr.Type)) {
	case "authentication_error":
		return llm.KindAuthentication
	case "permission_error", "billing_error":
		return llm.KindPermission
	case "rate_limit_error":
		return llm.KindRateLimit
	case "invalid_request_error", "not_found_error":
		return llm.KindInvalidRequest
	default:
		return llm.KindProvider
	}
}

func isContextLimitError(apiErr *APIError) bool {
	if apiErr == nil {
		return false
	}
	detail := strings.ToLower(apiErr.Type + " " + apiErr.Message)
	return strings.Contains(detail, "context window") ||
		strings.Contains(detail, "prompt is too long") ||
		strings.Contains(detail, "too many tokens")
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

// relabelProviderError keeps the Anthropic wire implementation independent of
// a compatible service's identity while presenting that identity at the public
// client boundary.
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
		Provider: "anthropic",
		Err:      fmt.Errorf(format, args...),
	}
}

func requestError(op, format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindInvalidRequest,
		Op:       op,
		Provider: "anthropic",
		Err:      fmt.Errorf(format, args...),
	}
}

func unsupported(op, message string) error {
	return &llm.Error{
		Kind:     llm.KindUnsupported,
		Op:       op,
		Provider: "anthropic",
		Err:      errors.New(message),
	}
}

func transportError(op string, err error) error {
	return &llm.Error{
		Kind:     llm.KindTransport,
		Op:       op,
		Provider: "anthropic",
		Err:      err,
	}
}

func malformedResponse(format string, args ...any) *llm.Error {
	return &llm.Error{
		Kind:     llm.KindMalformedResponse,
		Op:       "generate",
		Provider: "anthropic",
		Err:      fmt.Errorf(format, args...),
	}
}

// APIError is an error returned by Anthropic's API.
type APIError struct {
	Type      string `json:"type"`
	Message   string `json:"message"`
	RequestID string `json:"-"`
}

// Error returns the provider's message, type, or a generic description.
func (e *APIError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Message != "" {
		return e.Message
	}
	if e.Type != "" {
		return e.Type
	}
	return "Anthropic API error"
}

func (e *APIError) empty() bool {
	return e == nil || (e.Type == "" && e.Message == "")
}

var _ llm.Generator = (*Client)(nil)
