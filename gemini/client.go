package gemini

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/requestmeta"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	"google.golang.org/genai"
)

const (
	defaultProvider   = "gemini"
	defaultBaseURL    = "https://generativelanguage.googleapis.com/"
	defaultAPIVersion = "v1beta"
)

// Config configures a Gemini Developer API client. Model and APIKey are
// required. Provider identifies the service in ModelInfo and errors; it
// defaults to "gemini". BaseURL and APIVersion default to Google's public
// Gemini endpoint and v1beta API.
//
// Capabilities opts the configured model into optional protocol features;
// generation is always enabled, while streaming, tools, JSON mode, vision, and
// audio input must be listed explicitly. A non-nil HTTPClient and custom
// Headers are used as supplied, without being mutated by Client. A nil
// HTTPClient uses [http.DefaultClient], which has no overall request timeout;
// callers should use context deadlines or configure a client timeout.
type Config struct {
	Provider     string
	Model        string
	Capabilities []llm.Capability
	APIKey       string
	BaseURL      string
	APIVersion   string
	HTTPClient   *http.Client
	Headers      http.Header
}

// Client is an immutable Gemini Developer API client. It is safe for
// concurrent use when its configured HTTP client is safe for concurrent use.
type Client struct {
	provider     string
	model        string
	capabilities []llm.Capability
	sdkClient    *genai.Client
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
	if strings.ContainsAny(model, "?&") || strings.Contains(model, "..") {
		return nil, configError("model contains an invalid path or query component")
	}
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		return nil, configError("API key must not be empty")
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

	apiVersion := strings.TrimSpace(config.APIVersion)
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}
	if strings.ContainsAny(apiVersion, "/?#") {
		return nil, configError("API version must be one path segment")
	}

	httpClient := requestmeta.WrapClient(config.HTTPClient)
	headers := config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}

	sdkClient, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:     apiKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: httpClient,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    baseURL,
			APIVersion: apiVersion,
			Headers:    headers,
		},
	})
	if err != nil {
		return nil, configError("initialize Google Gen AI client: %w", err)
	}

	return &Client{
		provider:     provider,
		model:        model,
		capabilities: capabilities,
		sdkClient:    sdkClient,
	}, nil
}

func configureCapabilities(configured []llm.Capability) ([]llm.Capability, error) {
	capabilities := []llm.Capability{llm.CapabilityGeneration}
	for _, capability := range configured {
		switch capability {
		case llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools,
			llm.CapabilityJSON, llm.CapabilityVision, llm.CapabilityAudio:
		default:
			return nil, configError("capability %q is not implemented", capability)
		}
		if !slices.Contains(capabilities, capability) {
			capabilities = append(capabilities, capability)
		}
	}
	return capabilities, nil
}

// Info describes the configured model and its explicitly declared optional
// capabilities.
func (c *Client) Info() llm.ModelInfo {
	return llm.ModelInfo{
		Provider:     c.provider,
		Model:        c.model,
		API:          llm.APIGeminiGenerateContent,
		Capabilities: slices.Clone(c.capabilities),
	}
}

// Generate performs one non-streaming GenerateContent request.
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

	contents, generationConfig, err := requestToSDK("generate", request)
	if err != nil {
		return nil, err
	}
	response, err := c.generateContentSDK(ctx, contents, generationConfig)
	if contextErr := contextErr(ctx); contextErr != nil {
		return nil, contextErr
	}
	if err != nil {
		return nil, sdkError("generate", err)
	}
	converted, err := responseFromSDK(c.model, request, response)
	if err != nil {
		return nil, err
	}
	if err := contextErr(ctx); err != nil {
		return nil, err
	}
	if err := llm.ValidateToolCalls(request.Tools, converted.Message.ToolCalls); err != nil {
		return nil, malformedResponseFor("generate", "validate tool call arguments: %v", err)
	}
	return converted, nil
}

func (c *Client) generateContentSDK(
	ctx context.Context,
	contents []*genai.Content,
	config *genai.GenerateContentConfig,
) (response *genai.GenerateContentResponse, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			response = nil
			err = &sdkPanicError{value: recovered}
		}
	}()
	return c.sdkClient.Models.GenerateContent(ctx, c.model, contents, config)
}

// Stream starts one streaming GenerateContent request. Provider and transport
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

	contents, generationConfig, err := requestToSDK("stream", request)
	if err != nil {
		return nil, err
	}
	stream := internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return c.produceStream(producerCtx, request, contents, generationConfig, emit)
	})
	return llm.ValidateToolCallStream(stream, request.Tools, c.provider)
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
	if hasAudio && !c.hasCapability(llm.CapabilityAudio) {
		return unsupported(op, "configured model does not declare audio capability")
	}
	return nil
}

func (c *Client) hasCapability(target llm.Capability) bool {
	return slices.Contains(c.capabilities, target)
}

func sdkError(op string, err error) error {
	if err == nil {
		return nil
	}
	if _, ok := errors.AsType[*sdkPanicError](err); ok {
		return &llm.Error{Kind: llm.KindMalformedResponse, Op: op, Provider: defaultProvider, Err: err}
	}

	if apiErr, ok := errors.AsType[genai.APIError](err); ok {
		return &llm.Error{
			Kind:       classifyAPIError(apiErr),
			Op:         op,
			Provider:   defaultProvider,
			HTTPStatus: apiErr.Code,
			Err:        err,
		}
	}
	var apiErrPointer *genai.APIError
	if errors.As(err, &apiErrPointer) && apiErrPointer != nil {
		return &llm.Error{
			Kind:       classifyAPIError(*apiErrPointer),
			Op:         op,
			Provider:   defaultProvider,
			HTTPStatus: apiErrPointer.Code,
			Err:        err,
		}
	}

	var networkErr net.Error
	var urlErr *url.Error
	if errors.As(err, &networkErr) || errors.As(err, &urlErr) ||
		errors.Is(err, io.ErrUnexpectedEOF) {
		return &llm.Error{Kind: llm.KindTransport, Op: op, Provider: defaultProvider, Err: err}
	}

	detail := strings.ToLower(err.Error())
	if errors.Is(err, bufio.ErrTooLong) ||
		strings.Contains(detail, "error unmarshalling") ||
		strings.Contains(detail, "invalid stream chunk") ||
		strings.Contains(detail, "maptostruct") ||
		strings.Contains(detail, "response is too large") {
		return &llm.Error{Kind: llm.KindMalformedResponse, Op: op, Provider: defaultProvider, Err: err}
	}
	return &llm.Error{Kind: llm.KindTransport, Op: op, Provider: defaultProvider, Err: err}
}

type sdkPanicError struct {
	value any
}

func (err *sdkPanicError) Error() string {
	return fmt.Sprintf("Google Gen AI SDK panicked while decoding a response: %v", err.value)
}

func (err *sdkPanicError) Unwrap() error {
	cause, _ := err.value.(error)
	return cause
}

func classifyAPIError(apiErr genai.APIError) llm.ErrorKind {
	switch apiErr.Code {
	case http.StatusUnauthorized:
		return llm.KindAuthentication
	case http.StatusForbidden:
		return llm.KindPermission
	case http.StatusTooManyRequests:
		return llm.KindRateLimit
	case http.StatusBadRequest, http.StatusNotFound,
		http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		if isContextLimitError(apiErr) {
			return llm.KindContextLimit
		}
		return llm.KindInvalidRequest
	}
	switch strings.ToUpper(strings.TrimSpace(apiErr.Status)) {
	case "UNAUTHENTICATED":
		return llm.KindAuthentication
	case "PERMISSION_DENIED":
		return llm.KindPermission
	case "RESOURCE_EXHAUSTED":
		return llm.KindRateLimit
	case "INVALID_ARGUMENT", "NOT_FOUND", "FAILED_PRECONDITION", "OUT_OF_RANGE":
		if isContextLimitError(apiErr) {
			return llm.KindContextLimit
		}
		return llm.KindInvalidRequest
	default:
		if isContextLimitError(apiErr) {
			return llm.KindContextLimit
		}
		return llm.KindProvider
	}
}

func isContextLimitError(apiErr genai.APIError) bool {
	detail := strings.ToLower(apiErr.Status + " " + apiErr.Message)
	return strings.Contains(detail, "context window") ||
		strings.Contains(detail, "context length") ||
		strings.Contains(detail, "input token") ||
		strings.Contains(detail, "too many tokens")
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
		Provider: defaultProvider,
		Err:      fmt.Errorf(format, args...),
	}
}

func requestError(op, format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindInvalidRequest,
		Op:       op,
		Provider: defaultProvider,
		Err:      fmt.Errorf(format, args...),
	}
}

func malformedResponseFor(op, format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindMalformedResponse,
		Op:       op,
		Provider: defaultProvider,
		Err:      fmt.Errorf(format, args...),
	}
}

func unsupported(op, message string) error {
	return &llm.Error{
		Kind:     llm.KindUnsupported,
		Op:       op,
		Provider: defaultProvider,
		Err:      errors.New(message),
	}
}

var _ llm.Generator = (*Client)(nil)
