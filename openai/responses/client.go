// Package responses implements OpenAI's Responses API with llm-go's neutral
// model contracts. The Responses input-content union does not support audio, so
// CapabilityAudio is unsupported.
package responses

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	llm "github.com/XiaoConstantine/llm-go"
	internalresponses "github.com/XiaoConstantine/llm-go/internal/openairesponses"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	openaioption "github.com/openai/openai-go/v3/option"
	sdkresponses "github.com/openai/openai-go/v3/responses"
)

const (
	defaultProvider      = "openai"
	defaultBaseURL       = "https://api.openai.com/v1"
	providerDataAPI      = "openai-responses"
	maxProviderDataBytes = 16 << 20
)

// APIError is an error reported in a Responses event stream.
type APIError = internalresponses.APIError

// Config configures an OpenAI Responses client. Model and APIKey are required.
// Provider identifies the service in ModelInfo and errors and defaults to
// "openai". BaseURL defaults to https://api.openai.com/v1. Capabilities opts
// the model into optional protocol features; generation is always enabled,
// while streaming, tools, vision, and JSON mode must be listed explicitly.
//
// HTTPClient and Headers are used as supplied without being mutated. The client
// owns Authorization, Content-Type, and Accept. A nil HTTPClient uses
// [http.DefaultClient], so callers should use context deadlines when an
// unbounded request is not acceptable.
type Config struct {
	Provider     string
	Model        string
	Capabilities []llm.Capability
	APIKey       string
	BaseURL      string
	HTTPClient   *http.Client
	Headers      http.Header
}

// Options configures compatible Responses protocol extensions.
type Options struct {
	// EncryptedReasoning requests opaque reasoning state for replay on later
	// turns without requesting an OpenAI reasoning summary.
	EncryptedReasoning bool
}

// Client is an immutable OpenAI Responses client. It is safe for concurrent
// use when its HTTP client is safe for concurrent use.
type Client struct {
	provider                string
	model                   string
	capabilities            []llm.Capability
	responses               sdkresponses.ResponseService
	requestOptions          internalresponses.RequestOptions
	compatibility           llm.OpenAIResponsesCompatibility
	compatibilityConfigured bool
}

// New constructs a Client from config using OpenAI defaults.
func New(config Config) (*Client, error) {
	return NewWithOptions(config, Options{})
}

// NewWithOptions constructs a Client with compatible protocol extensions.
func NewWithOptions(config Config, options Options) (*Client, error) {
	return newClient(config, options, nil)
}

// NewWithCompatibility constructs a Client with model compatibility metadata.
// compatibility is copied during construction.
func NewWithCompatibility(config Config, compatibility *llm.OpenAIResponsesCompatibility) (*Client, error) {
	return newClient(config, Options{}, compatibility)
}

func newClient(config Config, options Options, configuredCompatibility *llm.OpenAIResponsesCompatibility) (*Client, error) {
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = defaultProvider
	}
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, configError(provider, "model must not be empty")
	}
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		return nil, configError(provider, "API key must not be empty")
	}
	capabilities, err := configureCapabilities(provider, config.Capabilities)
	if err != nil {
		return nil, err
	}
	modelCompatibility := llm.OpenAIResponsesCompatibility{}
	if configuredCompatibility != nil {
		modelCompatibility = *configuredCompatibility
		compatibility := &llm.ModelCompatibility{OpenAIResponses: &modelCompatibility}
		if err := compatibility.Validate(llm.APIOpenAIResponses); err != nil {
			return nil, configError(provider, "model compatibility: %v", err)
		}
	}
	encryptedReasoning := options.EncryptedReasoning
	if modelCompatibility.EncryptedReasoning == llm.CompatibilityEnabled {
		encryptedReasoning = true
	} else if modelCompatibility.EncryptedReasoning == llm.CompatibilityDisabled {
		encryptedReasoning = false
	}
	baseURL, err := serviceBaseURL(provider, config.BaseURL)
	if err != nil {
		return nil, err
	}
	if modelCompatibility.SessionAffinityFormat == llm.SessionAffinityDefault {
		if provider == "openrouter" || strings.Contains(strings.ToLower(baseURL), "openrouter.ai") {
			modelCompatibility.SessionAffinityFormat = llm.SessionAffinityOpenRouter
		} else {
			modelCompatibility.SessionAffinityFormat = llm.SessionAffinityOpenAI
		}
	}
	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	sdkOptions := []openaioption.RequestOption{
		openaioption.WithMaxRetries(0),
		openaioption.WithHTTPClient(httpClient),
		openaioption.WithBaseURL(baseURL),
	}
	for key, values := range config.Headers.Clone() {
		for _, value := range values {
			sdkOptions = append(sdkOptions, openaioption.WithHeaderAdd(key, value))
		}
	}
	sdkOptions = append(sdkOptions,
		openaioption.WithAPIKey(apiKey),
		openaioption.WithHeader("Content-Type", "application/json"),
		openaioption.WithHeader("Accept", "text/event-stream"),
	)

	return &Client{
		provider:     provider,
		model:        model,
		capabilities: capabilities,
		responses:    sdkresponses.NewResponseService(sdkOptions...),
		requestOptions: internalresponses.RequestOptions{
			EncryptedReasoning:  encryptedReasoning,
			JSONObjectOutput:    true,
			NamedToolChoice:     true,
			ExplicitPromptCache: modelCompatibility.ExplicitPromptCacheMode == llm.CompatibilityEnabled,
		},
		compatibility:           modelCompatibility,
		compatibilityConfigured: configuredCompatibility != nil,
	}, nil
}

func configureCapabilities(provider string, configured []llm.Capability) ([]llm.Capability, error) {
	capabilities := []llm.Capability{llm.CapabilityGeneration}
	seen := map[llm.Capability]struct{}{llm.CapabilityGeneration: {}}
	for _, capability := range configured {
		switch capability {
		case llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityJSON:
		default:
			return nil, configError(provider, "capability %q is not implemented", capability)
		}
		if _, exists := seen[capability]; exists {
			continue
		}
		seen[capability] = struct{}{}
		capabilities = append(capabilities, capability)
	}
	return capabilities, nil
}

func serviceBaseURL(provider, raw string) (string, error) {
	baseURL := strings.TrimSpace(raw)
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return "", configError(provider, "invalid base URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", configError(provider, "base URL scheme must be http or https")
	}
	if parsed.Host == "" {
		return "", configError(provider, "base URL must be absolute")
	}
	if parsed.User != nil {
		return "", configError(provider, "base URL must not contain user information")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" || parsed.ForceQuery {
		return "", configError(provider, "base URL must not contain a query or fragment")
	}
	path := strings.TrimRight(parsed.Path, "/")
	if strings.HasSuffix(path, "/responses") {
		path = strings.TrimSuffix(path, "responses")
	} else {
		path += "/"
	}
	parsed.Path = path
	parsed.RawPath = ""
	return parsed.String(), nil
}

// Info describes the configured model and optional capabilities.
func (c *Client) Info() llm.ModelInfo {
	info := llm.ModelInfo{
		Provider:     c.provider,
		Model:        c.model,
		Capabilities: append([]llm.Capability(nil), c.capabilities...),
	}
	if c.compatibilityConfigured {
		compatibility := c.compatibility
		info.Compatibility = &llm.ModelCompatibility{OpenAIResponses: &compatibility}
	}
	return info
}

// Generate performs one Responses request and assembles its event stream.
func (c *Client) Generate(ctx context.Context, request llm.Request) (*llm.Response, error) {
	params, err := c.prepare(ctx, "generate", request, false)
	if err != nil {
		return nil, err
	}
	response := &llm.Response{Model: c.model, Message: llm.Message{Role: llm.RoleAssistant}}
	err = c.produce(ctx, "generate", params, request.SessionID, func(chunk llm.Chunk) bool {
		internalresponses.MergeChunk(response, chunk)
		return true
	})
	if err != nil {
		return nil, err
	}
	return response, nil
}

// Stream starts one streaming Responses request. Provider and transport errors
// after validation are delivered by the returned stream.
func (c *Client) Stream(ctx context.Context, request llm.Request) (llm.Stream, error) {
	params, err := c.prepare(ctx, "stream", request, true)
	if err != nil {
		return nil, err
	}
	return internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return c.produce(producerCtx, "stream", params, request.SessionID, emit)
	}), nil
}

func (c *Client) prepare(ctx context.Context, op string, request llm.Request, requireStreaming bool) (sdkresponses.ResponseNewParams, error) {
	if err := request.Validate(); err != nil {
		return sdkresponses.ResponseNewParams{}, err
	}
	if err := internalresponses.ContextError(ctx); err != nil {
		return sdkresponses.ResponseNewParams{}, err
	}
	if requireStreaming && !c.hasCapability(llm.CapabilityStreaming) {
		return sdkresponses.ResponseNewParams{}, unsupported(c.provider, op, "configured model does not declare streaming capability")
	}
	if err := c.checkCapabilities(op, request); err != nil {
		return sdkresponses.ResponseNewParams{}, err
	}
	if err := checkRequest(c.provider, op, request); err != nil {
		return sdkresponses.ResponseNewParams{}, err
	}
	if c.compatibility.StrictTools == llm.CompatibilityDisabled {
		for index, tool := range request.Tools {
			if tool.Strict {
				return sdkresponses.ResponseNewParams{}, unsupported(c.provider, op, fmt.Sprintf("tool %d requires strict schemas, which the configured model does not support", index))
			}
		}
	}
	options := c.requestOptions
	if supportsReasoning(c.model) {
		options.ReasoningSummary = true
		if c.compatibility.EncryptedReasoning == llm.CompatibilityDefault {
			options.EncryptedReasoning = true
		}
	}
	return c.codec().Request(op, c.model, request, options)
}

func (c *Client) codec() internalresponses.Codec {
	return internalresponses.Codec{
		Provider:             c.provider,
		ProviderDataAPI:      providerDataAPI,
		MaxProviderDataBytes: maxProviderDataBytes,
	}
}

func (c *Client) checkCapabilities(op string, request llm.Request) error {
	usesTools := len(request.Tools) != 0
	usesImages := false
	for _, message := range request.Messages {
		usesTools = usesTools || len(message.ToolCalls) != 0 || len(message.ToolResults) != 0
		for _, part := range message.Content {
			usesImages = usesImages || part.Kind == llm.PartImage
		}
		for _, result := range message.ToolResults {
			for _, part := range result.Content {
				usesImages = usesImages || part.Kind == llm.PartImage
			}
		}
	}
	if usesTools && !c.hasCapability(llm.CapabilityTools) {
		return unsupported(c.provider, op, "configured model does not declare tool capability")
	}
	if usesImages && !c.hasCapability(llm.CapabilityVision) {
		return unsupported(c.provider, op, "configured model does not declare vision capability")
	}
	if request.ResponseFormat == llm.ResponseFormatJSON && !c.hasCapability(llm.CapabilityJSON) {
		return unsupported(c.provider, op, "configured model does not declare JSON capability")
	}
	if request.ReasoningBudgetTokens != 0 {
		return unsupported(c.provider, op, "reasoning token budgets are not supported by the Responses API")
	}
	if request.CacheRetention == llm.CacheRetentionLong && c.compatibility.LongCacheRetention == llm.CompatibilityDisabled {
		return unsupported(c.provider, op, "configured model does not support long prompt-cache retention")
	}
	return nil
}

func (c *Client) hasCapability(capability llm.Capability) bool {
	for _, configured := range c.capabilities {
		if configured == capability {
			return true
		}
	}
	return false
}

func checkRequest(provider, op string, request llm.Request) error {
	if key := request.CacheKey; utf8.RuneCountInString(key) > 64 {
		return requestError(provider, op, "prompt cache key must not exceed 64 characters")
	}
	if session := request.SessionID; request.CacheRetention != llm.CacheRetentionNone && request.CacheKey == "" && utf8.RuneCountInString(session) > 64 {
		return requestError(provider, op, "session ID used as a prompt cache key must not exceed 64 characters")
	}
	if request.PresencePenalty != nil || request.FrequencyPenalty != nil || len(request.Stop) != 0 {
		return unsupported(provider, op, "penalties and stop sequences are not supported by the Responses API")
	}
	for i, message := range request.Messages {
		for _, part := range message.Content {
			switch part.Kind {
			case llm.PartText:
			case llm.PartImage:
				if message.Role != llm.RoleUser {
					return unsupported(provider, op, "image content is supported only in user messages")
				}
			default:
				return unsupported(provider, op, "OpenAI Responses does not support audio message content")
			}
		}
		for _, result := range message.ToolResults {
			if result.CallID == "" {
				return requestError(provider, op, "messages[%d] tool result must have a call ID", i)
			}
			for _, part := range result.Content {
				if part.Kind == llm.PartAudio {
					return unsupported(provider, op, "audio tool results are not supported")
				}
			}
		}
		for j, call := range message.ToolCalls {
			if call.ID == "" {
				return requestError(provider, op, "messages[%d] tool call %d must have an ID", i, j)
			}
			if !validFunctionName(call.Name) {
				return requestError(provider, op, "messages[%d] tool call %d name %q must contain 1-64 letters, digits, underscores, or dashes", i, j, call.Name)
			}
		}
	}
	for i, tool := range request.Tools {
		if !validFunctionName(tool.Name) {
			return requestError(provider, op, "tool %d name %q must contain 1-64 letters, digits, underscores, or dashes", i, tool.Name)
		}
	}
	return checkToolHistory(provider, op, request.Messages)
}

func checkToolHistory(provider, op string, messages []llm.Message) error {
	pending := 0
	for i, message := range messages {
		if pending != 0 && message.Role != llm.RoleTool {
			return requestError(provider, op, "messages[%d] must contain tool results for %d pending calls", i, pending)
		}
		switch message.Role {
		case llm.RoleAssistant:
			pending = len(message.ToolCalls)
		case llm.RoleTool:
			pending -= len(message.ToolResults)
		}
	}
	if pending != 0 {
		return requestError(provider, op, "messages end with %d pending tool calls", pending)
	}
	return nil
}

func validFunctionName(name string) bool {
	if len(name) == 0 || len(name) > 64 {
		return false
	}
	for i := range len(name) {
		character := name[i]
		if character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || character == '_' || character == '-' {
			continue
		}
		return false
	}
	return true
}

func supportsReasoning(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	return strings.HasPrefix(model, "gpt-5") || strings.HasPrefix(model, "o1") ||
		strings.HasPrefix(model, "o3") || strings.HasPrefix(model, "o4")
}

func (c *Client) produce(ctx context.Context, op string, params sdkresponses.ResponseNewParams, sessionID string, emit internalstream.Emit) error {
	open := func(ctx context.Context, op string, params sdkresponses.ResponseNewParams) (*internalresponses.Stream, error) {
		return c.openStream(ctx, op, params, sessionID)
	}
	return c.codec().Produce(ctx, op, c.model, params, open, emit)
}

func (c *Client) openStream(ctx context.Context, op string, params sdkresponses.ResponseNewParams, sessionID string) (*internalresponses.Stream, error) {
	options := sessionAffinityOptions(c.compatibility.SessionAffinityFormat, sessionID)
	stream := c.responses.NewStreaming(ctx, params, options...)
	if stream.Err() == nil {
		return stream, nil
	}
	err := stream.Err()
	if contextErr := internalresponses.ContextError(ctx); contextErr != nil {
		err = contextErr
	} else {
		err = c.classifySDKError(op, err)
	}
	if closeErr := stream.Close(); closeErr != nil {
		closeErr = transportError(c.provider, op, fmt.Errorf("close event stream: %w", closeErr))
		if err != nil {
			err = errors.Join(err, closeErr)
		} else {
			err = closeErr
		}
	}
	return nil, err
}

func sessionAffinityOptions(format llm.SessionAffinityFormat, sessionID string) []openaioption.RequestOption {
	if sessionID == "" {
		return nil
	}
	switch format {
	case llm.SessionAffinityOpenRouter:
		return []openaioption.RequestOption{openaioption.WithHeader("X-Session-Id", sessionID)}
	case llm.SessionAffinityOpenAINoSession:
		return []openaioption.RequestOption{openaioption.WithHeader("X-Client-Request-Id", sessionID)}
	default:
		return []openaioption.RequestOption{
			openaioption.WithHeader("session_id", sessionID),
			openaioption.WithHeader("X-Client-Request-Id", sessionID),
		}
	}
}

func (c *Client) classifySDKError(op string, err error) error {
	var apiErr *sdkresponses.Error
	if !errors.As(err, &apiErr) {
		return transportError(c.provider, op, err)
	}
	kind := classifyStatus(apiErr.StatusCode, apiErr.Type, apiErr.Code, apiErr.Message)
	providerErr := &llm.Error{Kind: kind, Op: op, Provider: c.provider, HTTPStatus: apiErr.StatusCode, Err: apiErr}
	if apiErr.Response != nil {
		providerErr.RetryAfter = retryAfter(apiErr.Response.Header.Get("Retry-After"))
	}
	return providerErr
}

func classifyStatus(status int, errorType, code, message string) llm.ErrorKind {
	detail := strings.ToLower(errorType + " " + code + " " + message)
	if strings.Contains(detail, "context_length_exceeded") || strings.Contains(detail, "maximum context length") || strings.Contains(detail, "context window") {
		return llm.KindContextLimit
	}
	switch status {
	case http.StatusUnauthorized:
		return llm.KindAuthentication
	case http.StatusForbidden:
		return llm.KindPermission
	case http.StatusTooManyRequests:
		return llm.KindRateLimit
	case http.StatusBadRequest, http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		return llm.KindInvalidRequest
	default:
		return llm.KindProvider
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

func configError(provider, format string, args ...any) *llm.Error {
	return &llm.Error{Kind: llm.KindInvalidRequest, Op: "configure", Provider: provider, Err: fmt.Errorf(format, args...)}
}

func requestError(provider, op, format string, args ...any) *llm.Error {
	return &llm.Error{Kind: llm.KindInvalidRequest, Op: op, Provider: provider, Err: fmt.Errorf(format, args...)}
}

func unsupported(provider, op, message string) *llm.Error {
	return &llm.Error{Kind: llm.KindUnsupported, Op: op, Provider: provider, Err: errors.New(message)}
}

func transportError(provider, op string, err error) *llm.Error {
	return &llm.Error{Kind: llm.KindTransport, Op: op, Provider: provider, Err: err}
}
