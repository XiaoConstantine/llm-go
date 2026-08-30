// Package mistral implements Mistral's Conversations-compatible streaming chat
// API with llm-go's neutral model contracts.
package mistral

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"strings"
	"time"

	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/requestmeta"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
)

const (
	defaultProvider      = "mistral"
	defaultBaseURL       = "https://api.mistral.ai/v1"
	maxErrorBodyBytes    = 1 << 20
	maxResponseBodyBytes = 16 << 20
)

// Config configures a Mistral client. Model and APIKey are required. Provider
// defaults to "mistral" and BaseURL defaults to https://api.mistral.ai/v1.
// Reasoning declares that the configured model accepts Mistral reasoning
// controls and thinking blocks.
//
// Capabilities opts the model into optional protocol features; generation is
// always enabled, while streaming, tools, and vision must be listed explicitly.
// HTTPClient and Headers are used without mutation. A nil HTTPClient uses
// [http.DefaultClient], so callers should provide a context deadline when an
// unbounded request is not acceptable.
type Config struct {
	Provider     string
	Model        string
	Capabilities []llm.Capability
	Reasoning    bool
	APIKey       string
	BaseURL      string
	HTTPClient   *http.Client
	Headers      http.Header
}

// Client is an immutable Mistral client. It is safe for concurrent use when
// its configured HTTP client is safe for concurrent use.
type Client struct {
	provider     string
	model        string
	capabilities []llm.Capability
	reasoning    bool
	apiKey       string
	endpoint     string
	httpClient   *http.Client
	headers      http.Header
}

// New constructs a Mistral client.
func New(config Config) (_ *Client, err error) {
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = defaultProvider
	}
	defer func() { err = relabelProviderError(err, provider) }()
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, configError("model must not be empty")
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
	if parsed.User != nil {
		return nil, configError("base URL must not contain user information")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" || parsed.ForceQuery {
		return nil, configError("base URL must not contain a query or fragment")
	}
	endpoint, err := url.JoinPath(baseURL, "chat/completions")
	if err != nil {
		return nil, configError("invalid base URL: %w", err)
	}
	headers := config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	return &Client{provider: provider, model: model, capabilities: capabilities, reasoning: config.Reasoning,
		apiKey: apiKey, endpoint: endpoint, httpClient: requestmeta.WrapClient(config.HTTPClient), headers: headers}, nil
}

func configureCapabilities(configured []llm.Capability) ([]llm.Capability, error) {
	capabilities := []llm.Capability{llm.CapabilityGeneration}
	for _, capability := range configured {
		switch capability {
		case llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision:
		default:
			return nil, configError("capability %q is not implemented", capability)
		}
		if !slices.Contains(capabilities, capability) {
			capabilities = append(capabilities, capability)
		}
	}
	return capabilities, nil
}

// Info describes the configured model and capabilities.
func (c *Client) Info() llm.ModelInfo {
	return llm.ModelInfo{Provider: c.provider, Model: c.model, API: llm.APIMistralConversations,
		Capabilities: slices.Clone(c.capabilities), Reasoning: c.reasoning}
}

// Generate performs one request and collects Mistral's event stream.
func (c *Client) Generate(ctx context.Context, request llm.Request) (_ *llm.Response, err error) {
	defer func() { err = relabelProviderError(err, c.provider) }()
	payload, priorIDs, err := c.prepare(ctx, "generate", request, false)
	if err != nil {
		return nil, err
	}
	stream := internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return c.produce(producerCtx, "generate", request, payload, priorIDs, emit)
	})
	response, err := llm.Collect(stream, request.Tools)
	if err != nil {
		return nil, err
	}
	return response, nil
}

// Stream starts one Mistral event stream. Transport and provider errors after
// validation are delivered by the returned stream.
func (c *Client) Stream(ctx context.Context, request llm.Request) (_ llm.Stream, err error) {
	defer func() { err = relabelProviderError(err, c.provider) }()
	payload, priorIDs, err := c.prepare(ctx, "stream", request, true)
	if err != nil {
		return nil, err
	}
	stream := internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return c.produce(producerCtx, "stream", request, payload, priorIDs, emit)
	})
	return llm.ValidateToolCallStream(stream, request.Tools, c.provider)
}

func (c *Client) prepare(ctx context.Context, op string, request llm.Request, requireStreaming bool) ([]byte, map[string]struct{}, error) {
	if err := request.Validate(); err != nil {
		return nil, nil, err
	}
	if err := contextErr(ctx); err != nil {
		return nil, nil, err
	}
	if requireStreaming && !c.hasCapability(llm.CapabilityStreaming) {
		return nil, nil, unsupported(op, "configured model does not declare streaming capability")
	}
	if err := c.checkCapabilities(op, request); err != nil {
		return nil, nil, err
	}
	if err := checkRequest(op, request); err != nil {
		return nil, nil, err
	}
	return buildRequest(op, c.model, c.reasoning, c.hasCapability(llm.CapabilityVision), request)
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
		return unsupported(op, "configured model does not declare tool capability")
	}
	if usesImages && !c.hasCapability(llm.CapabilityVision) {
		return unsupported(op, "configured model does not declare vision capability")
	}
	if request.ResponseFormat == llm.ResponseFormatJSON {
		return unsupported(op, "Mistral Conversations does not expose portable JSON response mode")
	}
	if request.ReasoningEffort != llm.ReasoningEffortDefault && request.ReasoningEffort != llm.ReasoningEffortNone && !c.reasoning {
		return unsupported(op, "configured model does not support reasoning controls")
	}
	return nil
}

func checkRequest(op string, request llm.Request) error {
	if request.ReasoningBudgetTokens != 0 {
		return unsupported(op, "Mistral Conversations does not support explicit reasoning token budgets")
	}
	if request.CacheRetention != llm.CacheRetentionDefault || request.CacheKey != "" {
		return unsupported(op, "Mistral Conversations does not support portable prompt-cache controls")
	}
	if request.TopP != nil || request.PresencePenalty != nil || request.FrequencyPenalty != nil || len(request.Stop) != 0 {
		return unsupported(op, "top-p, penalties, and stop sequences are not supported by this adapter")
	}
	for messageIndex, message := range request.Messages {
		for partIndex, part := range message.Content {
			if part.Kind == llm.PartImage && message.Role != llm.RoleUser {
				return unsupported(op, fmt.Sprintf("messages[%d] content[%d] image is supported only in user messages", messageIndex, partIndex))
			}
			if part.Kind == llm.PartAudio {
				return unsupported(op, "Mistral Conversations does not support audio content")
			}
		}
		for _, result := range message.ToolResults {
			for _, part := range result.Content {
				if part.Kind == llm.PartAudio {
					return unsupported(op, "Mistral Conversations does not support audio tool results")
				}
			}
		}
	}
	return nil
}

func (c *Client) hasCapability(capability llm.Capability) bool {
	return slices.Contains(c.capabilities, capability)
}

func (c *Client) open(ctx context.Context, op string, payload []byte, sessionID string) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, transportError(op, "create request: %w", err)
	}
	request.Header = c.headers.Clone()
	request.Header.Set("Authorization", "Bearer "+c.apiKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	if sessionID != "" && request.Header.Get("X-Affinity") == "" {
		request.Header.Set("X-Affinity", sessionID)
	}
	response, err := c.httpClient.Do(request)
	if err != nil {
		if contextErr := contextErr(ctx); contextErr != nil {
			return nil, contextErr
		}
		return nil, transportError(op, "send request: %w", err)
	}
	if response == nil || response.Body == nil {
		return nil, transportError(op, "HTTP client returned no response body")
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		body, tooLarge, readErr := readLimited(response.Body, maxErrorBodyBytes)
		closeErr := response.Body.Close()
		if readErr != nil {
			return nil, errors.Join(transportError(op, "read error response: %w", readErr), closeError(op, closeErr))
		}
		return nil, errors.Join(responseError(op, response, body, tooLarge), closeError(op, closeErr))
	}
	mediaType, _, mediaErr := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if mediaErr != nil || !strings.EqualFold(mediaType, "text/event-stream") {
		closeErr := response.Body.Close()
		return nil, errors.Join(malformedStream(op, "response content type %q is not text/event-stream", response.Header.Get("Content-Type")), closeError(op, closeErr))
	}
	return response, nil
}

type errorEnvelope struct {
	Message string `json:"message"`
	Detail  string `json:"detail"`
	Error   *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func responseError(op string, response *http.Response, body []byte, tooLarge bool) error {
	var envelope errorEnvelope
	_ = jsonv2.Unmarshal(body, &envelope)
	message := strings.TrimSpace(envelope.Message)
	if message == "" {
		message = strings.TrimSpace(envelope.Detail)
	}
	if message == "" && envelope.Error != nil {
		message = strings.TrimSpace(envelope.Error.Message)
	}
	if message == "" {
		message = strings.TrimSpace(string(body))
	}
	if message == "" {
		message = response.Status
	}
	if tooLarge {
		message += " [truncated]"
	}
	detail := strings.ToLower(message)
	kind := llm.KindProvider
	switch response.StatusCode {
	case http.StatusUnauthorized:
		kind = llm.KindAuthentication
	case http.StatusForbidden:
		kind = llm.KindPermission
	case http.StatusTooManyRequests:
		kind = llm.KindRateLimit
	case http.StatusBadRequest, http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		kind = llm.KindInvalidRequest
	}
	if strings.Contains(detail, "context length") || strings.Contains(detail, "too large for model") || strings.Contains(detail, "too many tokens") {
		kind = llm.KindContextLimit
	}
	return &llm.Error{Kind: kind, Op: op, Provider: defaultProvider, HTTPStatus: response.StatusCode,
		RetryAfter: retryAfter(response.Header.Get("Retry-After")), Err: errors.New(message)}
}

func readLimited(reader io.Reader, limit int64) ([]byte, bool, error) {
	limited := io.LimitReader(reader, limit+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return nil, false, err
	}
	if int64(len(data)) > limit {
		return data[:limit], true, nil
	}
	return data, false, nil
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
	if delay > 0 {
		return delay
	}
	return 0
}

func contextErr(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		cause := context.Cause(ctx)
		if cause != nil && !errors.Is(cause, err) {
			return errors.Join(err, cause)
		}
		if cause != nil {
			return cause
		}
		return err
	}
	return nil
}

func relabelProviderError(err error, provider string) error {
	if err == nil || provider == defaultProvider {
		return err
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		causes := joined.Unwrap()
		relabeled := make([]error, len(causes))
		for index, cause := range causes {
			relabeled[index] = relabelProviderError(cause, provider)
		}
		return errors.Join(relabeled...)
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
	return &llm.Error{Kind: llm.KindInvalidRequest, Op: "configure", Provider: defaultProvider, Err: fmt.Errorf(format, args...)}
}

func requestError(op, format string, args ...any) error {
	return &llm.Error{Kind: llm.KindInvalidRequest, Op: op, Provider: defaultProvider, Err: fmt.Errorf(format, args...)}
}

func unsupported(op, message string) error {
	return &llm.Error{Kind: llm.KindUnsupported, Op: op, Provider: defaultProvider, Err: errors.New(message)}
}

func transportError(op, format string, args ...any) error {
	return &llm.Error{Kind: llm.KindTransport, Op: op, Provider: defaultProvider, Err: fmt.Errorf(format, args...)}
}

func malformedStream(op, format string, args ...any) error {
	return &llm.Error{Kind: llm.KindMalformedResponse, Op: op, Provider: defaultProvider, Err: fmt.Errorf(format, args...)}
}

var _ llm.Generator = (*Client)(nil)
