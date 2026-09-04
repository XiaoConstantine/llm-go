package codex

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/requestmeta"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	"github.com/gorilla/websocket"
	openaioption "github.com/openai/openai-go/v3/option"
	openairesponses "github.com/openai/openai-go/v3/responses"
)

const (
	defaultProvider              = "openai-codex"
	defaultBaseURL               = "https://chatgpt.com/backend-api"
	defaultOriginator            = "llm-go"
	maxErrorBodyBytes            = 1 << 20
	maxProviderDataBytes         = 16 << 20
	defaultWebSocketIdleLifetime = 5 * time.Minute
	defaultWebSocketMaximumAge   = 55 * time.Minute
	defaultWebSocketConnectTime  = 15 * time.Second
	defaultWebSocketMaxSessions  = 64
)

// TransportMode selects the Codex Responses transport.
type TransportMode string

const (
	// TransportSSE preserves the original HTTP event-stream transport and is the default.
	TransportSSE TransportMode = "sse"
	// TransportWebSocket uses WebSocket but always sends full canonical context.
	TransportWebSocket TransportMode = "websocket"
	// TransportWebSocketCached enables verified connection-scoped continuation deltas.
	TransportWebSocketCached TransportMode = "websocket-cached"
	// TransportAuto uses cached WebSocket and falls back to SSE only when the
	// WebSocket connection cannot be established before response.create is sent.
	TransportAuto TransportMode = "auto"
)

// Credentials authorize one request against a ChatGPT subscription. AccountID
// may be empty when AccessToken is a JWT containing chatgpt_account_id.
type Credentials struct {
	AccessToken string
	AccountID   string
}

// CredentialResolver returns current ChatGPT subscription credentials.
// rejectedAccessToken is nonempty after the server rejects a token with HTTP
// 401. Implementations that refresh credentials should rotate only when their
// current token still matches rejectedAccessToken. A resolver must be safe for
// concurrent use.
type CredentialResolver func(ctx context.Context, rejectedAccessToken string) (Credentials, error)

// Config configures a ChatGPT subscription Codex client. Model is required.
// Provider identifies the service in ModelInfo and errors and defaults to
// "openai-codex". Set either AccessToken (with optional AccountID) or
// ResolveCredentials, but not both. ResolveCredentials is called before every
// request and once more after an HTTP 401.
//
// BaseURL defaults to https://chatgpt.com/backend-api. Capabilities opts the
// configured model into optional protocol features; generation is always
// enabled, while streaming, tools, vision, and audio must be listed explicitly.
// Audio capability is model-gated and enables user-message WAV, MP3/MPEG,
// M4A/MP4, WebM, and Ogg inputs up to 50 MiB each. HTTPClient owns the SSE
// transport. WebSocketDialer, when non-nil, explicitly owns WebSocket dialing;
// otherwise Client best-effort copies settings from a concrete *http.Transport
// and cannot adapt arbitrary RoundTripper wrappers. Custom Headers are copied.
// Authorization, ChatGPT-Account-ID, OpenAI-Beta, Originator, Content-Type,
// Accept, and User-Agent are owned by Client and overwrite custom values. SSE
// uses [http.DefaultClient] when HTTPClient is nil, so callers should use context
// deadlines when an unbounded request is not acceptable. Transport defaults to SSE,
// preserving the original behavior. Strict WebSocket modes never fall back to
// SSE; Auto falls back only for a pre-send connection failure.
type Config struct {
	Provider           string
	Model              string
	Capabilities       []llm.Capability
	AccessToken        string
	AccountID          string
	ResolveCredentials CredentialResolver
	BaseURL            string
	HTTPClient         *http.Client
	// WebSocketDialer overrides WebSocket dialing. Client copies the dialer,
	// its TLS configuration, and ordinary option slices during construction.
	// Callback, pool, certificate, key, root, cache, and jar objects referenced
	// by the dialer must be concurrency-safe and must not be mutated during use.
	WebSocketDialer *websocket.Dialer
	Headers         http.Header
	Originator      string
	// Transport defaults to TransportSSE.
	Transport TransportMode
	// WebSocketIdleTime bounds how long an unused session socket is retained.
	// Zero selects five minutes.
	WebSocketIdleTime time.Duration
	// WebSocketMaxAge bounds total session socket lifetime. Zero selects 55 minutes.
	WebSocketMaxAge time.Duration
	// WebSocketConnectTimeout bounds the handshake. Zero selects 15 seconds.
	WebSocketConnectTimeout time.Duration
	// WebSocketMaxSessions bounds retained session sockets. Zero selects 64;
	// negative values are invalid. Busy sockets are never evicted.
	WebSocketMaxSessions int
}

// Client is a concurrency-safe ChatGPT subscription Codex client. Its
// configuration is immutable; WebSocket modes maintain an internal session
// connection cache. The credential resolver and HTTP client must also be safe
// for concurrent use.
type Client struct {
	provider                string
	model                   string
	capabilities            []llm.Capability
	resolveCredentials      CredentialResolver
	headers                 http.Header
	originator              string
	responses               openairesponses.ResponseService
	compatibility           llm.OpenAIResponsesCompatibility
	compatibilityConfigured bool
	transport               TransportMode
	webSocketURL            string
	webSocketIdleTime       time.Duration
	webSocketMaxAge         time.Duration
	webSocketConnectTimeout time.Duration
	webSocketMaxSessions    int
	webSocketDialer         websocket.Dialer
	webSockets              webSocketCache
}

// New constructs a Client from config using subscription protocol defaults.
func New(config Config) (*Client, error) {
	return newClient(config, nil)
}

// NewWithCompatibility constructs a Client with model compatibility metadata.
// compatibility is copied during construction.
func NewWithCompatibility(config Config, compatibility *llm.OpenAIResponsesCompatibility) (*Client, error) {
	return newClient(config, compatibility)
}

func newClient(config Config, configuredCompatibility *llm.OpenAIResponsesCompatibility) (_ *Client, err error) {
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
	compatibility := llm.OpenAIResponsesCompatibility{}
	if configuredCompatibility != nil {
		compatibility = *configuredCompatibility
		modelCompatibility := &llm.ModelCompatibility{OpenAIResponses: &compatibility}
		if err := modelCompatibility.Validate(llm.APIOpenAICodexResponses); err != nil {
			return nil, configError("model compatibility: %v", err)
		}
	}

	resolver, err := configureCredentials(config)
	if err != nil {
		return nil, err
	}
	sdkBaseURL, err := responseServiceBaseURL(config.BaseURL)
	if err != nil {
		return nil, err
	}
	transport, idleTime, maxAge, connectTimeout, maxSessions, err := configureTransport(config)
	if err != nil {
		return nil, err
	}
	webSocketURL, err := responseWebSocketURL(sdkBaseURL)
	if err != nil {
		return nil, err
	}
	webSocketDialer := configureWebSocketDialer(config.HTTPClient, config.WebSocketDialer)

	httpClient := requestmeta.WrapClient(config.HTTPClient)
	originator := strings.TrimSpace(config.Originator)
	if originator == "" {
		originator = defaultOriginator
	}

	return &Client{
		provider:                provider,
		model:                   model,
		capabilities:            capabilities,
		resolveCredentials:      resolver,
		headers:                 cloneHeader(config.Headers),
		originator:              originator,
		compatibility:           compatibility,
		compatibilityConfigured: configuredCompatibility != nil,
		transport:               transport,
		webSocketURL:            webSocketURL,
		webSocketIdleTime:       idleTime,
		webSocketMaxAge:         maxAge,
		webSocketConnectTimeout: connectTimeout,
		webSocketMaxSessions:    maxSessions,
		webSocketDialer:         webSocketDialer,
		webSockets:              newWebSocketCache(),
		responses: openairesponses.NewResponseService(
			openaioption.WithMaxRetries(0),
			openaioption.WithHTTPClient(httpClient),
			openaioption.WithBaseURL(sdkBaseURL),
		),
	}, nil
}

func configureCredentials(config Config) (CredentialResolver, error) {
	token := strings.TrimSpace(config.AccessToken)
	accountID := strings.TrimSpace(config.AccountID)
	if config.ResolveCredentials != nil {
		if token != "" || accountID != "" {
			return nil, configError("AccessToken and AccountID must be empty when ResolveCredentials is set")
		}
		return config.ResolveCredentials, nil
	}
	if token == "" {
		return nil, configError("access token or credential resolver is required")
	}
	if accountID == "" {
		var err error
		accountID, err = AccountIDFromToken(token)
		if err != nil {
			return nil, configError("resolve account ID from access token: %w", err)
		}
	}
	credentials := Credentials{AccessToken: token, AccountID: accountID}
	return func(_ context.Context, _ string) (Credentials, error) {
		return credentials, nil
	}, nil
}

func configureCapabilities(configured []llm.Capability) ([]llm.Capability, error) {
	capabilities := []llm.Capability{llm.CapabilityGeneration}
	seen := map[llm.Capability]struct{}{llm.CapabilityGeneration: {}}
	for _, capability := range configured {
		switch capability {
		case llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityVision, llm.CapabilityAudio:
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

func responseServiceBaseURL(raw string) (string, error) {
	baseURL := strings.TrimSpace(raw)
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return "", configError("invalid base URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", configError("base URL scheme must be http or https")
	}
	if parsed.Host == "" {
		return "", configError("base URL must be absolute")
	}
	if parsed.User != nil {
		return "", configError("base URL must not contain user information")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" || parsed.ForceQuery {
		return "", configError("base URL must not contain a query or fragment")
	}

	path := strings.TrimRight(parsed.Path, "/")
	switch {
	case strings.HasSuffix(path, "/codex/responses"):
		path = strings.TrimSuffix(path, "responses")
	case strings.HasSuffix(path, "/codex"):
		path += "/"
	default:
		path += "/codex/"
	}
	parsed.Path = path
	parsed.RawPath = ""
	return parsed.String(), nil
}

func configureTransport(config Config) (TransportMode, time.Duration, time.Duration, time.Duration, int, error) {
	transport := config.Transport
	if transport == "" {
		transport = TransportSSE
	}
	switch transport {
	case TransportSSE, TransportWebSocket, TransportWebSocketCached, TransportAuto:
	default:
		return "", 0, 0, 0, 0, configError("transport %q is not supported", transport)
	}
	idleTime := config.WebSocketIdleTime
	if idleTime == 0 {
		idleTime = defaultWebSocketIdleLifetime
	}
	maxAge := config.WebSocketMaxAge
	if maxAge == 0 {
		maxAge = defaultWebSocketMaximumAge
	}
	connectTimeout := config.WebSocketConnectTimeout
	if connectTimeout == 0 {
		connectTimeout = defaultWebSocketConnectTime
	}
	if idleTime < 0 || maxAge < 0 || connectTimeout < 0 {
		return "", 0, 0, 0, 0, configError("WebSocket durations must not be negative")
	}
	maxSessions := config.WebSocketMaxSessions
	if maxSessions < 0 {
		return "", 0, 0, 0, 0, configError("WebSocketMaxSessions must not be negative")
	}
	if maxSessions == 0 {
		maxSessions = defaultWebSocketMaxSessions
	}
	return transport, idleTime, maxAge, connectTimeout, maxSessions, nil
}

func responseWebSocketURL(serviceBaseURL string) (string, error) {
	parsed, err := url.Parse(serviceBaseURL)
	if err != nil {
		return "", configError("invalid WebSocket base URL: %w", err)
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/responses"
	switch parsed.Scheme {
	case "https":
		parsed.Scheme = "wss"
	case "http":
		parsed.Scheme = "ws"
	default:
		return "", configError("WebSocket base URL scheme must be http or https")
	}
	return parsed.String(), nil
}

func configureWebSocketDialer(client *http.Client, configured *websocket.Dialer) websocket.Dialer {
	if configured != nil {
		dialer := *configured
		if configured.TLSClientConfig != nil {
			dialer.TLSClientConfig = cloneTLSConfig(configured.TLSClientConfig)
		}
		dialer.Subprotocols = append([]string(nil), configured.Subprotocols...)
		return dialer
	}
	dialer := *websocket.DefaultDialer
	if client == nil {
		return dialer
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok || transport == nil {
		return dialer
	}
	dialer.Proxy = transport.Proxy
	dialer.NetDialContext = transport.DialContext
	dialer.TLSClientConfig = nil
	if transport.TLSClientConfig != nil {
		dialer.TLSClientConfig = cloneTLSConfig(transport.TLSClientConfig)
	}
	return dialer
}

func cloneTLSConfig(config *tls.Config) *tls.Config {
	if config == nil {
		return nil
	}
	clone := config.Clone()
	clone.NextProtos = append([]string(nil), config.NextProtos...)
	clone.CipherSuites = append([]uint16(nil), config.CipherSuites...)
	clone.CurvePreferences = append([]tls.CurveID(nil), config.CurvePreferences...)
	return clone
}

func cloneHeader(header http.Header) http.Header {
	clone := make(http.Header, len(header))
	for key, values := range header {
		canonical := http.CanonicalHeaderKey(key)
		clone[canonical] = append(clone[canonical], values...)
	}
	return clone
}

// Info describes the configured model and its explicitly declared optional
// capabilities.
func (c *Client) Info() llm.ModelInfo {
	info := llm.ModelInfo{
		Provider:     c.provider,
		Model:        c.model,
		API:          llm.APIOpenAICodexResponses,
		Capabilities: append([]llm.Capability(nil), c.capabilities...),
	}
	if c.compatibilityConfigured {
		compatibility := c.compatibility
		info.Compatibility = &llm.ModelCompatibility{OpenAIResponses: &compatibility}
	}
	return info
}

// Generate performs one Codex Responses request. The subscription endpoint is
// streaming-only, so Generate assembles its result from the same event stream
// used by Stream.
func (c *Client) Generate(ctx context.Context, request llm.Request) (_ *llm.Response, err error) {
	defer func() {
		err = relabelProviderError(err, c.provider)
	}()
	params, err := c.prepare(ctx, "generate", request, false)
	if err != nil {
		return nil, err
	}

	response := &llm.Response{Model: c.model, Message: llm.Message{Role: llm.RoleAssistant}}
	err = c.produce(ctx, "generate", params, request.SessionID, func(chunk llm.Chunk) bool {
		mergeChunk(response, chunk)
		return true
	})
	if err != nil {
		return nil, err
	}
	if err := llm.ValidateToolCalls(request.Tools, response.Message.ToolCalls); err != nil {
		return nil, malformedResponse("generate", "validate tool call arguments: %v", err)
	}
	return response, nil
}

// Stream starts one streaming Codex Responses request. Provider, credential,
// and transport failures after validation are delivered by the returned stream.
func (c *Client) Stream(ctx context.Context, request llm.Request) (_ llm.Stream, err error) {
	defer func() {
		err = relabelProviderError(err, c.provider)
	}()
	params, err := c.prepare(ctx, "stream", request, true)
	if err != nil {
		return nil, err
	}
	stream := internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return relabelProviderError(c.produce(producerCtx, "stream", params, request.SessionID, emit), c.provider)
	})
	return llm.ValidateToolCallStream(stream, request.Tools, c.provider)
}

func (c *Client) prepare(ctx context.Context, op string, request llm.Request, requireStreaming bool) (openairesponses.ResponseNewParams, error) {
	if err := request.Validate(); err != nil {
		return openairesponses.ResponseNewParams{}, err
	}
	if err := contextErr(ctx); err != nil {
		return openairesponses.ResponseNewParams{}, err
	}
	if requireStreaming && !c.hasCapability(llm.CapabilityStreaming) {
		return openairesponses.ResponseNewParams{}, unsupported(op, "configured model does not declare streaming capability")
	}
	if err := c.checkCapabilities(op, request); err != nil {
		return openairesponses.ResponseNewParams{}, err
	}
	if err := checkRequest(op, request); err != nil {
		return openairesponses.ResponseNewParams{}, err
	}
	if c.compatibility.StrictTools == llm.CompatibilityDisabled {
		for index, tool := range request.Tools {
			if tool.RequiresStrict() {
				return openairesponses.ResponseNewParams{}, unsupported(op, fmt.Sprintf("tool %d requires strict schemas, which the configured model does not support", index))
			}
		}
	}
	return requestToWireWithCompatibility(op, c.model, request, c.compatibility)
}

func (c *Client) checkCapabilities(op string, request llm.Request) error {
	if request.HasCacheBreakpoints() {
		return unsupported(op, "explicit cache breakpoint placement is not supported by the subscription endpoint")
	}
	usesTools := len(request.Tools) != 0
	usesImages := false
	usesAudio := false
	for _, message := range request.Messages {
		usesTools = usesTools || len(message.ToolCalls) != 0 || len(message.ToolResults) != 0
		for _, part := range message.Content {
			usesImages = usesImages || part.Kind == llm.PartImage
			usesAudio = usesAudio || part.Kind == llm.PartAudio
		}
		for _, result := range message.ToolResults {
			for _, part := range result.Content {
				usesImages = usesImages || part.Kind == llm.PartImage
				usesAudio = usesAudio || part.Kind == llm.PartAudio
			}
		}
	}
	if usesTools && !c.hasCapability(llm.CapabilityTools) {
		return unsupported(op, "configured model does not declare tool capability")
	}
	if usesImages && !c.hasCapability(llm.CapabilityVision) {
		return unsupported(op, "configured model does not declare vision capability")
	}
	if usesAudio && !c.hasCapability(llm.CapabilityAudio) {
		return unsupported(op, "configured model does not declare audio capability")
	}
	if request.ResponseFormat == llm.ResponseFormatJSON {
		return unsupported(op, "JSON response format is not implemented")
	}
	if request.ReasoningBudgetTokens != 0 {
		return unsupported(op, "reasoning token budgets are not supported by the subscription endpoint")
	}
	if request.CacheRetention == llm.CacheRetentionLong {
		return unsupported(op, "long prompt-cache retention is not supported by the subscription endpoint")
	}
	return nil
}

func (c *Client) hasCapability(capability llm.Capability) bool {
	return slices.Contains(c.capabilities, capability)
}

func checkRequest(op string, request llm.Request) error {
	if key := request.CacheKey; utf8.RuneCountInString(key) > 64 {
		return requestError(op, "prompt cache key must not exceed 64 characters")
	}
	if session := request.SessionID; request.CacheRetention != llm.CacheRetentionNone && request.CacheKey == "" && utf8.RuneCountInString(session) > 64 {
		return requestError(op, "session ID used as a prompt cache key must not exceed 64 characters")
	}
	if request.MaxOutputTokens != 0 {
		return unsupported(op, "max output tokens are not supported by the subscription endpoint")
	}
	if request.TopP != nil || request.PresencePenalty != nil || request.FrequencyPenalty != nil || len(request.Stop) != 0 {
		return unsupported(op, "top-p, penalties, and stop sequences are not supported by the subscription endpoint")
	}
	for i, message := range request.Messages {
		for _, part := range message.Content {
			switch part.Kind {
			case llm.PartText:
			case llm.PartImage:
				if message.Role != llm.RoleUser {
					return unsupported(op, "image content is supported only in user messages")
				}
			case llm.PartAudio:
				if message.Role != llm.RoleUser {
					return unsupported(op, "audio content is supported only in user messages")
				}
			}
		}
		for _, result := range message.ToolResults {
			if result.CallID == "" {
				return requestError(op, "messages[%d] tool result must have a call ID", i)
			}
			for _, part := range result.Content {
				if part.Kind == llm.PartAudio {
					return unsupported(op, "audio tool results are not supported")
				}
			}
		}
		for j, call := range message.ToolCalls {
			if call.ID == "" {
				return requestError(op, "messages[%d] tool call %d must have an ID", i, j)
			}
			if !validFunctionName(call.Name) {
				return requestError(op, "messages[%d] tool call %d name %q must contain 1-64 letters, digits, underscores, or dashes", i, j, call.Name)
			}
		}
	}
	for i, tool := range request.Tools {
		if !validFunctionName(tool.Name) {
			return requestError(op, "tool %d name %q must contain 1-64 letters, digits, underscores, or dashes", i, tool.Name)
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
	for i := range len(name) {
		character := name[i]
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') || character == '_' || character == '-' {
			continue
		}
		return false
	}
	return true
}

func mergeChunk(response *llm.Response, chunk llm.Chunk) {
	if chunk.ID != "" {
		response.ID = chunk.ID
	}
	if chunk.Model != "" {
		response.Model = chunk.Model
	}
	response.Message.Content = append(response.Message.Content, chunk.Content...)
	response.Message.ToolCalls = append(response.Message.ToolCalls, chunk.ToolCalls...)
	response.ReasoningSummary += chunk.ReasoningSummary
	if len(chunk.ProviderData) != 0 {
		response.Message.ProviderData = append(response.Message.ProviderData[:0], chunk.ProviderData...)
	}
	if chunk.FinishReason != "" {
		response.FinishReason = chunk.FinishReason
	}
	if chunk.Usage != nil {
		usage := *chunk.Usage
		response.Usage = &usage
	}
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
	if connectFailure, ok := err.(*webSocketConnectFailure); ok {
		return &webSocketConnectFailure{err: relabelProviderError(connectFailure.err, provider)}
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

func configError(format string, args ...any) *llm.Error {
	return &llm.Error{Kind: llm.KindInvalidRequest, Op: "configure", Provider: defaultProvider, Err: fmt.Errorf(format, args...)}
}

func requestError(op, format string, args ...any) *llm.Error {
	return &llm.Error{Kind: llm.KindInvalidRequest, Op: op, Provider: defaultProvider, Err: fmt.Errorf(format, args...)}
}

func unsupported(op, message string) *llm.Error {
	return &llm.Error{Kind: llm.KindUnsupported, Op: op, Provider: defaultProvider, Err: errors.New(message)}
}

func transportError(op string, err error) *llm.Error {
	return &llm.Error{Kind: llm.KindTransport, Op: op, Provider: defaultProvider, Err: err}
}

func malformedResponse(op, format string, args ...any) *llm.Error {
	return &llm.Error{Kind: llm.KindMalformedResponse, Op: op, Provider: defaultProvider, Err: fmt.Errorf(format, args...)}
}

func authenticationError(op string, err error) *llm.Error {
	return &llm.Error{Kind: llm.KindAuthentication, Op: op, Provider: defaultProvider, Err: err}
}
