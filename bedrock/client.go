// Package bedrock implements Amazon Bedrock's ConverseStream API.
// Filter and guardrail stops preserve text and usage with FinishReasonContentFilter,
// close interrupted text and tool events, and omit executable tool calls.
// Tool calls are committed only after a tool_use stop; empty arguments then
// become {}. Missing, null, or zero usage totals are derived from input, output,
// cache-read, and cache-write counts. A nonzero total that disagrees with those
// categories leaves Usage nil without discarding the response content.
//
// SDK retries are disabled. Use llm.WithRetry for explicit retries, including
// SDK-detected clock skew: the SDK retains the corrected signing time and its
// typed retry hint is preserved on the returned llm.Error.
package bedrock

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"regexp"
	"slices"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/adapterutil"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go/auth/bearer"
)

const defaultProvider = "amazon-bedrock"

// EventStream is one Bedrock event stream. Close must release its resources
// and return promptly; Client stops waiting for Events when its context ends.
type EventStream interface {
	Events() <-chan types.ConverseStreamOutput
	Close() error
	Err() error
}

// ConverseStreamAPI is the Bedrock operation used by Client. It permits
// deterministic transports and tests without changing request semantics.
// Implementations must honor context cancellation while opening the stream.
type ConverseStreamAPI interface {
	ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput) (EventStream, error)
}

type awsRuntime struct{ client *bedrockruntime.Client }

func (runtime awsRuntime) ConverseStream(ctx context.Context, input *bedrockruntime.ConverseStreamInput) (EventStream, error) {
	output, err := runtime.client.ConverseStream(ctx, input)
	if err != nil {
		return nil, err
	}
	if output == nil {
		return nil, nil
	}
	return output.GetStream(), nil
}

// Config configures Bedrock ConverseStream. Model is required. Region defaults
// through the AWS SDK chain, then to us-east-1 when neither Region nor Profile
// is set. APIKey is a Bedrock bearer token. Otherwise normal AWS credentials
// are used. SkipAuth installs dummy credentials for unauthenticated gateways.
// A nil HTTPClient preserves the AWS SDK's default client and CA-bundle
// configuration. Static and per-attempt headers apply only to Bedrock runtime
// requests, never to credential resolution. SDK retries are disabled; use
// llm.WithRetry for an explicit retry policy. SDK-detected clock skew carries a
// Retryable hint so an explicit retry can use the corrected signing time.
type Config struct {
	Provider     string
	Model        string
	Capabilities []llm.Capability
	Reasoning    bool
	APIKey       string
	Region       string
	Profile      string
	BaseURL      string
	HTTPClient   *http.Client
	Headers      http.Header
	SkipAuth     bool
	Runtime      ConverseStreamAPI
}

// Client is an immutable Bedrock client.
// Provider messages indicating oversized model input classify as KindContextLimit.
// The explicit "input is too long for requested model" and "prompt is too long"
// messages match regardless of provider error code, not just ValidationException;
// authentication, permission, and throttling errors still take precedence.
type Client struct {
	provider     string
	model        string
	capabilities []llm.Capability
	reasoning    bool
	region       string
	runtime      ConverseStreamAPI
}

// New constructs a Bedrock client.
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
	capabilities, err := configureCapabilities(config.Capabilities)
	if err != nil {
		return nil, err
	}
	if config.Runtime != nil {
		value := reflect.ValueOf(config.Runtime)
		if value.Kind() == reflect.Pointer && value.IsNil() {
			return nil, configError("runtime must not be nil")
		}
		return &Client{provider: provider, model: model, capabilities: capabilities, reasoning: config.Reasoning,
			region: strings.TrimSpace(config.Region), runtime: config.Runtime}, nil
	}
	region := strings.TrimSpace(config.Region)
	configuredRegion := region
	if configuredRegion == "" {
		configuredRegion = strings.TrimSpace(os.Getenv("AWS_REGION"))
	}
	if configuredRegion == "" {
		configuredRegion = strings.TrimSpace(os.Getenv("AWS_DEFAULT_REGION"))
	}
	profile := strings.TrimSpace(config.Profile)
	hasProfile := profile != "" || strings.TrimSpace(os.Getenv("AWS_PROFILE")) != ""
	baseURL := strings.TrimSpace(config.BaseURL)
	skipAuth := config.SkipAuth || os.Getenv("AWS_BEDROCK_SKIP_AUTH") == "1"
	if baseURL != "" {
		parsed, parseErr := url.Parse(baseURL)
		if parseErr != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" || parsed.User != nil {
			return nil, configError("base URL must be an absolute HTTP(S) URL without credentials, query, or fragment")
		}
	}
	loadOptions := []func(*awsconfig.LoadOptions) error{}
	if region != "" {
		loadOptions = append(loadOptions, awsconfig.WithRegion(region))
	} else if configuredRegion == "" && !hasProfile {
		if endpointRegion := standardEndpointRegion(baseURL); endpointRegion != "" {
			loadOptions = append(loadOptions, awsconfig.WithRegion(endpointRegion))
		} else {
			loadOptions = append(loadOptions, awsconfig.WithRegion("us-east-1"))
		}
	}
	if profile != "" {
		loadOptions = append(loadOptions, awsconfig.WithSharedConfigProfile(profile))
	}
	if config.HTTPClient != nil {
		clone := *config.HTTPClient
		loadOptions = append(loadOptions, awsconfig.WithHTTPClient(&clone))
	}
	if skipAuth {
		loadOptions = append(loadOptions, awsconfig.WithCredentialsProvider(credentials.NewStaticCredentialsProvider("dummy-access-key", "dummy-secret-key", "")))
	}
	awsConfig, err := awsconfig.LoadDefaultConfig(context.Background(), loadOptions...)
	if err != nil {
		return nil, configError("load AWS configuration: %w", err)
	}
	token := strings.TrimSpace(config.APIKey)
	if token == "" {
		token = strings.TrimSpace(os.Getenv("AWS_BEARER_TOKEN_BEDROCK"))
	}
	sdkClient := bedrockruntime.NewFromConfig(awsConfig, func(options *bedrockruntime.Options) {
		options.Retryer = aws.NopRetryer{}
		// Wrap only the resolved runtime client, not the credential chain's
		// client. NewFromConfig has already applied the SDK transport defaults.
		options.HTTPClient = headerClient{base: options.HTTPClient, headers: config.Headers.Clone()}
		if shouldPinEndpoint(baseURL, configuredRegion, profileIndicator(hasProfile)) {
			options.BaseEndpoint = aws.String(baseURL)
		}
		if token != "" && !skipAuth {
			options.BearerAuthTokenProvider = bearer.StaticTokenProvider{Token: bearer.Token{Value: token}}
			options.AuthSchemePreference = []string{"httpBearerAuth"}
		}
	})
	return &Client{provider: provider, model: model, capabilities: capabilities, reasoning: config.Reasoning,
		region: awsConfig.Region, runtime: awsRuntime{client: sdkClient}}, nil
}

func profileIndicator(configured bool) string {
	if configured {
		return "configured"
	}
	return ""
}

type headerClient struct {
	base    bedrockruntime.HTTPClient
	headers http.Header
}

func (client headerClient) Do(request *http.Request) (*http.Response, error) {
	attemptHeaders := llm.AttemptHeaders(request.Context())
	if len(attemptHeaders) == 0 && len(client.headers) == 0 {
		return client.base.Do(request)
	}
	clone := request.Clone(request.Context())
	for _, headers := range []http.Header{attemptHeaders, client.headers} {
		for name, values := range headers {
			if strings.EqualFold(name, "Authorization") || strings.EqualFold(name, "Host") || strings.EqualFold(name, "Content-Length") {
				continue
			}
			clone.Header.Del(name)
			for _, value := range values {
				clone.Header.Add(name, value)
			}
		}
	}
	return client.base.Do(clone)
}

var standardEndpoint = regexp.MustCompile(`^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$`)

func standardEndpointRegion(value string) string {
	parsed, err := url.Parse(value)
	if err != nil {
		return ""
	}
	match := standardEndpoint.FindStringSubmatch(strings.ToLower(parsed.Hostname()))
	if len(match) == 2 {
		return match[1]
	}
	return ""
}

func shouldPinEndpoint(baseURL, region, profile string) bool {
	if baseURL == "" {
		return false
	}
	return standardEndpointRegion(baseURL) == "" || region == "" && profile == ""
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

func (c *Client) Info() llm.ModelInfo {
	return llm.ModelInfo{Provider: c.provider, Model: c.model, API: llm.APIBedrockConverseStream,
		Capabilities: slices.Clone(c.capabilities), Reasoning: c.reasoning}
}

func (c *Client) Generate(ctx context.Context, request llm.Request) (_ *llm.Response, err error) {
	defer func() { err = relabelProviderError(err, c.provider) }()
	input, err := c.prepare(ctx, "generate", request, false)
	if err != nil {
		return nil, err
	}
	stream := internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return c.produce(producerCtx, "generate", request, input, emit)
	})
	response, err := llm.CollectStructural(stream)
	if err != nil {
		return nil, err
	}
	return response, nil
}

func (c *Client) Stream(ctx context.Context, request llm.Request) (_ llm.Stream, err error) {
	defer func() { err = relabelProviderError(err, c.provider) }()
	input, err := c.prepare(ctx, "stream", request, true)
	if err != nil {
		return nil, err
	}
	stream := internalstream.New(ctx, func(producerCtx context.Context, emit internalstream.Emit) error {
		return c.produce(producerCtx, "stream", request, input, emit)
	})
	return llm.ValidateToolCallStructureStream(stream, c.provider)
}

func (c *Client) prepare(ctx context.Context, op string, request llm.Request, streaming bool) (*bedrockruntime.ConverseStreamInput, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	if err := contextErr(ctx); err != nil {
		return nil, err
	}
	if streaming && !slices.Contains(c.capabilities, llm.CapabilityStreaming) {
		return nil, unsupported(op, "configured model does not declare streaming capability")
	}
	if err := c.checkCapabilities(op, request); err != nil {
		return nil, err
	}
	return buildInputFor(op, c.model, c.region, c.reasoning, request)
}

func (c *Client) checkCapabilities(op string, request llm.Request) error {
	if request.ProviderOptions != nil {
		return unsupported(op, "provider options are not supported by the Bedrock adapter")
	}
	if request.HasCacheBreakpoints() {
		return unsupported(op, "explicit cache breakpoint placement is not supported by the Bedrock adapter")
	}
	if documents, misplaced := adapterutil.FileParts(request); documents || misplaced {
		return unsupported(op, "document content is not supported by the Bedrock adapter")
	}
	tools, vision := len(request.Tools) != 0, false
	for _, message := range request.Messages {
		tools = tools || len(message.ToolCalls) != 0 || len(message.ToolResults) != 0
		for _, part := range message.Content {
			vision = vision || part.Kind == llm.PartImage
			if part.Kind == llm.PartAudio {
				return unsupported(op, "Bedrock ConverseStream does not support portable audio input")
			}
		}
		for _, result := range message.ToolResults {
			for _, part := range result.Content {
				vision = vision || part.Kind == llm.PartImage
				if part.Kind == llm.PartAudio {
					return unsupported(op, "Bedrock ConverseStream does not support audio tool results")
				}
			}
		}
	}
	if tools && !slices.Contains(c.capabilities, llm.CapabilityTools) {
		return unsupported(op, "configured model does not declare tool capability")
	}
	if vision && !slices.Contains(c.capabilities, llm.CapabilityVision) {
		return unsupported(op, "configured model does not declare vision capability")
	}
	if request.ResponseFormat == llm.ResponseFormatJSON {
		return unsupported(op, "Bedrock ConverseStream does not expose portable JSON response mode")
	}
	if (request.ReasoningBudgetTokens != 0 || request.ReasoningEffort != llm.ReasoningEffortDefault && request.ReasoningEffort != llm.ReasoningEffortNone) && !c.reasoning {
		return unsupported(op, "configured model does not support reasoning controls")
	}
	if request.CacheKey != "" || request.SessionID != "" {
		return unsupported(op, "Bedrock ConverseStream does not support portable cache keys or session affinity")
	}
	if request.PresencePenalty != nil || request.FrequencyPenalty != nil {
		return unsupported(op, "Bedrock ConverseStream does not support portable presence or frequency penalties")
	}
	if request.ParallelToolCalls != nil {
		return unsupported(op, "Bedrock ConverseStream does not support a portable parallel tool call control")
	}
	return nil
}

func contextErr(ctx context.Context) error {
	return adapterutil.ContextError(ctx)
}

func configError(format string, args ...any) error {
	return modelError(llm.KindInvalidRequest, "configure", format, args...)
}
func requestError(op, format string, args ...any) error {
	return modelError(llm.KindInvalidRequest, op, format, args...)
}
func unsupported(op, message string) error {
	return &llm.Error{Kind: llm.KindUnsupported, Op: op, Provider: defaultProvider, Err: errors.New(message)}
}
func malformed(op, format string, args ...any) error {
	return modelError(llm.KindMalformedResponse, op, format, args...)
}
func modelError(kind llm.ErrorKind, op, format string, args ...any) error {
	return &llm.Error{Kind: kind, Op: op, Provider: defaultProvider, Err: fmt.Errorf(format, args...)}
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
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Provider != defaultProvider {
		return err
	}
	clone := *modelErr
	clone.Provider = provider
	return &clone
}

var _ llm.Generator = (*Client)(nil)
