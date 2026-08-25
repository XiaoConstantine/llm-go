package models

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/anthropic"
	"github.com/XiaoConstantine/llm-go/gemini"
	"github.com/XiaoConstantine/llm-go/openai"
	openaicodex "github.com/XiaoConstantine/llm-go/openai/codex"
	openairesponses "github.com/XiaoConstantine/llm-go/openai/responses"
)

// API identifies a provider wire protocol.
type API string

const (
	// OpenAIResponses selects OpenAI's Responses API.
	OpenAIResponses API = "openai-responses"
	// OpenAIChatCompletions selects the OpenAI-compatible Chat Completions API.
	OpenAIChatCompletions API = "openai-chat-completions"
	// OpenAICodexResponses selects the ChatGPT subscription Codex Responses API.
	OpenAICodexResponses API = "openai-codex-responses"
	// AnthropicMessages selects the Anthropic-compatible Messages API.
	AnthropicMessages API = "anthropic-messages"
	// GeminiGenerateContent selects the Gemini Developer API GenerateContent protocol.
	GeminiGenerateContent API = "gemini-generate-content"
)

// Credentials contains current token-based provider credentials. AccountID may
// be empty when the selected protocol can derive it from AccessToken.
type Credentials struct {
	AccessToken string
	AccountID   string
}

// CredentialResolver returns current token-based provider credentials.
// rejectedAccessToken is nonempty after a provider rejects a token with HTTP
// 401. A resolver must be safe for concurrent use.
type CredentialResolver func(ctx context.Context, rejectedAccessToken string) (Credentials, error)

// ProviderConfig configures one provider. ID is the provider name used by
// llm.ModelInfo and llm.Error. API selects its wire protocol. APIKey configures
// API-key protocols. Credentials configures static token credentials for Codex
// Responses or Anthropic subscription OAuth. ResolveCredentials is supported
// only by Codex Responses; use CredentialManager for live Anthropic OAuth.
//
// BaseURL, HTTPClient, and Headers are forwarded to the selected provider
// implementation. New copies Headers. The caller remains responsible for safe
// concurrent use of HTTPClient. A nil HTTPClient ultimately uses
// [http.DefaultClient], so operations should have a context deadline when an
// unbounded request is not acceptable.
type ProviderConfig struct {
	ID                 string
	API                API
	APIKey             string
	Credentials        Credentials
	ResolveCredentials CredentialResolver
	BaseURL            string
	HTTPClient         *http.Client
	Headers            http.Header
}

type providerConfig struct {
	id                 string
	api                API
	apiKey             string
	credentials        Credentials
	resolveCredentials CredentialResolver
	baseURL            string
	httpClient         *http.Client
	headers            http.Header
}

// Collection is an immutable set of provider configurations. It is safe for
// concurrent use when its configured HTTP clients and credential resolvers are
// safe for concurrent use.
type Collection struct {
	providers   map[string]providerConfig
	credentials *CredentialManager
}

// New constructs a Collection. At least one provider is required. Provider IDs
// must be unique after surrounding whitespace is removed.
func New(configs ...ProviderConfig) (*Collection, error) {
	return NewWithCredentialManager(nil, configs...)
}

// NewWithCredentialManager constructs a Collection that resolves stored API-key
// or OAuth credentials when a generator is created. Explicit ProviderConfig
// credentials take precedence over the manager. Manager-backed credentials are
// resolved again for every provider request, so a long-lived generator observes
// API-key rotation and coalesced OAuth refreshes.
func NewWithCredentialManager(manager *CredentialManager, configs ...ProviderConfig) (*Collection, error) {
	if len(configs) == 0 {
		return nil, configureError("", "at least one provider is required")
	}

	providers := make(map[string]providerConfig, len(configs))
	for index, config := range configs {
		id := strings.TrimSpace(config.ID)
		if id == "" {
			return nil, configureError("", "providers[%d].ID must not be empty", index)
		}
		if _, exists := providers[id]; exists {
			return nil, configureError(id, "provider ID is configured more than once")
		}

		api := API(strings.TrimSpace(string(config.API)))
		switch api {
		case OpenAIResponses, OpenAIChatCompletions, OpenAICodexResponses, AnthropicMessages, GeminiGenerateContent:
		case "":
			return nil, configureError(id, "API must not be empty")
		default:
			return nil, configureError(id, "API %q is not supported", api)
		}
		if err := validateCredentials(id, api, config); err != nil {
			return nil, err
		}

		providers[id] = providerConfig{
			id:                 id,
			api:                api,
			apiKey:             config.APIKey,
			credentials:        config.Credentials,
			resolveCredentials: config.ResolveCredentials,
			baseURL:            config.BaseURL,
			httpClient:         config.HTTPClient,
			headers:            config.Headers.Clone(),
		}
	}

	return &Collection{providers: providers, credentials: manager}, nil
}

func validateCredentials(provider string, api API, config ProviderConfig) error {
	hasTokenCredentials := strings.TrimSpace(config.Credentials.AccessToken) != "" ||
		strings.TrimSpace(config.Credentials.AccountID) != "" || config.ResolveCredentials != nil
	if api == OpenAICodexResponses {
		if strings.TrimSpace(config.APIKey) != "" {
			return configureError(provider, "APIKey is not used by OpenAICodexResponses; use Credentials or ResolveCredentials")
		}
		if config.ResolveCredentials != nil &&
			(strings.TrimSpace(config.Credentials.AccessToken) != "" || strings.TrimSpace(config.Credentials.AccountID) != "") {
			return configureError(provider, "Credentials must be empty when ResolveCredentials is set")
		}
		return nil
	}
	if api == AnthropicMessages {
		if config.ResolveCredentials != nil {
			return configureError(provider, "ResolveCredentials is supported only by OpenAICodexResponses; use CredentialManager for live Anthropic OAuth")
		}
		if strings.TrimSpace(config.Credentials.AccessToken) != "" && strings.TrimSpace(config.Credentials.AccountID) == "" {
			return nil
		}
	}
	if hasTokenCredentials {
		return configureError(provider, "Credentials and ResolveCredentials are only supported by token-based protocols (Codex and Anthropic OAuth)")
	}
	return nil
}

// GeneratorFor constructs a provider-neutral generator from a catalog model.
// The model's API must match its configured provider protocol.
func (c *Collection) GeneratorFor(model llm.Model) (llm.Generator, error) {
	normalized, err := normalizeModel(model)
	if err != nil {
		return nil, resolveError(strings.TrimSpace(model.Provider), err.Error())
	}
	if c == nil {
		return nil, resolveError(normalized.Provider, "provider collection is nil")
	}
	config, exists := c.providers[normalized.Provider]
	if !exists {
		return nil, resolveError(normalized.Provider, "provider is not configured")
	}
	if string(normalized.API) != string(config.api) {
		return nil, resolveError(normalized.Provider,
			fmt.Sprintf("model API %q does not match configured API %q", normalized.API, config.api))
	}
	return c.Generator(normalized.Info())
}

// Generator constructs a provider-neutral generator for info. Provider and
// Model are required. Capabilities are validated by the selected protocol
// implementation.
func (c *Collection) Generator(info llm.ModelInfo) (llm.Generator, error) {
	return c.GeneratorContext(context.Background(), info)
}

// GeneratorContext constructs a generator after resolving any managed
// credential. Credential refresh honors ctx.
func (c *Collection) GeneratorContext(ctx context.Context, info llm.ModelInfo) (llm.Generator, error) {
	provider := strings.TrimSpace(info.Provider)
	if provider == "" {
		return nil, resolveError("", "model provider must not be empty")
	}
	if strings.TrimSpace(info.Model) == "" {
		return nil, resolveError(provider, "model name must not be empty")
	}
	if info.Cost != nil {
		if err := info.Cost.Validate(); err != nil {
			return nil, resolveError(provider, fmt.Sprintf("model cost: %v", err))
		}
	}
	if c == nil {
		return nil, resolveError(provider, "provider collection is nil")
	}

	config, exists := c.providers[provider]
	if !exists {
		return nil, resolveError(provider, "provider is not configured")
	}
	var err error
	config, err = c.resolveCredential(ctx, config)
	if err != nil {
		return nil, err
	}
	if err := info.Compatibility.Validate(llm.API(config.api)); err != nil {
		return nil, resolveError(provider, fmt.Sprintf("model compatibility: %v", err))
	}

	switch config.api {
	case OpenAIResponses:
		responsesConfig := openairesponses.Config{
			Provider:     config.id,
			Model:        info.Model,
			Capabilities: info.Capabilities,
			APIKey:       config.apiKey,
			BaseURL:      config.baseURL,
			HTTPClient:   config.httpClient,
			Headers:      config.headers,
		}
		var generator llm.Generator
		var err error
		if compatibility := openAIResponsesCompatibility(config.id, info.Compatibility); compatibility != nil {
			generator, err = openairesponses.NewWithCompatibility(responsesConfig, compatibility)
		} else {
			generator, err = openairesponses.NewWithOptions(responsesConfig, openairesponses.Options{EncryptedReasoning: config.id == ProviderXAI})
		}
		if err != nil {
			return nil, err
		}
		return withPricing(generator, info), nil
	case OpenAIChatCompletions:
		maxTokensField := openai.MaxTokensFieldCompletion
		if config.id == ProviderDeepSeek {
			maxTokensField = openai.MaxTokensFieldLegacy
		}
		var modelCompatibility *llm.OpenAIChatCompatibility
		if info.Compatibility != nil && info.Compatibility.OpenAIChat != nil {
			compatibility := *info.Compatibility.OpenAIChat
			if compatibility.MaxTokensField == "" {
				compatibility.MaxTokensField = maxTokensField
			}
			modelCompatibility = &compatibility
		}
		var generator llm.Generator
		var err error
		openAIConfig := openai.Config{
			Provider:     config.id,
			Model:        info.Model,
			Capabilities: info.Capabilities,
			APIKey:       config.apiKey,
			BaseURL:      config.baseURL,
			HTTPClient:   config.httpClient,
			Headers:      config.headers,
		}
		if modelCompatibility != nil {
			generator, err = openai.NewWithCompatibility(openAIConfig, modelCompatibility)
		} else {
			generator, err = openai.NewWithOptions(openAIConfig, openai.Options{MaxTokensField: maxTokensField})
		}
		if err != nil {
			return nil, err
		}
		return withPricing(generator, info), nil
	case OpenAICodexResponses:
		var resolver openaicodex.CredentialResolver
		if config.resolveCredentials != nil {
			resolver = func(ctx context.Context, rejectedAccessToken string) (openaicodex.Credentials, error) {
				credentials, err := config.resolveCredentials(ctx, rejectedAccessToken)
				return openaicodex.Credentials{
					AccessToken: credentials.AccessToken,
					AccountID:   credentials.AccountID,
				}, err
			}
		}
		codexConfig := openaicodex.Config{
			Provider:           config.id,
			Model:              info.Model,
			Capabilities:       info.Capabilities,
			AccessToken:        config.credentials.AccessToken,
			AccountID:          config.credentials.AccountID,
			ResolveCredentials: resolver,
			BaseURL:            config.baseURL,
			HTTPClient:         config.httpClient,
			Headers:            config.headers,
		}
		var generator llm.Generator
		var err error
		if info.Compatibility != nil && info.Compatibility.OpenAIResponses != nil {
			compatibility := *info.Compatibility.OpenAIResponses
			generator, err = openaicodex.NewWithCompatibility(codexConfig, &compatibility)
		} else {
			generator, err = openaicodex.New(codexConfig)
		}
		if err != nil {
			return nil, err
		}
		return withPricing(generator, info), nil
	case AnthropicMessages:
		generator, err := anthropic.NewWithOptions(anthropic.Config{
			Provider:     config.id,
			Model:        info.Model,
			Capabilities: info.Capabilities,
			APIKey:       config.apiKey,
			AccessToken:  config.credentials.AccessToken,
			BaseURL:      config.baseURL,
			HTTPClient:   config.httpClient,
			Headers:      config.headers,
		}, anthropic.Options{ModelCompatibility: anthropicCompatibility(info.Compatibility)})
		if err != nil {
			return nil, err
		}
		return withPricing(generator, info), nil
	case GeminiGenerateContent:
		generator, err := gemini.New(gemini.Config{
			Provider:     config.id,
			Model:        info.Model,
			Capabilities: info.Capabilities,
			APIKey:       config.apiKey,
			BaseURL:      config.baseURL,
			HTTPClient:   config.httpClient,
			Headers:      config.headers,
		})
		if err != nil {
			return nil, err
		}
		return withPricing(generator, info), nil
	default:
		panic("models: invalid configured API")
	}
}

func (c *Collection) resolveCredential(ctx context.Context, config providerConfig) (providerConfig, error) {
	if c.credentials == nil || config.apiKey != "" || config.credentials.AccessToken != "" || config.resolveCredentials != nil {
		return config, nil
	}
	credential, found, err := c.credentials.Resolve(ctx, config.id)
	if err != nil {
		return providerConfig{}, resolveError(config.id, fmt.Sprintf("resolve credential: %v", err))
	}
	if !found {
		return config, nil
	}
	switch credential.Type {
	case CredentialAPIKey:
		if config.api == OpenAICodexResponses {
			return providerConfig{}, resolveError(config.id, "Codex requires an OAuth credential")
		}
		config.apiKey = credential.APIKey
		config.httpClient = c.managedCredentialHTTPClient(config, CredentialAPIKey)
	case CredentialOAuth:
		switch config.api {
		case OpenAICodexResponses:
			config.resolveCredentials = c.managedCodexCredentialResolver(config.id)
			config.credentials = Credentials{}
		case AnthropicMessages:
			config.credentials = Credentials{AccessToken: credential.AccessToken, AccountID: credential.AccountID}
			config.httpClient = c.managedCredentialHTTPClient(config, CredentialOAuth)
		default:
			return providerConfig{}, resolveError(config.id, fmt.Sprintf("OAuth credentials are not supported by API %q", config.api))
		}
	default:
		return providerConfig{}, resolveError(config.id, fmt.Sprintf("credential type %q is not supported", credential.Type))
	}
	return config, nil
}

func (c *Collection) managedCodexCredentialResolver(provider string) CredentialResolver {
	return func(ctx context.Context, rejectedAccessToken string) (Credentials, error) {
		credential, found, err := c.credentials.resolveRejectedOAuth(ctx, provider, rejectedAccessToken)
		if err != nil {
			return Credentials{}, fmt.Errorf("resolve managed credential: %w", err)
		}
		if !found || credential.Type != CredentialOAuth {
			return Credentials{}, fmt.Errorf("managed Codex credential must be OAuth")
		}
		return Credentials{AccessToken: credential.AccessToken, AccountID: credential.AccountID}, nil
	}
}

type managedCredentialTransport struct {
	base     http.RoundTripper
	manager  *CredentialManager
	provider string
	api      API
	kind     CredentialType
}

func (c *Collection) managedCredentialHTTPClient(config providerConfig, kind CredentialType) *http.Client {
	base := config.httpClient
	if base == nil {
		base = http.DefaultClient
	}
	clone := *base
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	clone.Transport = &managedCredentialTransport{base: transport, manager: c.credentials, provider: config.id, api: config.api, kind: kind}
	return &clone
}

func (t *managedCredentialTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	credential, found, err := t.manager.Resolve(request.Context(), t.provider)
	if err != nil {
		return nil, fmt.Errorf("resolve managed credential: %w", err)
	}
	if !found || credential.Type != t.kind {
		return nil, fmt.Errorf("managed credential for %q changed type from %q", t.provider, t.kind)
	}
	clone := request.Clone(request.Context())
	clone.Header = request.Header.Clone()
	switch t.kind {
	case CredentialAPIKey:
		switch t.api {
		case AnthropicMessages:
			clone.Header.Del("Authorization")
			clone.Header.Set("X-Api-Key", credential.APIKey)
		case GeminiGenerateContent:
			for name := range clone.Header {
				if strings.EqualFold(name, "X-Goog-Api-Key") {
					delete(clone.Header, name)
				}
			}
			clone.Header.Set("X-Goog-Api-Key", credential.APIKey)
			// Keep the legacy query credential synchronized for compatible
			// gateways that route Gemini requests by key rather than header.
			urlClone := *request.URL
			query := urlClone.Query()
			query.Set("key", credential.APIKey)
			urlClone.RawQuery = query.Encode()
			clone.URL = &urlClone
		default:
			clone.Header.Set("Authorization", "Bearer "+credential.APIKey)
		}
	case CredentialOAuth:
		clone.Header.Del("X-Api-Key")
		clone.Header.Set("Authorization", "Bearer "+credential.AccessToken)
	}
	return t.base.RoundTrip(clone)
}

func anthropicCompatibility(compatibility *llm.ModelCompatibility) *llm.AnthropicCompatibility {
	if compatibility == nil || compatibility.Anthropic == nil {
		return nil
	}
	value := *compatibility.Anthropic
	return &value
}

func openAIResponsesCompatibility(provider string, compatibility *llm.ModelCompatibility) *llm.OpenAIResponsesCompatibility {
	if compatibility == nil || compatibility.OpenAIResponses == nil {
		return nil
	}
	modelCompatibility := *compatibility.OpenAIResponses
	if modelCompatibility.EncryptedReasoning == llm.CompatibilityDefault && provider == ProviderXAI {
		modelCompatibility.EncryptedReasoning = llm.CompatibilityEnabled
	}
	return &modelCompatibility
}

func configureError(provider, format string, args ...any) error {
	return &llm.Error{
		Kind:     llm.KindInvalidRequest,
		Op:       "configure",
		Provider: provider,
		Err:      fmt.Errorf(format, args...),
	}
}

func resolveError(provider, message string) error {
	return &llm.Error{
		Kind:     llm.KindInvalidRequest,
		Op:       "resolve",
		Provider: provider,
		Err:      errors.New(message),
	}
}
