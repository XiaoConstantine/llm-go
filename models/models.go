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
// API-key protocols. Credentials or ResolveCredentials configures token-based
// protocols; currently OpenAICodexResponses is the only such protocol.
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
	providers map[string]providerConfig
}

// New constructs a Collection. At least one provider is required. Provider IDs
// must be unique after surrounding whitespace is removed.
func New(configs ...ProviderConfig) (*Collection, error) {
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

	return &Collection{providers: providers}, nil
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
	if hasTokenCredentials {
		return configureError(provider, "Credentials and ResolveCredentials are only supported by token-based protocols")
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

	switch config.api {
	case OpenAIResponses:
		generator, err := openairesponses.NewWithOptions(openairesponses.Config{
			Provider:     config.id,
			Model:        info.Model,
			Capabilities: info.Capabilities,
			APIKey:       config.apiKey,
			BaseURL:      config.baseURL,
			HTTPClient:   config.httpClient,
			Headers:      config.headers,
		}, openairesponses.Options{EncryptedReasoning: config.id == ProviderXAI})
		if err != nil {
			return nil, err
		}
		return withPricing(generator, info), nil
	case OpenAIChatCompletions:
		maxTokensField := openai.MaxTokensFieldCompletion
		if config.id == ProviderDeepSeek {
			maxTokensField = openai.MaxTokensFieldLegacy
		}
		generator, err := openai.NewWithOptions(openai.Config{
			Provider:     config.id,
			Model:        info.Model,
			Capabilities: info.Capabilities,
			APIKey:       config.apiKey,
			BaseURL:      config.baseURL,
			HTTPClient:   config.httpClient,
			Headers:      config.headers,
		}, openai.Options{MaxTokensField: maxTokensField})
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
		generator, err := openaicodex.New(openaicodex.Config{
			Provider:           config.id,
			Model:              info.Model,
			Capabilities:       info.Capabilities,
			AccessToken:        config.credentials.AccessToken,
			AccountID:          config.credentials.AccountID,
			ResolveCredentials: resolver,
			BaseURL:            config.baseURL,
			HTTPClient:         config.httpClient,
			Headers:            config.headers,
		})
		if err != nil {
			return nil, err
		}
		return withPricing(generator, info), nil
	case AnthropicMessages:
		generator, err := anthropic.New(anthropic.Config{
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
