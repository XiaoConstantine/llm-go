package models

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/anthropic"
	"github.com/XiaoConstantine/llm-go/openai"
)

// API identifies a provider wire protocol.
type API string

const (
	// OpenAIChatCompletions selects the OpenAI-compatible Chat Completions API.
	OpenAIChatCompletions API = "openai-chat-completions"
	// AnthropicMessages selects the Anthropic-compatible Messages API.
	AnthropicMessages API = "anthropic-messages"
)

// ProviderConfig configures one provider. ID is the provider name used by
// llm.ModelInfo and llm.Error. API selects its wire protocol. APIKey is optional
// for local servers and gateways that authenticate with Headers instead.
//
// BaseURL, HTTPClient, and Headers are forwarded to the selected provider
// implementation. New copies Headers. The caller remains responsible for safe
// concurrent use of HTTPClient. A nil HTTPClient ultimately uses
// [http.DefaultClient], so operations should have a context deadline when an
// unbounded request is not acceptable.
type ProviderConfig struct {
	ID         string
	API        API
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Headers    http.Header
}

type providerConfig struct {
	id         string
	api        API
	apiKey     string
	baseURL    string
	httpClient *http.Client
	headers    http.Header
}

// Collection is an immutable set of provider configurations. It is safe for
// concurrent use when its configured HTTP clients are safe for concurrent use.
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
		case OpenAIChatCompletions, AnthropicMessages:
		case "":
			return nil, configureError(id, "API must not be empty")
		default:
			return nil, configureError(id, "API %q is not supported", api)
		}

		providers[id] = providerConfig{
			id:         id,
			api:        api,
			apiKey:     config.APIKey,
			baseURL:    config.BaseURL,
			httpClient: config.HTTPClient,
			headers:    config.Headers.Clone(),
		}
	}

	return &Collection{providers: providers}, nil
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
	if c == nil {
		return nil, resolveError(provider, "provider collection is nil")
	}

	config, exists := c.providers[provider]
	if !exists {
		return nil, resolveError(provider, "provider is not configured")
	}

	switch config.api {
	case OpenAIChatCompletions:
		generator, err := openai.New(openai.Config{
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
		return generator, nil
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
		return generator, nil
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
