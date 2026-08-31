package models

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"slices"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/anthropic"
	"github.com/XiaoConstantine/llm-go/gemini"
	"github.com/XiaoConstantine/llm-go/mistral"
	"github.com/XiaoConstantine/llm-go/openai"
	openaiAzure "github.com/XiaoConstantine/llm-go/openai/azure"
	openaicodex "github.com/XiaoConstantine/llm-go/openai/codex"
	openairesponses "github.com/XiaoConstantine/llm-go/openai/responses"
	"github.com/XiaoConstantine/llm-go/vertex"
)

// FactoryCredentialResolver returns a current provider credential. Custom
// factories may retain the callback but not returned secret-bearing values.
type FactoryCredentialResolver func(context.Context) (StoredCredential, bool, error)

// GeneratorFactoryConfig is an owned construction snapshot. Model, Headers,
// Credentials, and returned resolver values do not alias Collection internals.
type GeneratorFactoryConfig struct {
	Provider           string
	API                llm.API
	Model              llm.ModelInfo
	APIKey             string
	Credentials        Credentials
	ResolveCredentials CredentialResolver
	ResolveCredential  FactoryCredentialResolver
	BaseURL            string
	Project            string
	Location           string
	HTTPClient         *http.Client
	Headers            http.Header
}

// GeneratorFactory constructs a generator for one API. It may be called
// concurrently, must honor ctx, and must not mutate config storage.
type GeneratorFactory func(context.Context, GeneratorFactoryConfig) (llm.Generator, error)

// FactoryRegistration registers one nonempty API.
type FactoryRegistration struct {
	API     llm.API
	Factory GeneratorFactory
}

// FactoryRegistry is an immutable API-to-factory snapshot safe for concurrent use.
type FactoryRegistry struct {
	factories map[llm.API]GeneratorFactory
}

// NewFactoryRegistry returns the eight built-in factories plus registrations.
// Registering an API already present, including a built-in API, is an error.
func NewFactoryRegistry(registrations ...FactoryRegistration) (*FactoryRegistry, error) {
	factories := builtinFactories()
	for index, registration := range registrations {
		api := llm.API(strings.TrimSpace(string(registration.API)))
		if api == "" {
			return nil, fmt.Errorf("factory registrations[%d].API must not be empty", index)
		}
		if nilFunction(registration.Factory) {
			return nil, fmt.Errorf("factory registrations[%d] for API %q must not be nil", index, api)
		}
		if _, exists := factories[api]; exists {
			return nil, fmt.Errorf("factory API %q is registered more than once", api)
		}
		factories[api] = registration.Factory
	}
	return &FactoryRegistry{factories: factories}, nil
}

// APIs returns registered APIs in stable order.
func (r *FactoryRegistry) APIs() []llm.API {
	if r == nil {
		return nil
	}
	apis := make([]llm.API, 0, len(r.factories))
	for api := range r.factories {
		apis = append(apis, api)
	}
	slices.Sort(apis)
	return apis
}

func defaultFactoryRegistry() *FactoryRegistry {
	registry, err := NewFactoryRegistry()
	if err != nil {
		panic(err)
	}
	return registry
}

func (r *FactoryRegistry) factory(api llm.API) (GeneratorFactory, bool) {
	if r == nil {
		return nil, false
	}
	factory, ok := r.factories[api]
	return factory, ok
}

func builtinFactories() map[llm.API]GeneratorFactory {
	return map[llm.API]GeneratorFactory{
		llm.APIOpenAIResponses:       openAIResponsesFactory,
		llm.APIAzureOpenAIResponses:  azureOpenAIResponsesFactory,
		llm.APIOpenAIChatCompletions: openAIChatFactory,
		llm.APIOpenAICodexResponses:  codexFactory,
		llm.APIAnthropicMessages:     anthropicFactory,
		llm.APIGeminiGenerateContent: geminiFactory,
		llm.APIMistralConversations:  mistralFactory,
		llm.APIGoogleVertex:          vertexFactory,
	}
}

func azureOpenAIResponsesFactory(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
	providerConfig := openaiAzure.Config{Provider: config.Provider, Model: config.Model.Model,
		Capabilities: config.Model.Capabilities, APIKey: config.APIKey, BaseURL: config.BaseURL,
		HTTPClient: config.HTTPClient, Headers: config.Headers}
	if config.Model.Compatibility != nil && config.Model.Compatibility.OpenAIResponses != nil {
		value := *config.Model.Compatibility.OpenAIResponses
		return openaiAzure.NewWithCompatibility(providerConfig, &value)
	}
	return openaiAzure.New(providerConfig)
}

func openAIResponsesFactory(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
	providerConfig := openairesponses.Config{Provider: config.Provider, Model: config.Model.Model, Capabilities: config.Model.Capabilities,
		APIKey: config.APIKey, BaseURL: config.BaseURL, HTTPClient: config.HTTPClient, Headers: config.Headers}
	if compatibility := openAIResponsesCompatibility(config.Provider, config.Model.Compatibility); compatibility != nil {
		return openairesponses.NewWithCompatibility(providerConfig, compatibility)
	}
	return openairesponses.NewWithOptions(providerConfig, openairesponses.Options{EncryptedReasoning: config.Provider == ProviderXAI})
}

func openAIChatFactory(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
	maxTokensField := openai.MaxTokensFieldCompletion
	if config.Provider == ProviderDeepSeek {
		maxTokensField = openai.MaxTokensFieldLegacy
	}
	var compatibility *llm.OpenAIChatCompatibility
	if config.Model.Compatibility != nil && config.Model.Compatibility.OpenAIChat != nil {
		value := *config.Model.Compatibility.OpenAIChat
		if value.MaxTokensField == "" {
			value.MaxTokensField = maxTokensField
		}
		compatibility = &value
	}
	providerConfig := openai.Config{Provider: config.Provider, Model: config.Model.Model, Capabilities: config.Model.Capabilities,
		APIKey: config.APIKey, BaseURL: config.BaseURL, HTTPClient: config.HTTPClient, Headers: config.Headers}
	if compatibility != nil {
		return openai.NewWithCompatibility(providerConfig, compatibility)
	}
	return openai.NewWithOptions(providerConfig, openai.Options{MaxTokensField: maxTokensField})
}

func codexFactory(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
	var resolver openaicodex.CredentialResolver
	if config.ResolveCredentials != nil {
		resolver = func(ctx context.Context, rejected string) (openaicodex.Credentials, error) {
			credentials, err := config.ResolveCredentials(ctx, rejected)
			return openaicodex.Credentials{AccessToken: credentials.AccessToken, AccountID: credentials.AccountID}, err
		}
	}
	providerConfig := openaicodex.Config{Provider: config.Provider, Model: config.Model.Model, Capabilities: config.Model.Capabilities,
		AccessToken: config.Credentials.AccessToken, AccountID: config.Credentials.AccountID, ResolveCredentials: resolver,
		BaseURL: config.BaseURL, HTTPClient: config.HTTPClient, Headers: config.Headers}
	if config.Model.Compatibility != nil && config.Model.Compatibility.OpenAIResponses != nil {
		value := *config.Model.Compatibility.OpenAIResponses
		return openaicodex.NewWithCompatibility(providerConfig, &value)
	}
	return openaicodex.New(providerConfig)
}

func anthropicFactory(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
	return anthropic.NewWithOptions(anthropic.Config{Provider: config.Provider, Model: config.Model.Model, Capabilities: config.Model.Capabilities,
		APIKey: config.APIKey, AccessToken: config.Credentials.AccessToken, BaseURL: config.BaseURL,
		HTTPClient: config.HTTPClient, Headers: config.Headers}, anthropic.Options{ModelCompatibility: anthropicCompatibility(config.Model.Compatibility)})
}

func geminiFactory(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
	return gemini.New(gemini.Config{Provider: config.Provider, Model: config.Model.Model, Capabilities: config.Model.Capabilities,
		Reasoning: config.Model.Reasoning, APIKey: config.APIKey, BaseURL: config.BaseURL, HTTPClient: config.HTTPClient, Headers: config.Headers})
}

func mistralFactory(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
	return mistral.New(mistral.Config{Provider: config.Provider, Model: config.Model.Model,
		Capabilities: config.Model.Capabilities, Reasoning: config.Model.Reasoning, APIKey: config.APIKey,
		BaseURL: config.BaseURL, HTTPClient: config.HTTPClient, Headers: config.Headers})
}

func vertexFactory(_ context.Context, config GeneratorFactoryConfig) (llm.Generator, error) {
	return vertex.New(vertex.Config{Provider: config.Provider, Model: config.Model.Model,
		Capabilities: config.Model.Capabilities, Reasoning: config.Model.Reasoning, APIKey: config.APIKey, Project: config.Project, Location: config.Location,
		BaseURL: config.BaseURL, HTTPClient: config.HTTPClient, Headers: config.Headers})
}

func nilFunction(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	return reflected.Kind() == reflect.Func && reflected.IsNil()
}
