package models

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
)

// API identifies a provider wire protocol.
type API string

const (
	// OpenAIResponses selects OpenAI's Responses API.
	OpenAIResponses API = "openai-responses"
	// AzureOpenAIResponses selects Azure OpenAI's Responses API.
	AzureOpenAIResponses API = "azure-openai-responses"
	// OpenAIChatCompletions selects the OpenAI-compatible Chat Completions API.
	OpenAIChatCompletions API = "openai-chat-completions"
	// OpenAICodexResponses selects the ChatGPT subscription Codex Responses API.
	OpenAICodexResponses API = "openai-codex-responses"
	// AnthropicMessages selects the Anthropic-compatible Messages API.
	AnthropicMessages API = "anthropic-messages"
	// GeminiGenerateContent selects the Gemini Developer API GenerateContent protocol.
	GeminiGenerateContent API = "gemini-generate-content"
	// MistralConversations selects Mistral's Conversations-compatible chat protocol.
	MistralConversations API = "mistral-conversations"
	// GoogleVertex selects Google Vertex AI's GenerateContent protocol.
	GoogleVertex API = "google-vertex"
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
// BaseURL, Project, Location, HTTPClient, and Headers are forwarded to the
// selected provider implementation. Project and Location configure Vertex AI
// ADC routing. New copies Headers. The caller remains responsible for safe
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
	Project            string
	Location           string
	HTTPClient         *http.Client
	Headers            http.Header
	// AdditionalAPIs explicitly enables model-selected protocols for this
	// provider. Its storage is copied by Collection construction.
	AdditionalAPIs []ProviderAPIConfig
}

// ProviderAPIConfig configures one additional protocol route for a provider.
// Zero-valued connection and credential fields do not inherit from the default
// route; configure each route explicitly.
type ProviderAPIConfig struct {
	API                llm.API
	APIKey             string
	Credentials        Credentials
	ResolveCredentials CredentialResolver
	BaseURL            string
	Project            string
	Location           string
	HTTPClient         *http.Client
	Headers            http.Header
}

type providerConfig struct {
	id         string
	defaultAPI llm.API
	routes     map[llm.API]providerRoute
}

type providerRoute struct {
	api                llm.API
	apiKey             string
	credentials        Credentials
	resolveCredentials CredentialResolver
	baseURL            string
	project            string
	location           string
	httpClient         *http.Client
	headers            http.Header
}

// Collection is an immutable set of provider configurations. It is safe for
// concurrent use when its configured HTTP clients and credential resolvers are
// safe for concurrent use.
type Collection struct {
	providers   map[string]providerConfig
	credentials *CredentialManager
	registry    *FactoryRegistry
}

// New constructs a Collection. At least one provider is required. Provider IDs
// must be unique after surrounding whitespace is removed.
func New(configs ...ProviderConfig) (*Collection, error) {
	return NewWithCredentialManagerAndRegistry(nil, defaultFactoryRegistry(), configs...)
}

// NewWithRegistry constructs a Collection using registry and no managed credentials.
func NewWithRegistry(registry *FactoryRegistry, configs ...ProviderConfig) (*Collection, error) {
	return NewWithCredentialManagerAndRegistry(nil, registry, configs...)
}

// NewWithCredentialManager constructs a Collection with the default registry.
func NewWithCredentialManager(manager *CredentialManager, configs ...ProviderConfig) (*Collection, error) {
	return NewWithCredentialManagerAndRegistry(manager, defaultFactoryRegistry(), configs...)
}

// NewWithCredentialManagerAndRegistry constructs an immutable Collection.
func NewWithCredentialManagerAndRegistry(manager *CredentialManager, registry *FactoryRegistry, configs ...ProviderConfig) (*Collection, error) {
	if registry == nil {
		return nil, configureError("", "factory registry must not be nil")
	}
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
		defaultAPI := llm.API(strings.TrimSpace(string(config.API)))
		if defaultAPI == "" {
			return nil, configureError(id, "API must not be empty")
		}
		routes := make(map[llm.API]providerRoute, len(config.AdditionalAPIs)+1)
		defaultRoute := providerRoute{api: defaultAPI, apiKey: config.APIKey, credentials: config.Credentials,
			resolveCredentials: config.ResolveCredentials, baseURL: config.BaseURL, project: config.Project, location: config.Location,
			httpClient: config.HTTPClient, headers: config.Headers.Clone()}
		if err := validateRoute(id, defaultRoute, registry); err != nil {
			return nil, err
		}
		routes[defaultAPI] = defaultRoute
		for routeIndex, additional := range config.AdditionalAPIs {
			api := llm.API(strings.TrimSpace(string(additional.API)))
			if api == "" {
				return nil, configureError(id, "AdditionalAPIs[%d].API must not be empty", routeIndex)
			}
			if _, exists := routes[api]; exists {
				return nil, configureError(id, "API %q is configured more than once", api)
			}
			route := providerRoute{api: api, apiKey: additional.APIKey, credentials: additional.Credentials,
				resolveCredentials: additional.ResolveCredentials, baseURL: additional.BaseURL,
				project: additional.Project, location: additional.Location,
				httpClient: additional.HTTPClient, headers: additional.Headers.Clone()}
			if err := validateRoute(id, route, registry); err != nil {
				return nil, err
			}
			routes[api] = route
		}
		providers[id] = providerConfig{id: id, defaultAPI: defaultAPI, routes: routes}
	}
	return &Collection{providers: providers, credentials: manager, registry: registry}, nil
}

func validateRoute(provider string, route providerRoute, registry *FactoryRegistry) error {
	if _, exists := registry.factory(route.api); !exists {
		return configureError(provider, "API %q is not registered", route.api)
	}
	config := ProviderConfig{ID: provider, API: API(route.api), APIKey: route.apiKey, Credentials: route.credentials,
		ResolveCredentials: route.resolveCredentials, BaseURL: route.baseURL, Project: route.project, Location: route.location,
		HTTPClient: route.httpClient, Headers: route.headers}
	return validateCredentials(provider, API(route.api), config)
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
	if !builtinAPI(llm.API(api)) {
		if config.ResolveCredentials != nil && (config.Credentials.AccessToken != "" || config.Credentials.AccountID != "") {
			return configureError(provider, "Credentials must be empty when ResolveCredentials is set")
		}
		return nil
	}
	if hasTokenCredentials {
		return configureError(provider, "Credentials and ResolveCredentials are only supported by token-based protocols (Codex and Anthropic OAuth)")
	}
	return nil
}

func builtinAPI(api llm.API) bool {
	switch api {
	case llm.APIOpenAIResponses, llm.APIAzureOpenAIResponses, llm.APIOpenAIChatCompletions, llm.APIOpenAICodexResponses,
		llm.APIAnthropicMessages, llm.APIGeminiGenerateContent, llm.APIMistralConversations, llm.APIGoogleVertex:
		return true
	default:
		return false
	}
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
	if _, exists := config.routes[normalized.API]; !exists {
		return nil, resolveError(normalized.Provider, fmt.Sprintf("model API %q is not enabled for provider", normalized.API))
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
	model := strings.TrimSpace(info.Model)
	if model == "" {
		return nil, resolveError(provider, "model name must not be empty")
	}
	if ctx == nil {
		return nil, resolveError(provider, "context must not be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if info.ContextWindow < 0 || info.MaxOutputTokens < 0 || info.ContextWindow != 0 && info.MaxOutputTokens > info.ContextWindow {
		return nil, resolveError(provider, "model token limits are invalid")
	}
	if info.Cost != nil {
		if err := info.Cost.Validate(); err != nil {
			return nil, resolveError(provider, fmt.Sprintf("model cost: %v", err))
		}
	}
	if c == nil || c.registry == nil {
		return nil, resolveError(provider, "provider collection is nil or uninitialized")
	}
	config, exists := c.providers[provider]
	if !exists {
		return nil, resolveError(provider, "provider is not configured")
	}
	api := info.API
	if api == "" {
		api = config.defaultAPI
	}
	route, exists := config.routes[api]
	if !exists {
		return nil, resolveError(provider, fmt.Sprintf("model API %q is not enabled for provider", api))
	}
	factory, exists := c.registry.factory(api)
	if !exists {
		return nil, resolveError(provider, fmt.Sprintf("model API %q is not registered", api))
	}
	normalizedModel, err := normalizeModel(llm.Model{Provider: provider, ID: model, API: api,
		Capabilities: info.Capabilities, ContextWindow: info.ContextWindow, MaxOutputTokens: info.MaxOutputTokens,
		Reasoning: info.Reasoning, Cost: info.Cost, Compatibility: info.Compatibility})
	if err != nil {
		return nil, resolveError(provider, fmt.Sprintf("model metadata: %v", err))
	}
	ownedInfo := normalizedModel.Info()
	resolved, credentialResolver, err := c.resolveCredential(ctx, provider, route)
	if err != nil {
		return nil, err
	}
	factoryConfig := GeneratorFactoryConfig{
		Provider: provider, API: api, Model: cloneModelInfo(ownedInfo), APIKey: resolved.apiKey,
		Credentials: resolved.credentials, ResolveCredentials: resolved.resolveCredentials,
		ResolveCredential: credentialResolver, BaseURL: resolved.baseURL, HTTPClient: resolved.httpClient,
		Project: route.project, Location: route.location, Headers: resolved.headers.Clone(),
	}
	generator, err := callGeneratorFactory(ctx, factory, factoryConfig)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if nilInterface(generator) {
		return nil, resolveError(provider, fmt.Sprintf("factory for API %q returned a nil generator", api))
	}
	actual, err := callGeneratorInfo(provider, api, generator)
	if err != nil {
		return nil, err
	}
	if actual.Provider != provider || actual.Model != model || actual.API != api {
		return nil, resolveError(provider, fmt.Sprintf("factory for API %q returned inconsistent Info (provider=%q model=%q API=%q)", api, actual.Provider, actual.Model, actual.API))
	}
	return withPricingInfo(generator, ownedInfo, actual), nil
}

func callGeneratorFactory(ctx context.Context, factory GeneratorFactory, config GeneratorFactoryConfig) (generator llm.Generator, err error) {
	defer func() {
		if value := recover(); value != nil {
			generator = nil
			err = &llm.Error{Kind: llm.KindProvider, Op: "factory", Provider: config.Provider,
				Err: fmt.Errorf("generator factory for API %q panicked: %v", config.API, value)}
		}
	}()
	return factory(ctx, config)
}

func callGeneratorInfo(provider string, api llm.API, generator llm.Generator) (info llm.ModelInfo, err error) {
	defer func() {
		if value := recover(); value != nil {
			err = &llm.Error{Kind: llm.KindProvider, Op: "factory", Provider: provider,
				Err: fmt.Errorf("generator Info for API %q panicked: %v", api, value)}
		}
	}()
	return generator.Info(), nil
}

func cloneModelInfo(info llm.ModelInfo) llm.ModelInfo {
	info.Capabilities = append([]llm.Capability(nil), info.Capabilities...)
	if info.Cost != nil {
		cost := *info.Cost
		cost.Tiers = append([]llm.ModelCostTier(nil), cost.Tiers...)
		info.Cost = &cost
	}
	info.Compatibility = cloneCompatibility(info.Compatibility)
	return info
}

func nilInterface(value any) bool {
	if value == nil {
		return true
	}
	reflected := reflect.ValueOf(value)
	switch reflected.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return reflected.IsNil()
	default:
		return false
	}
}

func (c *Collection) resolveCredential(ctx context.Context, provider string, route providerRoute) (providerRoute, FactoryCredentialResolver, error) {
	resolver := c.factoryCredentialResolver(provider, route)
	if c.credentials == nil || route.apiKey != "" || route.credentials.AccessToken != "" || route.resolveCredentials != nil {
		return route, resolver, nil
	}
	credential, found, err := c.credentials.Resolve(ctx, provider)
	if err != nil {
		return providerRoute{}, nil, resolveError(provider, fmt.Sprintf("resolve credential: %v", err))
	}
	if !found {
		return route, resolver, nil
	}
	switch credential.Type {
	case CredentialAPIKey:
		if route.api == llm.APIOpenAICodexResponses {
			return providerRoute{}, nil, resolveError(provider, "Codex requires an OAuth credential")
		}
		route.apiKey = credential.APIKey
		if builtinAPI(route.api) {
			route.httpClient = c.managedCredentialHTTPClient(provider, route, CredentialAPIKey)
		}
	case CredentialOAuth:
		switch route.api {
		case llm.APIOpenAICodexResponses:
			route.resolveCredentials = c.managedCodexCredentialResolver(provider)
			route.credentials = Credentials{}
		case llm.APIAnthropicMessages:
			route.credentials = Credentials{AccessToken: credential.AccessToken, AccountID: credential.AccountID}
			route.httpClient = c.managedCredentialHTTPClient(provider, route, CredentialOAuth)
		default:
			if builtinAPI(route.api) {
				return providerRoute{}, nil, resolveError(provider, fmt.Sprintf("OAuth credentials are not supported by API %q", route.api))
			}
			route.credentials = Credentials{AccessToken: credential.AccessToken, AccountID: credential.AccountID}
		}
	default:
		return providerRoute{}, nil, resolveError(provider, fmt.Sprintf("credential type %q is not supported", credential.Type))
	}
	return route, resolver, nil
}

func (c *Collection) factoryCredentialResolver(provider string, route providerRoute) FactoryCredentialResolver {
	if c.credentials != nil && route.apiKey == "" && route.credentials.AccessToken == "" && route.resolveCredentials == nil {
		return func(ctx context.Context) (StoredCredential, bool, error) {
			credential, found, err := c.credentials.Resolve(ctx, provider)
			return cloneStoredCredential(credential), found, err
		}
	}
	if route.apiKey != "" {
		credential := StoredCredential{Type: CredentialAPIKey, APIKey: route.apiKey}
		return func(ctx context.Context) (StoredCredential, bool, error) {
			if err := ctx.Err(); err != nil {
				return StoredCredential{}, false, err
			}
			return credential, true, nil
		}
	}
	if route.credentials.AccessToken != "" {
		credential := StoredCredential{Type: CredentialOAuth, AccessToken: route.credentials.AccessToken, AccountID: route.credentials.AccountID}
		return func(ctx context.Context) (StoredCredential, bool, error) {
			if err := ctx.Err(); err != nil {
				return StoredCredential{}, false, err
			}
			return credential, true, nil
		}
	}
	if route.resolveCredentials != nil {
		return func(ctx context.Context) (StoredCredential, bool, error) {
			credential, err := route.resolveCredentials(ctx, "")
			if err != nil {
				return StoredCredential{}, false, err
			}
			return StoredCredential{Type: CredentialOAuth, AccessToken: credential.AccessToken, AccountID: credential.AccountID}, true, nil
		}
	}
	return nil
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
	api      llm.API
	kind     CredentialType
}

func (c *Collection) managedCredentialHTTPClient(provider string, route providerRoute, kind CredentialType) *http.Client {
	base := route.httpClient
	if base == nil {
		base = http.DefaultClient
	}
	clone := *base
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	clone.Transport = &managedCredentialTransport{base: transport, manager: c.credentials, provider: provider, api: route.api, kind: kind}
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
		case llm.APIAnthropicMessages:
			clone.Header.Del("Authorization")
			clone.Header.Set("X-Api-Key", credential.APIKey)
		case llm.APIGeminiGenerateContent:
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
