package models

import "strings"

const (
	ProviderOpenRouter   = "openrouter"
	ProviderGroq         = "groq"
	ProviderDeepSeek     = "deepseek"
	ProviderXAI          = "xai"
	ProviderCerebras     = "cerebras"
	ProviderFireworks    = "fireworks"
	ProviderMistral      = "mistral"
	ProviderGoogleVertex = "google-vertex"
)

// ProviderProfile contains the protocol defaults for a built-in provider. Its
// fields are exposed through accessors so profiles remain immutable values.
type ProviderProfile struct {
	id      string
	api     API
	baseURL string
}

// ID returns the canonical provider ID.
func (p ProviderProfile) ID() string { return p.id }

// API returns the provider's wire protocol.
func (p ProviderProfile) API() API { return p.api }

// BaseURL returns the provider's default API base URL.
func (p ProviderProfile) BaseURL() string { return p.baseURL }

// Config constructs an independently owned provider configuration. Callers may
// override its BaseURL, HTTPClient, or Headers before passing it to New.
func (p ProviderProfile) Config(apiKey string) ProviderConfig {
	return ProviderConfig{
		ID:      p.id,
		API:     p.api,
		APIKey:  apiKey,
		BaseURL: p.baseURL,
	}
}

var builtinProviderProfiles = [...]ProviderProfile{
	{id: ProviderOpenRouter, api: OpenAIChatCompletions, baseURL: "https://openrouter.ai/api/v1"},
	{id: ProviderGroq, api: OpenAIChatCompletions, baseURL: "https://api.groq.com/openai/v1"},
	{id: ProviderDeepSeek, api: OpenAIChatCompletions, baseURL: "https://api.deepseek.com"},
	{id: ProviderXAI, api: OpenAIResponses, baseURL: "https://api.x.ai/v1"},
	{id: ProviderCerebras, api: OpenAIChatCompletions, baseURL: "https://api.cerebras.ai/v1"},
	{id: ProviderFireworks, api: OpenAIChatCompletions, baseURL: "https://api.fireworks.ai/inference/v1"},
	{id: ProviderMistral, api: MistralConversations, baseURL: "https://api.mistral.ai/v1"},
	{id: ProviderGoogleVertex, api: GoogleVertex},
}

// BuiltinProvider returns protocol defaults for a canonical provider ID.
func BuiltinProvider(id string) (ProviderProfile, bool) {
	id = strings.TrimSpace(id)
	for _, profile := range builtinProviderProfiles {
		if profile.id == id {
			return profile, true
		}
	}
	return ProviderProfile{}, false
}

// BuiltinProviders returns all built-in profiles in stable order. The returned
// slice is owned by the caller.
func BuiltinProviders() []ProviderProfile {
	return append([]ProviderProfile(nil), builtinProviderProfiles[:]...)
}
