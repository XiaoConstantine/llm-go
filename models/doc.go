// Package models constructs provider-neutral model generators from a reusable
// collection of provider configurations.
//
// Applications configure provider protocols and credentials once, then resolve
// llm.Generator values from llm.ModelInfo. Catalog stores immutable model
// metadata, protocol compatibility, and optional per-million-token pricing
// independently of credentials; resolved generators apply model compatibility
// and attach category costs to usage. BuiltinCatalog provides generated,
// validated metadata from the versioned source in models/catalogsource.
// Built-in profiles configure OpenRouter, Groq, DeepSeek, xAI, Cerebras, and
// Fireworks endpoints and protocol compatibility. CredentialStore defines
// concurrency-safe persistence, while CredentialManager coalesces provider OAuth
// refreshes and can be attached to Collection. AnthropicOAuth and CodexOAuth
// provide non-interactive PKCE exchange and refresh primitives; their default
// subscription endpoints/client IDs are provider-private, unofficial, unstable,
// and replaceable. Applications own browser, callback, and prompt UI. The concrete provider packages remain available
// for protocol-specific configuration that this package does not expose.
package models
