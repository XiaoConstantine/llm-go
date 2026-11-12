// Package models constructs provider-neutral model generators from a reusable
// collection of provider configurations.
//
// Applications configure provider protocols and credentials once, then resolve
// llm.Generator values from llm.ModelInfo. Catalog stores immutable model
// metadata, protocol compatibility, and optional per-million-token pricing
// independently of credentials; resolved generators apply model compatibility
// and attach category costs to usage. FactoryRegistry supports immutable custom
// API registration and explicitly configured mixed-protocol provider routes.
// CatalogManager atomically publishes validated immutable snapshots from static
// baselines, persisted provider overlays, and conditional provider sources.
// BuiltinCatalog provides generated, validated metadata from the versioned
// source in models/catalogsource.
// Built-in profiles configure OpenRouter, Groq, DeepSeek, xAI, Cerebras, and
// Fireworks endpoints and protocol compatibility. CredentialStore defines
// concurrency-safe persistence, while CredentialManager coalesces provider OAuth
// refreshes and can be attached to Collection. AnthropicOAuth and CodexOAuth
// provide non-interactive PKCE exchange and refresh primitives; their default
// subscription endpoints/client IDs are provider-private, unofficial, unstable,
// and replaceable. Applications own browser, callback, and prompt UI. The
// concrete provider packages remain available for protocol-specific
// configuration that this package does not expose.
//
// Pricing is optional enrichment: for a priced model, non-nil Usage with nil
// Cost means cost calculation was skipped. Invalid pricing tables are rejected
// when configured. The pricing wrapper does not repair custom-generator usage:
// Generate preserves inconsistent usage with nil Cost, while Stream followed by
// llm.Collect still rejects that usage under the collector's strict contract.
package models
