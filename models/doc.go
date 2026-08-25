// Package models constructs provider-neutral model generators from a reusable
// collection of provider configurations.
//
// Applications configure provider protocols and credentials once, then resolve
// llm.Generator values from llm.ModelInfo. Catalog stores immutable model
// metadata and optional per-million-token pricing independently of credentials;
// generators resolved from priced models attach category costs to usage.
// Built-in profiles configure OpenRouter, Groq, DeepSeek, xAI, Cerebras, and
// Fireworks endpoints and protocol compatibility. CredentialStore defines
// concurrency-safe persistence, while CredentialManager coalesces provider OAuth refreshes.
// The concrete provider packages remain available
// for protocol-specific configuration that this package does not expose.
package models
