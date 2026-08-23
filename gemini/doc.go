// Package gemini implements llm.Generator with Google's official Gemini SDK.
//
// It targets the Gemini Developer API. Vertex AI configuration is intentionally
// outside this package's current surface. Assistant messages may contain opaque
// ProviderData that preserves Gemini thought signatures and part ordering;
// callers should pass that data back unchanged with conversation history.
package gemini
