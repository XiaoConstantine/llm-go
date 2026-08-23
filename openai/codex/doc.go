// Package codex adapts OpenAI's ChatGPT subscription Codex Responses protocol
// to llm's provider-neutral generation interfaces. Callers own the ChatGPT
// authentication lifecycle and supply either current credentials or a resolver;
// this package does not perform or persist OAuth logins.
package codex
