// Package codex adapts OpenAI's ChatGPT subscription Codex Responses protocol
// to llm's provider-neutral generation interfaces. Callers own the ChatGPT
// authentication lifecycle and supply either current credentials or a resolver;
// this package does not perform or persist OAuth logins. Models explicitly
// configured with CapabilityAudio accept user-message WAV, MP3/MPEG, M4A/MP4,
// WebM, and Ogg input through the subscription Responses backend.
//
// TransportSSE is the compatibility-preserving default. TransportWebSocket
// reuses an idle session connection but always sends full context.
// TransportWebSocketCached and TransportAuto may send a verified continuation
// delta only when the canonical request exactly extends the preceding request
// and response. Requests with per-attempt headers use transient sockets. Auto
// falls back to SSE only when the WebSocket handshake fails before
// response.create is sent; strict WebSocket modes never fall back, and ambiguous
// response.create write failures are not retried internally. Call Client.Close
// to release cached and active WebSocket resources early.
package codex
