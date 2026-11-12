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
// Idle connections continue reading control frames. Peer closure or data
// observed without an active lease evicts the connection. Frames arriving after
// a new lease begins are not correlated with the preceding response.
// A reused connection closed before any response event reports a transport
// error, without resending the request or falling back to SSE.
//
// SSE wire events, including all lines and delimiters, are limited to 17 MiB:
// the WebSocket JSON budget of 16 MiB plus 1 MiB for SSE framing. Larger events
// are malformed responses. HTTP error bodies are limited to 1 MiB.
package codex
