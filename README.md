# llm-go

`llm-go` is a provider-neutral Go library for model generation, streaming, tool
use, multimodal messages, reasoning, and usage accounting. Provider SDK types
stay behind a small common API, so applications can switch protocols without
changing their conversation model.

> Requires Go 1.27 or newer. The public API is still under active development.

## Highlights

- One request, response, tool, error, and model contract across providers
- Synchronous generation and lifecycle-safe streaming with typed events
- Multi-turn reasoning replay, prompt caching, and usage/cost accounting
- JSON Schema coercion and validation, plus tolerant JSON repair helpers
- Immutable model catalogs, provider profiles, and extensible protocol factories
- Concurrency-safe credentials, OAuth refresh, retries, and token budgeting
- Caller-owned HTTP clients, contexts, and returned data

## Install

```sh
go get github.com/XiaoConstantine/llm-go
```

## Protocols and providers

`llm-go` separates wire-protocol adapters from model and provider metadata. Every
adapter supports generation, typed stream events, classified errors, and token
usage. In the matrix below, ✓ means the adapter implements the feature and —
means it does not. Model-scoped features must still be declared by the configured
model; a ✓ does not imply that every model exposed by an endpoint supports it.

| Package / protocol | Streaming | Background | Tools | JSON mode | Image input | Audio input | Reasoning | Cache tokens | Authentication |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | --- | --- | --- |
| [`openai/responses`](./openai/responses) · OpenAI Responses | ✓ | ✓ | ✓ | ✓ | ✓ | — | Summaries and encrypted replay | Read/write | API key |
| [`openai/azure`](./openai/azure) · Azure OpenAI Responses | ✓ | — | ✓ | ✓ | ✓ | — | Summaries and encrypted replay | Read/write | Azure API key |
| [`openai`](./openai) · OpenAI-compatible Chat Completions | ✓ | — | ✓ | ✓ | ✓ | WAV/MP3 input | Controls and provider-state replay | Read/write | API key or custom headers |
| [`openai/codex`](./openai/codex) · ChatGPT subscription Codex Responses | ✓ | — | ✓ | — | ✓ | WAV/MP3/M4A/WebM/Ogg input | Summaries and encrypted replay | Read/write | Access token or credential resolver |
| [`anthropic`](./anthropic) · Anthropic-compatible Messages | ✓ | — | ✓ | — | ✓ | — | Signed thinking replay | Read/write, including 1h writes | API key, OAuth token, or custom headers |
| [`gemini`](./gemini) · Gemini Developer API | ✓ | — | ✓ | ✓ | ✓ | ✓ | Controls and thought-signature replay | Read | API key |
| [`vertex`](./vertex) · Google Vertex AI | ✓ | — | ✓ | ✓ | ✓ | ✓ | Controls and thought-signature replay | Read | API key, ADC, or authenticated HTTP client |
| [`bedrock`](./bedrock) · Amazon Bedrock ConverseStream | ✓ | — | ✓ | — | ✓ | — | Controls and signed replay | Read/write | AWS credentials, profile, bearer token, or unauthenticated gateway |
| [`mistral`](./mistral) · Mistral Conversations | ✓ | — | ✓ | — | ✓ | — | Thinking blocks and controls | — | API key |

JSON mode refers to `llm.ResponseFormatJSON`, not tool argument schemas. Tool
calls work in generation and streaming, including partial argument events where
the protocol exposes them. Strict tools and reasoning replay are model- and
protocol-dependent. Cache retention markers are emitted only where supported;
unsupported cache keys and session-affinity mappings fail before provider I/O.
OpenAI Responses also supports durable background jobs through
`llm.BackgroundGenerator`.

### Request options and exact reasoning

`Request` accepts protocol-specific options without exposing provider SDK types:

| Request field | Adapters | Behavior |
| --- | --- | --- |
| `OpenAIChat` | `openai` | Logit bias, logprobs, user, verbosity, text prediction, store, metadata, safety identifier, service tier, and validated extra fields |
| `OpenAIResponses` | `openai/responses`, `openai/azure`, `openai/codex` | Text verbosity and service tier |
| `Anthropic` | `anthropic` | Thinking display and validated extra fields |
| `TopK` | `anthropic` | Optional nonnegative integer; an explicit zero is retained |
| `ParallelToolCalls` | The five adapters above | Optional boolean; `nil` omits the field, while explicit `false` and `true` are preserved |

For Anthropic, parallel control maps to `disable_parallel_tool_use` inside the
tool choice and is emitted only when tools are enabled. Extra-field constructors
copy JSON values, reject fields owned by the typed request, and preserve literal
top-level keys and null values. Anthropic extra keys must be plain names.
Unsupported protocol options are rejected before provider I/O.

`ReasoningPolicyExact` is supported by those same five adapters. It preserves
the requested effort instead of converting `minimal` to `low`, rejects thinking
formats that cannot preserve the effort, and requires an explicit token budget
for non-adaptive Anthropic thinking. Effort and budget cannot be combined under
this policy. Model compatibility and endpoint restrictions still apply.
Gemini, Vertex, Bedrock, and Mistral reject the exact policy and the new fields
in the table; the zero-value policy retains their existing behavior.

Responses requests with a nil `ParallelToolCalls` omit `parallel_tool_calls`.
Set the pointer explicitly when a particular wire value is required.

### Document input

Use `Part{Kind: PartFile, Data: data, MediaType: mime, Filename: name}` in a user
message. Data must be nonempty; text documents and filenames must be valid UTF-8.
Documents are not accepted in assistant, system, or tool-result content.

| Adapter | Document MIME types |
| --- | --- |
| `openai` | `application/pdf`, `text/*` |
| `openai/responses`, `openai/azure` | `application/pdf` |
| `anthropic` | `application/pdf`, `text/*` |

These are serializer capabilities, not a guarantee that every model accepts
documents. Codex and the other adapters do not implement document input.
With OpenAI Chat, PDF Data starting with `file-` is sent as a provider file ID;
other PDF Data is sent inline.
Inline PDFs without a filename receive `part-<index>.pdf`. Anthropic maps text
documents to `text/plain` and derives a sanitized document title from Filename,
falling back to `Document` when it is empty.

### Codex transport modes

`openai/codex.Config.Transport` supports these modes; the default remains SSE.

| Mode | Behavior |
| --- | --- |
| `TransportSSE` | Original HTTP event stream |
| `TransportWebSocket` | Transient socket with full context; never falls back to SSE |
| `TransportWebSocketCached` | Session socket with verified incremental context; never falls back to SSE |
| `TransportAuto` | Cached WebSocket when possible; SSE fallback only when connect or handshake fails before `response.create` |

Cached sockets require `Request.SessionID`, are isolated by client, session,
account, and credential. `WebSocketMaxSessions` defaults to 64; busy overflow and
requests with per-attempt headers use transient sockets. Use `WebSocketDialer`
for WebSocket-specific proxy, mTLS, or custom dialing. `Client.Close` is
idempotent and closes all owned sockets and timers.

### Portable orchestration helpers

These helpers are opt-in and do not change provider defaults.

| API | Purpose |
| --- | --- |
| `llm.WithRetry` | Bounded retries; streams retry only before their first chunk; attempt hooks may add validated headers |
| `llm.Collect` | Closes and assembles a stream, preserving partial results and validating completed calls against supplied schemas |
| `llm.CollectStructural` | Collects with structure and strict-JSON checks, leaving argument-schema validation to the execution owner |
| `llm.Tool.Strictness` | Requests `prefer` or `require` constrained tool arguments while retaining legacy `Tool.Strict` behavior |
| `llm.TransformHistory` | Owns cross-model history, normalizes tool IDs, repairs missing/orphaned results, and reports every change |
| `llm.BudgetRequest` | Uses a caller-supplied token estimator and model limits to clamp an owned request copy |
| `llm.ValidateToolArguments` | Coerces model-generated arguments through JSON Schema, then validates and returns an owned value |
| `llm.IsContextOverflow` | Detects classified, explicit, and silent context-window overflow |
| `llm.RepairJSON` / `llm.ParseStreamingJSON` | Repairs malformed string escapes and reads partial streaming JSON |

Tool inputs use JSON Schema 2020-12 by default, support declared drafts and
local references offline, and use Go/RE2-compatible patterns.

`Generate`, `Stream`, background responses, and `CollectStructural(stream)` return
complete, structurally valid tool calls with their original arguments. They do
not enforce argument schemas. Execution owners must call
`ValidateToolCalls(tools, calls)` before invoking tools and can return an error
`ToolResult` on rejection so the model can correct its call. Tool declarations,
strict JSON, provider protocol checks, and constrained-generation settings are
unchanged. `ValidateToolCallStream` remains an explicit schema-validation wrapper
for callers that deliberately want schema failures to terminate collection.

**Migration:** automatic response argument-schema validation has been removed.
`Collect(stream, tools)` retains its published signature and schema-validation
behavior, including rejection of undeclared calls when tools is nil. Agent loops
that need corrective tool results can use `CollectStructural(stream)` and
validate arguments explicitly at the execution boundary. No request flag or
diagnostic metadata is needed; do not execute an unchecked call just because
generation succeeded.

### Custom protocols and dynamic catalogs

`models.NewFactoryRegistry` creates an immutable registry containing the nine
built-in protocol factories plus caller registrations. `ProviderConfig.API`
remains the default route when `ModelInfo.API` is empty; `AdditionalAPIs`
explicitly enables mixed-protocol models for the same provider. Factories receive
owned construction snapshots and an optional live credential resolver.

`models.CatalogManager` keeps `Catalog` immutable while atomically publishing a
static baseline plus validated provider overlays. `Refresh` restores persisted
state before conditional fetches, supports ETag/Last-Modified, force and
provider-selective refresh, and `NoNetwork` restore-only operation. `Available`
uses `CredentialManager` and optional secret-free provider filters. Refreshes run
only when called; the manager owns no permanent goroutines and needs no Close.
The [`modelsdev`](./models/modelsdev) package provides a `CatalogSource`
implementation to dynamically fetch models from [models.dev](https://models.dev).

### Built-in provider profiles

`models.BuiltinProvider` supplies protocol and endpoint defaults for these
services:

| Provider ID | Protocol | Default endpoint |
| --- | --- | --- |
| `openrouter` | OpenAI Chat Completions | `https://openrouter.ai/api/v1` |
| `groq` | OpenAI Chat Completions | `https://api.groq.com/openai/v1` |
| `deepseek` | OpenAI Chat Completions | `https://api.deepseek.com` |
| `xai` | OpenAI Responses | `https://api.x.ai/v1` |
| `cerebras` | OpenAI Chat Completions | `https://api.cerebras.ai/v1` |
| `fireworks` | OpenAI Chat Completions | `https://api.fireworks.ai/inference/v1` |
| `mistral` | Mistral Conversations | `https://api.mistral.ai/v1` |
| `google-vertex` | Google Vertex AI | SDK-derived regional endpoint |
| `amazon-bedrock` | Bedrock ConverseStream | `https://bedrock-runtime.us-east-1.amazonaws.com` |

### Local and compatible servers

Local servers that expose an OpenAI-compatible `/v1/chat/completions` endpoint
use the [`openai`](./openai) adapter. They do not need a built-in profile, and
the adapter permits an empty API key; authenticated servers can instead use an
API key or custom headers.

| Server | Typical `BaseURL` | Support |
| --- | --- | --- |
| Ollama | `http://localhost:11434/v1` | OpenAI-compatible endpoint; the native `/api/chat` API is not implemented |
| llama.cpp `llama-server` | `http://localhost:8080/v1` | OpenAI-compatible endpoint |

Choose a provider ID, configure `models.OpenAIChatCompletions`, and supply the
local model through `llm.ModelInfo`. Streaming, tools, JSON mode, vision, and
reasoning depend on what that server and model implement, so declare only the
capabilities they actually support. The same configuration works for other
compatible servers.

The built-in catalog covers OpenAI, Anthropic, Amazon Bedrock, Gemini, Vertex
AI, Mistral, Codex, and the OpenAI-compatible profiled providers. Azure
Responses uses caller-supplied model metadata. Custom `BaseURL`, headers, and
HTTP clients support compatible gateways.

## Quick start

The generated catalog can configure the protocol, capabilities, compatibility,
and pricing for a model:

```go
package main

import (
    "context"
    "log"
    "os"
    "time"

    llm "github.com/XiaoConstantine/llm-go"
    "github.com/XiaoConstantine/llm-go/models"
)

func main() {
    model, ok := models.BuiltinCatalog().Model("openai", "gpt-5-mini")
    if !ok {
        log.Fatal("model not found")
    }

    providers, err := models.New(models.ProviderConfig{
        ID:     "openai",
        API:    models.OpenAIResponses,
        APIKey: os.Getenv("OPENAI_API_KEY"),
    })
    if err != nil {
        log.Fatal(err)
    }

    generator, err := providers.GeneratorFor(model)
    if err != nil {
        log.Fatal(err)
    }

    ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
    defer cancel()

    response, err := generator.Generate(ctx, llm.Request{
        Messages: []llm.Message{{
            Role:    llm.RoleUser,
            Content: []llm.Part{{Text: "Say hello in one sentence."}},
        }},
    })
    if err != nil {
        log.Fatal(err)
    }
    log.Print(response.Text())
}
```

Concrete provider packages can also be constructed directly when protocol-level
configuration is needed.

## Streaming

Every successful stream must be closed, including after `Recv` returns an error.
`Close` is idempotent, unblocks a pending `Recv`, and waits for transport cleanup.
A cleanup error returned by `Close` does not replace the stream's sticky terminal
state.

```go
stream, err := generator.Stream(ctx, request)
if err != nil {
    return err
}

for {
    chunk, recvErr := stream.Recv()
    if errors.Is(recvErr, io.EOF) {
        break
    }
    if recvErr != nil {
        _ = stream.Close()
        return recvErr
    }
    for _, part := range chunk.Content {
        fmt.Print(part.Text)
    }
}
if err := stream.Close(); err != nil {
    return err
}
```

`Chunk.Events` exposes typed lifecycle events for text, reasoning, and partial
or completed tool calls. Completed values remain available through the ordinary
chunk fields for response assembly.

Anthropic emits cumulative usage as soon as it is reported, including a
usage-only chunk at message start. Keep the last non-nil `Chunk.Usage` rather
than summing snapshots; it remains meaningful if a later event fails or the
caller cancels. Early usage is not a success marker. Applications must allow
chunks without text or a finish reason. Since `WithRetry` retries only before
the first committed chunk, an early usage chunk also ends that retry window.

## Errors and cancellation

Provider and protocol failures are classified as `*llm.Error`. Context
cancellation, deadlines, custom cancellation causes, and underlying transport
errors remain discoverable with `errors.Is` or `errors.As`.
`llm.IsContextOverflow` handles both classified failures and providers that
silently fill or truncate the context window.

The library does not impose a default request timeout. Supply an appropriate
context deadline or configure the provided `http.Client`.

## Model catalogs and credentials

`models.BuiltinCatalog()` returns an immutable generated catalog whose read
methods return independently owned model metadata. `models.Catalog` and
`models.Collection` are
immutable and safe for concurrent use when caller-supplied HTTP clients and
credential resolvers are also safe.

`models.CredentialStore` supports API-key and OAuth credentials.
`models.MemoryCredentialStore` provides an in-memory implementation, while
`models.CredentialManager` coalesces concurrent refreshes, supports live
credential rotation for long-lived generators, and safely handles token
rotation. `models.AnthropicOAuth` and `models.CodexOAuth` expose non-interactive
PKCE exchange/refresh primitives. Their subscription endpoints and client IDs
are provider-private, unofficial, unstable, and replaceable; applications remain
responsible for browser, callback, prompt UI, and persistent storage.

`models.FindEnvAPIKeys` lists configured variables in precedence order,
`models.EnvAPIKey` returns the preferred value, and
`models.HasAmbientCredentials` reports selected Vertex ADC and Bedrock
environment or file hints without exposing secrets.

## Development

```sh
make fmt       # Format Go sources.
make fmt-check # Check formatting without changing files.
make lint      # Run the pinned golangci-lint version used by CI.
go test ./...
go test -race ./...
go vet ./...
```

The built-in catalog source is `models/catalogsource/catalog.json`. Regenerate
and verify it with:

```sh
go generate ./models
git diff --exit-code -- models/catalog_generated.go
```
