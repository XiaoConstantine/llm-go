# llm-go

`llm-go` is a provider-neutral Go library for model generation, streaming, tool
use, multimodal messages, reasoning, and usage accounting. Provider SDK types
stay behind a small common API, so applications can switch protocols without
changing their conversation model.

> Requires Go 1.27 or newer. The public API is still under active development.

## Highlights

- Provider-neutral requests, responses, tools, errors, and model metadata
- Synchronous generation and lifecycle-safe streaming
- Typed text, reasoning, and tool-call stream events with stable indexes
- Provider-specific reasoning state preservation for multi-turn conversations
- Cache-aware token usage and tiered cost calculation
- Immutable model catalogs with capabilities, context limits, pricing, and
  protocol compatibility metadata
- Built-in provider profiles for OpenRouter, Groq, DeepSeek, xAI, Cerebras, and
  Fireworks
- Concurrency-safe credential storage and coalesced OAuth token refresh
- Opt-in bounded retries with pre-output-only stream retry and safe attempt hooks
- Stream collection, completed tool-schema validation, history transformation,
  and pluggable model-aware token budgeting
- Caller-owned HTTP clients, headers, contexts, and returned data

## Install

```sh
go get github.com/XiaoConstantine/llm-go
```

## Protocols and providers

`llm-go` separates wire-protocol adapters from model and provider metadata. Every
adapter supports generation, typed stream events, classified errors, and token
usage. In the matrix below, ✓ means the adapter implements the feature and —
means it does not. Optional features must still be declared by the configured
model; a ✓ does not imply that every model exposed by an endpoint supports it.

| Package / protocol | Streaming | Tools | JSON mode | Image input | Audio input | Reasoning | Cache tokens | Authentication |
| --- | :---: | :---: | :---: | :---: | :---: | --- | --- | --- |
| [`openai/responses`](./openai/responses) · OpenAI Responses | ✓ | ✓ | ✓ | ✓ | — | Summaries and encrypted replay | Read/write | API key |
| [`openai`](./openai) · OpenAI-compatible Chat Completions | ✓ | ✓ | ✓ | ✓ | WAV/MP3 input | Controls and provider-state replay | Read/write | API key or custom headers |
| [`openai/codex`](./openai/codex) · ChatGPT subscription Codex Responses | ✓ | ✓ | — | ✓ | WAV/MP3/M4A/WebM/Ogg input | Summaries and encrypted replay | Read/write | Access token or credential resolver |
| [`anthropic`](./anthropic) · Anthropic-compatible Messages | ✓ | ✓ | — | ✓ | — | Signed thinking replay | Read/write, including 1h writes | API key, OAuth token, or custom headers |
| [`gemini`](./gemini) · Gemini Developer API | ✓ | ✓ | ✓ | ✓ | ✓ | Controls and thought-signature replay | Read | API key |

JSON mode refers to `llm.ResponseFormatJSON`, not tool argument schemas. Tool
calls are available in both generation and streaming; protocols that expose
partial arguments emit `llm.StreamEventToolCallDelta`, and `llm.Tool.Strict` is
forwarded when model compatibility permits it. Reasoning controls and replay are
model-dependent and preserved in `llm.Message.ProviderData`. `CacheRetention`,
`CacheKey`, and `SessionID` provide independent prompt-cache partition and
session-affinity controls where a provider mapping is verified. Both may be set;
`CacheRetentionNone` disables cache-key emission while preserving session
affinity. Unsupported mappings fail before provider I/O.

### Portable orchestration helpers

These helpers are opt-in and do not change provider defaults.

| API | Purpose |
| --- | --- |
| `llm.WithRetry` | Bounded retries; streams retry only before their first chunk; attempt hooks may add validated headers |
| `llm.Collect` | Closes and assembles a stream, preserves partial results, and validates completed tool calls |
| `llm.Tool.Strictness` | Requests `prefer` or `require` constrained tool arguments while retaining legacy `Tool.Strict` behavior |
| `llm.TransformHistory` | Returns an owned cross-model history plus warnings for lossy changes |
| `llm.BudgetRequest` | Uses a caller-supplied token estimator and model limits to clamp an owned request copy |

Tool inputs use JSON Schema 2020-12 by default, support declared drafts and
local references offline, and use Go/RE2-compatible patterns.

### Built-in provider profiles

`models.BuiltinProvider` supplies the protocol and default endpoint for these
OpenAI-compatible services:

| Provider ID | Protocol | Default endpoint |
| --- | --- | --- |
| `openrouter` | OpenAI Chat Completions | `https://openrouter.ai/api/v1` |
| `groq` | OpenAI Chat Completions | `https://api.groq.com/openai/v1` |
| `deepseek` | OpenAI Chat Completions | `https://api.deepseek.com` |
| `xai` | OpenAI Responses | `https://api.x.ai/v1` |
| `cerebras` | OpenAI Chat Completions | `https://api.cerebras.ai/v1` |
| `fireworks` | OpenAI Chat Completions | `https://api.fireworks.ai/inference/v1` |

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

The built-in model catalog covers OpenAI, Anthropic, Google Gemini, and all six
profiled providers. The Codex adapter is available for caller-supplied model
metadata but has no built-in catalog entries. Custom `BaseURL`, headers, and HTTP
clients support compatible gateways; native Azure OpenAI, Amazon Bedrock, and
Vertex AI transports are not currently implemented.

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

## Errors and cancellation

Provider and protocol failures are classified as `*llm.Error`. Context
cancellation, deadlines, custom cancellation causes, and underlying transport
errors remain discoverable with `errors.Is` or `errors.As`.

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

## Development

```sh
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
