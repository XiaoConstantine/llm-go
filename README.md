# llm-go

`llm-go` is a reusable, provider-neutral Go library for generation, streaming,
tool use, and multimodal messages. It is independent of dspy-go and of any agent
or prompt-programming framework.

Requires Go 1.27 or newer. The API is still under active development.

## Install

```sh
go get github.com/XiaoConstantine/llm-go
```

The root package defines the public contracts: `Generator`, requests, messages,
tools, responses, stream chunks, and classified errors. Provider packages
translate those types to provider protocols. The official Anthropic and OpenAI
Go SDKs are private transport dependencies; their types do not appear in the
provider-neutral API.

## Providers

Optional capabilities must be declared in each provider's `Config.Capabilities`;
generation is always enabled.

| Package | Implemented surface | Meaningful limitations |
| --- | --- | --- |
| `anthropic` | Messages generation, non-streaming client tool use, and text streaming | No image/audio input, JSON response mode, or streamed tool calls |
| `openai/responses` | OpenAI Responses generation and streaming, function tools, reasoning summaries, and image input | No JSON response mode or audio input |
| `openai` | OpenAI-compatible Chat Completions generation and streaming, function tools, JSON mode, and image input | Intended for compatibility endpoints; no Responses reasoning items or audio input |

Both providers accept an optional API key, custom `BaseURL`, `*http.Client`, and
headers. This supports compatible gateways and path-prefixed endpoints while
keeping authentication and transport policy under caller control. Provider SDK
environment defaults and implicit retries are disabled; configuration comes from
the provider `Config`.

## Example

```go
package main

import (
    "context"
    "log"
    "os"
    "time"

    llm "github.com/XiaoConstantine/llm-go"
    openairesponses "github.com/XiaoConstantine/llm-go/openai/responses"
)

func main() {
    client, err := openairesponses.New(openairesponses.Config{
        Model:  "gpt-5.4-mini",
        APIKey: os.Getenv("OPENAI_API_KEY"),
    })
    if err != nil {
        log.Fatal(err)
    }

    ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
    defer cancel()

    response, err := client.Generate(ctx, llm.Request{
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

The library does not impose a default request timeout. Contexts control request
cancellation and deadlines; cancellation remains available through `errors.Is`.
Provider failures are classified as `*llm.Error`.
Callers must close every successful `llm.Stream`, including after `Recv` returns
an error. `Close` is idempotent, releases transport resources, and unblocks a
pending `Recv`.
