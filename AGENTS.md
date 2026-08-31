# AGENTS.md

This file provides guidance to agents working in this repository.

## Purpose and precedence

- MUST means required.
- SHOULD means recommended unless there is a concrete reason to deviate.
- MAY means optional.
- This root file defines repository-wide defaults. A deeper `AGENTS.md` takes
  precedence for its subtree.

## Repository map

- The root `llm` package owns the provider-neutral public contracts: requests,
  responses, streams, tools, errors, validation, retries, budgeting, and usage.
- `anthropic`, `bedrock`, `gemini`, `mistral`, `openai`, and `vertex` are
  protocol adapters. Provider SDK and wire types MUST remain behind these
  package boundaries.
- `openai/responses`, `openai/azure`, and `openai/codex` are distinct protocol
  or transport variants; do not assume behavior is interchangeable between
  them.
- `openai/azure` delegates Responses behavior to `openai/responses`, `vertex`
  delegates GenerateContent behavior to `gemini`, and both `openai/responses`
  and `openai/codex` use `internal/openairesponses`. Changes to shared paths
  require focused checks for every dependent adapter.
- `internal` contains shared implementation details that are not public API.
- `models` owns provider construction and configuration, credentials, factory
  registries, catalog storage and sources, and application of compatibility and
  pricing metadata. The public metadata vocabulary remains in the root package.

# Development guidance

## Non-negotiables

1. Correctness and lifecycle safety come first.
2. Do not invent APIs, defaults, provider behavior, wire mappings, or test
   workflows. Confirm them in code, tests, provider documentation, or the
   requested contract.
3. Preserve provider-neutral behavior. A quirk verified for one adapter MUST
   NOT silently become a cross-provider rule.
4. Keep diffs minimal. Avoid unrelated refactors, broad renames, dependency
   churn, or formatting-only changes.
5. Leave verifiable evidence. Run checks proportional to the change and report
   the exact commands.

## Scope and design

- Implement the smallest complete change for the current requirement.
- Treat exported names and documented behavior as compatibility boundaries even
  though the README says the public API is still under active development.
- Read the neighboring adapter implementations, their tests, and existing
  helpers before introducing a new abstraction.
- Keep the root API provider-neutral; keep SDK-specific and wire-specific data
  in the relevant adapter unless opaque replay through `ProviderData` is the
  established contract.
- Do not add production APIs, flags, interfaces, or hooks solely to make a test
  convenient.
- When behavior applies to both generation and streaming, inspect and test both
  paths. Do not infer one from the other.
- Keep unrelated modernization and cleanup out of the requested change.

## Go and public contracts

- The governing Go version is the version in `go.mod`. Do not use language or
  standard-library features newer than that version.
- Format Go code with `gofmt`; follow standard Go naming, package, error, and
  documentation conventions.
- Preserve `errors.Is` and `errors.As` discoverability when wrapping errors.
  Provider failures use the most specific applicable `*llm.Error` kind, while
  context cancellation, deadlines, custom causes, and underlying transport
  failures remain discoverable.
- Generator implementations follow the validation and cancellation ordering
  documented on `llm.Generator`: request validation, already-done context,
  capability checks, then model-specific checks, all before provider I/O.
- Request-reachable storage is borrowed only for the lifetime documented in
  `doc.go`; returned values belong to the caller. Copy mutable slices, maps,
  headers, byte data, and JSON when ownership crosses that boundary.
- Clients advertised as immutable or concurrency-safe MUST remain so. Do not
  add mutable shared state without explicit synchronization and lifecycle
  tests.
- Add or update doc comments when exported behavior changes. Wire format,
  accepted input, output shape, defaults, persistence, and error classification
  are behavior changes, not mechanical refactors.
- Avoid new dependencies unless they are necessary. If dependencies change,
  update `go.mod` and `go.sum` deliberately and review the resulting diff.

## Streams and resources

- Every non-nil stream returned by `Stream` MUST eventually be closed, including
  after `Recv` returns an error. Tests and examples MUST model this contract;
  lifecycle tests MAY control the exact `Close` point explicitly.
- Preserve the stream semantics documented in `model.go`: ordered committed
  chunks, exactly one sticky terminal state, idempotent `Close`, pending `Recv`
  unblocking, and cleanup completion before `Close` returns.
- Producers MUST honor cancellation, stop after `Emit` returns false, and join
  goroutines they start. Prefer `internal/stream` where its contract fits.
- Close or release HTTP bodies, SDK streams, and WebSockets on every terminal
  path; stop timers and join goroutines. Drain bodies when the transport
  contract requires it. Do not replace the primary terminal error with a
  cleanup error; join errors only where the established contract preserves both.

## Provider adapter changes

- Keep request construction, response conversion, stream event conversion,
  error classification, capability checks, and compatibility switches aligned.
- Verify wire behavior with focused tests that inspect actual serialized
  requests and representative provider responses or events.
- Reuse local HTTP test servers or injected runtimes. Unit tests MUST NOT depend
  on real credentials, provider availability, or external network access.
- Propagate caller contexts and honor cancellation. Do not mutate caller-supplied
  HTTP clients, headers, or transport configuration; preserve the adapter's
  documented cloning and provider-owned-header precedence. Do not introduce
  hidden retries or default timeouts; retry behavior is opt-in unless an
  existing adapter contract says otherwise.
- When adding a protocol or capability, trace all relevant surfaces: package
  docs, model capabilities and compatibility, factory registration, error
  mapping, generation, streaming, and README support tables.

## Tests

- Prefer extending the nearest existing test suite and helpers over creating
  new scaffolding.
- Keep tests focused, deterministic, and safe under `-race`. Avoid sleeps when a
  channel, barrier, fake clock, or explicit signal can prove the ordering.
- For bug fixes, make the focused regression fail for the intended reason before
  applying the fix when practical.
- Test observable behavior: returned values, classified and wrapped errors,
  serialized requests, stream events, terminal states, cancellation, ownership,
  and cleanup. Do not treat log text or keyword presence as a behavioral proxy.
- Cover malformed and partial provider data as well as the success path when a
  converter or stream state machine changes.
- Use `-count=1` when checking stateful, cache-sensitive, or concurrency-sensitive
  behavior. Run the race detector for changes involving streams, caches,
  credentials, goroutines, or shared clients.

## Generated model catalog

- `models/catalogsource/catalog.json` is the canonical source. Increment its
  `revision` whenever catalog data changes.
- Change `schema_version` only when the source JSON structure changes.
- Maintain `openai-codex` route metadata directly; those entries are
  intentionally excluded from models.dev synchronization.
- `models/catalog_generated.go` is generated; MUST NOT be edited by hand.
- After changing or synchronizing the source, regenerate it and run the focused
  freshness test:

  ```sh
  go generate ./models
  go test ./models/cmd/cataloggen
  ```

- CI runs `go generate ./models` followed by
  `git diff --exit-code -- models/catalog_generated.go` from a clean checkout.
  The broader `go test ./...` also runs the catalog freshness test.

- `models/cmd/modelsdevsync` can update pricing and limits from models.dev. Its
  network-backed mode is an explicit catalog-maintenance operation, not part of
  normal tests; prefer its `-file` mode when reproducibility matters.
- Review generated diffs for unexpected model removals, capability changes,
  compatibility changes, limits, and pricing before accepting them.

## Verification

Start with the narrowest relevant package or test, then broaden checks in
proportion to risk. The following repository-root commands are the local
equivalents of CI's build, test, race, formatting, vet, and lint checks:

```sh
go build -v ./...
go test -v -count=1 ./...
go test -v -race -count=1 ./...
make fmt-check
go vet ./...
make lint
```

CI runs the build, ordinary tests, and race tests on both Ubuntu and macOS.
`make lint` uses the pinned golangci-lint version from `Makefile`; CI invokes the
same version through its GitHub Action. `make check` runs `fmt-check` and `lint`,
but does not run tests or `go vet`.

## Documentation and command snippets

- Commands in docs SHOULD be copy-pasteable from the repository root unless
  explicitly scoped elsewhere.
- Use explicit placeholders such as `<package>`, `<TestName>`, and `<path>`.
- Keep the README provider matrix and feature descriptions synchronized with
  externally visible support.
- Keep terminology consistent with the public types and package docs; distinguish
  protocol support from model capability and model compatibility.
