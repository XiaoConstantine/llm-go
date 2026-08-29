# Built-in catalog source

`catalog.json` is the canonical, llm-go-owned source for the generated built-in
model catalog. Its schema is versioned independently of any upstream package:

- `schema_version` changes only when the JSON structure changes.
- `revision` increments whenever catalog data changes.
- compatibility fields use llm-go's provider-neutral compatibility vocabulary.

The initial snapshot was normalized from provider metadata used during the
pi-ai parity assessment and then validated against llm-go's own model schema.
There is no runtime or build dependency on pi-ai.

The `openai-codex` entries are route-specific metadata maintained directly in
this catalog from the ChatGPT/Codex route definitions used by pi-ai. They are
not aliases of the direct `openai` entries and are intentionally excluded from
models.dev synchronization because that dataset has no Codex subscription
provider.

To synchronize existing models with upstream pricing and limits from models.dev:

```sh
go run ./models/cmd/modelsdevsync -catalog ./models/catalogsource/catalog.json
go generate ./models
```

Run `go generate ./models` after editing or syncing the source. The `cataloggen` tests
compare regenerated output with `catalog_generated.go`, so `go test ./...`
fails when generated data is stale.
