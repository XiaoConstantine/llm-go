# Built-in catalog source

`catalog.json` is the canonical, llm-go-owned source for the generated built-in
model catalog. Its schema is versioned independently of any upstream package:

- `schema_version` changes only when the JSON structure changes.
- `revision` increments whenever catalog data changes.
- compatibility fields use llm-go's provider-neutral compatibility vocabulary.

The initial snapshot was normalized from provider metadata used during the
pi-ai parity assessment and then validated against llm-go's own model schema.
There is no runtime or build dependency on pi-ai.

Run `go generate ./models` after editing the source. The `cataloggen` tests
compare regenerated output with `catalog_generated.go`, so `go test ./...`
fails when generated data is stale.
