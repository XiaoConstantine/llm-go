# modelsdevsync

`modelsdevsync` synchronizes `llm-go`'s built-in model catalog (`catalogsource/catalog.json`)
with upstream data from [models.dev](https://models.dev).

Models marked `source: "models.dev"` in schema version 2 are managed. A sync
removes managed entries that disappear, have upstream `status: "deprecated"`,
or fail the same eligibility checks as new discoveries: tool support, exactly
text output, and valid positive context/output limits. Removals are reported,
written even without additions/updates, and increment the revision once.

Entries without `source` are curated and remain unchanged. New discoveries get
the marker automatically; existing curated entries are never silently adopted.
`openai-codex` is always manually maintained. Selection and retirement are scoped
to the provider route, not just the model ID.

Supply a complete snapshot for each selected provider. Missing providers and
null/missing model maps cannot retire managed entries and cause an error;
an explicit empty models object is authoritative. `-add-new` only controls
discovery, not retirement. `-dry-run` reports the resulting set without writing
files. See the [ownership/migration guide](../../catalogsource/README.md#ownership-and-retirement).

## Usage

### Reconcile managed models with latest membership, pricing & limits

```sh
go run ./models/cmd/modelsdevsync -catalog ./models/catalogsource/catalog.json
```

### Dry-run to inspect updates without modifying catalog

```sh
go run ./models/cmd/modelsdevsync -catalog ./models/catalogsource/catalog.json -dry-run
```

### Synchronize specific providers

```sh
go run ./models/cmd/modelsdevsync -catalog ./models/catalogsource/catalog.json -providers openai,anthropic,google
```

### Add newly discovered models from models.dev

New entries must have `tool_call: true` and exactly `modalities.output: ["text"]`,
as well as valid context and output limits. The same filter applies to manual
and scheduled runs, including retirement of ineligible managed entries.
Curated entries remain unchanged. The report lists excluded new model IDs.

When upstream omits `cache_read`, its retained base or tier rate is capped at the
merged input rate; the report names each model whose price was clamped. A non-empty
list of surviving upstream tiers defines the set of thresholds, with matching
thresholds inheriting baseline rates. Baseline tiers are retained only when upstream
omits tiers or all upstream tiers are filtered out. Retained tiers at or above the
merged context window are dropped, including when pricing is absent or rejected.
Explicit invalid prices are not clamped: the cost update is skipped and reported,
while valid limits, reasoning, and vision updates still apply. New models with
invalid costs are skipped.

```sh
go run ./models/cmd/modelsdevsync -catalog ./models/catalogsource/catalog.json -add-new
```

### Offline synchronization using a local JSON file

```sh
go run ./models/cmd/modelsdevsync -catalog ./models/catalogsource/catalog.json -file <path-to-complete-api.json>
```

## After synchronizing

After updating `catalog.json`, regenerate the static catalog code:

```sh
go generate ./models
go test ./...
```
