# modelsdevsync

`modelsdevsync` synchronizes `llm-go`'s built-in model catalog (`catalogsource/catalog.json`)
with upstream data from [models.dev](https://models.dev).

## Usage

### Synchronize existing models with latest pricing & limits

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
and scheduled runs; existing entries remain eligible for updates regardless of
these discovery requirements. The report lists excluded new model IDs.

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
go run ./models/cmd/modelsdevsync -catalog ./models/catalogsource/catalog.json -file /path/to/api.json
```

## After synchronizing

After updating `catalog.json`, regenerate the static catalog code:

```sh
go generate ./models
go test ./...
```
