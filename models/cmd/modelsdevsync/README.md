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
