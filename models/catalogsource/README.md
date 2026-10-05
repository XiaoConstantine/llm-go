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

Add `-add-new` to also discover models that are not already in the catalog.

Run `go generate ./models` after editing or syncing the source. The `cataloggen` tests
compare regenerated output with `catalog_generated.go`, so `go test ./...`
fails when generated data is stale.

## Scheduled updates

The [Update model catalog workflow](../../.github/workflows/model-catalog-update.yml)
checks models.dev every six hours and can also run through GitHub Actions'
**Run workflow** button. It updates Anthropic, direct OpenAI, and Google entries,
regenerates the Go catalog, and opens or updates one pull request on
`automation/model-catalog` after the build, tests, race tests, formatting check,
and vet pass. It creates no pull request when there are no changes and leaves
merging to a reviewer.

Automatic additions require upstream `tool_call: true` and
`modalities.output: ["text"]`. This keeps embeddings, image generation, and music
models out of automatic additions. Existing catalog entries remain eligible
for metadata updates regardless of that filter. Other new models and providers,
including `openai-codex`, require separate maintenance. Discovery depends on when
the model appears in models.dev; a provider announcement alone does not trigger
an update. Review new models against provider documentation for protocol,
capabilities, compatibility, pricing, and limits before merging.

After merging the workflow into the default branch, enable **Settings → Actions
→ General → Workflow permissions → Allow GitHub Actions to create and approve
pull requests**. The workflow grants its `GITHUB_TOKEN` contents and pull-request
write permissions and uses that token by default. It never approves or merges
pull requests.

To let the normal pull-request CI start without a manual approval, configure an
optional repository secret named `MODEL_CATALOG_TOKEN` with a fine-grained
personal access token limited to this repository, with **Contents: Read and
write** and **Pull requests: Read and write**. Without that secret, GitHub's
`GITHUB_TOKEN` pull-request runs require approval; the catalog workflow still
performs validation before publishing the pull request. See
[GitHub's workflow-trigger documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

Each changed run retains the raw `modelsdev.json` snapshot, filtered
`modelsdev-candidates.json`, timestamp, SHA-256, synchronization report, and
pull-request body when available as a workflow artifact for 14 days. The pull
request links to the run and includes the source hash and validation commands.
To reproduce synchronization, download and extract that run's artifact, then
run from the repository root:

```sh
go run ./models/cmd/modelsdevsync \
  -catalog ./models/catalogsource/catalog.json \
  -file <artifact-directory>/modelsdev-candidates.json \
  -providers anthropic,openai,google -add-new
go generate ./models
go test -count=1 ./models/cmd/modelsdevsync ./models/cmd/cataloggen ./models/modelsdev
```

The existing release workflow publishes only when a `v*` tag is pushed. Merging
a catalog pull request updates the default branch; publishing a tagged library
release remains a separate step.
