// Package modelsdev provides integration with the models.dev open model
// directory and API, allowing dynamic catalog loading and synchronization for
// llm-go.
//
// NewSource preserves models.BuiltinCatalog() metadata by default, updating only
// limits and non-null cost rates for known models. Missing rates retain baseline
// values; omitted cache-read rates are capped at merged input rates. A non-empty
// list of surviving upstream tiers defines the set of thresholds, with matching
// thresholds inheriting baseline rates. Baseline tiers are retained only when
// upstream omits tiers or all upstream tiers are filtered out. Retained tiers at
// or above the merged, known context window are dropped. Explicit zero rates
// replace baseline rates. WithBaseline selects a baseline or disables preservation:
//
//	source := modelsdev.NewSource() // Protect built-in metadata.
//	unprotected := modelsdev.NewSource(modelsdev.WithBaseline(nil))
//
// A custom API or compatibility resolver disables the implicit built-in baseline.
// Explicit WithBaseline takes precedence over resolvers for matching models,
// regardless of option order. Pass the same catalog to WithBaseline and
// models.CatalogManagerConfig.Baseline. Invalid model records are skipped without
// discarding valid records from the same provider. An invalid price rejects the
// entire record, including its limits, rather than publishing a partial update.
// If every unfiltered record fails conversion, Fetch returns KindMalformedResponse
// so CatalogManager retains the previous overlay and stored snapshot.
//
// CatalogStore persists the curated fields captured at fetch time. After a
// library or baseline upgrade, offline restore or HTTP 304 can restore old fields
// over the new baseline. Clear the store and recreate the manager, or force a
// network refresh using a source constructed against the new baseline. Conditional
// refresh alone does not repair stale curated metadata; CatalogManager's default
// wholesale replacement semantics are unchanged.
package modelsdev
