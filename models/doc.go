// Package models constructs provider-neutral model generators from a reusable
// collection of provider configurations.
//
// Applications configure provider protocols and credentials once, then resolve
// llm.Generator values from llm.ModelInfo. The concrete provider packages remain
// available for protocol-specific configuration that this package does not
// expose.
package models
