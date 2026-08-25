package models

//go:generate go run ./cmd/cataloggen -source ./catalogsource/catalog.json -output ./catalog_generated.go

var builtinCatalog = mustBuiltinCatalog()

// BuiltinCatalog returns the immutable generated model catalog. Catalog read
// methods return independently owned values.
func BuiltinCatalog() *Catalog { return builtinCatalog }

func mustBuiltinCatalog() *Catalog {
	catalog, err := NewCatalog(generatedBuiltinModels...)
	if err != nil {
		panic("models: invalid generated built-in catalog: " + err.Error())
	}
	return catalog
}
