package adapterutil

import (
	"encoding/json"
	"reflect"
	"slices"
	"testing"
)

func TestWireFieldsIncludesOmittedAndUntaggedMembers(t *testing.T) {
	type wire struct {
		Model   string `json:"model"`
		Stream  bool   `json:"stream,omitzero"`
		Untyped any
		Skipped string `json:"-"`
		hidden  string
	}
	_ = wire{}.hidden
	fields := WireFields(reflect.TypeFor[wire]())
	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	slices.Sort(names)
	if !slices.Equal(names, []string{"Untyped", "model", "stream"}) {
		t.Fatalf("fields = %v", names)
	}
}

func TestExtraFieldsCheckAppendAndClone(t *testing.T) {
	reserved := map[string]struct{}{"model": {}}
	for name, fields := range map[string]map[string]json.RawMessage{
		"reserved":     {"model": json.RawMessage(`"other"`)},
		"empty name":   {"": json.RawMessage(`1`)},
		"empty value":  {"a": nil},
		"two values":   {"a": json.RawMessage(`1,"model":"other"`)},
		"closed early": {"a": json.RawMessage(`1}`)},
	} {
		if err := CheckExtraFields(fields, reserved); err == nil {
			t.Errorf("%s: CheckExtraFields accepted %s", name, fields)
		}
	}
	fields := map[string]json.RawMessage{"b": json.RawMessage(`[1]`), "a\"": json.RawMessage(`null`)}
	if err := CheckExtraFields(fields, reserved); err != nil {
		t.Fatal(err)
	}
	body := []byte(`{"model":"m"}`)
	merged, err := AppendExtraFields(body, fields)
	if err != nil || string(merged) != `{"model":"m","a\"":null,"b":[1]}` || string(body) != `{"model":"m"}` {
		t.Fatalf("merged = %s, body = %s, error = %v", merged, body, err)
	}
	if empty, err := AppendExtraFields([]byte(`{}`), map[string]json.RawMessage{"a": json.RawMessage(`1`)}); err != nil || string(empty) != `{"a":1}` {
		t.Fatalf("empty object merge = %s, %v", empty, err)
	}
	if _, err := AppendExtraFields([]byte(`[]`), fields); err == nil {
		t.Fatal("AppendExtraFields accepted a non-object body")
	}
	clone := CloneExtraFields(fields)
	clone["b"][1] = '2'
	clone["c"] = nil
	if string(fields["b"]) != `[1]` || len(fields) != 2 || CloneExtraFields(nil) != nil {
		t.Fatalf("clone aliased input: %s", fields)
	}
}
