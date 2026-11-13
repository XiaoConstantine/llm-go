package adapterutil

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"
)

// WireFields returns the top-level JSON member names a wire struct declares,
// whether or not a given request emits them. Adapters reserve these names so
// extra fields cannot replace a member the adapter controls, such as stream.
func WireFields(wire reflect.Type) map[string]struct{} {
	names := make(map[string]struct{}, wire.NumField())
	for index := range wire.NumField() {
		field := wire.Field(index)
		if !field.IsExported() {
			continue
		}
		name, _, _ := strings.Cut(field.Tag.Get("json"), ",")
		if name == "-" {
			continue
		}
		if name == "" {
			name = field.Name
		}
		names[name] = struct{}{}
	}
	return names
}

// CheckExtraFields reports the first invalid extra field in key order: an
// empty or reserved name, or a value that is not one JSON value.
func CheckExtraFields(fields map[string]json.RawMessage, reserved map[string]struct{}) error {
	for _, name := range sortedKeys(fields) {
		if name == "" {
			return fmt.Errorf("extra field name must not be empty")
		}
		if _, ok := reserved[name]; ok {
			return fmt.Errorf("extra field %q is set by the adapter and cannot be replaced", name)
		}
		if !json.Valid(fields[name]) {
			return fmt.Errorf("extra field %q is not a valid JSON value", name)
		}
	}
	return nil
}

// AppendExtraFields adds checked extra fields, in key order, to an encoded
// JSON object and returns the result. It does not modify body.
func AppendExtraFields(body []byte, fields map[string]json.RawMessage) ([]byte, error) {
	if len(fields) == 0 {
		return body, nil
	}
	end := len(body) - 1
	if end < 1 || body[0] != '{' || body[end] != '}' {
		return nil, fmt.Errorf("request body is not a JSON object")
	}
	merged := append([]byte(nil), body[:end]...)
	for _, name := range sortedKeys(fields) {
		key, err := json.Marshal(name)
		if err != nil {
			return nil, err
		}
		if len(merged) > 1 {
			merged = append(merged, ',')
		}
		merged = append(append(append(merged, key...), ':'), fields[name]...)
	}
	return append(merged, '}'), nil
}

// CloneExtraFields returns a deep copy of fields.
func CloneExtraFields(fields map[string]json.RawMessage) map[string]json.RawMessage {
	if fields == nil {
		return nil
	}
	clone := make(map[string]json.RawMessage, len(fields))
	for name, value := range fields {
		clone[name] = append(json.RawMessage(nil), value...)
	}
	return clone
}

func sortedKeys(fields map[string]json.RawMessage) []string {
	names := make([]string, 0, len(fields))
	for name := range fields {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}
