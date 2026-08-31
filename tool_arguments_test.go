package llm

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestValidateToolArgumentsCoercesNestedValues(t *testing.T) {
	tool := Tool{Name: "configure", InputSchema: json.RawMessage(`{
		"type":"object",
		"properties":{
			"count":{"type":"integer","minimum":1},
			"enabled":{"type":"boolean"},
			"label":{"type":"string"},
			"items":{"type":"array","items":{"type":"number"}}
		},
		"additionalProperties":{"type":"boolean"},
		"required":["count","enabled","label","items"]
	}`)}
	call := ToolCall{Name: "configure", Arguments: json.RawMessage(`{"count":"2","enabled":"true","label":3,"items":["1.5",false],"extra":1}`)}
	got, err := ValidateToolArguments(tool, call)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"count": float64(2), "enabled": true, "label": "3", "items": []any{1.5, float64(0)}, "extra": true}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ValidateToolArguments() = %#v, want %#v", got, want)
	}
}

func TestValidateToolArgumentsUnionsAndOwnership(t *testing.T) {
	tool := Tool{Name: "choose", InputSchema: json.RawMessage(`{"type":"object","properties":{"value":{"anyOf":[{"type":"integer","minimum":2},{"type":"boolean"}]}},"required":["value"]}`)}
	call := ToolCall{Name: "choose", Arguments: json.RawMessage(`{"value":"true"}`)}
	got, err := ValidateToolCallArguments([]Tool{tool}, call)
	if err != nil {
		t.Fatal(err)
	}
	object := got.(map[string]any)
	if object["value"] != true {
		t.Fatalf("coerced value = %#v", object["value"])
	}
	object["value"] = false
	if string(call.Arguments) != `{"value":"true"}` {
		t.Fatal("input arguments were mutated")
	}
}

func TestValidateToolArgumentsNullableAndNumericStrings(t *testing.T) {
	tool := Tool{Name: "convert", InputSchema: json.RawMessage(`{"type":"object","properties":{"nullable":{"type":"string","nullable":true},"hex":{"type":"integer"},"large":{"type":"string"}},"required":["nullable","hex","large"]}`)}
	call := ToolCall{Name: "convert", Arguments: json.RawMessage(`{"nullable":null,"hex":" 0x10 ","large":1e21}`)}
	got, err := ValidateToolArguments(tool, call)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"nullable": nil, "hex": float64(16), "large": "1e+21"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ValidateToolArguments() = %#v, want %#v", got, want)
	}
}

func TestValidateToolArgumentsUntypedObjectsAndPrefixItems(t *testing.T) {
	tool := Tool{Name: "tuple", InputSchema: json.RawMessage(`{"properties":{"values":{"type":"array","prefixItems":[{"type":"integer"},{"type":"boolean"}],"items":{"type":"string"}}},"required":["values"]}`)}
	call := ToolCall{Name: "tuple", Arguments: json.RawMessage(`{"values":["2","true",3,false]}`)}
	got, err := ValidateToolArguments(tool, call)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]any{"values": []any{float64(2), true, "3", "false"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ValidateToolArguments() = %#v, want %#v", got, want)
	}
}

func TestValidateToolArgumentsFailures(t *testing.T) {
	tool := Tool{Name: "count", InputSchema: json.RawMessage(`{"type":"object","properties":{"value":{"type":"integer","minimum":2}},"required":["value"]}`)}
	if _, err := ValidateToolArguments(tool, ToolCall{Name: "count", Arguments: json.RawMessage(`{"value":"1"}`)}); err == nil {
		t.Fatal("invalid coerced arguments succeeded")
	}
	if _, err := ValidateToolCallArguments([]Tool{tool}, ToolCall{Name: "missing", Arguments: json.RawMessage(`{}`)}); err == nil {
		t.Fatal("undeclared tool succeeded")
	}
	if _, err := ValidateToolArguments(tool, ToolCall{Name: "count", Arguments: json.RawMessage(`{`)}); err == nil {
		t.Fatal("malformed arguments succeeded")
	}
}
