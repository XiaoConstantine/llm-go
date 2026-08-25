package llm

import (
	"strings"
	"testing"
)

func TestToolCallSchemaValidation(t *testing.T) {
	tools := []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","required":["count"],"properties":{"count":{"type":"integer","minimum":1},"tag":{"anyOf":[{"const":"a"},{"type":"null"}]}},"additionalProperties":false}`)}}
	if err := ValidateToolCalls(tools, []ToolCall{{Name: "lookup", Arguments: []byte(`{"count":2,"tag":null}`)}}); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct{ name, arguments, want string }{
		{"missing", `{}`, `$.count`}, {"path", `{"count":0}`, `$.count`},
		{"additional", `{"count":1,"extra":true}`, `$.extra`}, {"malformed", `{`, `strict JSON`},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateToolCalls(tools, []ToolCall{{Name: "lookup", Arguments: []byte(test.arguments)}})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want %q", err, test.want)
			}
		})
	}
	for _, schema := range []string{`{"type":7}`, `{"$ref":"#/$defs/value"}`} {
		request := validGenerationRequest()
		request.Tools = []Tool{{Name: "bad", InputSchema: []byte(schema)}}
		if err := request.Validate(); err == nil {
			t.Fatalf("malformed schema %s succeeded", schema)
		}
	}
}

func TestToolStrictnessValidationAndLegacyBehavior(t *testing.T) {
	request := validGenerationRequest()
	request.Tools = []Tool{{Name: "tool", InputSchema: []byte(`{}`), Strict: true, Strictness: ToolStrictPrefer}}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "both strict and strictness") {
		t.Fatalf("conflicting strictness error = %v", err)
	}
	request.Tools[0].Strict = false
	request.Tools[0].Strictness = "future"
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "strictness") {
		t.Fatalf("unknown strictness error = %v", err)
	}
	legacy := Tool{Strict: true}
	if !legacy.RequiresStrict() || !legacy.StrictEnabled(true) || legacy.StrictEnabled(false) {
		t.Fatalf("legacy strict behavior = require %v enabled %v/%v", legacy.RequiresStrict(), legacy.StrictEnabled(true), legacy.StrictEnabled(false))
	}
}

func TestPortableSchemaCombinatorsAndBounds(t *testing.T) {
	schema := []byte(`{"type":"object","properties":{"items":{"type":"array","minItems":1,"maxItems":2,"uniqueItems":true,"items":{"type":"string","minLength":2,"maxLength":3,"pattern":"^[a-z]+$"}},"value":{"allOf":[{"type":"number","minimum":1},{"type":["number","null"],"maximum":3}]},"choice":{"oneOf":[{"const":"x"},{"enum":["y","z"]}]}},"required":["items","value","choice"]}`)
	tools := []Tool{{Name: "bounded", InputSchema: schema}}
	if err := ValidateToolCalls(tools, []ToolCall{{Name: "bounded", Arguments: []byte(`{"items":["ab"],"value":2,"choice":"y"}`)}}); err != nil {
		t.Fatal(err)
	}
	for _, arguments := range []string{
		`{"items":["a"],"value":2,"choice":"y"}`,
		`{"items":["ab","ab"],"value":2,"choice":"y"}`,
		`{"items":["ab"],"value":4,"choice":"y"}`,
		`{"items":["ab"],"value":2,"choice":"no"}`,
	} {
		if err := ValidateToolCalls(tools, []ToolCall{{Name: "bounded", Arguments: []byte(arguments)}}); err == nil {
			t.Fatalf("ValidateToolCalls(%s) succeeded", arguments)
		}
	}
}

func TestJSONSchemaDialectsLocalRefsAndFormats(t *testing.T) {
	tests := []struct {
		name    string
		schema  string
		valid   string
		invalid string
	}{
		{
			name:   "local JSON pointer reference",
			schema: `{"$defs":{"name":{"type":"string","minLength":2}},"type":"object","properties":{"name":{"$ref":"#/$defs/name"}},"required":["name"]}`,
			valid:  `{"name":"ok"}`, invalid: `{"name":"x"}`,
		},
		{
			name:   "draft7 tuple contains and conditional",
			schema: `{"$schema":"http://json-schema.org/draft-07/schema#","type":"array","items":[{"type":"string"},{"type":"number"}],"additionalItems":false,"contains":{"type":"number"},"if":{"minItems":2},"then":{"maxItems":2}}`,
			valid:  `["x",2]`, invalid: `["x"]`,
		},
		{
			name:   "draft7 conditional",
			schema: `{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"kind":{"type":"string"},"value":{}},"if":{"properties":{"kind":{"const":"number"}}},"then":{"properties":{"value":{"type":"number"}}},"else":{"properties":{"value":{"type":"string"}}}}`,
			valid:  `{"kind":"number","value":2}`, invalid: `{"kind":"number","value":"two"}`,
		},
		{
			name:   "draft2020 prefix items",
			schema: `{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"array","prefixItems":[{"const":"x"}],"items":{"type":"integer"}}`,
			valid:  `["x",2]`, invalid: `["y",2]`,
		},
		{
			name:   "draft7 asserted format",
			schema: `{"$schema":"http://json-schema.org/draft-07/schema#","type":"string","format":"email"}`,
			valid:  `"person@example.com"`, invalid: `"not-an-email"`,
		},
		{
			name:    "nullable extension",
			schema:  `{"type":"string","nullable":true}`,
			valid:   `null`,
			invalid: `7`,
		},
		{
			name:    "const data remains untouched",
			schema:  `{"const":{"nullable":true,"type":"literal"}}`,
			valid:   `{"nullable":true,"type":"literal"}`,
			invalid: `null`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tools := []Tool{{Name: "validate", InputSchema: []byte(test.schema)}}
			if err := ValidateToolCalls(tools, []ToolCall{{Name: "validate", Arguments: []byte(test.valid)}}); err != nil {
				t.Fatalf("valid instance: %v", err)
			}
			if err := ValidateToolCalls(tools, []ToolCall{{Name: "validate", Arguments: []byte(test.invalid)}}); err == nil {
				t.Fatal("invalid instance succeeded")
			}
		})
	}

	for _, schema := range []string{
		`{"$ref":"https://example.com/schema.json"}`,
		`{"$ref":"#/$defs/missing"}`,
		`{"type":"string","pattern":"^(?=ab)ab$"}`,
	} {
		request := validGenerationRequest()
		request.Tools = []Tool{{Name: "offline", InputSchema: []byte(schema)}}
		if err := request.Validate(); err == nil {
			t.Fatalf("unresolved schema %s succeeded", schema)
		}
	}
}
