package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v6"
	"github.com/santhosh-tekuri/jsonschema/v6/kind"
)

const toolSchemaURL = "https://llm-go.invalid/tool-schema.json"

// ValidateToolCalls validates complete tool-call arguments against declared
// tool schemas. Partial streaming argument deltas must not be passed here.
// Schemas default to JSON Schema draft 2020-12 and may select draft 4, 6, 7,
// 2019-09, or 2020-12 with $schema. References must resolve within the supplied
// schema; filesystem and network loading are disabled. Pattern keywords use
// Go regexp syntax.
func ValidateToolCalls(tools []Tool, calls []ToolCall) error {
	validator, err := newToolCallValidator(tools)
	if err != nil {
		return err
	}
	return validator.validate(calls)
}

type toolCallValidator map[string]*jsonschema.Schema

func newToolCallValidator(tools []Tool) (toolCallValidator, error) {
	compiled := make(toolCallValidator, len(tools))
	for index, tool := range tools {
		if _, exists := compiled[tool.Name]; exists {
			return nil, fmt.Errorf("tools[%d].name %q is duplicated", index, tool.Name)
		}
		schema, err := compileToolSchema(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("tools[%d].input schema: %w", index, err)
		}
		compiled[tool.Name] = schema
	}
	return compiled, nil
}

func (compiled toolCallValidator) validate(calls []ToolCall) error {
	for index, call := range calls {
		if err := validateToolCall(call); err != nil {
			return fmt.Errorf("tool calls[%d]: %w", index, err)
		}
		schema, exists := compiled[call.Name]
		if !exists {
			return fmt.Errorf("tool calls[%d] %q is not declared", index, call.Name)
		}
		value, err := decodeSchemaJSON(call.Arguments)
		if err != nil {
			return fmt.Errorf("tool calls[%d] %q arguments: %w", index, call.Name, err)
		}
		if err := schema.Validate(value); err != nil {
			return fmt.Errorf("tool calls[%d] %q arguments at %s: %w", index, call.Name, validationPath(err), err)
		}
	}
	return nil
}

func compileToolSchema(raw []byte) (*jsonschema.Schema, error) {
	if err := validateJSON(raw); err != nil {
		return nil, err
	}
	value, err := decodeSchemaJSON(raw)
	if err != nil {
		return nil, err
	}
	switch value.(type) {
	case map[string]any, bool:
	default:
		return nil, fmt.Errorf("must contain a JSON Schema object or boolean")
	}
	applyNullableExtension(value)

	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	compiler.UseLoader(offlineSchemaLoader{})
	if err := compiler.AddResource(toolSchemaURL, value); err != nil {
		return nil, fmt.Errorf("add schema resource: %w", err)
	}
	schema, err := compiler.Compile(toolSchemaURL)
	if err != nil {
		return nil, fmt.Errorf("compile schema: %w", err)
	}
	return schema, nil
}

func validateJSONSchema(raw []byte) error {
	_, err := compileToolSchema(raw)
	return err
}

func decodeSchemaJSON(raw []byte) (any, error) {
	value, err := jsonschema.UnmarshalJSON(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("must contain strict JSON: %w", err)
	}
	return value, nil
}

// applyNullableExtension preserves the existing portable OpenAPI-style
// nullable:true shorthand when it accompanies a type declaration. Unknown JSON
// Schema annotations otherwise remain untouched.
func applyNullableExtension(value any) {
	if schemas, ok := value.([]any); ok {
		for _, schema := range schemas {
			applyNullableExtension(schema)
		}
		return
	}
	schema, ok := value.(map[string]any)
	if !ok {
		return
	}
	if nullable, ok := schema["nullable"].(bool); ok && nullable {
		switch schemaType := schema["type"].(type) {
		case string:
			if schemaType != "null" {
				schema["type"] = []any{schemaType, "null"}
			}
		case []any:
			hasNull := false
			for _, item := range schemaType {
				hasNull = hasNull || item == "null"
			}
			if !hasNull {
				schema["type"] = append(schemaType, "null")
			}
		}
	}
	for _, keyword := range []string{"additionalProperties", "additionalItems", "items", "contains", "not", "if", "then", "else", "propertyNames", "unevaluatedProperties", "unevaluatedItems", "contentSchema"} {
		applyNullableExtension(schema[keyword])
	}
	for _, keyword := range []string{"allOf", "anyOf", "oneOf", "prefixItems"} {
		if children, ok := schema[keyword].([]any); ok {
			for _, child := range children {
				applyNullableExtension(child)
			}
		}
	}
	for _, keyword := range []string{"properties", "patternProperties", "dependentSchemas", "dependencies", "$defs", "definitions"} {
		if children, ok := schema[keyword].(map[string]any); ok {
			for _, child := range children {
				applyNullableExtension(child)
			}
		}
	}
}

type offlineSchemaLoader struct{}

func (offlineSchemaLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("external schema reference %q is unavailable: schema loading is disabled", url)
}

func validationPath(err error) string {
	var validationErr *jsonschema.ValidationError
	if !errors.As(err, &validationErr) {
		return "$"
	}
	for len(validationErr.Causes) != 0 {
		validationErr = validationErr.Causes[0]
	}
	tokens := append([]string(nil), validationErr.InstanceLocation...)
	switch detail := validationErr.ErrorKind.(type) {
	case *kind.Required:
		if len(detail.Missing) != 0 {
			tokens = append(tokens, detail.Missing[0])
		}
	case *kind.AdditionalProperties:
		if len(detail.Properties) != 0 {
			tokens = append(tokens, detail.Properties[0])
		}
	}
	var path strings.Builder
	path.WriteByte('$')
	for _, token := range tokens {
		if _, err := strconv.Atoi(token); err == nil {
			path.WriteByte('[')
			path.WriteString(token)
			path.WriteByte(']')
		} else if simpleJSONPathName(token) {
			path.WriteByte('.')
			path.WriteString(token)
		} else {
			encoded, _ := json.Marshal(token)
			path.WriteByte('[')
			path.Write(encoded)
			path.WriteByte(']')
		}
	}
	return path.String()
}

func simpleJSONPathName(value string) bool {
	if value == "" || value[0] != '_' && (value[0] < 'A' || value[0] > 'Z') && (value[0] < 'a' || value[0] > 'z') {
		return false
	}
	for index := 1; index < len(value); index++ {
		character := value[index]
		if character != '_' && (character < 'A' || character > 'Z') && (character < 'a' || character > 'z') && (character < '0' || character > '9') {
			return false
		}
	}
	return true
}
