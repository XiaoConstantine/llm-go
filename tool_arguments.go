package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ValidateToolCallArguments finds call's declared tool, coerces its arguments
// according to the tool's JSON Schema, validates the converted value, and
// returns independently owned JSON-compatible storage.
func ValidateToolCallArguments(tools []Tool, call ToolCall) (any, error) {
	for _, tool := range tools {
		if tool.Name == call.Name {
			return ValidateToolArguments(tool, call)
		}
	}
	return nil, fmt.Errorf("tool %q is not declared", call.Name)
}

// ValidateToolArguments coerces call arguments according to tool's JSON Schema,
// validates the converted value, and returns independently owned
// JSON-compatible storage. It does not modify tool, call, or nested input.
func ValidateToolArguments(tool Tool, call ToolCall) (any, error) {
	if err := validateTools([]Tool{tool}); err != nil {
		return nil, fmt.Errorf("tool: %w", err)
	}
	if err := validateToolCall(call); err != nil {
		return nil, fmt.Errorf("tool call: %w", err)
	}
	var value any
	if err := json.Unmarshal(call.Arguments, &value); err != nil {
		return nil, fmt.Errorf("tool %q arguments: %w", call.Name, err)
	}
	schemaValue, err := decodeSchemaJSON(tool.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("tool %q schema: %w", tool.Name, err)
	}
	applyNullableExtension(schemaValue)
	coerced := coerceJSONSchemaValue(value, schemaValue)
	compiled, err := compileToolSchema(tool.InputSchema)
	if err != nil {
		return nil, fmt.Errorf("tool %q schema: %w", tool.Name, err)
	}
	if err := compiled.Validate(coerced); err != nil {
		return nil, fmt.Errorf("tool %q arguments at %s: %w", call.Name, validationPath(err), err)
	}
	return coerced, nil
}

func coerceJSONSchemaValue(value, schemaValue any) any {
	schema, ok := schemaValue.(map[string]any)
	if !ok {
		return value
	}
	if allOf, ok := schema["allOf"].([]any); ok {
		for _, nested := range allOf {
			value = coerceJSONSchemaValue(value, nested)
		}
	}
	for _, keyword := range []string{"anyOf", "oneOf"} {
		if alternatives, ok := schema[keyword].([]any); ok {
			value = coerceJSONSchemaUnion(value, alternatives)
		}
	}
	types := schemaTypes(schema["type"])
	if len(types) != 0 && !matchesAnyJSONType(value, types) {
		for _, schemaType := range types {
			if candidate, changed := coerceJSONPrimitive(value, schemaType); changed {
				value = candidate
				break
			}
		}
	}
	if object, ok := value.(map[string]any); ok && (len(types) == 0 || containsString(types, "object")) {
		properties, _ := schema["properties"].(map[string]any)
		for name, propertySchema := range properties {
			if property, exists := object[name]; exists {
				object[name] = coerceJSONSchemaValue(property, propertySchema)
			}
		}
		if additional, ok := schema["additionalProperties"].(map[string]any); ok {
			for name, property := range object {
				if _, declared := properties[name]; !declared {
					object[name] = coerceJSONSchemaValue(property, additional)
				}
			}
		}
	}
	if array, ok := value.([]any); ok && (len(types) == 0 || containsString(types, "array")) {
		prefixItems, _ := schema["prefixItems"].([]any)
		for index := range min(len(array), len(prefixItems)) {
			array[index] = coerceJSONSchemaValue(array[index], prefixItems[index])
		}
		switch items := schema["items"].(type) {
		case []any:
			for index := range min(len(array), len(items)) {
				array[index] = coerceJSONSchemaValue(array[index], items[index])
			}
		case map[string]any:
			for index := len(prefixItems); index < len(array); index++ {
				array[index] = coerceJSONSchemaValue(array[index], items)
			}
		}
	}
	return value
}

func coerceJSONSchemaUnion(value any, alternatives []any) any {
	for _, alternative := range alternatives {
		candidate := coerceJSONSchemaValue(cloneJSONValue(value), alternative)
		raw, err := json.Marshal(alternative)
		if err != nil {
			continue
		}
		validator, err := compileToolSchema(raw)
		if err == nil && validator.Validate(candidate) == nil {
			return candidate
		}
	}
	return value
}

func schemaTypes(value any) []string {
	switch value := value.(type) {
	case string:
		return []string{value}
	case []any:
		result := make([]string, 0, len(value))
		for _, item := range value {
			if schemaType, ok := item.(string); ok {
				result = append(result, schemaType)
			}
		}
		return result
	default:
		return nil
	}
}

func matchesAnyJSONType(value any, types []string) bool {
	for _, schemaType := range types {
		switch schemaType {
		case "number":
			if _, ok := value.(float64); ok {
				return true
			}
		case "integer":
			if number, ok := value.(float64); ok && math.Trunc(number) == number {
				return true
			}
		case "boolean":
			if _, ok := value.(bool); ok {
				return true
			}
		case "string":
			if _, ok := value.(string); ok {
				return true
			}
		case "null":
			if value == nil {
				return true
			}
		case "array":
			if _, ok := value.([]any); ok {
				return true
			}
		case "object":
			if _, ok := value.(map[string]any); ok {
				return true
			}
		}
	}
	return false
}

func coerceJSONPrimitive(value any, schemaType string) (any, bool) {
	switch schemaType {
	case "number", "integer":
		if value == nil {
			return float64(0), true
		}
		if boolean, ok := value.(bool); ok {
			if boolean {
				return float64(1), true
			}
			return float64(0), true
		}
		if text, ok := value.(string); ok && strings.TrimSpace(text) != "" {
			trimmed := strings.TrimSpace(text)
			number, err := strconv.ParseFloat(trimmed, 64)
			if err != nil {
				if integer, integerErr := strconv.ParseInt(trimmed, 0, 64); integerErr == nil {
					number, err = float64(integer), nil
				}
			}
			if err == nil && !math.IsInf(number, 0) && !math.IsNaN(number) && (schemaType != "integer" || math.Trunc(number) == number) {
				return number, true
			}
		}
	case "boolean":
		if value == nil {
			return false, true
		}
		if text, ok := value.(string); ok && (text == "true" || text == "false") {
			return text == "true", true
		}
		if number, ok := value.(float64); ok && (number == 0 || number == 1) {
			return number == 1, true
		}
	case "string":
		if value == nil {
			return "", true
		}
		switch value := value.(type) {
		case bool:
			return strconv.FormatBool(value), true
		case float64:
			return strconv.FormatFloat(value, 'g', -1, 64), true
		}
	case "null":
		switch value := value.(type) {
		case string:
			if value == "" {
				return nil, true
			}
		case float64:
			if value == 0 {
				return nil, true
			}
		case bool:
			if !value {
				return nil, true
			}
		}
	}
	return value, false
}

func cloneJSONValue(value any) any {
	switch value := value.(type) {
	case map[string]any:
		clone := make(map[string]any, len(value))
		for key, item := range value {
			clone[key] = cloneJSONValue(item)
		}
		return clone
	case []any:
		clone := make([]any, len(value))
		for index, item := range value {
			clone[index] = cloneJSONValue(item)
		}
		return clone
	default:
		return value
	}
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
