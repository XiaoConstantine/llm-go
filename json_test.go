package llm

import (
	"reflect"
	"strings"
	"testing"
)

func TestRepairJSON(t *testing.T) {
	input := "{\"line\":\"one\ntwo\",\"path\":\"C:\\temp\\q\",\"valid\":\"\\u263a\"}"
	want := `{"line":"one\ntwo","path":"C:\temp\\q","valid":"\u263a"}`
	if got := RepairJSON(input); got != want {
		t.Fatalf("RepairJSON() = %q, want %q", got, want)
	}
	if got := RepairJSON(want); got != want {
		t.Fatalf("RepairJSON() is not idempotent: %q", got)
	}
}

func TestParseJSONWithRepair(t *testing.T) {
	var value map[string]string
	if err := ParseJSONWithRepair("{\"line\":\"one\ntwo\",\"path\":\"C:\\q\"}", &value); err != nil {
		t.Fatal(err)
	}
	if value["line"] != "one\ntwo" || value["path"] != `C:\q` {
		t.Fatalf("value = %#v", value)
	}
	if err := ParseJSONWithRepair(`{`, &value); err == nil {
		t.Fatal("incomplete completed JSON succeeded")
	}
}

func TestParseStreamingJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  any
	}{
		{name: "empty", want: map[string]any{}},
		{name: "null", input: `null`, want: map[string]any{}},
		{name: "complete", input: `{"key":"value"}`, want: map[string]any{"key": "value"}},
		{name: "partial string", input: `{"key":"val`, want: map[string]any{"key": "val"}},
		{name: "missing value", input: `{"a":1,"b":`, want: map[string]any{"a": float64(1)}},
		{name: "nested", input: `[{"key1":"value1","key2":["value2`, want: []any{map[string]any{"key1": "value1", "key2": []any{"value2"}}}},
		{name: "partial boolean", input: `{"ok":tr`, want: map[string]any{"ok": true}},
		{name: "partial exponent", input: `{"n":12e`, want: map[string]any{"n": float64(12)}},
		{name: "malformed partial string", input: "{\"line\":\"a\nb", want: map[string]any{}},
		{name: "malformed", input: `wrong`, want: map[string]any{}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := ParseStreamingJSON(test.input); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("ParseStreamingJSON(%q) = %#v, want %#v", test.input, got, test.want)
			}
		})
	}
}

func TestParseStreamingJSONDoesNotRetainLargeInput(t *testing.T) {
	input := `{"value":"` + strings.Repeat("x", 1<<20)
	value := ParseStreamingJSON(input).(map[string]any)
	if len(value["value"].(string)) != 1<<20 {
		t.Fatal("partial string length changed")
	}
}

func TestParseStreamingJSONBoundsNesting(t *testing.T) {
	input := strings.Repeat("[", maxPartialJSONDepth+100)
	if value := ParseStreamingJSON(input); value == nil {
		t.Fatal("deep partial input returned nil")
	}
}
