package llm

import (
	"errors"
	"io"
	"reflect"
	"testing"
)

func TestToolCallStructureLeavesSchemasToExplicitValidation(t *testing.T) {
	tools := []Tool{{Name: "history_retrieve", InputSchema: []byte(`{"type":"object","properties":{"limit":{"type":"integer","minimum":0,"maximum":6144}},"required":["limit"],"additionalProperties":false}`)}}
	for _, test := range []struct {
		name, arguments string
		schemaOnly      bool
	}{
		{"maximum", `{"limit":14000}`, true},
		{"minimum", `{"limit":-1}`, true},
		{"type", `{"limit":"6144"}`, true},
		{"required", `{}`, true},
		{"extra property", `{"limit":1,"other":true}`, true},
		{"wrong root type", `[]`, true},
		{"truncated JSON", `{"limit":`, false},
		{"duplicate key", `{"limit":1,"limit":2}`, false},
		{"invalid UTF-8", "{\"limit\":\"\xff\"}", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			calls := []ToolCall{{ID: "call", Name: "history_retrieve", Arguments: []byte(test.arguments)}}
			if err := ValidateToolCalls(tools, calls); err == nil {
				t.Fatal("explicit validation accepted invalid arguments")
			}
			if err := ValidateToolCallStructure(calls); (err == nil) != test.schemaOnly {
				t.Fatalf("structure validation = %v, schemaOnly = %t", err, test.schemaOnly)
			}
			if string(calls[0].Arguments) != test.arguments {
				t.Fatalf("arguments changed to %s", calls[0].Arguments)
			}
		})
	}
	calls := []ToolCall{{Name: "unknown", Arguments: []byte(`{}`)}}
	if err := ValidateToolCallStructure(calls); err != nil {
		t.Fatal(err)
	}
	if err := ValidateToolCalls(tools, calls); err == nil {
		t.Fatal("explicit validation accepted an undeclared name")
	}
	calls[0].Name = ""
	if err := ValidateToolCallStructure(calls); err == nil {
		t.Fatal("structure validation accepted an empty name")
	}
}

func TestToolCallStructureStreamPreservesResponseButRejectsMalformedJSON(t *testing.T) {
	for _, arguments := range []string{`{}`, `{"city":`} {
		t.Run(arguments, func(t *testing.T) {
			call := ToolCall{ID: "call", Name: "lookup", Arguments: []byte(arguments)}
			usage := &Usage{InputTokens: 10, OutputTokens: 4, TotalTokens: 14}
			underlying := &scriptedStream{chunks: []Chunk{
				{Events: []StreamEvent{{Kind: StreamEventToolCallDelta, Delta: arguments}}},
				{Events: []StreamEvent{{Kind: StreamEventToolCallEnd, ToolCall: &call}}},
				{ToolCalls: []ToolCall{call}, Usage: usage, ProviderData: []byte(`{"replay":"opaque"}`), FinishReason: FinishReasonToolCall},
			}}
			stream, err := ValidateToolCallStructureStream(underlying, "provider")
			if err != nil {
				t.Fatal(err)
			}
			response, err := CollectStructural(stream)
			if !underlying.closed {
				t.Fatal("Collect did not close the underlying stream")
			}
			if arguments != `{}` {
				modelErr, ok := errors.AsType[*Error](err)
				if !ok || modelErr.Kind != KindMalformedResponse || modelErr.Op != "stream" {
					t.Fatalf("malformed terminal = %v", err)
				}
				if _, again := stream.Recv(); again != err {
					t.Fatalf("terminal changed: %v then %v", err, again)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(response.Message.ToolCalls, []ToolCall{call}) || !reflect.DeepEqual(response.Usage, usage) || string(response.Message.ProviderData) != `{"replay":"opaque"}` || response.FinishReason != FinishReasonToolCall {
				t.Fatalf("collected response = %#v", response)
			}
			if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
				t.Fatalf("terminal = %v", err)
			}
		})
	}
}
