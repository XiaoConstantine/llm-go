package llm

import (
	"context"
	"errors"
	"testing"
)

func TestCollectAssemblesAndPreservesPartialFailures(t *testing.T) {
	usage1 := &Usage{InputTokens: 2, OutputTokens: 1, TotalTokens: 3}
	usage2 := &Usage{InputTokens: 2, OutputTokens: 2, TotalTokens: 4}
	stream := &scriptedStream{chunks: []Chunk{
		{ID: "id", Model: "model", Content: []Part{{Text: "a"}}, ReasoningSummary: "r1", Usage: usage1},
		{ID: "id", Model: "model", Content: []Part{{Text: "b"}}, ReasoningSummary: "r2", ProviderData: []byte(`{"x":1}`), FinishReason: FinishReasonStop, Usage: usage2},
	}}
	response, err := Collect(stream, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !stream.closed || response.Text() != "ab" || response.ReasoningSummary != "r1r2" || response.Message.Role != RoleAssistant || string(response.Message.ProviderData) != `{"x":1}` || response.Usage.OutputTokens != 2 {
		t.Fatalf("Collect() = %#v, closed %v", response, stream.closed)
	}

	bad := &scriptedStream{chunks: []Chunk{{ID: "one", Content: []Part{{Text: "kept"}}}, {ID: "two"}}}
	partial, err := Collect(bad, nil)
	if err == nil || partial.Text() != "kept" || partial.ID != "one" || !bad.closed {
		t.Fatalf("partial = %#v, error = %v, closed = %v", partial, err, bad.closed)
	}
}

func TestCollectRejectsDuplicateProviderDataAndDecreasingUsage(t *testing.T) {
	usage1 := &Usage{InputTokens: 2, OutputTokens: 2, TotalTokens: 4}
	usage2 := &Usage{InputTokens: 2, OutputTokens: 1, TotalTokens: 3}
	for _, stream := range []*scriptedStream{
		{chunks: []Chunk{{ProviderData: []byte(`{"a":1}`)}, {ProviderData: []byte(`{"b":2}`)}}},
		{chunks: []Chunk{{Usage: usage1}, {Usage: usage2}}},
	} {
		if _, err := Collect(stream, nil); err == nil {
			t.Fatal("Collect() accepted malformed chunks")
		}
		if !stream.closed {
			t.Fatal("Collect() did not close malformed stream")
		}
	}
}

func TestCollectReturnsPartialResponseOnCancellation(t *testing.T) {
	stream := &scriptedStream{chunks: []Chunk{{Content: []Part{{Text: "partial"}}}}, errs: []error{context.Canceled}}
	response, err := Collect(stream, nil)
	if !errors.Is(err, context.Canceled) || response.Text() != "partial" || !stream.closed {
		t.Fatalf("Collect() = %#v, %v; closed %v", response, err, stream.closed)
	}
}

func TestCollectJoinsCloseErrorAfterProviderFailure(t *testing.T) {
	providerErr := retryableTestError()
	closeErr := errors.New("close failed")
	stream := &scriptedStream{errs: []error{providerErr}, closeErr: closeErr}
	_, err := Collect(stream, nil)
	if !errors.Is(err, providerErr) || !errors.Is(err, closeErr) {
		t.Fatalf("Collect() error = %v", err)
	}
}

func TestCollectStructuralLeavesArgumentSchemasToExecutionOwner(t *testing.T) {
	tools := []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","properties":{"city":{"type":"string","minLength":2}},"required":["city"]}`)}}
	stream := &scriptedStream{chunks: []Chunk{{ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(`{"city":"x"}`)}}}}}
	response, err := CollectStructural(stream)
	if err != nil || len(response.Message.ToolCalls) != 1 || string(response.Message.ToolCalls[0].Arguments) != `{"city":"x"}` || !stream.closed {
		t.Fatalf("Collect() = %#v, %v", response, err)
	}
	if err := ValidateToolCalls(tools, response.Message.ToolCalls); err == nil {
		t.Fatal("explicit schema validation accepted a too-short city")
	}
}

func TestCollectPreservesPublishedSchemaValidation(t *testing.T) {
	// Preserve the exact public signature, including function-value assignments.
	collectCalls := Collect
	tools := []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","properties":{"city":{"type":"string","minLength":2}},"required":["city"]}`)}}
	for _, test := range []struct {
		name      string
		tools     []Tool
		arguments string
		wantError bool
	}{
		{"valid", tools, `{"city":"Paris"}`, false},
		{"schema mismatch", tools, `{"city":"x"}`, true},
		{"undeclared", nil, `{"city":"Paris"}`, true},
		{"invalid schema", []Tool{{Name: "lookup", InputSchema: []byte(`{`)}}, `{"city":"Paris"}`, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			closeErr := errors.New("close failed")
			stream := &scriptedStream{chunks: []Chunk{{Content: []Part{{Text: "kept"}}}, {ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(test.arguments)}}}}}
			if test.wantError {
				stream.closeErr = closeErr
			}
			response, err := collectCalls(stream, test.tools)
			if response == nil || !stream.closed {
				t.Fatalf("response=%+v closed=%t error=%v", response, stream.closed, err)
			}
			if test.wantError {
				failure, ok := errors.AsType[*Error](err)
				if !ok || failure.Kind != KindMalformedResponse || failure.Op != "collect" || !errors.Is(err, closeErr) || len(response.Message.ToolCalls) != 0 {
					t.Fatalf("response=%+v error=%v", response, err)
				}
				if test.name != "invalid schema" && response.Text() != "kept" {
					t.Fatalf("partial response=%+v", response)
				}
			} else if err != nil || response.Text() != "kept" || len(response.Message.ToolCalls) != 1 || string(response.Message.ToolCalls[0].Arguments) != test.arguments {
				t.Fatalf("response=%+v error=%v", response, err)
			}
		})
	}
}
