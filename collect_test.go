package llm

import (
	"context"
	"errors"
	"strings"
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

func TestCollectValidatesCompletedToolCalls(t *testing.T) {
	tools := []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","properties":{"city":{"type":"string","minLength":2}},"required":["city"]}`)}}
	stream := &scriptedStream{chunks: []Chunk{{ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(`{"city":"x"}`)}}}}}
	partial, err := Collect(stream, tools)
	if err == nil || !strings.Contains(err.Error(), `$.city`) || len(partial.Message.ToolCalls) != 0 {
		t.Fatalf("Collect() = %#v, %v", partial, err)
	}
}
