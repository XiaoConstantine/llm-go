package llm

import (
	"context"
	"errors"
	"io"
	"testing"
)

func TestToolCallStreamRejectsInvalidCompletedCallsButNotDeltas(t *testing.T) {
	tools := []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","properties":{"count":{"type":"integer","minimum":1}},"required":["count"]}`)}}
	underlying := &scriptedStream{chunks: []Chunk{
		{Events: []StreamEvent{{Kind: StreamEventToolCallDelta, Delta: `{"count":`}}},
		{ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(`{"count":0}`)}}},
	}}
	stream, err := ValidateToolCallStream(underlying, tools, "provider")
	if err != nil {
		t.Fatal(err)
	}
	first, err := stream.Recv()
	if err != nil || len(first.Events) != 1 {
		t.Fatalf("delta Recv() = %#v, %v", first, err)
	}
	chunk, err := stream.Recv()
	if len(chunk.ToolCalls) != 0 || err == nil {
		t.Fatalf("invalid completed Recv() = %#v, %v", chunk, err)
	}
	var modelErr *Error
	if !errors.As(err, &modelErr) || modelErr.Kind != KindMalformedResponse || modelErr.Op != "stream" || modelErr.Provider != "provider" {
		t.Fatalf("terminal classification = %#v", modelErr)
	}
	if !underlying.closed {
		t.Fatal("invalid completed stream was not closed")
	}
	_, again := stream.Recv()
	if again != err {
		t.Fatalf("terminal not sticky: %v then %v", err, again)
	}
	if closeErr := stream.Close(); closeErr != nil {
		t.Fatalf("Close() = %v", closeErr)
	}
}

func TestToolCallStreamCloseDefersToExistingUnderlyingTerminal(t *testing.T) {
	providerErr := &Error{Kind: KindProvider, Err: errors.New("provider terminal")}
	for _, test := range []struct {
		name     string
		terminal error
	}{
		{name: "canceled before close", terminal: context.Canceled},
		{name: "provider terminal before close", terminal: providerErr},
	} {
		t.Run(test.name, func(t *testing.T) {
			underlying := &scriptedStream{errs: []error{test.terminal}}
			stream, err := ValidateToolCallStream(underlying, nil, "provider")
			if err != nil {
				t.Fatal(err)
			}
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() = %v", err)
			}
			if _, err := stream.Recv(); !errors.Is(err, test.terminal) {
				t.Fatalf("Recv() after Close = %v, want %v", err, test.terminal)
			}
			first := func() error { _, err := stream.Recv(); return err }()
			if !errors.Is(first, test.terminal) {
				t.Fatalf("sticky Recv() = %v, want %v", first, test.terminal)
			}
			if err := stream.Close(); err != nil {
				t.Fatalf("repeated Close() = %v", err)
			}
		})
	}
}

func TestToolCallStreamCloseWinsValidationRaceAndKeepsCloseErrorSeparate(t *testing.T) {
	closeErr := errors.New("cleanup failed")
	underlying := &blockingCloseStream{
		recvStarted: make(chan struct{}), closeStarted: make(chan struct{}),
		releaseClose: make(chan struct{}), releaseRecv: make(chan struct{}), closeErr: closeErr,
		chunk: Chunk{ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(`{}`)}}},
	}
	tools := []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","required":["city"]}`)}}
	stream, err := ValidateToolCallStream(underlying, tools, "provider")
	if err != nil {
		t.Fatal(err)
	}
	recvDone := make(chan struct {
		chunk Chunk
		err   error
	}, 1)
	go func() {
		chunk, err := stream.Recv()
		recvDone <- struct {
			chunk Chunk
			err   error
		}{chunk, err}
	}()
	<-underlying.recvStarted
	closed := make(chan error, 1)
	go func() { closed <- stream.Close() }()
	<-underlying.closeStarted
	close(underlying.releaseClose)
	if err := <-closed; !errors.Is(err, closeErr) {
		t.Fatalf("Close() = %v", err)
	}
	result := <-recvDone
	if !errors.Is(result.err, io.EOF) || errors.Is(result.err, closeErr) || len(result.chunk.ToolCalls) != 0 {
		t.Fatalf("Recv() = %#v, %v", result.chunk, result.err)
	}
	if err := stream.Close(); !errors.Is(err, closeErr) {
		t.Fatalf("repeated Close() = %v", err)
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("sticky Recv() = %v", err)
	}
}

func TestToolCallStreamValidationWinsAndCloseErrorIsOnlyFromClose(t *testing.T) {
	closeErr := errors.New("cleanup failed")
	underlying := &scriptedStream{chunks: []Chunk{{ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(`{}`)}}}}, closeErr: closeErr}
	tools := []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","required":["city"]}`)}}
	stream, err := ValidateToolCallStream(underlying, tools, "provider")
	if err != nil {
		t.Fatal(err)
	}
	chunk, terminal := stream.Recv()
	if terminal == nil || errors.Is(terminal, closeErr) || len(chunk.ToolCalls) != 0 {
		t.Fatalf("Recv() = %#v, %v", chunk, terminal)
	}
	var modelErr *Error
	if !errors.As(terminal, &modelErr) || modelErr.Kind != KindMalformedResponse {
		t.Fatalf("terminal = %#v", modelErr)
	}
	if err := stream.Close(); !errors.Is(err, closeErr) {
		t.Fatalf("Close() = %v", err)
	}
	if _, err := stream.Recv(); err != terminal {
		t.Fatalf("sticky terminal = %v, want %v", err, terminal)
	}
}

func TestToolCallStreamValidatesCompletedEndEvents(t *testing.T) {
	tools := []Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object","required":["city"]}`)}}
	underlying := &scriptedStream{chunks: []Chunk{{Events: []StreamEvent{{Kind: StreamEventToolCallEnd, ToolCall: &ToolCall{Name: "lookup", Arguments: []byte(`{}`)}}}}}}
	stream, err := ValidateToolCallStream(underlying, tools, "provider")
	if err != nil {
		t.Fatal(err)
	}
	if chunk, err := stream.Recv(); err == nil || len(chunk.Events) != 0 {
		t.Fatalf("Recv() = %#v, %v", chunk, err)
	}
}
