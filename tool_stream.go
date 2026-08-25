package llm

import (
	"errors"
	"fmt"
	"io"
	"sync"
)

// ValidateToolCallStream wraps stream so every completed ToolCall in a Chunk or
// ToolCallEnd event is validated before the chunk is delivered. Partial argument
// delta events are never validated. The caller retains the normal obligation to
// close the returned stream. If setup fails, stream is closed before returning.
func ValidateToolCallStream(stream Stream, tools []Tool, provider string) (Stream, error) {
	if stream == nil {
		return nil, fmt.Errorf("tool-call stream must not be nil")
	}
	validator, err := newToolCallValidator(tools)
	if err != nil {
		closeErr := stream.Close()
		if closeErr != nil {
			err = errors.Join(err, closeErr)
		}
		return nil, err
	}
	return &toolCallStream{Stream: stream, validator: validator, provider: provider, closeDone: make(chan struct{})}, nil
}

type toolCallStream struct {
	Stream
	validator toolCallValidator
	provider  string

	mu           sync.Mutex
	terminal     error
	closeStarted bool
	recvDone     chan struct{}
	closeDone    chan struct{}

	underlyingCloseOnce sync.Once
	underlyingCloseErr  error
	closeOnce           sync.Once
}

func (s *toolCallStream) Recv() (Chunk, error) {
	s.mu.Lock()
	if s.terminal != nil {
		err := s.terminal
		s.mu.Unlock()
		return Chunk{}, err
	}
	s.recvDone = make(chan struct{})
	s.mu.Unlock()

	chunk, err := s.Stream.Recv()
	if err != nil {
		s.mu.Lock()
		if s.terminal == nil {
			s.terminal = err
		}
		terminal := s.terminal
		s.finishRecvLocked()
		s.mu.Unlock()
		return Chunk{}, terminal
	}
	validationErr := s.validator.validate(completedToolCalls(chunk))
	s.mu.Lock()
	if validationErr != nil {
		if s.terminal == nil {
			if s.closeStarted {
				s.terminal = io.EOF
			} else {
				s.terminal = &Error{Kind: KindMalformedResponse, Op: "stream", Provider: s.provider,
					Err: fmt.Errorf("validate completed tool call arguments: %w", validationErr)}
			}
		}
		terminal := s.terminal
		s.finishRecvLocked()
		s.mu.Unlock()
		s.closeUnderlying()
		return Chunk{}, terminal
	}
	s.finishRecvLocked()
	s.mu.Unlock()
	return chunk, nil
}

// Close records that Close started before waiting on the wrapped stream. A
// validation failure discovered after that point cannot replace Close's EOF.
// Diagnostics from releasing the wrapped stream are returned only by Close.
func (s *toolCallStream) Close() error {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		s.closeStarted = true
		recvDone := s.recvDone
		s.mu.Unlock()

		s.closeUnderlying()
		if recvDone != nil {
			<-recvDone
		}

		// Do not synthesize EOF here. With no active validation race, the
		// wrapped stream owns the terminal transition; the next Recv caches its
		// sticky EOF, provider failure, or canceled-before-Close context error.
		close(s.closeDone)
	})
	<-s.closeDone
	return s.underlyingCloseErr
}

func (s *toolCallStream) closeUnderlying() {
	s.underlyingCloseOnce.Do(func() {
		s.underlyingCloseErr = s.Stream.Close()
	})
}

func (s *toolCallStream) finishRecvLocked() {
	if s.recvDone != nil {
		close(s.recvDone)
		s.recvDone = nil
	}
}

func completedToolCalls(chunk Chunk) []ToolCall {
	completed := append([]ToolCall(nil), chunk.ToolCalls...)
	for _, event := range chunk.Events {
		if event.Kind == StreamEventToolCallEnd && event.ToolCall != nil {
			completed = append(completed, *event.ToolCall)
		}
	}
	return completed
}

var _ Stream = (*toolCallStream)(nil)
