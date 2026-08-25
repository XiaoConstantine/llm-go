package llm

import (
	"context"
	"errors"
	"io"
	"sync"
)

type scriptedGenerator struct {
	mu            sync.Mutex
	generate      func(context.Context, Request, int) (*Response, error)
	streams       []func() Stream
	streamErrors  []error
	generateCalls int
	streamCalls   int
}

func (g *scriptedGenerator) Info() ModelInfo {
	return ModelInfo{Provider: "test", Model: "model", API: "test", Capabilities: []Capability{CapabilityGeneration, CapabilityStreaming}}
}

func (g *scriptedGenerator) Generate(ctx context.Context, request Request) (*Response, error) {
	g.mu.Lock()
	g.generateCalls++
	call := g.generateCalls
	fn := g.generate
	g.mu.Unlock()
	return fn(ctx, request, call)
}

func (g *scriptedGenerator) Stream(context.Context, Request) (Stream, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.streamCalls++
	index := g.streamCalls - 1
	if index < len(g.streamErrors) && g.streamErrors[index] != nil {
		return nil, g.streamErrors[index]
	}
	return g.streams[index](), nil
}

type scriptedStream struct {
	mu       sync.Mutex
	chunks   []Chunk
	errs     []error
	index    int
	closed   bool
	closeErr error
}

func (s *scriptedStream) Recv() (Chunk, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.index < len(s.chunks) {
		chunk := s.chunks[s.index]
		s.index++
		return chunk, nil
	}
	index := s.index - len(s.chunks)
	s.index++
	if index < len(s.errs) {
		return Chunk{}, s.errs[index]
	}
	return Chunk{}, io.EOF
}

func (s *scriptedStream) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return s.closeErr
}

func (s *scriptedStream) isClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed
}

type blockingCloseStream struct {
	recvStarted  chan struct{}
	closeStarted chan struct{}
	releaseClose chan struct{}
	releaseRecv  chan struct{}
	chunk        Chunk
	recvErr      error
	closeErr     error
}

func (s *blockingCloseStream) Recv() (Chunk, error) {
	close(s.recvStarted)
	<-s.releaseRecv
	return s.chunk, s.recvErr
}

func (s *blockingCloseStream) Close() error {
	close(s.closeStarted)
	close(s.releaseRecv)
	<-s.releaseClose
	return s.closeErr
}

func retryableTestError() error {
	retryable := true
	return &Error{Kind: KindTransport, Provider: "test", Retryable: &retryable, Err: errors.New("temporary")}
}

func validGenerationRequest() Request {
	return Request{Messages: []Message{{Role: RoleUser, Content: []Part{{Text: "hello"}}}}}
}
