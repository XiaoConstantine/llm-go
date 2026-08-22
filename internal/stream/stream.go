// Package stream implements the internal producer side of llm.Stream.
package stream

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"sync"

	llm "github.com/XiaoConstantine/llm-go"
)

const queueCapacity = 16

// Emit commits a chunk to a stream. It returns false after the stream reaches a
// terminal state. A successful call transfers ownership of all storage reachable
// from the chunk; the producer must not mutate it afterward.
type Emit func(llm.Chunk) bool

// Producer emits chunks until generation finishes. Returning nil completes the
// stream with io.EOF; returning an error completes it with that error. A
// Producer must return promptly after its context is done or Emit returns false.
// It must not use Emit after returning and must join any goroutines it starts
// before returning. If Close stops the producer, a non-cancellation error it
// returns is reported by Close without replacing the stream's terminal state.
type Producer func(context.Context, Emit) error

// New starts producer and returns its stream. The producer runs at most once.
func New(parent context.Context, producer Producer) llm.Stream {
	if producer == nil {
		panic("stream: nil producer")
	}

	ctx, cancel := context.WithCancel(parent)
	s := &pipe{
		parent:       parent,
		ctx:          ctx,
		cancel:       cancel,
		producerDone: make(chan struct{}),
		watcherDone:  make(chan struct{}),
	}
	s.changed = sync.NewCond(&s.mu)

	if err := parent.Err(); err != nil {
		s.finish(contextError(parent))
		cancel()
		close(s.producerDone)
		close(s.watcherDone)
		return s
	}

	go s.watchParent()
	go s.run(producer)
	return s
}

type pipe struct {
	parent context.Context
	ctx    context.Context
	cancel context.CancelFunc

	mu             sync.Mutex
	changed        *sync.Cond
	queue          [queueCapacity]llm.Chunk
	queueHead      int
	queueSize      int
	terminal       error
	terminalSet    bool
	parentTerminal bool
	producerErr    error

	producerDone chan struct{}
	watcherDone  chan struct{}
	closeOnce    sync.Once
	closeErr     error
}

func (s *pipe) Recv() (llm.Chunk, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for s.queueSize == 0 && !s.terminalSet {
		s.changed.Wait()
	}
	if s.queueSize == 0 {
		return llm.Chunk{}, s.terminal
	}

	chunk := s.queue[s.queueHead]
	s.queue[s.queueHead] = llm.Chunk{}
	s.queueHead = (s.queueHead + 1) % queueCapacity
	s.queueSize--
	s.changed.Broadcast()
	return chunk, nil
}

func (s *pipe) Close() error {
	s.closeOnce.Do(func() {
		s.mu.Lock()
		parentErr := s.parent.Err()
		closeInitiated := !s.terminalSet && parentErr == nil
		if !s.terminalSet {
			if parentErr != nil {
				s.finishParentLocked(contextError(s.parent))
			} else {
				s.finishLocked(io.EOF)
			}
		}
		s.mu.Unlock()

		s.cancel()
		<-s.producerDone
		<-s.watcherDone

		s.mu.Lock()
		producerErr := s.producerErr
		reportDiagnostic := closeInitiated || s.parentTerminal
		s.mu.Unlock()
		if reportDiagnostic {
			s.closeErr = closeDiagnostic(producerErr, context.Cause(s.ctx))
		}
	})
	return s.closeErr
}

func (s *pipe) run(producer Producer) {
	defer close(s.producerDone)

	if s.parent.Err() != nil {
		s.finishParent(contextError(s.parent))
		return
	}
	if s.ctx.Err() != nil {
		if s.parent.Err() != nil {
			s.finishParent(contextError(s.parent))
		}
		return
	}

	err := producer(s.ctx, s.emit)
	s.mu.Lock()
	s.producerErr = err
	s.mu.Unlock()
	if s.parent.Err() != nil {
		s.finishParent(contextError(s.parent))
		return
	}
	if err != nil {
		s.finish(err)
		return
	}
	s.finish(io.EOF)
}

func (s *pipe) watchParent() {
	defer close(s.watcherDone)
	select {
	case <-s.parent.Done():
		s.finishParent(contextError(s.parent))
		s.cancel()
		<-s.producerDone
	case <-s.producerDone:
		if s.parent.Err() != nil {
			s.finishParent(contextError(s.parent))
			s.cancel()
		}
	}
}

func (s *pipe) emit(chunk llm.Chunk) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	for s.queueSize == queueCapacity && !s.terminalSet && s.ctx.Err() == nil && s.parent.Err() == nil {
		s.changed.Wait()
	}
	if s.terminalSet || s.ctx.Err() != nil || s.parent.Err() != nil {
		return false
	}

	tail := (s.queueHead + s.queueSize) % queueCapacity
	s.queue[tail] = chunk
	s.queueSize++
	s.changed.Broadcast()
	return true
}

func (s *pipe) finish(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.finishLocked(err)
}

func (s *pipe) finishParent(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.finishParentLocked(err)
}

func (s *pipe) finishParentLocked(err error) {
	if s.terminalSet {
		return
	}
	s.finishLocked(err)
	s.parentTerminal = true
}

func (s *pipe) finishLocked(err error) {
	if s.terminalSet {
		return
	}
	if err == nil {
		err = io.EOF
	}
	s.terminal = err
	s.terminalSet = true
	s.changed.Broadcast()
}

func closeDiagnostic(err, cancellation error) error {
	if err == nil {
		return nil
	}
	if sameError(err, cancellation) {
		return nil
	}
	if modelErr, ok := err.(*llm.Error); ok {
		if modelErr.Err == nil {
			return modelErr
		}
		diagnostic := closeDiagnostic(modelErr.Err, cancellation)
		if diagnostic == nil {
			return nil
		}
		clone := *modelErr
		clone.Err = diagnostic
		return &clone
	}
	if joined, ok := err.(interface{ Unwrap() []error }); ok {
		diagnostics := make([]error, 0, len(joined.Unwrap()))
		for _, cause := range joined.Unwrap() {
			if diagnostic := closeDiagnostic(cause, cancellation); diagnostic != nil {
				diagnostics = append(diagnostics, diagnostic)
			}
		}
		return errors.Join(diagnostics...)
	}
	if wrapped, ok := err.(interface{ Unwrap() error }); ok {
		unwrapped := wrapped.Unwrap()
		if unwrapped != nil {
			diagnostic := closeDiagnostic(unwrapped, cancellation)
			if diagnostic == nil {
				return nil
			}
			if message := err.Error(); strings.HasSuffix(message, unwrapped.Error()) {
				return fmt.Errorf("%s%w", strings.TrimSuffix(message, unwrapped.Error()), diagnostic)
			}
			return diagnostic
		}
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, io.EOF) ||
		cancellation != nil && errors.Is(err, cancellation) {
		return nil
	}
	return err
}

func sameError(left, right error) bool {
	if left == nil || right == nil {
		return left == nil && right == nil
	}
	leftType := reflect.TypeOf(left)
	return leftType == reflect.TypeOf(right) && leftType.Comparable() && left == right
}

func contextError(ctx context.Context) error {
	err := ctx.Err()
	cause := context.Cause(ctx)
	if cause == nil {
		return err
	}
	if errors.Is(cause, err) {
		return cause
	}
	return errors.Join(err, cause)
}

var _ llm.Stream = (*pipe)(nil)
