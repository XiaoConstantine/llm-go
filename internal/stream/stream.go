// Package stream implements the internal producer side of llm.Stream.
package stream

import (
	"context"
	"errors"
	"io"
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
// before returning.
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

	mu          sync.Mutex
	changed     *sync.Cond
	queue       [queueCapacity]llm.Chunk
	queueHead   int
	queueSize   int
	terminal    error
	terminalSet bool

	producerDone chan struct{}
	watcherDone  chan struct{}
	closeOnce    sync.Once
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
		if !s.terminalSet {
			if s.parent.Err() != nil {
				s.finishLocked(contextError(s.parent))
			} else {
				s.finishLocked(io.EOF)
			}
		}
		s.mu.Unlock()

		s.cancel()
		<-s.producerDone
		<-s.watcherDone
	})
	return nil
}

func (s *pipe) run(producer Producer) {
	defer close(s.producerDone)

	if s.parent.Err() != nil {
		s.finish(contextError(s.parent))
		return
	}
	if s.ctx.Err() != nil {
		if s.parent.Err() != nil {
			s.finish(contextError(s.parent))
		}
		return
	}

	err := producer(s.ctx, s.emit)
	if s.parent.Err() != nil {
		s.finish(contextError(s.parent))
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
	case <-s.producerDone:
	}
	if s.parent.Err() != nil {
		s.finish(contextError(s.parent))
		s.cancel()
	}
}

func (s *pipe) emit(chunk llm.Chunk) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	for s.queueSize == queueCapacity && !s.terminalSet {
		s.changed.Wait()
	}
	if s.terminalSet {
		return false
	}
	if s.parent.Err() != nil {
		s.finishLocked(contextError(s.parent))
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
