package stream

import (
	"context"
	"errors"
	"io"
	"reflect"
	"sync"
	"testing"
	"testing/synctest"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestStreamDeliversChunksBeforeError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		terminal := errors.New("provider failed")
		stream := New(context.Background(), func(_ context.Context, emit Emit) error {
			emit(llm.Chunk{Content: []llm.Part{{Text: "first"}}})
			emit(llm.Chunk{Content: []llm.Part{{Text: "second"}}})
			return terminal
		})

		for _, want := range []string{"first", "second"} {
			chunk, err := stream.Recv()
			if err != nil {
				t.Fatalf("Recv() error = %v", err)
			}
			if got := chunk.Content[0].Text; got != want {
				t.Fatalf("Recv() text = %q, want %q", got, want)
			}
		}
		for range 2 {
			chunk, err := stream.Recv()
			if !reflect.ValueOf(chunk).IsZero() {
				t.Fatalf("Recv() chunk = %#v, want zero", chunk)
			}
			if !errors.Is(err, terminal) {
				t.Fatalf("Recv() error = %v, want %v", err, terminal)
			}
		}
		if err := stream.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})
}

func TestStreamCleanCompletionIsSticky(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		stream := New(context.Background(), func(context.Context, Emit) error { return nil })
		for range 2 {
			if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
				t.Fatalf("Recv() error = %v, want io.EOF", err)
			}
		}
		for range 2 {
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		}
	})
}

func TestCloseRetainsCommittedChunk(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		committed := make(chan struct{})
		stream := New(context.Background(), func(ctx context.Context, emit Emit) error {
			if !emit(llm.Chunk{Content: []llm.Part{{Text: "committed"}}}) {
				return errors.New("first Emit returned false")
			}
			close(committed)
			<-ctx.Done()
			return ctx.Err()
		})
		<-committed

		if err := stream.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		chunk, err := stream.Recv()
		if err != nil {
			t.Fatalf("Recv() error = %v", err)
		}
		if got := chunk.Content[0].Text; got != "committed" {
			t.Fatalf("Recv() text = %q, want committed", got)
		}
		if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
			t.Fatalf("Recv() error = %v, want io.EOF", err)
		}
	})
}

func TestCloseUnblocksRecvAndWaitsForProducer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cleanupStarted := make(chan struct{})
		releaseCleanup := make(chan struct{})
		stream := New(context.Background(), func(ctx context.Context, _ Emit) error {
			<-ctx.Done()
			close(cleanupStarted)
			<-releaseCleanup
			return ctx.Err()
		})

		received := make(chan error, 1)
		go func() {
			_, err := stream.Recv()
			received <- err
		}()
		synctest.Wait()

		closed := make(chan struct{})
		go func() {
			_ = stream.Close()
			close(closed)
		}()
		<-cleanupStarted
		synctest.Wait()
		select {
		case <-closed:
			t.Fatal("Close returned before producer cleanup")
		default:
		}
		if err := <-received; !errors.Is(err, io.EOF) {
			t.Fatalf("Recv() error = %v, want io.EOF", err)
		}

		close(releaseCleanup)
		synctest.Wait()
		select {
		case <-closed:
		default:
			t.Fatal("Close did not return after producer cleanup")
		}
	})
}

func TestParentCancellationAndClosePrecedence(t *testing.T) {
	t.Run("cancellation first", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			stream := New(ctx, waitForCancellation)
			cancel()
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
			if _, err := stream.Recv(); !errors.Is(err, context.Canceled) {
				t.Fatalf("Recv() error = %v, want context.Canceled", err)
			}
		})
	})

	t.Run("close first", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			stream := New(ctx, waitForCancellation)
			if err := stream.Close(); err != nil {
				t.Fatalf("Close() error = %v", err)
			}
			cancel()
			if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
				t.Fatalf("Recv() error = %v, want io.EOF", err)
			}
		})
	})
}

func TestNewWithCanceledContextDoesNotRunProducer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		called := false
		stream := New(ctx, func(context.Context, Emit) error {
			called = true
			return nil
		})

		if called {
			t.Fatal("New called producer with an already-canceled context")
		}
		if _, err := stream.Recv(); !errors.Is(err, context.Canceled) {
			t.Fatalf("Recv() error = %v, want context.Canceled", err)
		}
		if err := stream.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
	})
}

func TestCloseUnblocksProducerWithFullQueue(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		queueFilled := make(chan struct{})
		extraAccepted := make(chan bool, 1)
		stream := New(context.Background(), func(_ context.Context, emit Emit) error {
			for i := range queueCapacity {
				if !emit(llm.Chunk{ID: string(rune(i + 1))}) {
					return errors.New("Emit returned false before queue filled")
				}
			}
			close(queueFilled)
			extraAccepted <- emit(llm.Chunk{ID: "extra"})
			return nil
		})
		<-queueFilled
		synctest.Wait()

		if err := stream.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if accepted := <-extraAccepted; accepted {
			t.Fatal("Emit accepted a chunk after Close")
		}

		for range queueCapacity {
			if _, err := stream.Recv(); err != nil {
				t.Fatalf("Recv() error = %v", err)
			}
		}
		if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
			t.Fatalf("Recv() error = %v, want io.EOF", err)
		}
	})
}

func TestConcurrentClose(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		stream := New(context.Background(), waitForCancellation)
		results := make(chan error, 8)
		for range cap(results) {
			go func() { results <- stream.Close() }()
		}
		for range cap(results) {
			if err := <-results; err != nil {
				t.Fatalf("Close() error = %v", err)
			}
		}
		if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
			t.Fatalf("Recv() error = %v, want io.EOF", err)
		}
	})
}

func TestCancellationCauseIsPreserved(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cause := errors.New("caller stopped")
		ctx, cancel := context.WithCancelCause(context.Background())
		stream := New(ctx, waitForCancellation)
		cancel(cause)

		_, err := stream.Recv()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Recv() error = %v, want context.Canceled", err)
		}
		if !errors.Is(err, cause) {
			t.Fatalf("Recv() error = %v, want cause %v", err, cause)
		}
		if closeErr := stream.Close(); closeErr != nil {
			t.Fatalf("Close() error = %v", closeErr)
		}
	})
}

func TestWrappedCancellationCauseIsPreserved(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cause := errors.Join(errors.New("caller stopped"), context.Canceled)
		ctx, cancel := context.WithCancelCause(context.Background())
		stream := New(ctx, waitForCancellation)
		cancel(cause)

		_, err := stream.Recv()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Recv() error = %v, want context.Canceled", err)
		}
		if !errors.Is(err, cause) {
			t.Fatalf("Recv() error = %v, want cause %v", err, cause)
		}
		if closeErr := stream.Close(); closeErr != nil {
			t.Fatalf("Close() error = %v", closeErr)
		}
	})
}

func TestWatcherRechecksCancellationAfterProducerDone(t *testing.T) {
	for range 100 {
		parent, cancelParent := context.WithCancel(context.Background())
		ctx, cancel := context.WithCancel(parent)
		s := &pipe{
			parent:       parent,
			ctx:          ctx,
			cancel:       cancel,
			producerDone: make(chan struct{}),
			watcherDone:  make(chan struct{}),
		}
		s.changed = sync.NewCond(&s.mu)

		cancelParent()
		close(s.producerDone)
		s.watchParent()

		s.mu.Lock()
		terminal := s.terminal
		s.mu.Unlock()
		if !errors.Is(terminal, context.Canceled) {
			t.Fatalf("terminal error = %v, want context.Canceled", terminal)
		}
	}
}

func waitForCancellation(ctx context.Context, _ Emit) error {
	<-ctx.Done()
	return ctx.Err()
}
