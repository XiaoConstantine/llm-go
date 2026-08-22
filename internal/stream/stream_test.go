package stream

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
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

func TestParentCancellationUnblocksRecvAndCloseWaitsForProducer(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cause := errors.New("caller stopped")
		cleanupErr := errors.New("cleanup failed")
		ctx, cancel := context.WithCancelCause(context.Background())
		cleanupStarted := make(chan struct{})
		releaseCleanup := make(chan struct{})
		stream := New(ctx, func(ctx context.Context, _ Emit) error {
			<-ctx.Done()
			close(cleanupStarted)
			<-releaseCleanup
			return errors.Join(ctx.Err(), context.Cause(ctx), cleanupErr)
		})

		received := make(chan error, 1)
		go func() {
			_, err := stream.Recv()
			received <- err
		}()
		synctest.Wait()

		cancel(cause)
		closed := make(chan error, 1)
		go func() { closed <- stream.Close() }()
		<-cleanupStarted
		synctest.Wait()

		select {
		case err := <-closed:
			t.Fatalf("Close returned before producer cleanup: %v", err)
		default:
		}
		terminal := <-received
		if !errors.Is(terminal, context.Canceled) || !errors.Is(terminal, cause) || errors.Is(terminal, cleanupErr) {
			t.Fatalf("Recv() error = %v, want cancellation and cause without pending cleanup diagnostic", terminal)
		}

		close(releaseCleanup)
		if err := <-closed; !errors.Is(err, cleanupErr) || errors.Is(err, context.Canceled) || errors.Is(err, cause) {
			t.Fatalf("Close() error = %v, want only cleanup diagnostic", err)
		}
		_, terminal = stream.Recv()
		if !errors.Is(terminal, context.Canceled) || !errors.Is(terminal, cause) || errors.Is(terminal, cleanupErr) {
			t.Fatalf("sticky terminal = %v, want cancellation and cause without cleanup diagnostic", terminal)
		}
	})
}

func TestCloseReturnsProducerCleanupError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cleanupErr := errors.New("cleanup failed")
		started := make(chan struct{})
		stream := New(context.Background(), func(ctx context.Context, _ Emit) error {
			close(started)
			<-ctx.Done()
			return cleanupErr
		})
		<-started

		results := make(chan error, 8)
		for range cap(results) {
			go func() { results <- stream.Close() }()
		}
		for range cap(results) {
			if err := <-results; !errors.Is(err, cleanupErr) {
				t.Fatalf("Close() error = %v, want %v", err, cleanupErr)
			}
		}
		if err := stream.Close(); !errors.Is(err, cleanupErr) {
			t.Fatalf("repeated Close() error = %v, want %v", err, cleanupErr)
		}
		if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
			t.Fatalf("Recv() error = %v, want io.EOF", err)
		}
	})
}

func TestCloseReturnsCleanupDiagnosticJoinedWithCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cleanupErr := errors.New("cleanup failed")
		started := make(chan struct{})
		stream := New(context.Background(), func(ctx context.Context, _ Emit) error {
			close(started)
			<-ctx.Done()
			return errors.Join(ctx.Err(), cleanupErr)
		})
		<-started

		if err := stream.Close(); !errors.Is(err, cleanupErr) || errors.Is(err, context.Canceled) {
			t.Fatalf("Close() error = %v, want only cleanup diagnostic", err)
		}
		if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
			t.Fatalf("Recv() error = %v, want io.EOF", err)
		}
	})
}

func TestClosePreservesNestedModelCleanupDiagnostic(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cleanupErr := errors.New("cleanup failed")
		started := make(chan struct{})
		stream := New(context.Background(), func(ctx context.Context, _ Emit) error {
			close(started)
			<-ctx.Done()
			return &llm.Error{
				Kind:     llm.KindTransport,
				Op:       "stream",
				Provider: "custom",
				Err:      fmt.Errorf("cleanup wrapper: %w", errors.Join(ctx.Err(), cleanupErr)),
			}
		})
		<-started

		err := stream.Close()
		var modelErr *llm.Error
		if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindTransport || modelErr.Provider != "custom" ||
			!errors.Is(err, cleanupErr) || errors.Is(err, context.Canceled) || !strings.Contains(err.Error(), "cleanup wrapper") {
			t.Fatalf("Close() error = %#v, want wrapped custom transport cleanup diagnostic", err)
		}
	})
}

func TestCloseDiagnosticFiltersCustomCancellationCause(t *testing.T) {
	sentinel := errors.New("caller stopped")
	cleanupErr := errors.New("cleanup failed")
	for _, cause := range []error{
		sentinel,
		fmt.Errorf("wrapped cancellation: %w", sentinel),
		errors.Join(sentinel, context.Canceled),
	} {
		diagnostic := closeDiagnostic(errors.Join(context.Canceled, cause, cleanupErr), cause)
		if !errors.Is(diagnostic, cleanupErr) || errors.Is(diagnostic, context.Canceled) || errors.Is(diagnostic, cause) {
			t.Fatalf("closeDiagnostic(%T) = %v, want only cleanup error", cause, diagnostic)
		}
	}
}

func TestCloseDiagnosticPreservesErrorsWithNilCauses(t *testing.T) {
	leaf := &nilUnwrapError{message: "panic diagnostic"}
	if diagnostic := closeDiagnostic(leaf, context.Canceled); diagnostic != leaf {
		t.Fatalf("closeDiagnostic(nil unwrapper) = %#v, want original", diagnostic)
	}
	modelErr := &llm.Error{Kind: llm.KindTransport, Op: "stream", Provider: "custom"}
	if diagnostic := closeDiagnostic(modelErr, context.Canceled); diagnostic != modelErr {
		t.Fatalf("closeDiagnostic(causeless model error) = %#v, want original", diagnostic)
	}
}

func TestParentCancellationUnblocksFullQueueAndPreservesCleanupError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cause := errors.New("caller stopped")
		cleanupErr := errors.New("cleanup failed")
		ctx, cancel := context.WithCancelCause(context.Background())
		queueFilled := make(chan struct{})
		stream := New(ctx, func(ctx context.Context, emit Emit) error {
			for index := range queueCapacity {
				if !emit(llm.Chunk{ID: fmt.Sprint(index)}) {
					return errors.New("Emit stopped before queue filled")
				}
			}
			close(queueFilled)
			if emit(llm.Chunk{ID: "overflow"}) {
				return errors.New("Emit accepted output after cancellation")
			}
			return errors.Join(ctx.Err(), context.Cause(ctx), cleanupErr)
		})
		<-queueFilled
		cancel(cause)

		for index := range queueCapacity {
			chunk, err := stream.Recv()
			if err != nil || chunk.ID != fmt.Sprint(index) {
				t.Fatalf("Recv() = (%#v, %v), want committed chunk %d", chunk, err, index)
			}
		}
		_, err := stream.Recv()
		if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || errors.Is(err, cleanupErr) {
			t.Fatalf("terminal error = %v, want cancellation and cause without cleanup error", err)
		}
		if err := stream.Close(); !errors.Is(err, cleanupErr) || errors.Is(err, context.Canceled) || errors.Is(err, cause) {
			t.Fatalf("Close() error = %v, want only cleanup diagnostic", err)
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

type nilUnwrapError struct{ message string }

func (err *nilUnwrapError) Error() string { return err.message }
func (*nilUnwrapError) Unwrap() error     { return nil }

func waitForCancellation(ctx context.Context, _ Emit) error {
	<-ctx.Done()
	return ctx.Err()
}
