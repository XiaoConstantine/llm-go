package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRetryPolicyValidation(t *testing.T) {
	generator := &scriptedGenerator{}
	for _, policy := range []RetryPolicy{
		{}, {MaxAttempts: -1}, {MaxAttempts: 2, InitialBackoff: -1},
		{MaxAttempts: 2, InitialBackoff: 2, MaxBackoff: 1},
		{MaxAttempts: 2, Multiplier: .5}, {MaxAttempts: 2, Jitter: 1.1},
	} {
		if _, err := WithRetry(generator, policy); err == nil {
			t.Errorf("WithRetry(%+v) succeeded", policy)
		}
	}
}

func TestRetryHonorsExplicitProviderHints(t *testing.T) {
	generator := &scriptedGenerator{generate: func(context.Context, Request, int) (*Response, error) { return nil, nil }}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 2})
	if err != nil {
		t.Fatal(err)
	}
	implementation := retrying.(*retryGenerator)
	yes, no := true, false
	if !implementation.retryable(&Error{Kind: KindInvalidRequest, Retryable: &yes}) {
		t.Fatal("explicit positive retry hint was ignored")
	}
	if implementation.retryable(&Error{Kind: KindRateLimit, Retryable: &no}) {
		t.Fatal("explicit negative retry hint was ignored")
	}
}

func TestRetryAmbiguousTransportRequiresExplicitOptIn(t *testing.T) {
	transportErr := &Error{Kind: KindTransport, Err: errors.New("connection lost after POST")}
	generator := &scriptedGenerator{generate: func(context.Context, Request, int) (*Response, error) { return nil, transportErr }}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := retrying.Generate(context.Background(), validGenerationRequest()); !errors.Is(err, transportErr) {
		t.Fatalf("Generate() = %v", err)
	}
	if generator.generateCalls != 1 {
		t.Fatalf("Generate attempts = %d, want 1", generator.generateCalls)
	}

	first := &scriptedStream{errs: []error{transportErr}}
	generator = &scriptedGenerator{streams: []func() Stream{func() Stream { return first }, func() Stream { return &scriptedStream{} }}}
	retrying, _ = WithRetry(generator, RetryPolicy{MaxAttempts: 3})
	stream, err := retrying.Stream(context.Background(), validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Recv(); !errors.Is(err, transportErr) {
		t.Fatalf("Stream Recv() = %v", err)
	}
	if generator.streamCalls != 1 {
		t.Fatalf("Stream attempts = %d, want 1", generator.streamCalls)
	}
	_ = stream.Close()

	generator = &scriptedGenerator{generate: func(_ context.Context, _ Request, call int) (*Response, error) {
		if call == 1 {
			return nil, transportErr
		}
		return &Response{Message: Message{Role: RoleAssistant}}, nil
	}}
	retrying, _ = WithRetry(generator, RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Nanosecond, MaxBackoff: time.Nanosecond, ShouldRetry: func(err error) bool { return errors.Is(err, transportErr) }})
	if _, err := retrying.Generate(context.Background(), validGenerationRequest()); err != nil || generator.generateCalls != 2 {
		t.Fatalf("explicit transport retry = %v, calls %d", err, generator.generateCalls)
	}
}

func TestRetryGenerateDoesNotRetryUnsafeFailure(t *testing.T) {
	generator := &scriptedGenerator{generate: func(context.Context, Request, int) (*Response, error) {
		return nil, &Error{Kind: KindMalformedResponse, Err: errors.New("bad response")}
	}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := retrying.Generate(context.Background(), validGenerationRequest()); err == nil {
		t.Fatal("Generate() succeeded")
	}
	if generator.generateCalls != 1 {
		t.Fatalf("unsafe attempts = %d, want 1", generator.generateCalls)
	}
}

func TestRetryGenerateCancellationDuringDelay(t *testing.T) {
	started := make(chan struct{}, 1)
	providerErr := &Error{Kind: KindRateLimit, RetryAfter: 30 * time.Minute, Err: errors.New("wait")}
	generator := &scriptedGenerator{generate: func(context.Context, Request, int) (*Response, error) {
		started <- struct{}{}
		return nil, providerErr
	}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 3, InitialBackoff: time.Hour, MaxBackoff: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { _, err := retrying.Generate(ctx, validGenerationRequest()); done <- err }()
	<-started
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Generate() error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Generate did not stop during retry delay")
	}
	if generator.generateCalls != 1 {
		t.Fatalf("attempts = %d, want 1", generator.generateCalls)
	}
}

func TestRetryCancellationPreservesCustomCauseAndContextKind(t *testing.T) {
	cause := errors.New("caller stopped")
	generator := &scriptedGenerator{generate: func(context.Context, Request, int) (*Response, error) { return nil, retryableTestError() }}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Hour, MaxBackoff: time.Hour})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancelCause(context.Background())
	done := make(chan error, 1)
	go func() { _, err := retrying.Generate(ctx, validGenerationRequest()); done <- err }()
	for {
		generator.mu.Lock()
		calls := generator.generateCalls
		generator.mu.Unlock()
		if calls != 0 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	cancel(cause)
	err = <-done
	if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
		t.Fatalf("Generate() cancellation = %v", err)
	}
}

func TestRetryStreamCanceledBeforeCloseWinsPendingRecvRace(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	underlying := &blockingCloseStream{
		recvStarted: make(chan struct{}), closeStarted: make(chan struct{}),
		releaseClose: make(chan struct{}), releaseRecv: make(chan struct{}), recvErr: context.Canceled,
	}
	generator := &scriptedGenerator{streams: []func() Stream{func() Stream { return underlying }}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 1})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := retrying.Stream(ctx, validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	recvDone := make(chan error, 1)
	go func() { _, err := stream.Recv(); recvDone <- err }()
	<-underlying.recvStarted
	cancel()
	closeDone := make(chan error, 1)
	go func() { closeDone <- stream.Close() }()
	<-underlying.closeStarted
	select {
	case err := <-recvDone:
		t.Fatalf("Recv returned before Close established terminal: %v", err)
	default:
	}
	close(underlying.releaseClose)
	if err := <-closeDone; err != nil {
		t.Fatalf("Close() = %v", err)
	}
	if err := <-recvDone; !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv() = %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("repeated Close() = %v", err)
	}
	if _, err := stream.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("sticky Recv() = %v", err)
	}
}

func TestRetryStreamProviderTerminalRacingCloseWins(t *testing.T) {
	providerErr := &Error{Kind: KindTransport, Err: errors.New("provider failed")}
	underlying := &blockingCloseStream{
		recvStarted: make(chan struct{}), closeStarted: make(chan struct{}),
		releaseClose: make(chan struct{}), releaseRecv: make(chan struct{}), recvErr: providerErr,
	}
	generator := &scriptedGenerator{streams: []func() Stream{func() Stream { return underlying }}}
	retrying, _ := WithRetry(generator, RetryPolicy{MaxAttempts: 1})
	stream, _ := retrying.Stream(context.Background(), validGenerationRequest())
	recvDone := make(chan error, 1)
	go func() { _, err := stream.Recv(); recvDone <- err }()
	<-underlying.recvStarted
	closeDone := make(chan error, 1)
	go func() { closeDone <- stream.Close() }()
	<-underlying.closeStarted
	close(underlying.releaseClose)
	_ = <-closeDone
	if err := <-recvDone; !errors.Is(err, providerErr) {
		t.Fatalf("Recv() = %v", err)
	}
}

func TestRetryStreamClosePreservesExistingContextError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	underlying := &scriptedStream{errs: []error{context.Canceled}}
	generator := &scriptedGenerator{streams: []func() Stream{func() Stream { return underlying }}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 1})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := retrying.Stream(ctx, validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	_ = stream.Close()
	if _, err := stream.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv() after canceled Close = %v", err)
	}
}

func TestRetryStreamCreationAndProviderDelayHint(t *testing.T) {
	opened := &scriptedStream{chunks: []Chunk{{Content: []Part{{Text: "ok"}}}}}
	generator := &scriptedGenerator{streamErrors: []error{retryableTestError(), nil}, streams: []func() Stream{nil, func() Stream { return opened }}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Nanosecond, MaxBackoff: 10 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := retrying.Stream(context.Background(), validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Recv(); err != nil || generator.streamCalls != 2 {
		t.Fatalf("creation retry Recv error/calls = %v/%d", err, generator.streamCalls)
	}
	_ = stream.Close()
	implementation := retrying.(*retryGenerator)
	delay, err := implementation.delay(1, &Error{Kind: KindRateLimit, RetryAfter: 7 * time.Millisecond})
	if err != nil || delay != 7*time.Millisecond {
		t.Fatalf("provider RetryAfter delay = %v, %v", delay, err)
	}
}

func TestRetryAfterAboveBoundStopsWithoutEarlyRetry(t *testing.T) {
	providerErr := &Error{Kind: KindRateLimit, RetryAfter: time.Minute, Err: errors.New("slow down")}
	generator := &scriptedGenerator{generate: func(context.Context, Request, int) (*Response, error) { return nil, providerErr }}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 3, MaxBackoff: 2 * time.Second})
	if err != nil {
		t.Fatal(err)
	}
	_, err = retrying.Generate(context.Background(), validGenerationRequest())
	if !errors.Is(err, providerErr) || !strings.Contains(err.Error(), "exceeds configured maximum") || generator.generateCalls != 1 {
		t.Fatalf("Generate() = %v; calls %d", err, generator.generateCalls)
	}

	first := &scriptedStream{errs: []error{providerErr}}
	generator = &scriptedGenerator{streams: []func() Stream{func() Stream { return first }, func() Stream { return &scriptedStream{} }}}
	retrying, _ = WithRetry(generator, RetryPolicy{MaxAttempts: 3, MaxBackoff: 2 * time.Second})
	stream, err := retrying.Stream(context.Background(), validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	_, err = stream.Recv()
	if !errors.Is(err, providerErr) || !strings.Contains(err.Error(), "exceeds configured maximum") || generator.streamCalls != 1 || !first.closed {
		t.Fatalf("Stream Recv() = %v; calls/closed %d/%v", err, generator.streamCalls, first.closed)
	}
	_ = stream.Close()
}

func TestRetryStreamCancellationDuringRetryAfter(t *testing.T) {
	providerErr := &Error{Kind: KindRateLimit, RetryAfter: 30 * time.Minute, Err: errors.New("wait")}
	first := &scriptedStream{errs: []error{providerErr}}
	generator := &scriptedGenerator{streams: []func() Stream{func() Stream { return first }, func() Stream { return &scriptedStream{} }}}
	retrying, _ := WithRetry(generator, RetryPolicy{MaxAttempts: 2, MaxBackoff: time.Hour})
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := retrying.Stream(ctx, validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := stream.Recv(); done <- err }()
	for !first.isClosed() {
		time.Sleep(time.Millisecond)
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) || !errors.Is(err, providerErr) {
			t.Fatalf("Recv() = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Recv did not stop during RetryAfter delay")
	}
	_ = stream.Close()
}

func TestRetryStreamPreOutputAndNoRetryAfterCommit(t *testing.T) {
	first := &scriptedStream{errs: []error{retryableTestError()}}
	second := &scriptedStream{chunks: []Chunk{{Content: []Part{{Text: "ok"}}}}}
	generator := &scriptedGenerator{streams: []func() Stream{func() Stream { return first }, func() Stream { return second }}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Nanosecond, MaxBackoff: time.Nanosecond})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := retrying.Stream(context.Background(), validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	chunk, err := stream.Recv()
	if err != nil || chunk.Content[0].Text != "ok" {
		t.Fatalf("Recv() = %#v, %v", chunk, err)
	}
	if !first.closed || generator.streamCalls != 2 {
		t.Fatalf("abandoned closed/calls = %v/%d", first.closed, generator.streamCalls)
	}
	_ = stream.Close()

	committed := &scriptedStream{chunks: []Chunk{{Content: []Part{{Text: "first"}}}}, errs: []error{retryableTestError()}}
	generator = &scriptedGenerator{streams: []func() Stream{func() Stream { return committed }, func() Stream { return &scriptedStream{} }}}
	retrying, _ = WithRetry(generator, RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Nanosecond, MaxBackoff: time.Nanosecond})
	stream, _ = retrying.Stream(context.Background(), validGenerationRequest())
	if _, err := stream.Recv(); err != nil {
		t.Fatal(err)
	}
	if _, err := stream.Recv(); err == nil {
		t.Fatal("post-output Recv succeeded")
	}
	if generator.streamCalls != 1 {
		t.Fatalf("post-output attempts = %d, want 1", generator.streamCalls)
	}
	_ = stream.Close()
}

func TestRetryStreamCloseCancelsAttemptHook(t *testing.T) {
	first := &scriptedStream{errs: []error{retryableTestError()}}
	generator := &scriptedGenerator{streams: []func() Stream{func() Stream { return first }, func() Stream { return &scriptedStream{} }}}
	hookStarted := make(chan struct{})
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Nanosecond, MaxBackoff: time.Nanosecond, Hook: func(ctx context.Context, attempt Attempt) (http.Header, error) {
		if attempt.Number == 1 {
			return nil, nil
		}
		close(hookStarted)
		<-ctx.Done()
		return nil, contextError(ctx)
	}})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := retrying.Stream(context.Background(), validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	recvDone := make(chan error, 1)
	go func() { _, err := stream.Recv(); recvDone <- err }()
	<-hookStarted
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() = %v", err)
	}
	select {
	case err := <-recvDone:
		if !errors.Is(err, io.EOF) {
			t.Fatalf("Recv() = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Close did not unblock attempt hook")
	}
}

func TestRetryStopsWhenAbandonedStreamCloseFails(t *testing.T) {
	closeErr := errors.New("abandoned close")
	first := &scriptedStream{errs: []error{retryableTestError()}, closeErr: closeErr}
	generator := &scriptedGenerator{streams: []func() Stream{func() Stream { return first }, func() Stream { return &scriptedStream{} }}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Nanosecond, MaxBackoff: time.Nanosecond})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := retrying.Stream(context.Background(), validGenerationRequest())
	if err != nil {
		t.Fatal(err)
	}
	_, err = stream.Recv()
	if !errors.Is(err, closeErr) || generator.streamCalls != 1 || !first.closed {
		t.Fatalf("Recv() = %v; calls/closed = %d/%v", err, generator.streamCalls, first.closed)
	}
	_ = stream.Close()
}

func TestRetryAttemptHookHeaderIsolation(t *testing.T) {
	returned := http.Header{"X-Trace": {"original"}}
	var seen http.Header
	generator := &scriptedGenerator{generate: func(ctx context.Context, _ Request, _ int) (*Response, error) {
		seen = AttemptHeaders(ctx)
		seen.Set("X-Trace", "changed")
		return &Response{Message: Message{Role: RoleAssistant}}, nil
	}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 1, Hook: func(context.Context, Attempt) (http.Header, error) { return returned, nil }})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := retrying.Generate(context.Background(), validGenerationRequest()); err != nil {
		t.Fatal(err)
	}
	if returned.Get("X-Trace") != "original" {
		t.Fatalf("hook header mutated: %#v", returned)
	}
	if seen.Get("X-Trace") != "changed" {
		t.Fatalf("fake did not receive isolated header: %#v", seen)
	}

	hookCalls := 0
	validationFirst, _ := WithRetry(generator, RetryPolicy{MaxAttempts: 1, Hook: func(context.Context, Attempt) (http.Header, error) {
		hookCalls++
		return nil, nil
	}})
	if _, err := validationFirst.Generate(context.Background(), Request{}); err == nil || hookCalls != 0 {
		t.Fatalf("invalid request error/hook calls = %v/%d", err, hookCalls)
	}

	invalid, _ := WithRetry(generator, RetryPolicy{MaxAttempts: 1, Hook: func(context.Context, Attempt) (http.Header, error) {
		return http.Header{"Authorization": {"secret"}}, nil
	}})
	if _, err := invalid.Generate(context.Background(), validGenerationRequest()); err == nil || !strings.Contains(err.Error(), "reserved") {
		t.Fatalf("reserved header error = %v", err)
	}
	if generator.generateCalls != 1 {
		t.Fatalf("provider called for invalid header; calls = %d", generator.generateCalls)
	}
}

func TestRetryAttemptHookConcurrentIsolation(t *testing.T) {
	type traceKey struct{}
	const calls = 32
	seen := make(chan string, calls)
	generator := &scriptedGenerator{generate: func(ctx context.Context, _ Request, _ int) (*Response, error) {
		seen <- AttemptHeaders(ctx).Get("X-Trace")
		return &Response{Message: Message{Role: RoleAssistant}}, nil
	}}
	retrying, err := WithRetry(generator, RetryPolicy{MaxAttempts: 1, Hook: func(ctx context.Context, _ Attempt) (http.Header, error) {
		return http.Header{"X-Trace": {ctx.Value(traceKey{}).(string)}}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	var wait sync.WaitGroup
	for index := range calls {
		wait.Add(1)
		go func() {
			defer wait.Done()
			value := fmt.Sprintf("trace-%d", index)
			ctx := context.WithValue(context.Background(), traceKey{}, value)
			if _, err := retrying.Generate(ctx, validGenerationRequest()); err != nil {
				t.Errorf("Generate() error = %v", err)
			}
		}()
	}
	wait.Wait()
	close(seen)
	values := make(map[string]bool)
	for value := range seen {
		values[value] = true
	}
	if len(values) != calls {
		t.Fatalf("isolated header count = %d, want %d: %#v", len(values), calls, values)
	}
}
