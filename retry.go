package llm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// Attempt describes one call made by a retrying Generator. Number is one-based.
// Model and its nested storage are a fresh copy owned by the hook and may be
// retained or mutated without affecting the generator.
type Attempt struct {
	Operation   string
	Number      int
	MaxAttempts int
	Model       ModelInfo
}

// AttemptHook is called after Request validation and immediately before each
// provider attempt. It may observe the attempt and return request headers, but
// cannot inspect, replace, or mutate the validated Request payload. Hooks may be
// called concurrently and therefore must be safe for concurrent use. A hook
// must honor ctx cancellation and return promptly so Stream.Close can unblock
// retry opening attempts. Returned headers are borrowed only until the hook
// returns and are cloned before use.
type AttemptHook func(context.Context, Attempt) (http.Header, error)

// RetryPolicy configures an opt-in retrying Generator. MaxAttempts includes the
// first attempt and must be positive. Zero backoff fields use conservative
// defaults. MaxBackoff also bounds an acceptable provider RetryAfter; a larger
// requested minimum stops retrying instead of sleeping less than requested.
// Ambiguous transport failures are not retried by default because a POST may
// already have been accepted; use an explicit provider Retryable hint or
// ShouldRetry when the caller can accept that risk. Jitter is a fractional range
// from zero through one. ShouldRetry, if non-nil, replaces the default retry
// classification and must be concurrency-safe.
type RetryPolicy struct {
	MaxAttempts    int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Multiplier     float64
	Jitter         float64
	ShouldRetry    func(error) bool
	Hook           AttemptHook
}

type retryGenerator struct {
	generator Generator
	policy    RetryPolicy
}

// WithRetry wraps generator with an explicit bounded retry policy. Providers
// remain single-attempt when this wrapper is not used.
func WithRetry(generator Generator, policy RetryPolicy) (Generator, error) {
	if generator == nil {
		return nil, fmt.Errorf("retry generator must not be nil")
	}
	normalized, err := normalizeRetryPolicy(policy)
	if err != nil {
		return nil, err
	}
	retrying := &retryGenerator{generator: generator, policy: normalized}
	if background, ok := generator.(BackgroundGenerator); ok {
		return &retryBackgroundGenerator{retryGenerator: retrying, background: background}, nil
	}
	return retrying, nil
}

func normalizeRetryPolicy(policy RetryPolicy) (RetryPolicy, error) {
	if policy.MaxAttempts <= 0 {
		return RetryPolicy{}, fmt.Errorf("retry max attempts must be positive")
	}
	if policy.InitialBackoff < 0 || policy.MaxBackoff < 0 {
		return RetryPolicy{}, fmt.Errorf("retry backoff must not be negative")
	}
	if policy.InitialBackoff == 0 {
		policy.InitialBackoff = 100 * time.Millisecond
	}
	if policy.MaxBackoff == 0 {
		policy.MaxBackoff = 2 * time.Second
	}
	if policy.MaxBackoff < policy.InitialBackoff {
		return RetryPolicy{}, fmt.Errorf("retry max backoff must not be less than initial backoff")
	}
	if policy.Multiplier == 0 {
		policy.Multiplier = 2
	}
	if math.IsNaN(policy.Multiplier) || math.IsInf(policy.Multiplier, 0) || policy.Multiplier < 1 {
		return RetryPolicy{}, fmt.Errorf("retry multiplier must be finite and at least one")
	}
	if math.IsNaN(policy.Jitter) || math.IsInf(policy.Jitter, 0) || policy.Jitter < 0 || policy.Jitter > 1 {
		return RetryPolicy{}, fmt.Errorf("retry jitter must be between zero and one")
	}
	return policy, nil
}

func (g *retryGenerator) Info() ModelInfo { return cloneInfo(g.generator.Info()) }

func (g *retryGenerator) Generate(ctx context.Context, request Request) (*Response, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	var last error
	for attempt := 1; attempt <= g.policy.MaxAttempts; attempt++ {
		attemptCtx, err := g.attemptContext(ctx, "generate", attempt)
		if err != nil {
			return nil, err
		}
		response, err := g.generator.Generate(attemptCtx, request)
		if err == nil {
			return response, nil
		}
		last = err
		if response != nil || attempt == g.policy.MaxAttempts || !g.retryable(err) {
			return response, err
		}
		delay, delayErr := g.delay(attempt, err)
		if delayErr != nil {
			return nil, delayErr
		}
		if err := waitRetry(ctx, delay); err != nil {
			return nil, errors.Join(err, last)
		}
	}
	return nil, last
}

func (g *retryGenerator) Stream(ctx context.Context, request Request) (Stream, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	opctx, cancel := context.WithCancel(ctx)
	var last error
	for attempt := 1; attempt <= g.policy.MaxAttempts; attempt++ {
		attemptCtx, err := g.attemptContext(opctx, "stream", attempt)
		if err != nil {
			cancel()
			return nil, err
		}
		stream, err := g.generator.Stream(attemptCtx, request)
		if err == nil {
			return &retryStream{generator: g, request: request, ctx: opctx, cancel: cancel, current: stream, attempts: attempt, closeCh: make(chan struct{}), closeDone: make(chan struct{})}, nil
		}
		last = err
		if attempt == g.policy.MaxAttempts || !g.retryable(err) {
			cancel()
			return nil, err
		}
		delay, delayErr := g.delay(attempt, err)
		if delayErr != nil {
			cancel()
			return nil, delayErr
		}
		if err := waitRetry(opctx, delay); err != nil {
			cancel()
			return nil, errors.Join(err, last)
		}
	}
	cancel()
	return nil, last
}

type retryBackgroundGenerator struct {
	*retryGenerator
	background BackgroundGenerator
}

func (g *retryBackgroundGenerator) StartBackground(ctx context.Context, request Request) (*BackgroundResult, error) {
	if err := request.Validate(); err != nil {
		return nil, err
	}
	return g.doBackground(ctx, "start_background", func(attemptCtx context.Context) (*BackgroundResult, error) {
		return g.background.StartBackground(attemptCtx, request)
	})
}

func (g *retryBackgroundGenerator) FetchBackground(ctx context.Context, handle BackgroundHandle) (*BackgroundResult, error) {
	if err := g.validateBackgroundHandle("fetch_background", handle); err != nil {
		return nil, err
	}
	return g.doBackground(ctx, "fetch_background", func(attemptCtx context.Context) (*BackgroundResult, error) {
		return g.background.FetchBackground(attemptCtx, handle)
	})
}

func (g *retryBackgroundGenerator) CancelBackground(ctx context.Context, handle BackgroundHandle) (*BackgroundResult, error) {
	if err := g.validateBackgroundHandle("cancel_background", handle); err != nil {
		return nil, err
	}
	return g.doBackground(ctx, "cancel_background", func(attemptCtx context.Context) (*BackgroundResult, error) {
		return g.background.CancelBackground(attemptCtx, handle)
	})
}

func (g *retryBackgroundGenerator) validateBackgroundHandle(operation string, handle BackgroundHandle) error {
	info := g.generator.Info()
	if err := handle.validate(); err != nil {
		return &Error{Kind: KindInvalidRequest, Op: operation, Provider: info.Provider, Err: err}
	}
	if handle.Provider != info.Provider || handle.Model != info.Model || handle.API != info.API {
		return &Error{Kind: KindInvalidRequest, Op: operation, Provider: info.Provider,
			Err: fmt.Errorf("background handle belongs to provider/model/API %q/%q/%q", handle.Provider, handle.Model, handle.API)}
	}
	return nil
}

func (g *retryBackgroundGenerator) doBackground(ctx context.Context, operation string, call func(context.Context) (*BackgroundResult, error)) (*BackgroundResult, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	var last error
	for attempt := 1; attempt <= g.policy.MaxAttempts; attempt++ {
		attemptCtx, err := g.attemptContext(ctx, operation, attempt)
		if err != nil {
			return nil, err
		}
		result, err := call(attemptCtx)
		if err == nil {
			return result, nil
		}
		last = err
		if result != nil || attempt == g.policy.MaxAttempts || !g.retryable(err) {
			return result, err
		}
		delay, delayErr := g.delay(attempt, err)
		if delayErr != nil {
			return nil, delayErr
		}
		if err := waitRetry(ctx, delay); err != nil {
			return nil, errors.Join(err, last)
		}
	}
	return nil, last
}

func (g *retryGenerator) attemptContext(ctx context.Context, operation string, number int) (context.Context, error) {
	if g.policy.Hook == nil {
		return ctx, nil
	}
	headers, err := g.policy.Hook(ctx, Attempt{Operation: operation, Number: number, MaxAttempts: g.policy.MaxAttempts, Model: cloneInfo(g.generator.Info())})
	if err != nil {
		return nil, err
	}
	headers, err = validateAttemptHeaders(headers)
	if err != nil {
		return nil, &Error{Kind: KindInvalidRequest, Op: operation, Err: fmt.Errorf("attempt headers: %w", err)}
	}
	return withAttemptHeaders(ctx, headers), nil
}

func validateAttemptHeaders(headers http.Header) (http.Header, error) {
	if len(headers) == 0 {
		return nil, nil
	}
	result := make(http.Header, len(headers))
	for name, values := range headers {
		if !validHeaderName(name) {
			return nil, fmt.Errorf("header name %q is invalid", name)
		}
		lower := strings.ToLower(name)
		switch lower {
		case "authorization", "proxy-authorization", "x-api-key", "api-key", "x-goog-api-key", "google-api-key", "x-goog-user-project",
			"cookie", "set-cookie", "host", "content-type", "content-length", "accept", "user-agent", "transfer-encoding", "connection", "upgrade",
			"anthropic-version", "anthropic-beta", "openai-beta", "openai-organization", "openai-project",
			"chatgpt-account-id", "cf-aig-authorization", "originator", "x-app", "idempotency-key",
			"session_id", "session-id", "x-session-id", "x-client-request-id", "x-session-affinity":
			return nil, fmt.Errorf("header %q is reserved", name)
		}
		for _, value := range values {
			if !validHeaderValue(value) {
				return nil, fmt.Errorf("header %q contains an invalid value", name)
			}
			result.Add(name, value)
		}
	}
	return result, nil
}

func validHeaderName(name string) bool {
	if name == "" {
		return false
	}
	for i := range len(name) {
		c := name[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || strings.ContainsRune("!#$%&'*+-.^_`|~", rune(c)) {
			continue
		}
		return false
	}
	return true
}

func validHeaderValue(value string) bool {
	if !utf8.ValidString(value) {
		return false
	}
	for i := range len(value) {
		if value[i] == '\t' || value[i] >= 0x20 && value[i] != 0x7f {
			continue
		}
		return false
	}
	return true
}

func (g *retryGenerator) retryable(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if g.policy.ShouldRetry != nil {
		return g.policy.ShouldRetry(err)
	}
	var modelErr *Error
	if !errors.As(err, &modelErr) {
		return false
	}
	if modelErr.Retryable != nil {
		return *modelErr.Retryable
	}
	if modelErr.RetryAfter > 0 || modelErr.Kind == KindRateLimit {
		return true
	}
	switch modelErr.HTTPStatus {
	case 408, 425, 429, 500, 502, 503, 504:
		return true
	default:
		return false
	}
}

func (g *retryGenerator) delay(failedAttempt int, err error) (time.Duration, error) {
	delay := float64(g.policy.InitialBackoff)
	for i := 1; i < failedAttempt && delay < float64(g.policy.MaxBackoff); i++ {
		delay *= g.policy.Multiplier
	}
	if delay > float64(g.policy.MaxBackoff) || math.IsInf(delay, 0) {
		delay = float64(g.policy.MaxBackoff)
	}
	if g.policy.Jitter != 0 {
		factor := 1 - g.policy.Jitter + 2*g.policy.Jitter*rand.Float64()
		delay *= factor
		if delay > float64(g.policy.MaxBackoff) {
			delay = float64(g.policy.MaxBackoff)
		}
	}
	var modelErr *Error
	if errors.As(err, &modelErr) && modelErr.RetryAfter > 0 {
		if modelErr.RetryAfter > g.policy.MaxBackoff {
			return 0, errors.Join(err, fmt.Errorf("provider requested retry delay %s exceeds configured maximum %s", modelErr.RetryAfter, g.policy.MaxBackoff))
		}
		if modelErr.RetryAfter > time.Duration(delay) {
			delay = float64(modelErr.RetryAfter)
		}
	}
	return time.Duration(delay), nil
}

func waitRetry(ctx context.Context, delay time.Duration) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return contextError(ctx)
	}
}

type retryStream struct {
	generator *retryGenerator
	request   Request
	ctx       context.Context
	cancel    context.CancelFunc
	closeCh   chan struct{}
	closeDone chan struct{}

	mu        sync.Mutex
	current   Stream
	attempts  int
	committed bool
	closed    bool
	terminal  error
	recvDone  chan struct{}
	closeErr  error
	closeOnce sync.Once
}

func (s *retryStream) Recv() (Chunk, error) {
	for {
		s.mu.Lock()
		if s.terminal != nil {
			err := s.terminal
			s.mu.Unlock()
			return Chunk{}, err
		}
		current := s.current
		s.recvDone = make(chan struct{})
		s.mu.Unlock()

		chunk, err := current.Recv()
		s.mu.Lock()
		if err == nil {
			s.committed = true
			s.finishRecvLocked()
			s.mu.Unlock()
			return chunk, nil
		}
		if s.closed {
			if !errors.Is(err, io.EOF) && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) && s.terminal == nil {
				s.terminal = err
			}
			s.finishRecvLocked()
			s.mu.Unlock()
			return Chunk{}, s.waitForClose(nil)
		}
		committed := s.committed
		attempt := s.attempts
		if errors.Is(err, io.EOF) || committed || attempt >= s.generator.policy.MaxAttempts || !s.generator.retryable(err) {
			if s.terminal == nil {
				s.terminal = err
			}
			terminal := s.terminal
			s.finishRecvLocked()
			s.mu.Unlock()
			return Chunk{}, terminal
		}
		s.finishRecvLocked()
		s.mu.Unlock()
		if closeErr := current.Close(); closeErr != nil {
			joined := errors.Join(err, closeErr)
			s.setTerminal(joined)
			return Chunk{}, joined
		}
		delay, delayErr := s.generator.delay(attempt, err)
		if delayErr != nil {
			s.setTerminal(delayErr)
			return Chunk{}, delayErr
		}
		if waitErr := s.wait(delay); waitErr != nil {
			if s.isClosed() {
				return Chunk{}, s.waitForClose(nil)
			}
			joined := errors.Join(waitErr, err)
			s.setTerminal(joined)
			return Chunk{}, joined
		}

		for {
			attempt++
			attemptCtx, hookErr := s.generator.attemptContext(s.ctx, "stream", attempt)
			if hookErr != nil {
				if s.isClosed() {
					return Chunk{}, s.waitForClose(hookErr)
				}
				s.setTerminal(hookErr)
				return Chunk{}, hookErr
			}
			next, openErr := s.generator.generator.Stream(attemptCtx, s.request)
			if openErr == nil {
				s.mu.Lock()
				if s.closed {
					s.mu.Unlock()
					_ = next.Close()
					return Chunk{}, s.waitForClose(nil)
				}
				s.current = next
				s.attempts = attempt
				s.mu.Unlock()
				break
			}
			if s.isClosed() {
				return Chunk{}, s.waitForClose(openErr)
			}
			if attempt >= s.generator.policy.MaxAttempts || !s.generator.retryable(openErr) {
				s.setTerminal(openErr)
				return Chunk{}, openErr
			}
			delay, delayErr := s.generator.delay(attempt, openErr)
			if delayErr != nil {
				s.setTerminal(delayErr)
				return Chunk{}, delayErr
			}
			if waitErr := s.wait(delay); waitErr != nil {
				if s.isClosed() {
					return Chunk{}, s.waitForClose(nil)
				}
				joined := errors.Join(waitErr, openErr)
				s.setTerminal(joined)
				return Chunk{}, joined
			}
		}
	}
}

func (s *retryStream) wait(delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-s.ctx.Done():
		return contextError(s.ctx)
	case <-s.closeCh:
		return io.EOF
	}
}

func (s *retryStream) Close() error {
	s.closeOnce.Do(func() {
		contextErr := contextError(s.ctx)
		s.mu.Lock()
		s.closed = true
		if contextErr != nil && s.terminal == nil {
			s.terminal = contextErr
		}
		current := s.current
		recvDone := s.recvDone
		close(s.closeCh)
		s.cancel()
		s.mu.Unlock()
		if current != nil {
			s.closeErr = current.Close()
		}
		if recvDone != nil {
			<-recvDone
		}
		s.mu.Lock()
		if s.terminal == nil {
			s.terminal = io.EOF
		}
		s.mu.Unlock()
		close(s.closeDone)
	})
	return s.closeErr
}

func (s *retryStream) finishRecvLocked() {
	if s.recvDone != nil {
		close(s.recvDone)
		s.recvDone = nil
	}
}

func (s *retryStream) waitForClose(candidate error) error {
	if candidate != nil && !errors.Is(candidate, io.EOF) && !errors.Is(candidate, context.Canceled) && !errors.Is(candidate, context.DeadlineExceeded) {
		s.setTerminal(candidate)
	}
	<-s.closeDone
	s.mu.Lock()
	terminal := s.terminal
	s.mu.Unlock()
	return terminal
}

func (s *retryStream) setTerminal(err error) {
	s.mu.Lock()
	if s.terminal == nil {
		s.terminal = err
	}
	s.mu.Unlock()
}

func (s *retryStream) isClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed
}

func contextError(ctx context.Context) error {
	err := ctx.Err()
	if err == nil {
		return nil
	}
	cause := context.Cause(ctx)
	if cause == nil || errors.Is(cause, err) {
		if cause != nil {
			return cause
		}
		return err
	}
	return errors.Join(err, cause)
}

func cloneInfo(info ModelInfo) ModelInfo {
	info.Capabilities = append([]Capability(nil), info.Capabilities...)
	info.Cost = cloneModelCost(info.Cost)
	info.Compatibility = cloneModelCompatibility(info.Compatibility)
	return info
}

var (
	_ Generator           = (*retryGenerator)(nil)
	_ BackgroundGenerator = (*retryBackgroundGenerator)(nil)
)
var _ Stream = (*retryStream)(nil)
