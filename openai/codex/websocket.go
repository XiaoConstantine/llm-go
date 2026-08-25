package codex

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	internalresponses "github.com/XiaoConstantine/llm-go/internal/openairesponses"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	"github.com/gorilla/websocket"
	openairesponses "github.com/openai/openai-go/v3/responses"
)

const webSocketBeta = "responses_websockets=2026-02-06"

type webSocketCacheKey struct {
	session    string
	account    string
	credential [32]byte
}

type webSocketContinuation struct {
	request       webSocketRequestBody
	responseID    string
	responseItems []jsontext.Value
}

type cachedWebSocket struct {
	conn         *websocket.Conn
	busy         bool
	createdAt    time.Time
	sequence     uint64
	idleTimer    *time.Timer
	continuation *webSocketContinuation
}

type webSocketCache struct {
	mu           sync.Mutex
	closed       bool
	entries      map[webSocketCacheKey]*cachedWebSocket
	connections  map[*websocket.Conn]struct{}
	nextSequence uint64
}

func newWebSocketCache() webSocketCache {
	return webSocketCache{entries: make(map[webSocketCacheKey]*cachedWebSocket), connections: make(map[*websocket.Conn]struct{})}
}

// Close closes all cached Codex WebSocket sessions. It is idempotent and safe
// to call concurrently. A closed Client cannot start new WebSocket requests;
// its SSE transport remains unaffected.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	c.webSockets.mu.Lock()
	if c.webSockets.closed {
		c.webSockets.mu.Unlock()
		return nil
	}
	c.webSockets.closed = true
	connections := make([]*websocket.Conn, 0, len(c.webSockets.connections))
	for key, entry := range c.webSockets.entries {
		delete(c.webSockets.entries, key)
		if entry.idleTimer != nil {
			entry.idleTimer.Stop()
			entry.idleTimer = nil
		}
	}
	for conn := range c.webSockets.connections {
		delete(c.webSockets.connections, conn)
		connections = append(connections, conn)
	}
	c.webSockets.mu.Unlock()
	for _, conn := range connections {
		_ = conn.Close()
	}
	return nil
}

type webSocketLease struct {
	client    *Client
	key       webSocketCacheKey
	conn      *websocket.Conn
	entry     *cachedWebSocket
	transient bool
	released  bool
}

func (l *webSocketLease) release(keep bool, continuation *webSocketContinuation) {
	if l == nil || l.released {
		return
	}
	l.released = true
	if l.transient || l.entry == nil {
		l.client.closeWebSocket(l.conn)
		return
	}
	cache := &l.client.webSockets
	cache.mu.Lock()
	current := cache.entries[l.key]
	if current != l.entry || cache.closed || !keep {
		if current == l.entry {
			delete(cache.entries, l.key)
		}
		if l.entry.idleTimer != nil {
			l.entry.idleTimer.Stop()
			l.entry.idleTimer = nil
		}
		cache.mu.Unlock()
		l.client.closeWebSocket(l.conn)
		return
	}
	l.entry.continuation = cloneWebSocketContinuation(continuation)
	l.entry.busy = false
	remainingAge := l.client.webSocketMaxAge - time.Since(l.entry.createdAt)
	delay := l.client.webSocketIdleTime
	if remainingAge < delay {
		delay = remainingAge
	}
	if delay <= 0 {
		delete(cache.entries, l.key)
		cache.mu.Unlock()
		l.client.closeWebSocket(l.conn)
		return
	}
	entry := l.entry
	key := l.key
	entry.idleTimer = time.AfterFunc(delay, func() {
		cache.mu.Lock()
		if cache.entries[key] != entry || entry.busy {
			cache.mu.Unlock()
			return
		}
		delete(cache.entries, key)
		entry.idleTimer = nil
		cache.mu.Unlock()
		l.client.closeWebSocket(entry.conn)
	})
	cache.mu.Unlock()
}

func cloneWebSocketContinuation(value *webSocketContinuation) *webSocketContinuation {
	if value == nil {
		return nil
	}
	return &webSocketContinuation{
		request:       value.request.clone(),
		responseID:    value.responseID,
		responseItems: cloneJSONValues(value.responseItems),
	}
}

func (c *Client) registerWebSocket(conn *websocket.Conn) bool {
	c.webSockets.mu.Lock()
	defer c.webSockets.mu.Unlock()
	if c.webSockets.closed {
		return false
	}
	c.webSockets.connections[conn] = struct{}{}
	return true
}

func (c *Client) closeWebSocket(conn *websocket.Conn) {
	if conn == nil {
		return
	}
	c.webSockets.mu.Lock()
	delete(c.webSockets.connections, conn)
	c.webSockets.mu.Unlock()
	_ = conn.Close()
}

func (c *Client) webSocketClosed() bool {
	c.webSockets.mu.Lock()
	defer c.webSockets.mu.Unlock()
	return c.webSockets.closed
}

func credentialCacheKey(session string, credentials Credentials) webSocketCacheKey {
	return webSocketCacheKey{session: session, account: credentials.AccountID,
		credential: sha256.Sum256([]byte(credentials.AccessToken))}
}

type webSocketConnectFailure struct{ err error }

func (e *webSocketConnectFailure) Error() string { return e.err.Error() }
func (e *webSocketConnectFailure) Unwrap() error { return e.err }

func (c *Client) acquireWebSocket(ctx context.Context, op, sessionID string) (*webSocketLease, Credentials, error) {
	if c.webSocketClosed() {
		return nil, Credentials{}, transportError(op, errors.New("Codex WebSocket client is closed"))
	}
	credentials, err := c.currentCredentials(ctx, op, "", nil)
	if err != nil {
		return nil, Credentials{}, err
	}
	key := credentialCacheKey(sessionID, credentials)
	attemptScoped := len(llm.AttemptHeaders(ctx)) != 0
	if sessionID != "" && !attemptScoped {
		if lease := c.takeCachedWebSocket(key); lease != nil {
			return lease, credentials, nil
		}
	}

	conn, resolved, err := c.connectWebSocket(ctx, op, sessionID, credentials)
	if err != nil {
		return nil, Credentials{}, err
	}
	if !c.registerWebSocket(conn) {
		_ = conn.Close()
		return nil, Credentials{}, transportError(op, errors.New("Codex WebSocket client is closed"))
	}
	key = credentialCacheKey(sessionID, resolved)
	if sessionID == "" || attemptScoped {
		return &webSocketLease{client: c, key: key, conn: conn, transient: true}, resolved, nil
	}

	entry := &cachedWebSocket{conn: conn, busy: true, createdAt: time.Now()}
	c.webSockets.mu.Lock()
	if c.webSockets.closed {
		c.webSockets.mu.Unlock()
		c.closeWebSocket(conn)
		return nil, Credentials{}, transportError(op, errors.New("Codex WebSocket client is closed"))
	}
	if current := c.webSockets.entries[key]; current != nil {
		// Another request won the connection race. Keep its session connection and
		// use this newly authenticated socket transiently to avoid frame interleaving.
		c.webSockets.mu.Unlock()
		return &webSocketLease{client: c, key: key, conn: conn, transient: true}, resolved, nil
	}
	var evicted *cachedWebSocket
	if len(c.webSockets.entries) >= c.webSocketMaxSessions {
		for _, candidate := range c.webSockets.entries {
			if candidate.busy || evicted != nil && candidate.sequence >= evicted.sequence {
				continue
			}
			evicted = candidate
		}
		if evicted == nil {
			// Capacity is fully occupied by active requests. The new authenticated
			// connection remains transient rather than disrupting active work.
			c.webSockets.mu.Unlock()
			return &webSocketLease{client: c, key: key, conn: conn, transient: true}, resolved, nil
		}
		for evictedKey, candidate := range c.webSockets.entries {
			if candidate == evicted {
				delete(c.webSockets.entries, evictedKey)
				break
			}
		}
		if evicted.idleTimer != nil {
			evicted.idleTimer.Stop()
			evicted.idleTimer = nil
		}
	}
	c.webSockets.nextSequence++
	entry.sequence = c.webSockets.nextSequence
	c.webSockets.entries[key] = entry
	c.webSockets.mu.Unlock()
	if evicted != nil {
		c.closeWebSocket(evicted.conn)
	}
	return &webSocketLease{client: c, key: key, conn: conn, entry: entry}, resolved, nil
}

func (c *Client) takeCachedWebSocket(key webSocketCacheKey) *webSocketLease {
	c.webSockets.mu.Lock()
	if c.webSockets.closed {
		c.webSockets.mu.Unlock()
		return nil
	}
	entry := c.webSockets.entries[key]
	if entry == nil || entry.busy {
		c.webSockets.mu.Unlock()
		return nil
	}
	if time.Since(entry.createdAt) >= c.webSocketMaxAge {
		delete(c.webSockets.entries, key)
		if entry.idleTimer != nil {
			entry.idleTimer.Stop()
			entry.idleTimer = nil
		}
		c.webSockets.mu.Unlock()
		c.closeWebSocket(entry.conn)
		return nil
	}
	if entry.idleTimer != nil {
		entry.idleTimer.Stop()
		entry.idleTimer = nil
	}
	entry.busy = true
	c.webSockets.mu.Unlock()
	return &webSocketLease{client: c, key: key, conn: entry.conn, entry: entry}
}

func (c *Client) currentCredentials(ctx context.Context, op, rejected string, rejectedErr error) (Credentials, error) {
	credentials, err := c.resolveCredentials(ctx, rejected)
	if err != nil {
		if contextErr := contextErr(ctx); contextErr != nil {
			return Credentials{}, contextErr
		}
		resolvedErr := authenticationError(op, fmt.Errorf("resolve credentials: %w", err))
		if rejectedErr != nil {
			return Credentials{}, errors.Join(rejectedErr, resolvedErr)
		}
		return Credentials{}, resolvedErr
	}
	credentials, err = normalizeCredentials(credentials)
	if err != nil {
		resolvedErr := authenticationError(op, err)
		if rejectedErr != nil {
			return Credentials{}, errors.Join(rejectedErr, resolvedErr)
		}
		return Credentials{}, resolvedErr
	}
	if rejected != "" && credentials.AccessToken == rejected {
		return Credentials{}, rejectedErr
	}
	return credentials, nil
}

func (c *Client) connectWebSocket(ctx context.Context, op, sessionID string, initial Credentials) (*websocket.Conn, Credentials, error) {
	credentials := initial
	var rejectedErr error
	for attempt := 0; attempt < 2; attempt++ {
		headers, err := c.webSocketHeaders(ctx, credentials, sessionID)
		if err != nil {
			return nil, Credentials{}, transportError(op, err)
		}
		dialCtx := ctx
		cancel := func() {}
		if c.webSocketConnectTimeout > 0 {
			dialCtx, cancel = context.WithTimeout(ctx, c.webSocketConnectTimeout)
		}
		conn, response, dialErr := c.webSocketDialer.DialContext(dialCtx, c.webSocketURL, headers)
		cancel()
		if dialErr == nil {
			conn.SetReadLimit(maxProviderDataBytes + 1)
			return conn, credentials, nil
		}
		if contextErr := contextErr(ctx); contextErr != nil {
			if response != nil && response.Body != nil {
				_ = response.Body.Close()
			}
			return nil, Credentials{}, contextErr
		}
		if response == nil {
			return nil, Credentials{}, &webSocketConnectFailure{err: transportError(op, fmt.Errorf("connect WebSocket: %w", dialErr))}
		}
		var body []byte
		var tooLarge bool
		if response.Body != nil {
			var readErr error
			body, tooLarge, readErr = readLimited(response.Body, maxErrorBodyBytes)
			closeErr := response.Body.Close()
			if readErr != nil {
				return nil, Credentials{}, transportError(op, fmt.Errorf("read WebSocket handshake error: %w", readErr))
			}
			if closeErr != nil {
				return nil, Credentials{}, transportError(op, fmt.Errorf("close WebSocket handshake response: %w", closeErr))
			}
		}
		requestErr := responseErrorBytes(op, response, body, tooLarge)
		if response.StatusCode != http.StatusUnauthorized || attempt != 0 {
			return nil, Credentials{}, requestErr
		}
		rejectedErr = requestErr
		rejected := credentials.AccessToken
		credentials, err = c.currentCredentials(ctx, op, rejected, rejectedErr)
		if err != nil {
			return nil, Credentials{}, err
		}
	}
	return nil, Credentials{}, authenticationError(op, errors.New("credentials were rejected"))
}

func responseErrorBytes(op string, response *http.Response, body []byte, tooLarge bool) *llm.Error {
	clone := *response
	clone.Body = io.NopCloser(bytes.NewReader(body))
	return responseError(op, &clone, body, tooLarge)
}

func (c *Client) webSocketHeaders(ctx context.Context, credentials Credentials, sessionID string) (http.Header, error) {
	headers := cloneHeader(c.headers)
	for name, values := range llm.AttemptHeaders(ctx) {
		headers.Del(name)
		for _, value := range values {
			headers.Add(name, value)
		}
	}
	for key := range headers {
		canonical := http.CanonicalHeaderKey(key)
		if strings.HasPrefix(canonical, "Sec-Websocket-") || canonical == "Connection" || canonical == "Upgrade" {
			delete(headers, key)
		}
	}
	headers.Set("Authorization", "Bearer "+credentials.AccessToken)
	headers.Set("ChatGPT-Account-ID", credentials.AccountID)
	headers.Set("OpenAI-Beta", webSocketBeta)
	headers.Set("Originator", c.originator)
	headers.Set("User-Agent", "llm-go")
	headers.Set("Content-Type", "application/json")
	if sessionID != "" {
		headers.Set("Session-Id", sessionID)
		headers.Set("X-Client-Request-Id", sessionID)
	} else {
		requestID, err := randomRequestID()
		if err != nil {
			return nil, fmt.Errorf("create WebSocket request ID: %w", err)
		}
		headers.Del("Session-Id")
		headers.Set("X-Client-Request-Id", requestID)
	}
	return headers, nil
}

func randomRequestID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}

type webSocketRequestBody struct {
	fields map[string]jsontext.Value
	input  []jsontext.Value
}

func requestBody(params openairesponses.ResponseNewParams) (webSocketRequestBody, error) {
	raw, err := jsonv2.Marshal(params)
	if err != nil {
		return webSocketRequestBody{}, err
	}
	fields := make(map[string]jsontext.Value)
	if err := jsonv2.Unmarshal(raw, &fields); err != nil {
		return webSocketRequestBody{}, err
	}
	fields["store"] = jsontext.Value("false")
	fields["stream"] = jsontext.Value("true")
	var input []jsontext.Value
	if rawInput := fields["input"]; len(rawInput) != 0 {
		if err := jsonv2.Unmarshal(rawInput, &input); err != nil {
			return webSocketRequestBody{}, fmt.Errorf("decode canonical input: %w", err)
		}
	}
	return webSocketRequestBody{fields: cloneJSONMap(fields), input: cloneJSONValues(input)}, nil
}

func (b webSocketRequestBody) clone() webSocketRequestBody {
	return webSocketRequestBody{fields: cloneJSONMap(b.fields), input: cloneJSONValues(b.input)}
}

func cloneJSONMap(input map[string]jsontext.Value) map[string]jsontext.Value {
	result := make(map[string]jsontext.Value, len(input))
	for key, value := range input {
		result[key] = append(jsontext.Value(nil), value...)
	}
	return result
}

func cloneJSONValues(input []jsontext.Value) []jsontext.Value {
	result := make([]jsontext.Value, len(input))
	for index, value := range input {
		result[index] = append(jsontext.Value(nil), value...)
	}
	return result
}

func (b webSocketRequestBody) responseCreate(previousResponseID string, input []jsontext.Value) ([]byte, error) {
	fields := cloneJSONMap(b.fields)
	fields["type"] = jsontext.Value(`"response.create"`)
	fields["store"] = jsontext.Value("false")
	if previousResponseID == "" {
		delete(fields, "previous_response_id")
	} else {
		encoded, _ := jsonv2.Marshal(previousResponseID)
		fields["previous_response_id"] = encoded
	}
	encodedInput, err := jsonv2.Marshal(input)
	if err != nil {
		return nil, err
	}
	fields["input"] = encodedInput
	return jsonv2.Marshal(fields)
}

func cachedInputDelta(current webSocketRequestBody, continuation *webSocketContinuation) ([]jsontext.Value, bool) {
	if continuation == nil || continuation.responseID == "" || !requestFieldsEqual(current.fields, continuation.request.fields) {
		return nil, false
	}
	baseline := append(cloneJSONValues(continuation.request.input), continuation.responseItems...)
	if len(current.input) < len(baseline) {
		return nil, false
	}
	for index := range baseline {
		if !bytes.Equal(current.input[index], baseline[index]) {
			return nil, false
		}
	}
	return cloneJSONValues(current.input[len(baseline):]), true
}

func requestFieldsEqual(left, right map[string]jsontext.Value) bool {
	ignored := func(key string) bool { return key == "input" || key == "previous_response_id" }
	count := func(values map[string]jsontext.Value) int {
		n := 0
		for key := range values {
			if !ignored(key) {
				n++
			}
		}
		return n
	}
	if count(left) != count(right) {
		return false
	}
	for key, value := range left {
		if ignored(key) {
			continue
		}
		if !bytes.Equal(value, right[key]) {
			return false
		}
	}
	return true
}

func (c *Client) produceWebSocket(ctx context.Context, op string, params openairesponses.ResponseNewParams, sessionID string, emit internalstream.Emit) error {
	body, err := requestBody(params)
	if err != nil {
		return requestError(op, "encode WebSocket request: %v", err)
	}
	allowContinuation := true
	missingContinuationRetried := false
	for {
		lease, _, err := c.acquireWebSocket(ctx, op, sessionID)
		if err != nil {
			return err
		}
		previousResponseID := ""
		input := cloneJSONValues(body.input)
		usedContinuation := false
		if allowContinuation && (c.transport == TransportWebSocketCached || c.transport == TransportAuto) && lease.entry != nil {
			if delta, ok := cachedInputDelta(body, lease.entry.continuation); ok {
				input = delta
				previousResponseID = lease.entry.continuation.responseID
				usedContinuation = true
			} else {
				lease.entry.continuation = nil
			}
		}
		payload, err := body.responseCreate(previousResponseID, input)
		if err != nil {
			lease.release(false, nil)
			return requestError(op, "encode response.create: %v", err)
		}
		stream := newWebSocketEventStream(ctx, op, lease.conn)
		if err := lease.conn.WriteMessage(websocket.TextMessage, payload); err != nil {
			_ = stream.Close()
			lease.release(false, nil)
			if contextErr := contextErr(ctx); contextErr != nil {
				return contextErr
			}
			// A failed frame write is ambiguous: the peer may have received the
			// request. Evict the lease, but never retry this caller's operation.
			return transportError(op, fmt.Errorf("write response.create: %w", err))
		}
		committed := false
		emitRejected := false
		wrappedEmit := func(chunk llm.Chunk) bool {
			accepted := emit(chunk)
			if accepted {
				committed = true
			} else {
				emitRejected = true
			}
			return accepted
		}
		err = responseCodec().ProduceEvents(ctx, op, c.model, params, stream, wrappedEmit)
		keep := err == nil && stream.completed && !emitRejected && ctx.Err() == nil
		var continuation *webSocketContinuation
		if keep && (c.transport == TransportWebSocketCached || c.transport == TransportAuto) && stream.responseID != "" {
			continuation = &webSocketContinuation{request: body.clone(), responseID: stream.responseID, responseItems: cloneJSONValues(stream.responseItems)}
		}
		lease.release(keep, continuation)
		if err == nil {
			return nil
		}
		if usedContinuation && !committed && !missingContinuationRetried && previousResponseNotFound(err) {
			missingContinuationRetried = true
			allowContinuation = false
			continue
		}
		return err
	}
}

func previousResponseNotFound(err error) bool {
	var apiErr *APIError
	return errors.As(err, &apiErr) && apiErr.Code == "previous_response_not_found"
}

type webSocketEventStream struct {
	ctx           context.Context
	op            string
	conn          *websocket.Conn
	current       openairesponses.ResponseStreamEventUnion
	err           error
	terminal      bool
	completed     bool
	responseID    string
	responseItems []jsontext.Value
	stopWatcher   chan struct{}
	watcherDone   chan struct{}
	closeOnce     sync.Once
}

func newWebSocketEventStream(ctx context.Context, op string, conn *websocket.Conn) *webSocketEventStream {
	stream := &webSocketEventStream{ctx: ctx, op: op, conn: conn, stopWatcher: make(chan struct{}), watcherDone: make(chan struct{})}
	go func() {
		defer close(stream.watcherDone)
		select {
		case <-ctx.Done():
			_ = conn.Close()
		case <-stream.stopWatcher:
		}
	}()
	return stream
}

func (s *webSocketEventStream) Next() bool {
	if s.err != nil || s.terminal {
		return false
	}
	messageType, data, err := s.conn.ReadMessage()
	if err != nil {
		if contextErr := contextErr(s.ctx); contextErr != nil {
			s.err = contextErr
		} else if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
			// A clean close without a terminal event is handled by Codec as malformed.
			s.err = nil
		} else if strings.Contains(strings.ToLower(err.Error()), "read limit") || websocket.IsCloseError(err, websocket.CloseMessageTooBig) {
			s.err = malformedResponse(s.op, "WebSocket event exceeds %d bytes", maxProviderDataBytes)
		} else {
			s.err = err
		}
		return false
	}
	if messageType != websocket.TextMessage && messageType != websocket.BinaryMessage {
		s.err = malformedResponse(s.op, "WebSocket delivered unsupported message type %d", messageType)
		return false
	}
	if len(data) > maxProviderDataBytes {
		s.err = malformedResponse(s.op, "WebSocket event exceeds %d bytes", maxProviderDataBytes)
		return false
	}
	var event openairesponses.ResponseStreamEventUnion
	if err := jsonv2.Unmarshal(data, &event); err != nil {
		s.err = malformedResponse(s.op, "decode WebSocket event: %v", err)
		return false
	}
	if event.Type == "" {
		s.err = malformedResponse(s.op, "WebSocket event is missing type")
		return false
	}
	s.current = event
	s.captureTerminal(data, event.Type)
	return true
}

func (s *webSocketEventStream) captureTerminal(data []byte, eventType string) {
	switch eventType {
	case "response.completed", "response.done", "response.incomplete":
		s.terminal = true
		s.completed = eventType != "response.incomplete"
		var envelope struct {
			Response struct {
				ID     string           `json:"id"`
				Output []jsontext.Value `json:"output"`
			} `json:"response"`
		}
		if err := jsonv2.Unmarshal(data, &envelope); err == nil {
			s.responseID = envelope.Response.ID
			s.responseItems = cloneJSONValues(envelope.Response.Output)
		}
	}
}

func (s *webSocketEventStream) Current() openairesponses.ResponseStreamEventUnion { return s.current }
func (s *webSocketEventStream) Err() error                                        { return s.err }
func (s *webSocketEventStream) Close() error {
	s.closeOnce.Do(func() {
		close(s.stopWatcher)
		<-s.watcherDone
	})
	return nil
}

var _ internalresponses.EventStream = (*webSocketEventStream)(nil)
