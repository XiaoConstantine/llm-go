package codex

import (
	"bytes"
	"context"
	"crypto/tls"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/gorilla/websocket"
)

var testUpgrader = websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}

func wsClient(t *testing.T, server *httptest.Server, transport TransportMode, options ...func(*Config)) *Client {
	t.Helper()
	config := Config{Model: "gpt-codex", AccessToken: "token", AccountID: "account", BaseURL: server.URL,
		HTTPClient: server.Client(), Transport: transport, Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, Originator: "test-origin"}
	for _, option := range options {
		option(&config)
	}
	client, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func readCreate(t *testing.T, conn *websocket.Conn) map[string]any {
	t.Helper()
	_, data, err := conn.ReadMessage()
	if err != nil {
		t.Fatalf("ReadMessage() error = %v", err)
	}
	var body map[string]any
	if err := jsonv2.Unmarshal(data, &body); err != nil {
		t.Fatalf("decode response.create: %v: %s", err, data)
	}
	return body
}

func writeWSEvent(t *testing.T, conn *websocket.Conn, binary bool, event string) {
	t.Helper()
	kind := websocket.TextMessage
	if binary {
		kind = websocket.BinaryMessage
	}
	if err := conn.WriteMessage(kind, []byte(event)); err != nil {
		t.Fatalf("WriteMessage() error = %v", err)
	}
}

func completeWS(t *testing.T, conn *websocket.Conn, responseID, text string) {
	t.Helper()
	item := fmt.Sprintf(`{"type":"message","id":"msg_%s","role":"assistant","status":"completed","content":[{"type":"output_text","text":%q,"annotations":[]}]}`, responseID, text)
	writeWSEvent(t, conn, false, fmt.Sprintf(`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_%s","role":"assistant","status":"in_progress","content":[]}}`, responseID))
	writeWSEvent(t, conn, false, fmt.Sprintf(`{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg_%s","delta":%q}`, responseID, text))
	writeWSEvent(t, conn, false, fmt.Sprintf(`{"type":"response.output_text.done","output_index":0,"content_index":0,"item_id":"msg_%s","text":%q}`, responseID, text))
	writeWSEvent(t, conn, false, fmt.Sprintf(`{"type":"response.output_item.done","output_index":0,"item":%s}`, item))
	writeWSEvent(t, conn, false, fmt.Sprintf(`{"type":"response.completed","response":{"id":%q,"status":"completed","output":[%s],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`, responseID, item))
}

func TestResponseWebSocketURL(t *testing.T) {
	for raw, want := range map[string]string{
		"https://example.com/backend-api/codex/": "wss://example.com/backend-api/codex/responses",
		"http://example.com/codex/":              "ws://example.com/codex/responses",
	} {
		got, err := responseWebSocketURL(raw)
		if err != nil || got != want {
			t.Errorf("responseWebSocketURL(%q) = %q, %v; want %q", raw, got, err, want)
		}
	}
}

func TestWebSocketHandshakeBodyAndTypedDecoding(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/codex/responses" {
			t.Errorf("path = %q", r.URL.Path)
		}
		for name, want := range map[string]string{"Authorization": "Bearer token", "ChatGPT-Account-ID": "account", "OpenAI-Beta": webSocketBeta, "Originator": "test-origin", "Session-Id": "session", "X-Client-Request-Id": "session", "X-Trace": "trace"} {
			if got := r.Header.Get(name); got != want {
				t.Errorf("%s = %q, want %q", name, got, want)
			}
		}
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		body := readCreate(t, conn)
		if body["type"] != "response.create" || body["store"] != false || body["stream"] != true || body["model"] != "gpt-codex" {
			t.Errorf("body = %#v", body)
		}
		writeWSEvent(t, conn, true, `{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`)
		writeWSEvent(t, conn, false, `{"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"item_id":"rs_1","delta":"thinking"}`)
		writeWSEvent(t, conn, false, `{"type":"response.reasoning_summary_text.done","output_index":0,"summary_index":0,"item_id":"rs_1","text":"thinking"}`)
		writeWSEvent(t, conn, false, `{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[]}}`)
		writeWSEvent(t, conn, false, `{"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":""}}`)
		writeWSEvent(t, conn, false, `{"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc_1","delta":"{\"path\":\"README.md\"}"}`)
		writeWSEvent(t, conn, false, `{"type":"response.function_call_arguments.done","output_index":1,"item_id":"fc_1","name":"read","arguments":"{\"path\":\"README.md\"}"}`)
		writeWSEvent(t, conn, false, `{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{\"path\":\"README.md\"}"}}`)
		writeWSEvent(t, conn, false, `{"type":"response.output_item.added","output_index":2,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}`)
		writeWSEvent(t, conn, true, `{"type":"response.output_text.delta","output_index":2,"content_index":0,"item_id":"msg_1","delta":"hello"}`)
		writeWSEvent(t, conn, false, `{"type":"response.output_text.done","output_index":2,"content_index":0,"item_id":"msg_1","text":"hello"}`)
		writeWSEvent(t, conn, false, `{"type":"response.output_item.done","output_index":2,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`)
		writeWSEvent(t, conn, false, `{"type":"response.completed","response":{"id":"resp","model":"served","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[]},{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}`)
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket, func(c *Config) { c.Headers = http.Header{"X-Trace": {"trace"}} })
	stream, err := client.Stream(context.Background(), llm.Request{SessionID: "session", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}, Tools: []llm.Tool{{Name: "read", InputSchema: []byte(`{"type":"object"}`)}}})
	if err != nil {
		t.Fatal(err)
	}
	response := &llm.Response{Message: llm.Message{Role: llm.RoleAssistant}}
	eventKinds := make(map[llm.StreamEventKind]bool)
	for {
		chunk, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		if recvErr != nil {
			t.Fatal(recvErr)
		}
		mergeChunk(response, chunk)
		for _, event := range chunk.Events {
			eventKinds[event.Kind] = true
		}
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []llm.StreamEventKind{llm.StreamEventStart, llm.StreamEventReasoningDelta, llm.StreamEventToolCallDelta, llm.StreamEventToolCallEnd, llm.StreamEventTextDelta, llm.StreamEventDone} {
		if !eventKinds[kind] {
			t.Errorf("missing stream event %q", kind)
		}
	}
	if response.ID != "resp" || response.Model != "served" || response.Text() != "hello" || response.ReasoningSummary != "thinking" || len(response.Message.ToolCalls) != 1 || response.Usage == nil || response.Usage.TotalTokens != 5 {
		t.Fatalf("response = %#v", response)
	}
	if got := string(response.Message.ToolCalls[0].Arguments); got != `{"path":"README.md"}` {
		t.Fatalf("arguments = %s", got)
	}
	if len(response.Message.ProviderData) == 0 {
		t.Fatal("missing reasoning/provider replay data")
	}
}

func TestOwnedHeadersReplaceEveryCaseOnSSEAndWebSocket(t *testing.T) {
	for _, transport := range []TransportMode{TransportSSE, TransportWebSocket} {
		t.Run(string(transport), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				beta := "responses=experimental"
				if transport == TransportWebSocket {
					beta = webSocketBeta
				}
				for name, want := range map[string]string{
					"Authorization": "Bearer token", "ChatGPT-Account-ID": "account", "OpenAI-Beta": beta,
					"Originator": "test-origin", "Content-Type": "application/json", "User-Agent": "llm-go",
				} {
					if values := r.Header.Values(name); len(values) != 1 || values[0] != want {
						t.Errorf("%s = %q, want [%q]", name, values, want)
					}
				}
				if transport == TransportWebSocket {
					conn, err := testUpgrader.Upgrade(w, r, nil)
					if err != nil {
						return
					}
					defer func() { _ = conn.Close() }()
					_, _, _ = conn.ReadMessage()
					completeWS(t, conn, "resp", "ok")
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				writeSSE(t, w, `{"type":"response.completed","response":{"status":"completed"}}`)
			}))
			defer server.Close()
			client := wsClient(t, server, transport, func(c *Config) {
				c.Headers = http.Header{
					"authorization": {"evil-a"}, "Openai-Beta": {"evil-b"}, "CHATGPT-account-ID": {"evil-c"},
					"originator": {"evil-d"}, "content-type": {"evil-e"}, "user-AGENT": {"evil-f"},
				}
			})
			if _, err := client.Generate(context.Background(), textRequest("hello")); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestWebSocketAttemptHeadersUseTransientConnections(t *testing.T) {
	traces := make(chan string, 2)
	var connections atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		traces <- r.Header.Get("X-Trace")
		connections.Add(1)
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		_, _, _ = conn.ReadMessage()
		completeWS(t, conn, "resp", "ok")
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocketCached)
	request := textRequest("hello")
	request.SessionID = "session"
	for _, trace := range []string{"first", "second"} {
		generator, err := llm.WithRetry(client, llm.RetryPolicy{MaxAttempts: 1, Hook: func(context.Context, llm.Attempt) (http.Header, error) { return http.Header{"X-Trace": {trace}}, nil }})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := generator.Generate(context.Background(), request); err != nil {
			t.Fatal(err)
		}
	}
	if first, second := <-traces, <-traces; first != "first" || second != "second" {
		t.Fatalf("handshake traces = %q, %q", first, second)
	}
	if connections.Load() != 2 {
		t.Fatalf("connections = %d, want 2 transient handshakes", connections.Load())
	}
	client.webSockets.mu.Lock()
	cached := len(client.webSockets.entries)
	client.webSockets.mu.Unlock()
	if cached != 0 {
		t.Fatalf("attempt-scoped requests cached %d connections", cached)
	}
}

func TestWebSocketSessionCapacityEvictsOldestIdle(t *testing.T) {
	closed := make(chan string, 3)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		session := r.Header.Get("Session-Id")
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close(); closed <- session }()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
			completeWS(t, conn, "resp-"+session, "ok")
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket, func(c *Config) { c.WebSocketMaxSessions = 2 })
	for _, session := range []string{"one", "two", "three"} {
		request := textRequest(session)
		request.SessionID = session
		if _, err := client.Generate(context.Background(), request); err != nil {
			t.Fatal(err)
		}
	}
	select {
	case session := <-closed:
		if session != "one" {
			t.Fatalf("evicted session = %q, want one", session)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("oldest idle session was not evicted")
	}
	client.webSockets.mu.Lock()
	retained := make(map[string]bool)
	for key := range client.webSockets.entries {
		retained[key.session] = true
	}
	count := len(client.webSockets.entries)
	client.webSockets.mu.Unlock()
	if count != 2 || retained["one"] || !retained["two"] || !retained["three"] {
		t.Fatalf("retained sessions = %v", retained)
	}
}

func TestWebSocketSessionCapacityUsesTransientWhenAllBusy(t *testing.T) {
	seen := make(chan string, 2)
	releaseFirst := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		session := r.Header.Get("Session-Id")
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		seen <- session
		if session == "one" {
			<-releaseFirst
		}
		completeWS(t, conn, "resp-"+session, "ok")
		if session == "one" {
			for {
				if _, _, err := conn.ReadMessage(); err != nil {
					return
				}
			}
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket, func(c *Config) { c.WebSocketMaxSessions = 1 })
	firstDone := make(chan error, 1)
	go func() {
		request := textRequest("one")
		request.SessionID = "one"
		_, err := client.Generate(context.Background(), request)
		firstDone <- err
	}()
	if session := <-seen; session != "one" {
		t.Fatalf("first session = %q", session)
	}
	second := textRequest("two")
	second.SessionID = "two"
	if _, err := client.Generate(context.Background(), second); err != nil {
		t.Fatal(err)
	}
	if session := <-seen; session != "two" {
		t.Fatalf("second session = %q", session)
	}
	client.webSockets.mu.Lock()
	count := len(client.webSockets.entries)
	retainedFirst := false
	for key := range client.webSockets.entries {
		retainedFirst = retainedFirst || key.session == "one"
	}
	client.webSockets.mu.Unlock()
	if count != 1 || !retainedFirst {
		t.Fatalf("busy capacity state: count=%d retainedFirst=%v", count, retainedFirst)
	}
	close(releaseFirst)
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
}

func TestConfiguredWebSocketDialerIsUsedAndOwned(t *testing.T) {
	called := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/codex/responses" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Sec-WebSocket-Protocol"); got != "owned-protocol" {
			t.Errorf("subprotocol = %q", got)
		}
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		_, _, _ = conn.ReadMessage()
		completeWS(t, conn, "resp", "ok")
	}))
	defer server.Close()
	dialer := &websocket.Dialer{
		NetDialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			called <- address
			return (&net.Dialer{}).DialContext(ctx, network, address)
		},
		TLSClientConfig: &tls.Config{ServerName: "owned.example", NextProtos: []string{"owned-next"}, MinVersion: tls.VersionTLS12},
		Subprotocols:    []string{"owned-protocol"},
	}
	client := wsClient(t, server, TransportWebSocket, func(c *Config) { c.WebSocketDialer = dialer })
	dialer.NetDialContext = func(context.Context, string, string) (net.Conn, error) { return nil, errors.New("mutated dialer") }
	dialer.TLSClientConfig.ServerName = "mutated.example"
	dialer.TLSClientConfig.NextProtos[0] = "mutated-next"
	dialer.Subprotocols[0] = "mutated-protocol"
	if client.webSocketDialer.TLSClientConfig.ServerName != "owned.example" || client.webSocketDialer.TLSClientConfig.NextProtos[0] != "owned-next" || client.webSocketDialer.Subprotocols[0] != "owned-protocol" {
		t.Fatal("Client aliases caller WebSocketDialer storage")
	}
	if _, err := client.Generate(context.Background(), textRequest("hello")); err != nil {
		t.Fatal(err)
	}
	select {
	case address := <-called:
		if address == "" {
			t.Fatal("empty dial address")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("configured dialer was not used")
	}
}

func TestWebSocketConnectionReuseAndNoSessionIsolation(t *testing.T) {
	var connections atomic.Int32
	bodies := make(chan map[string]any, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		connections.Add(1)
		defer func() { _ = conn.Close() }()
		for i := 0; ; i++ {
			_, data, err := conn.ReadMessage()
			if err != nil {
				return
			}
			var body map[string]any
			if err := jsonv2.Unmarshal(data, &body); err != nil {
				t.Errorf("decode response.create: %v", err)
				return
			}
			bodies <- body
			completeWS(t, conn, fmt.Sprintf("%d", i), "ok")
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket)
	request := textRequest("hello")
	request.SessionID = "same"
	for range 2 {
		if _, err := client.Generate(context.Background(), request); err != nil {
			t.Fatal(err)
		}
	}
	if connections.Load() != 1 {
		t.Fatalf("session connections=%d", connections.Load())
	}
	for range 2 {
		body := <-bodies
		if _, ok := body["previous_response_id"]; ok || len(body["input"].([]any)) != 1 {
			t.Fatalf("plain WebSocket body = %#v", body)
		}
	}
	without := textRequest("hello")
	for range 2 {
		if _, err := client.Generate(context.Background(), without); err != nil {
			t.Fatal(err)
		}
	}
	if connections.Load() != 3 {
		t.Fatalf("all connections=%d, want 3", connections.Load())
	}
}

func TestWebSocketCachedDeltaAndMismatchFullContext(t *testing.T) {
	bodies := make(chan map[string]any, 3)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		for i := range 3 {
			body := readCreate(t, conn)
			bodies <- body
			completeWS(t, conn, fmt.Sprintf("resp%d", i+1), fmt.Sprintf("answer%d", i+1))
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocketCached)
	firstRequest := textRequest("one")
	firstRequest.SessionID = "session"
	first, err := client.Generate(context.Background(), firstRequest)
	if err != nil {
		t.Fatal(err)
	}
	secondRequest := llm.Request{SessionID: "session", Messages: []llm.Message{firstRequest.Messages[0], first.Message, {Role: llm.RoleUser, Content: []llm.Part{{Text: "two"}}}}}
	second, err := client.Generate(context.Background(), secondRequest)
	if err != nil {
		t.Fatal(err)
	}
	temperature := 0.5
	thirdRequest := llm.Request{SessionID: "session", Temperature: &temperature, Messages: append(append([]llm.Message(nil), secondRequest.Messages...), second.Message, llm.Message{Role: llm.RoleUser, Content: []llm.Part{{Text: "three"}}})}
	if _, err := client.Generate(context.Background(), thirdRequest); err != nil {
		t.Fatal(err)
	}
	firstBody, secondBody, thirdBody := <-bodies, <-bodies, <-bodies
	if _, ok := firstBody["previous_response_id"]; ok {
		t.Fatalf("first previous = %#v", firstBody)
	}
	if secondBody["previous_response_id"] != "resp1" || len(secondBody["input"].([]any)) != 1 {
		t.Fatalf("delta body = %#v", secondBody)
	}
	if _, ok := thirdBody["previous_response_id"]; ok || len(thirdBody["input"].([]any)) <= 1 {
		t.Fatalf("mismatch body = %#v", thirdBody)
	}
}

func TestWebSocketPreviousResponseNotFoundRetriesFullContext(t *testing.T) {
	bodies := make(chan map[string]any, 3)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		for {
			body := readCreate(t, conn)
			bodies <- body
			n := requests.Add(1)
			if n == 1 {
				completeWS(t, conn, "resp1", "one")
				continue
			}
			if n == 2 {
				writeWSEvent(t, conn, false, `{"type":"error","code":"previous_response_not_found","message":"gone"}`)
				return
			}
			completeWS(t, conn, "resp2", "two")
			return
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocketCached)
	request := textRequest("one")
	request.SessionID = "session"
	first, err := client.Generate(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	secondRequest := llm.Request{SessionID: "session", Messages: []llm.Message{request.Messages[0], first.Message, {Role: llm.RoleUser, Content: []llm.Part{{Text: "two"}}}}}
	if _, err := client.Generate(context.Background(), secondRequest); err != nil {
		t.Fatal(err)
	}
	_, delta, full := <-bodies, <-bodies, <-bodies
	if delta["previous_response_id"] != "resp1" {
		t.Fatalf("delta=%#v", delta)
	}
	if _, ok := full["previous_response_id"]; ok || len(full["input"].([]any)) != 3 {
		t.Fatalf("retry=%#v", full)
	}
}

func TestWebSocketMalformedEarlyCloseAndProviderErrors(t *testing.T) {
	tests := []struct {
		name   string
		action func(*testing.T, *websocket.Conn)
		kind   llm.ErrorKind
	}{
		{"malformed json", func(t *testing.T, c *websocket.Conn) { _ = c.WriteMessage(websocket.TextMessage, []byte("{")) }, llm.KindMalformedResponse},
		{"missing type", func(t *testing.T, c *websocket.Conn) { writeWSEvent(t, c, false, `{"value":1}`) }, llm.KindMalformedResponse},
		{"early clean close", func(t *testing.T, c *websocket.Conn) {
			_ = c.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, "done"), time.Now().Add(time.Second))
		}, llm.KindMalformedResponse},
		{"connection error", func(_ *testing.T, c *websocket.Conn) { _ = c.Close() }, llm.KindTransport},
		{"failed", func(t *testing.T, c *websocket.Conn) {
			writeWSEvent(t, c, false, `{"type":"response.failed","response":{"status":"failed","error":{"code":"rate_limit_exceeded","message":"busy"}}}`)
		}, llm.KindRateLimit},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				c, e := testUpgrader.Upgrade(w, r, nil)
				if e != nil {
					return
				}
				defer func() { _ = c.Close() }()
				_, _, _ = c.ReadMessage()
				test.action(t, c)
			}))
			defer server.Close()
			client := wsClient(t, server, TransportWebSocket)
			response, err := client.Generate(context.Background(), textRequest("hello"))
			if response != nil {
				t.Fatalf("response=%#v", response)
			}
			requireModelError(t, err, test.kind, "generate")
		})
	}
}

type blockingResponseWriteConn struct {
	net.Conn
	handshake        []byte
	handshakeWritten bool
	blocked          chan struct{}
	unblock          chan struct{}
	blockOnce        sync.Once
	closeOnce        sync.Once
}

func (c *blockingResponseWriteConn) Write(data []byte) (int, error) {
	if c.handshakeWritten {
		c.blockOnce.Do(func() { close(c.blocked) })
		<-c.unblock
		return 0, net.ErrClosed
	}
	n, err := c.Conn.Write(data)
	c.handshake = append(c.handshake, data[:n]...)
	if bytes.Contains(c.handshake, []byte("\r\n\r\n")) {
		c.handshake = nil
		c.handshakeWritten = true
	}
	return n, err
}

func (c *blockingResponseWriteConn) Close() error {
	c.closeOnce.Do(func() { close(c.unblock) })
	return c.Conn.Close()
}

func TestWebSocketCancelAndCloseInterruptBlockedResponseCreateWrite(t *testing.T) {
	for _, operation := range []string{"cancel", "close"} {
		t.Run(operation, func(t *testing.T) {
			blocked := make(chan struct{})
			unblock := make(chan struct{})
			serverRelease := make(chan struct{})
			serverDone := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := testUpgrader.Upgrade(w, r, nil)
				if err != nil {
					close(serverDone)
					return
				}
				<-serverRelease // Deliberately never read the large response.create frame.
				_ = conn.Close()
				close(serverDone)
			}))
			defer server.Close()
			transport := &http.Transport{DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
				conn, err := (&net.Dialer{}).DialContext(ctx, network, address)
				if err != nil {
					return nil, err
				}
				return &blockingResponseWriteConn{Conn: conn, blocked: blocked, unblock: unblock}, nil
			}}
			client := wsClient(t, server, TransportWebSocket, func(c *Config) { c.HTTPClient = &http.Client{Transport: transport} })
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			request := textRequest(strings.Repeat("x", 16<<20))
			stream, err := client.Stream(ctx, request)
			if err != nil {
				t.Fatal(err)
			}
			select {
			case <-blocked:
			case <-time.After(3 * time.Second):
				t.Fatal("response.create write did not block")
			}
			done := make(chan error, 1)
			if operation == "cancel" {
				cancel()
				go func() {
					_, recvErr := stream.Recv()
					if !errors.Is(recvErr, context.Canceled) {
						done <- fmt.Errorf("Recv: %w", recvErr)
						return
					}
					done <- stream.Close()
				}()
			} else {
				go func() { done <- stream.Close() }()
			}
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("cancel/Close did not unblock WriteMessage")
			}
			close(serverRelease)
			select {
			case <-serverDone:
			case <-time.After(3 * time.Second):
				t.Fatal("server socket did not close")
			}
			client.webSockets.mu.Lock()
			active, cached := len(client.webSockets.connections), len(client.webSockets.entries)
			client.webSockets.mu.Unlock()
			if active != 0 || cached != 0 {
				t.Fatalf("WebSocket resources leaked: active=%d cached=%d", active, cached)
			}
		})
	}
}

func TestWebSocketCancellationAndStreamCloseCloseConnection(t *testing.T) {
	requestSeen := make(chan struct{})
	connectionClosed := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, e := testUpgrader.Upgrade(w, r, nil)
		if e != nil {
			return
		}
		defer close(connectionClosed)
		defer func() { _ = c.Close() }()
		_, _, _ = c.ReadMessage()
		close(requestSeen)
		for {
			if _, _, e = c.ReadMessage(); e != nil {
				return
			}
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket)
	stream, err := client.Stream(context.Background(), llm.Request{SessionID: "session", Messages: textRequest("hello").Messages})
	if err != nil {
		t.Fatal(err)
	}
	<-requestSeen
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-connectionClosed:
	case <-time.After(2 * time.Second):
		t.Fatal("connection not closed")
	}
	client.webSockets.mu.Lock()
	cached := len(client.webSockets.entries)
	client.webSockets.mu.Unlock()
	if cached != 0 {
		t.Fatalf("cached connections after Close = %d", cached)
	}
}

func TestWebSocketContextCancellationClosesConnection(t *testing.T) {
	requestSeen := make(chan struct{})
	connectionClosed := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer close(connectionClosed)
		defer func() { _ = conn.Close() }()
		_, _, _ = conn.ReadMessage()
		close(requestSeen)
		for {
			if _, _, err = conn.ReadMessage(); err != nil {
				return
			}
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket)
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.Stream(ctx, llm.Request{SessionID: "session", Messages: textRequest("hello").Messages})
	if err != nil {
		t.Fatal(err)
	}
	<-requestSeen
	cancel()
	if _, err := stream.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv() error = %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-connectionClosed:
	case <-time.After(2 * time.Second):
		t.Fatal("connection not closed")
	}
}

func TestWebSocketBusySessionUsesTransientConnection(t *testing.T) {
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	var connections atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, e := testUpgrader.Upgrade(w, r, nil)
		if e != nil {
			return
		}
		connections.Add(1)
		defer func() { _ = c.Close() }()
		_, _, _ = c.ReadMessage()
		started <- struct{}{}
		<-release
		completeWS(t, c, "resp", "ok")
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket)
	request := textRequest("hello")
	request.SessionID = "session"
	done := make(chan error, 2)
	for range 2 {
		go func() { _, e := client.Generate(context.Background(), request); done <- e }()
	}
	<-started
	<-started
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if connections.Load() != 2 {
		t.Fatalf("connections=%d", connections.Load())
	}
}

func TestWebSocketFailedCachedWriteIsNotRetried(t *testing.T) {
	var connections atomic.Int32
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		connections.Add(1)
		defer func() { _ = conn.Close() }()
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
			requests.Add(1)
			completeWS(t, conn, "resp", "ok")
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket)
	request := textRequest("hello")
	request.SessionID = "session"
	if _, err := client.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	client.webSockets.mu.Lock()
	var dead *websocket.Conn
	for _, entry := range client.webSockets.entries {
		dead = entry.conn
	}
	client.webSockets.mu.Unlock()
	if dead == nil {
		t.Fatal("missing cached connection")
	}
	_ = dead.Close()
	if response, err := client.Generate(context.Background(), request); err == nil || response != nil {
		t.Fatalf("ambiguous failed write = %#v, %v; want nil error response", response, err)
	}
	if connections.Load() != 1 || requests.Load() != 1 {
		t.Fatalf("failed write retried: connections=%d requests=%d", connections.Load(), requests.Load())
	}
	if _, err := client.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if connections.Load() != 2 || requests.Load() != 2 {
		t.Fatalf("next caller did not reconnect once: connections=%d requests=%d", connections.Load(), requests.Load())
	}
}

func TestWebSocketDoesNotReuseProviderErrorOrIncompleteResponse(t *testing.T) {
	for _, kind := range []string{"provider", "incomplete"} {
		t.Run(kind, func(t *testing.T) {
			var connections atomic.Int32
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				conn, err := testUpgrader.Upgrade(w, r, nil)
				if err != nil {
					return
				}
				connections.Add(1)
				defer func() { _ = conn.Close() }()
				if _, _, err := conn.ReadMessage(); err != nil {
					return
				}
				if requests.Add(1) == 1 {
					if kind == "provider" {
						writeWSEvent(t, conn, false, `{"type":"error","code":"server_error","message":"failed"}`)
					} else {
						writeWSEvent(t, conn, false, `{"type":"response.incomplete","response":{"id":"partial","status":"incomplete","output":[],"incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`)
					}
					return
				}
				completeWS(t, conn, "ok", "done")
			}))
			defer server.Close()
			client := wsClient(t, server, TransportWebSocket)
			request := textRequest("hello")
			request.SessionID = "session"
			first, err := client.Generate(context.Background(), request)
			if kind == "provider" {
				if err == nil || first != nil {
					t.Fatalf("first = %#v, %v", first, err)
				}
			} else if err != nil || first.FinishReason != llm.FinishReasonLength {
				t.Fatalf("first = %#v, %v", first, err)
			}
			if _, err := client.Generate(context.Background(), request); err != nil {
				t.Fatal(err)
			}
			if connections.Load() != 2 {
				t.Fatalf("connections = %d, want 2", connections.Load())
			}
		})
	}
}

func TestWebSocketEventSizeLimit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := testUpgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		_, _, _ = conn.ReadMessage()
		_ = conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"`+strings.Repeat("x", maxProviderDataBytes)+`"}`))
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket)
	_, err := client.Generate(context.Background(), textRequest("hello"))
	requireModelError(t, err, llm.KindMalformedResponse, "generate")
}

func TestWebSocketSessionCredentialAndClientIsolation(t *testing.T) {
	var connections atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, e := testUpgrader.Upgrade(w, r, nil)
		if e != nil {
			return
		}
		connections.Add(1)
		defer func() { _ = c.Close() }()
		for {
			if _, _, e = c.ReadMessage(); e != nil {
				return
			}
			completeWS(t, c, "resp", "ok")
		}
	}))
	defer server.Close()
	var token atomic.Int32
	var account atomic.Int32
	config := func() Config {
		return Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client(), Transport: TransportWebSocket, ResolveCredentials: func(context.Context, string) (Credentials, error) {
			return Credentials{AccessToken: fmt.Sprintf("token-%d", token.Load()), AccountID: fmt.Sprintf("account-%d", account.Load())}, nil
		}}
	}
	first, _ := New(config())
	defer func() { _ = first.Close() }()
	second, _ := New(config())
	defer func() { _ = second.Close() }()
	request := textRequest("hello")
	request.SessionID = "one"
	if _, e := first.Generate(context.Background(), request); e != nil {
		t.Fatal(e)
	}
	request.SessionID = "two"
	if _, e := first.Generate(context.Background(), request); e != nil {
		t.Fatal(e)
	}
	token.Store(1)
	request.SessionID = "one"
	if _, e := first.Generate(context.Background(), request); e != nil {
		t.Fatal(e)
	}
	account.Store(1)
	if _, e := first.Generate(context.Background(), request); e != nil {
		t.Fatal(e)
	}
	if _, e := second.Generate(context.Background(), request); e != nil {
		t.Fatal(e)
	}
	if connections.Load() != 5 {
		t.Fatalf("connections=%d,want5", connections.Load())
	}
}

func TestWebSocketIdleAndMaximumAgeCleanup(t *testing.T) {
	for _, which := range []string{"idle", "age"} {
		t.Run(which, func(t *testing.T) {
			closed := make(chan struct{}, 1)
			var connections atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				c, e := testUpgrader.Upgrade(w, r, nil)
				if e != nil {
					return
				}
				connections.Add(1)
				defer func() { closed <- struct{}{} }()
				defer func() { _ = c.Close() }()
				for {
					if _, _, e = c.ReadMessage(); e != nil {
						return
					}
					completeWS(t, c, "resp", "ok")
				}
			}))
			defer server.Close()
			client := wsClient(t, server, TransportWebSocket, func(c *Config) {
				if which == "idle" {
					c.WebSocketIdleTime = 20 * time.Millisecond
					c.WebSocketMaxAge = time.Second
				} else {
					c.WebSocketIdleTime = time.Second
					c.WebSocketMaxAge = 20 * time.Millisecond
				}
			})
			r := textRequest("x")
			r.SessionID = "session"
			if _, e := client.Generate(context.Background(), r); e != nil {
				t.Fatal(e)
			}
			select {
			case <-closed:
			case <-time.After(2 * time.Second):
				t.Fatal("cached connection not expired")
			}
			if _, e := client.Generate(context.Background(), r); e != nil {
				t.Fatal(e)
			}
			if connections.Load() != 2 {
				t.Fatalf("connections=%d", connections.Load())
			}
		})
	}
}

func TestAutoFallbackOnlyBeforeWebSocketCommitment(t *testing.T) {
	t.Run("preconnect fallback", func(t *testing.T) {
		var posts atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
				h, _, e := w.(http.Hijacker).Hijack()
				if e == nil {
					_ = h.Close()
				}
				return
			}
			posts.Add(1)
			w.Header().Set("Content-Type", "text/event-stream")
			writeSSE(t, w, `{"type":"response.completed","response":{"status":"completed"}}`)
		}))
		defer server.Close()
		client := wsClient(t, server, TransportAuto)
		if _, e := client.Generate(context.Background(), textRequest("x")); e != nil {
			t.Fatal(e)
		}
		if posts.Load() != 1 {
			t.Fatalf("posts=%d", posts.Load())
		}
	})
	t.Run("no fallback after send", func(t *testing.T) {
		var posts atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
				posts.Add(1)
				return
			}
			c, e := testUpgrader.Upgrade(w, r, nil)
			if e != nil {
				return
			}
			_, _, _ = c.ReadMessage()
			writeWSEvent(t, c, false, `{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg","role":"assistant","status":"in_progress","content":[]}}`)
			writeWSEvent(t, c, false, `{"type":"response.output_text.delta","output_index":0,"content_index":0,"item_id":"msg","delta":"committed"}`)
			_ = c.Close()
		}))
		defer server.Close()
		client := wsClient(t, server, TransportAuto)
		stream, err := client.Stream(context.Background(), textRequest("x"))
		if err != nil {
			t.Fatal(err)
		}
		chunk, err := stream.Recv()
		if err != nil || len(chunk.Content) != 1 || chunk.Content[0].Text != "committed" {
			t.Fatalf("first Recv() = %#v, %v", chunk, err)
		}
		if _, err := stream.Recv(); err == nil {
			t.Fatal("second Recv succeeded")
		}
		if err := stream.Close(); err != nil {
			t.Fatal(err)
		}
		if posts.Load() != 0 {
			t.Fatalf("posts=%d", posts.Load())
		}
	})
	t.Run("malformed event does not fall back", func(t *testing.T) {
		var posts atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
				posts.Add(1)
				return
			}
			conn, err := testUpgrader.Upgrade(w, r, nil)
			if err != nil {
				return
			}
			defer func() { _ = conn.Close() }()
			_, _, _ = conn.ReadMessage()
			_ = conn.WriteMessage(websocket.TextMessage, []byte("{"))
		}))
		defer server.Close()
		client := wsClient(t, server, TransportAuto)
		if _, err := client.Generate(context.Background(), textRequest("x")); err == nil {
			t.Fatal("Generate succeeded")
		}
		if posts.Load() != 0 {
			t.Fatalf("posts = %d", posts.Load())
		}
	})
	t.Run("auth and provider handshakes do not fall back", func(t *testing.T) {
		for _, status := range []int{http.StatusForbidden, http.StatusInternalServerError} {
			var posts atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
					http.Error(w, `{"error":{"message":"rejected"}}`, status)
					return
				}
				posts.Add(1)
			}))
			client := wsClient(t, server, TransportAuto)
			if _, err := client.Generate(context.Background(), textRequest("x")); err == nil {
				t.Fatalf("status %d succeeded", status)
			}
			if posts.Load() != 0 {
				t.Fatalf("status %d posts = %d", status, posts.Load())
			}
			_ = client.Close()
			server.Close()
		}
	})
	t.Run("strict never falls back", func(t *testing.T) {
		var posts atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
				h, _, e := w.(http.Hijacker).Hijack()
				if e == nil {
					_ = h.Close()
				}
				return
			}
			posts.Add(1)
		}))
		defer server.Close()
		client := wsClient(t, server, TransportWebSocket)
		if _, e := client.Generate(context.Background(), textRequest("x")); e == nil {
			t.Fatal("Generate succeeded")
		}
		if posts.Load() != 0 {
			t.Fatalf("posts=%d", posts.Load())
		}
	})
}

func TestWebSocketHandshakeUnauthorizedRefreshesOnce(t *testing.T) {
	var handshakes atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if handshakes.Add(1) == 1 {
			http.Error(w, `{"error":{"code":"expired","message":"expired"}}`, http.StatusUnauthorized)
			return
		}
		if r.Header.Get("Authorization") != "Bearer fresh" {
			t.Errorf("authorization=%q", r.Header.Get("Authorization"))
		}
		c, e := testUpgrader.Upgrade(w, r, nil)
		if e != nil {
			return
		}
		defer func() { _ = c.Close() }()
		_, _, _ = c.ReadMessage()
		completeWS(t, c, "resp", "ok")
	}))
	defer server.Close()
	var rejectedMu sync.Mutex
	var rejected []string
	client := wsClient(t, server, TransportWebSocket, func(c *Config) {
		c.AccessToken = ""
		c.AccountID = ""
		c.ResolveCredentials = func(_ context.Context, value string) (Credentials, error) {
			rejectedMu.Lock()
			rejected = append(rejected, value)
			rejectedMu.Unlock()
			token := "stale"
			if value != "" {
				token = "fresh"
			}
			return Credentials{AccessToken: token, AccountID: "account"}, nil
		}
	})
	if _, e := client.Generate(context.Background(), textRequest("x")); e != nil {
		t.Fatal(e)
	}
	rejectedMu.Lock()
	defer rejectedMu.Unlock()
	if fmt.Sprint(rejected) != "[ stale]" {
		t.Fatalf("rejected=%q", rejected)
	}
}

func TestClientCloseIdempotentClosesCachedSockets(t *testing.T) {
	closed := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, e := testUpgrader.Upgrade(w, r, nil)
		if e != nil {
			return
		}
		defer close(closed)
		defer func() { _ = c.Close() }()
		_, _, _ = c.ReadMessage()
		completeWS(t, c, "resp", "ok")
		for {
			if _, _, e = c.ReadMessage(); e != nil {
				return
			}
		}
	}))
	defer server.Close()
	client := wsClient(t, server, TransportWebSocket)
	request := textRequest("x")
	request.SessionID = "session"
	if _, e := client.Generate(context.Background(), request); e != nil {
		t.Fatal(e)
	}
	if e := client.Close(); e != nil {
		t.Fatal(e)
	}
	if e := client.Close(); e != nil {
		t.Fatal(e)
	}
	select {
	case <-closed:
	case <-time.After(2 * time.Second):
		t.Fatal("Close did not close socket")
	}
}
