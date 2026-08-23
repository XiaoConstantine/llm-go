package codex

import (
	"context"
	"encoding/base64"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestNewValidatesConfig(t *testing.T) {
	tests := []struct {
		name   string
		config Config
		want   string
	}{
		{name: "empty model", config: Config{AccessToken: testAccessToken("account")}, want: "model must not be empty"},
		{name: "missing credentials", config: Config{Model: "model"}, want: "access token or credential resolver is required"},
		{name: "invalid token", config: Config{Model: "model", AccessToken: "opaque"}, want: "access token is not a JWT"},
		{name: "ambiguous credentials", config: Config{Model: "model", AccessToken: "token", ResolveCredentials: func(context.Context, string) (Credentials, error) { return Credentials{}, nil }}, want: "must be empty"},
		{name: "unsupported capability", config: Config{Model: "model", AccessToken: testAccessToken("account"), Capabilities: []llm.Capability{llm.CapabilityJSON}}, want: "not implemented"},
		{name: "relative URL", config: Config{Model: "model", AccessToken: testAccessToken("account"), BaseURL: "/api"}, want: "scheme must be http or https"},
		{name: "URL credentials", config: Config{Model: "model", AccessToken: testAccessToken("account"), BaseURL: "https://user@example.com"}, want: "user information"},
		{name: "URL query", config: Config{Model: "model", AccessToken: testAccessToken("account"), BaseURL: "https://example.com?x=1"}, want: "query or fragment"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, err := New(test.config)
			if client != nil {
				t.Fatalf("New() = %#v, want nil", client)
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("New() error = %v, want substring %q", err, test.want)
			}
			modelErr := requireModelError(t, err, llm.KindInvalidRequest, "configure")
			if modelErr.Provider != defaultProvider {
				t.Fatalf("New() error provider = %q, want %q", modelErr.Provider, defaultProvider)
			}
		})
	}
}

func TestInfoCopiesCapabilitiesAndRelabelsProvider(t *testing.T) {
	client, err := New(Config{
		Provider:     "gateway",
		Model:        "gpt-codex",
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools, llm.CapabilityStreaming},
		AccessToken:  testAccessToken("account"),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	info := client.Info()
	if info.Provider != "gateway" || info.Model != "gpt-codex" {
		t.Fatalf("Info() = %#v", info)
	}
	want := []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming, llm.CapabilityTools}
	if fmt.Sprint(info.Capabilities) != fmt.Sprint(want) {
		t.Fatalf("Info().Capabilities = %v, want %v", info.Capabilities, want)
	}
	info.Capabilities[0] = "mutated"
	if client.Info().Capabilities[0] != llm.CapabilityGeneration {
		t.Fatal("Info returned Client-owned capability storage")
	}

	_, err = client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}, MaxOutputTokens: 1})
	modelErr := requireModelError(t, err, llm.KindUnsupported, "generate")
	if modelErr.Provider != "gateway" {
		t.Fatalf("Generate() error provider = %q, want gateway", modelErr.Provider)
	}
}

func TestGenerateUsesSubscriptionResponsesAndReplaysProviderData(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		requestNumber := requests.Add(1)
		if request.URL.Path != "/codex/responses" {
			t.Errorf("request path = %q, want /codex/responses", request.URL.Path)
		}
		for name, want := range map[string]string{
			"Authorization":      "Bearer access-token",
			"ChatGPT-Account-ID": "account-123",
			"OpenAI-Beta":        "responses=experimental",
			"Originator":         "test-suite",
			"Accept":             "text/event-stream",
			"X-Trace":            "trace",
		} {
			if got := request.Header.Values(name); len(got) != 1 || got[0] != want {
				t.Errorf("header %s = %q, want [%q]", name, got, want)
			}
		}

		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("ReadAll(request.Body) error = %v", err)
			return
		}
		var payload map[string]any
		if err := jsonv2.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if payload["model"] != "gpt-codex" || payload["stream"] != true || payload["store"] != false {
			t.Errorf("request model/stream/store = %v/%v/%v", payload["model"], payload["stream"], payload["store"])
		}
		input, ok := payload["input"].([]any)
		if !ok {
			t.Errorf("request input = %#v, want array", payload["input"])
			return
		}
		if requestNumber == 1 {
			if payload["instructions"] != "Use the read tool." {
				t.Errorf("instructions = %#v", payload["instructions"])
			}
			if len(input) != 1 || input[0].(map[string]any)["role"] != "user" {
				t.Errorf("first request input = %#v", input)
			}
			tools := payload["tools"].([]any)
			if len(tools) != 1 || tools[0].(map[string]any)["name"] != "read" || tools[0].(map[string]any)["strict"] != true {
				t.Errorf("tools = %#v", tools)
			}
		} else {
			if len(input) != 5 {
				t.Errorf("replay input length = %d, want 5: %#v", len(input), input)
			} else {
				wantTypes := []string{"reasoning", "message", "function_call", "function_call_output"}
				for i, want := range wantTypes {
					if got := input[i].(map[string]any)["type"]; got != want {
						t.Errorf("replay input[%d].type = %#v, want %q", i, got, want)
					}
				}
				if got := input[4].(map[string]any)["role"]; got != "user" {
					t.Errorf("replay input[4].role = %#v, want user", got)
				}
				if got := input[0].(map[string]any)["encrypted_content"]; got != "opaque" {
					t.Errorf("replay reasoning encrypted_content = %#v, want opaque", got)
				}
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		if requestNumber == 1 {
			writeSSE(t, w,
				`{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`,
				`{"type":"response.output_text.delta","delta":"hello"}`,
				`{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`,
				`{"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{\"path\":\"README.md\"}"}}`,
				`{"type":"response.completed","response":{"id":"resp_1","model":"served-codex","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque","summary":[]}],"usage":{"input_tokens":10,"output_tokens":4,"total_tokens":14}}}`,
			)
			return
		}
		writeSSE(t, w,
			`{"type":"response.output_text.delta","delta":"done"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_2","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}}`,
			`{"type":"response.completed","response":{"id":"resp_2","status":"completed"}}`,
		)
	}))
	defer server.Close()

	client, err := New(Config{
		Model:        "gpt-codex",
		Capabilities: []llm.Capability{llm.CapabilityTools},
		AccessToken:  "access-token",
		AccountID:    "account-123",
		BaseURL:      server.URL,
		HTTPClient:   server.Client(),
		Originator:   "test-suite",
		Headers: http.Header{
			"x-trace":            {"trace"},
			"authorization":      {"wrong"},
			"chatgpt-account-id": {"wrong"},
			"openai-beta":        {"wrong"},
			"originator":         {"wrong"},
			"accept":             {"wrong"},
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	tool := llm.Tool{Name: "read", Description: "read a file", InputSchema: jsontext.Value(`{"type":"object"}`), Strict: true}
	first, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.Part{{Text: "Use the read tool."}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "Begin"}}},
		},
		Tools: []llm.Tool{tool},
	})
	if err != nil {
		t.Fatalf("Generate(first) error = %v", err)
	}
	if first.ID != "resp_1" || first.Model != "served-codex" || first.Text() != "hello" || first.FinishReason != llm.FinishReasonToolCall {
		t.Fatalf("Generate(first) = %#v", first)
	}
	if first.Usage == nil || *first.Usage != (llm.Usage{InputTokens: 10, OutputTokens: 4, TotalTokens: 14}) {
		t.Fatalf("Generate(first).Usage = %#v", first.Usage)
	}
	if len(first.Message.ToolCalls) != 1 {
		t.Fatalf("Generate(first).ToolCalls = %#v", first.Message.ToolCalls)
	}
	call := first.Message.ToolCalls[0]
	if call.ID != "call_1" || call.Name != "read" || string(call.Arguments) != `{"path":"README.md"}` {
		t.Fatalf("Generate(first).ToolCalls[0] = %#v", call)
	}
	var continuation providerDataEnvelope
	if err := jsonv2.Unmarshal(first.Message.ProviderData, &continuation); err != nil || continuation.API != providerDataAPI || continuation.Version != providerDataVersion || continuation.Model != "gpt-codex" || len(continuation.Output) != 3 {
		t.Fatalf("provider data = %s, error = %v", first.Message.ProviderData, err)
	}

	second, err := client.Generate(context.Background(), llm.Request{
		Messages: []llm.Message{
			first.Message,
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: call.ID, Name: call.Name, Content: []llm.Part{{Text: "contents"}}}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "Continue"}}},
		},
		Tools: []llm.Tool{tool},
	})
	if err != nil {
		t.Fatalf("Generate(second) error = %v", err)
	}
	if second.Text() != "done" || second.FinishReason != llm.FinishReasonStop {
		t.Fatalf("Generate(second) = %#v", second)
	}
	if requests.Load() != 2 {
		t.Fatalf("request count = %d, want 2", requests.Load())
	}
}

func TestGenerateRetriesUnauthorizedWithRejectedToken(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if requests.Add(1) == 1 {
			if got := request.Header.Get("Authorization"); got != "Bearer stale" {
				t.Errorf("first Authorization = %q", got)
			}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"error":{"type":"authentication_error","code":"expired","message":"expired"}}`)
			return
		}
		if got := request.Header.Get("Authorization"); got != "Bearer fresh" {
			t.Errorf("second Authorization = %q", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w, `{"type":"response.completed","response":{"status":"completed"}}`)
	}))
	defer server.Close()

	var mu sync.Mutex
	var rejected []string
	client, err := New(Config{
		Model:      "gpt-codex",
		BaseURL:    server.URL,
		HTTPClient: server.Client(),
		ResolveCredentials: func(_ context.Context, rejectedToken string) (Credentials, error) {
			mu.Lock()
			rejected = append(rejected, rejectedToken)
			mu.Unlock()
			token := "stale"
			if rejectedToken != "" {
				token = "fresh"
			}
			return Credentials{AccessToken: token, AccountID: "account"}, nil
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.FinishReason != llm.FinishReasonStop {
		t.Fatalf("Generate().FinishReason = %q", response.FinishReason)
	}
	mu.Lock()
	defer mu.Unlock()
	if fmt.Sprint(rejected) != `[ stale]` {
		t.Fatalf("resolver rejected tokens = %q, want [\"\" \"stale\"]", rejected)
	}
}

func TestGenerateDoesNotRetryRejectedCredentials(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"type":"authentication_error","message":"expired"}}`)
	}))
	defer server.Close()

	client, err := New(Config{
		Model:       "gpt-codex",
		AccessToken: "stale",
		AccountID:   "account",
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if response != nil {
		t.Fatalf("Generate() = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindAuthentication, "generate")
	if got := requests.Load(); got != 1 {
		t.Fatalf("request count = %d, want 1", got)
	}
}

func TestGenerateClassifiesHTTPAndStreamErrors(t *testing.T) {
	tests := []struct {
		name        string
		status      int
		body        string
		contentType string
		wantKind    llm.ErrorKind
	}{
		{name: "rate limit response", status: http.StatusTooManyRequests, contentType: "application/json", body: `{"error":{"code":"rate_limit_exceeded","message":"busy"}}`, wantKind: llm.KindRateLimit},
		{name: "context response", status: http.StatusBadRequest, contentType: "application/json", body: `{"error":{"code":"context_length_exceeded","message":"too long"}}`, wantKind: llm.KindContextLimit},
		{name: "stream auth", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"type\":\"error\",\"code\":\"authentication_error\",\"message\":\"expired\"}\n\n", wantKind: llm.KindAuthentication},
		{name: "nested stream rate limit", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"type\":\"error\",\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"busy\"}}\n\n", wantKind: llm.KindRateLimit},
		{name: "malformed stream", status: http.StatusOK, contentType: "text/event-stream", body: "data: {not-json}\n\n", wantKind: llm.KindMalformedResponse},
		{name: "unfinished output item", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n", wantKind: llm.KindMalformedResponse},
		{name: "changed output identity", status: http.StatusOK, contentType: "text/event-stream", body: "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_2\"}}\n\n", wantKind: llm.KindMalformedResponse},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", test.contentType)
				w.Header().Set("Retry-After", "3")
				w.WriteHeader(test.status)
				_, _ = io.WriteString(w, test.body)
			}))
			defer server.Close()
			client, err := New(Config{Model: "model", AccessToken: "token", AccountID: "account", BaseURL: server.URL, HTTPClient: server.Client()})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			response, err := client.Generate(context.Background(), textRequest("hello"))
			if response != nil {
				t.Fatalf("Generate() = %#v, want nil", response)
			}
			modelErr := requireModelError(t, err, test.wantKind, "generate")
			if test.status >= 400 && modelErr.HTTPStatus != test.status {
				t.Fatalf("HTTPStatus = %d, want %d", modelErr.HTTPStatus, test.status)
			}
			if test.status == http.StatusTooManyRequests && modelErr.RetryAfter != 3*time.Second {
				t.Fatalf("RetryAfter = %v, want 3s", modelErr.RetryAfter)
			}
		})
	}
}

func TestGenerateIncompleteDropsUnfinishedProviderData(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}`,
			`{"type":"response.output_text.delta","output_index":0,"delta":"partial"}`,
			`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`,
		)
	}))
	defer server.Close()

	client, err := New(Config{
		Model:       "model",
		AccessToken: "token",
		AccountID:   "account",
		BaseURL:     server.URL,
		HTTPClient:  server.Client(),
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	response, err := client.Generate(context.Background(), textRequest("hello"))
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.Text() != "partial" || response.FinishReason != llm.FinishReasonLength {
		t.Fatalf("Generate() = %#v", response)
	}
	if len(response.Message.ProviderData) != 0 {
		t.Fatalf("Generate().Message.ProviderData = %s, want empty", response.Message.ProviderData)
	}
}

func TestStreamOrdersToolCallsByOutputIndex(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w,
			`{"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","id":"fc_2","call_id":"call_2","name":"second","arguments":"{}"}}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"first","arguments":"{}"}}`,
			`{"type":"response.completed","response":{"status":"completed"}}`,
		)
	}))
	defer server.Close()
	client, err := New(Config{
		Model: "model", AccessToken: "token", AccountID: "account", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	request := textRequest("hello")
	request.Tools = []llm.Tool{
		{Name: "first", InputSchema: jsontext.Value(`{"type":"object"}`)},
		{Name: "second", InputSchema: jsontext.Value(`{"type":"object"}`)},
	}
	stream, err := client.Stream(context.Background(), request)
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	defer stream.Close()
	chunk, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv() error = %v", err)
	}
	if len(chunk.ToolCalls) != 2 || chunk.ToolCalls[0].Name != "first" || chunk.ToolCalls[1].Name != "second" {
		t.Fatalf("Recv().ToolCalls = %#v", chunk.ToolCalls)
	}
	if chunk.FinishReason != llm.FinishReasonToolCall {
		t.Fatalf("Recv().FinishReason = %q", chunk.FinishReason)
	}
	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("second Recv() error = %v, want EOF", err)
	}
}

func TestStreamCancellationReachesHTTPServer(t *testing.T) {
	requestStarted := make(chan struct{})
	requestCanceled := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		close(requestStarted)
		<-request.Context().Done()
		close(requestCanceled)
	}))
	defer server.Close()
	client, err := New(Config{
		Model: "model", AccessToken: "token", AccountID: "account", BaseURL: server.URL, HTTPClient: server.Client(),
		Capabilities: []llm.Capability{llm.CapabilityStreaming},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := client.Stream(ctx, textRequest("hello"))
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	select {
	case <-requestStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("HTTP request did not start")
	}
	cancel()
	if _, err := stream.Recv(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Recv() error = %v, want context.Canceled", err)
	}
	if err := stream.Close(); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("Close() error = %v", err)
	}
	select {
	case <-requestCanceled:
	case <-time.After(5 * time.Second):
		t.Fatal("context cancellation did not reach HTTP server")
	}
}

func TestGenerateRejectsMalformedProviderDataBeforeCredentials(t *testing.T) {
	var resolutions atomic.Int32
	client, err := New(Config{
		Model: "model",
		ResolveCredentials: func(context.Context, string) (Credentials, error) {
			resolutions.Add(1)
			return Credentials{AccessToken: "token", AccountID: "account"}, nil
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	tests := []jsontext.Value{
		jsontext.Value(`{"api":"openai-codex-responses","version":1,"model":"model","output":[]}`),
		jsontext.Value(`{"api":"openai-codex-responses","version":1,"model":"model","output":["not-an-object"]}`),
	}
	for _, providerData := range tests {
		response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleAssistant, ProviderData: providerData}}})
		if response != nil {
			t.Fatalf("Generate(%s) = %#v, want nil", providerData, response)
		}
		requireModelError(t, err, llm.KindInvalidRequest, "generate")
	}
	if resolutions.Load() != 0 {
		t.Fatalf("credential resolutions = %d, want 0", resolutions.Load())
	}
}

func TestRequestToWireIgnoresForeignProviderData(t *testing.T) {
	params, err := requestToWire("generate", "model", llm.Request{Messages: []llm.Message{{
		Role:         llm.RoleAssistant,
		Content:      []llm.Part{{Text: "fallback"}},
		ProviderData: jsontext.Value(`{"api":"another-provider","version":1}`),
	}}})
	if err != nil {
		t.Fatalf("requestToWire() error = %v", err)
	}
	items := params.Input.OfInputItemList
	if len(items) != 1 || items[0].OfMessage == nil || items[0].OfMessage.Role != "assistant" {
		t.Fatalf("request input = %#v, want fallback assistant message", items)
	}
}

func TestAccountIDFromToken(t *testing.T) {
	accountID, err := AccountIDFromToken(testAccessToken(" account-42 "))
	if err != nil || accountID != "account-42" {
		t.Fatalf("AccountIDFromToken() = %q, %v", accountID, err)
	}
	tests := []string{
		"opaque",
		"header.invalid!base64.signature",
		jwtWithPayload(`{"sub":"user"}`),
		jwtWithPayload(`{"https://api.openai.com/auth":{"chatgpt_account_id":""}}`),
		jwtWithPayload(`{"https://api.openai.com/auth":{"chatgpt_account_id":"a","chatgpt_account_id":"b"}}`),
	}
	for _, token := range tests {
		if accountID, err := AccountIDFromToken(token); err == nil || accountID != "" {
			t.Errorf("AccountIDFromToken(%q) = %q, %v, want error", token, accountID, err)
		}
	}
}

func textRequest(text string) llm.Request {
	return llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: text}}}}}
}

func testAccessToken(accountID string) string {
	payload, err := jsonv2.Marshal(map[string]any{accountClaim: map[string]any{"chatgpt_account_id": accountID}})
	if err != nil {
		panic(err)
	}
	return jwtWithPayload(string(payload))
}

func jwtWithPayload(payload string) string {
	return "header." + base64.RawURLEncoding.EncodeToString([]byte(payload)) + ".signature"
}

func writeSSE(t *testing.T, writer io.Writer, events ...string) {
	t.Helper()
	for _, event := range events {
		if _, err := fmt.Fprintf(writer, "data: %s\n\n", event); err != nil {
			t.Errorf("write SSE event: %v", err)
			return
		}
	}
}

func requireModelError(t *testing.T, err error, kind llm.ErrorKind, op string) *llm.Error {
	t.Helper()
	if err == nil {
		t.Fatalf("error = nil, want %s", kind)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) {
		t.Fatalf("error = %T %v, want *llm.Error", err, err)
	}
	if modelErr.Kind != kind || modelErr.Op != op {
		t.Fatalf("error = %#v, want kind %s and op %q", modelErr, kind, op)
	}
	return modelErr
}
