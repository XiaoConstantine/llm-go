package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/anthropic"
	"github.com/XiaoConstantine/llm-go/bedrock"
	"github.com/XiaoConstantine/llm-go/gemini"
	"github.com/XiaoConstantine/llm-go/mistral"
	"github.com/XiaoConstantine/llm-go/openai"
	"github.com/XiaoConstantine/llm-go/openai/azure"
	"github.com/XiaoConstantine/llm-go/openai/codex"
	"github.com/XiaoConstantine/llm-go/openai/responses"
	"github.com/XiaoConstantine/llm-go/vertex"
)

// extra decodes a JSON object into provider extra fields.
func extra(object string) map[string]json.RawMessage {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(object), &fields); err != nil {
		panic(err)
	}
	return fields
}

func TestRequestOptionsCloneOwnsExtraFields(t *testing.T) {
	for name, options := range map[string]llm.ProviderOptions{
		"anthropic": &anthropic.RequestOptions{ServiceTier: "auto", ExtraFields: extra(`{"vendor_flag":[1]}`)},
		"openai":    &openai.RequestOptions{ServiceTier: "auto", ExtraFields: extra(`{"vendor_flag":[1]}`)},
	} {
		var original, clone map[string]json.RawMessage
		switch owned := options.CloneProviderOptions().(type) {
		case *anthropic.RequestOptions:
			original, clone = options.(*anthropic.RequestOptions).ExtraFields, owned.ExtraFields
		case *openai.RequestOptions:
			original, clone = options.(*openai.RequestOptions).ExtraFields, owned.ExtraFields
		}
		if string(clone["vendor_flag"]) != "[1]" {
			t.Fatalf("%s: cloned extra fields = %s", name, clone)
		}
		clone["vendor_flag"][1] = '2'
		if string(original["vendor_flag"]) != "[1]" {
			t.Fatalf("%s: clone aliased extra fields", name)
		}
	}
}

// conformanceOptions is a provider options type that no adapter owns.
type conformanceOptions struct{}

func (o *conformanceOptions) CloneProviderOptions() llm.ProviderOptions { return o }

type adapterFailure struct {
	name, body, retryHeader string
	status                  int
	kind                    llm.ErrorKind
	retryAfter              time.Duration
	inBand                  bool
	attempts                int
}

// Exercise public constructors and both Generator paths, including adapters
// which deliver an HTTP failure on Recv rather than from Stream itself.
func TestAdapterConformance(t *testing.T) {
	t.Setenv("AWS_MAX_ATTEMPTS", "1")
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("AWS_BEARER_TOKEN_BEDROCK", "")
	adapters := []struct {
		name, protocol string
		new            func(*httptest.Server, []llm.Capability) (llm.Generator, error)
	}{
		{"anthropic", "anthropic", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return anthropic.New(anthropic.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
		{"bedrock", "bedrock", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return bedrock.New(bedrock.Config{Model: "model", Region: "us-east-1", SkipAuth: true, BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
		{"gemini", "google", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return gemini.New(gemini.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
		{"vertex", "google", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return vertex.New(vertex.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
		{"mistral", "mistral", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return mistral.New(mistral.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
		{"openai", "openai", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return openai.New(openai.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
		{"responses", "responses", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return responses.New(responses.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
		{"azure", "responses", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return azure.New(azure.Config{Model: "model", APIKey: "key", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
		{"codex", "responses", func(s *httptest.Server, caps []llm.Capability) (llm.Generator, error) {
			return codex.New(codex.Config{Model: "model", AccessToken: "token", AccountID: "account", BaseURL: s.URL, HTTPClient: s.Client(), Capabilities: caps})
		}},
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
	for _, adapter := range adapters {
		t.Run(adapter.name, func(t *testing.T) {
			for _, streaming := range []bool{false, true} {
				op := "generate"
				if streaming {
					op = "stream"
				}
				t.Run(op, func(t *testing.T) {
					for _, test := range conformanceFailures(adapter.protocol, adapter.name != "openai" || streaming) {
						t.Run(test.name, func(t *testing.T) {
							var attempts atomic.Int32
							server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
								attempts.Add(1)
								w.Header().Set("Content-Type", "application/json")
								if test.inBand {
									w.Header().Set("Content-Type", "text/event-stream")
								}
								if test.retryHeader != "" {
									w.Header().Set("Retry-After", test.retryHeader)
								}
								w.WriteHeader(test.status)
								_, _ = io.WriteString(w, test.body)
							}))
							defer server.Close()
							client, err := adapter.new(server, []llm.Capability{llm.CapabilityStreaming})
							if err != nil {
								t.Fatal(err)
							}
							if test.attempts > 0 {
								client, err = llm.WithRetry(client, llm.RetryPolicy{MaxAttempts: 2, InitialBackoff: time.Nanosecond})
								if err != nil {
									t.Fatal(err)
								}
							}
							err = conformanceCall(t, client, context.Background(), request, streaming)
							if test.attempts > 0 && attempts.Load() != int32(test.attempts) {
								t.Errorf("attempts = %d, want %d", attempts.Load(), test.attempts)
							}
							var modelErr *llm.Error
							wantStatus := test.status
							if test.inBand {
								wantStatus = 0 // An event error does not assert an HTTP failure status.
							}
							if !errors.As(err, &modelErr) || modelErr.Kind != test.kind || modelErr.HTTPStatus != wantStatus || modelErr.Provider != client.Info().Provider {
								t.Fatalf("error = %v (%#v), want %v HTTP %d provider %q", err, modelErr, test.kind, wantStatus, client.Info().Provider)
							}
							if modelErr.RetryAfter != test.retryAfter {
								t.Fatalf("RetryAfter = %v, want %v", modelErr.RetryAfter, test.retryAfter)
							}
						})
					}
					t.Run("ordering and cancellation causes", func(t *testing.T) {
						var calls atomic.Int32
						server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
							calls.Add(1)
							w.WriteHeader(http.StatusInternalServerError)
						}))
						defer server.Close()
						client, err := adapter.new(server, nil)
						if err != nil {
							t.Fatal(err)
						}
						for _, cause := range []error{errors.New("caller stopped"), fmt.Errorf("caller stopped: %w", context.Canceled)} {
							ctx, cancel := context.WithCancelCause(context.Background())
							cancel(cause)
							invalid := conformanceCall(t, client, ctx, llm.Request{}, streaming)
							var modelErr *llm.Error
							if !errors.As(invalid, &modelErr) || modelErr.Kind != llm.KindInvalidRequest || modelErr.Op != "validate" || errors.Is(invalid, context.Canceled) {
								t.Fatalf("validation must precede cancellation: %v", invalid)
							}
							unsupported := request
							unsupported.ResponseFormat = llm.ResponseFormatJSON
							// Valid neutrally, but outside several adapters' temperature range:
							// capability checks must precede model-specific validation.
							temperature := 100.0
							unsupported.Temperature = &temperature
							err = conformanceCall(t, client, ctx, unsupported, streaming)
							if !errors.Is(err, context.Canceled) || !errors.Is(err, cause) {
								t.Fatalf("cancellation must preserve %v and precede capability checks: %v", cause, err)
							}
							// A valid but unsupported request must fail before I/O too.
							err = conformanceCall(t, client, context.Background(), unsupported, streaming)
							if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported {
								t.Fatalf("unsupported request = %v", err)
							}
						}
						if calls.Load() != 0 {
							t.Fatalf("preflight made %d HTTP requests", calls.Load())
						}
					})
					t.Run("parallel tool calls", func(t *testing.T) {
						// Adapters either send the control or reject it before I/O;
						// none may drop it silently.
						wire := map[string]string{
							"anthropic": `"disable_parallel_tool_use":true`,
							"openai":    `"parallel_tool_calls":false`,
							"responses": `"parallel_tool_calls":false`,
							"azure":     `"parallel_tool_calls":false`,
							"codex":     `"parallel_tool_calls":false`,
						}[adapter.name]
						var calls atomic.Int32
						var body atomic.Pointer[string]
						server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							calls.Add(1)
							data, _ := io.ReadAll(r.Body)
							text := string(data)
							body.Store(&text)
							w.WriteHeader(http.StatusInternalServerError)
						}))
						defer server.Close()
						client, err := adapter.new(server, []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools})
						if err != nil {
							t.Fatal(err)
						}
						sequential := request
						sequential.Tools = []llm.Tool{{Name: "lookup", InputSchema: []byte(`{"type":"object"}`)}}
						sequential.ParallelToolCalls = new(false)
						err = conformanceCall(t, client, context.Background(), sequential, streaming)
						var modelErr *llm.Error
						if wire == "" {
							if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported || calls.Load() != 0 {
								t.Fatalf("error = %v after %d HTTP requests, want unsupported before I/O", err, calls.Load())
							}
							return
						}
						if sent := body.Load(); sent == nil || !strings.Contains(*sent, wire) {
							t.Fatalf("request body = %v (error %v), want %s", sent, err, wire)
						}
					})
					t.Run("document parts", func(t *testing.T) {
						// Adapters either send the document or reject it before I/O;
						// none may drop it silently.
						wire := map[string]string{
							"anthropic": `"type":"document"`,
							"openai":    `"type":"file"`,
							"responses": `"type":"input_file"`,
							"azure":     `"type":"input_file"`,
						}[adapter.name]
						document := llm.Part{Kind: llm.PartFile, Data: []byte("%PDF-1"), MediaType: "application/pdf"}
						inUser := request
						inUser.Messages = []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{document, {Kind: llm.PartText, Text: "summarize"}}}}
						inSystem := request
						inSystem.Messages = []llm.Message{{Role: llm.RoleSystem, Content: []llm.Part{document}}, {Role: llm.RoleUser, Content: []llm.Part{{Kind: llm.PartText, Text: "summarize"}}}}
						declared := []llm.Capability{llm.CapabilityStreaming, llm.CapabilityDocuments}
						for _, test := range []struct {
							name         string
							capabilities []llm.Capability
							request      llm.Request
							sent         bool
						}{
							{"user message", declared, inUser, wire != ""},
							{"undeclared capability", []llm.Capability{llm.CapabilityStreaming}, inUser, false},
							{"system message", declared, inSystem, false},
						} {
							t.Run(test.name, func(t *testing.T) {
								var calls atomic.Int32
								var body atomic.Pointer[string]
								server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
									calls.Add(1)
									data, _ := io.ReadAll(r.Body)
									text := string(data)
									body.Store(&text)
									w.WriteHeader(http.StatusInternalServerError)
								}))
								defer server.Close()
								client, err := adapter.new(server, test.capabilities)
								if err != nil {
									if wire == "" {
										return // The adapter cannot declare the capability at all.
									}
									t.Fatal(err)
								}
								err = conformanceCall(t, client, context.Background(), test.request, streaming)
								if !test.sent {
									var modelErr *llm.Error
									if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported || calls.Load() != 0 {
										t.Fatalf("error = %v after %d HTTP requests, want unsupported before I/O", err, calls.Load())
									}
									return
								}
								if sent := body.Load(); sent == nil || !strings.Contains(*sent, wire) || !strings.Contains(*sent, "JVBERi0x") {
									t.Fatalf("request body = %v (error %v), want %s with the document bytes", sent, err, wire)
								}
							})
						}
					})
					t.Run("provider options", func(t *testing.T) {
						// An adapter sends its own options and rejects any other
						// adapter's options before I/O; none may drop them silently.
						type optionsCase struct {
							name    string
							options llm.ProviderOptions
							wire    string // expected in the body; empty means rejected
							invalid bool   // rejected as an invalid request, not unsupported
							absent  string // must not be in the body
						}
						cases := []optionsCase{{name: "another adapter's options", options: &conformanceOptions{}}}
						switch adapter.name {
						case "anthropic":
							cases[0].options = &openai.RequestOptions{ServiceTier: "flex"}
							cases = append(cases,
								optionsCase{name: "own options", options: &anthropic.RequestOptions{ServiceTier: "standard_only"}, wire: `"service_tier":"standard_only"`},
								optionsCase{name: "nil own options", options: (*anthropic.RequestOptions)(nil), wire: `"messages"`, absent: "service_tier"},
								optionsCase{name: "extra field", options: &anthropic.RequestOptions{ExtraFields: extra(`{"vendor_flag":{"on":true}}`)}, wire: `,"vendor_flag":{"on":true}}`},
								optionsCase{name: "extra field replaces stream", options: &anthropic.RequestOptions{ExtraFields: extra(`{"stream":true}`)}, invalid: true},
								optionsCase{name: "extra field replaces an omitted field", options: &anthropic.RequestOptions{ExtraFields: extra(`{"vendor_flag":1,"thinking":{"type":"enabled"}}`)}, invalid: true},
								optionsCase{name: "extra field is not JSON", options: &anthropic.RequestOptions{ExtraFields: map[string]json.RawMessage{"vendor_flag": []byte(`1,"stream":true`)}}, invalid: true})
						case "openai":
							cases[0].options = &anthropic.RequestOptions{ServiceTier: "auto"}
							cases = append(cases,
								optionsCase{name: "own options", options: &openai.RequestOptions{ServiceTier: "flex"}, wire: `"service_tier":"flex"`},
								optionsCase{name: "nil own options", options: (*openai.RequestOptions)(nil), wire: `"messages"`, absent: "service_tier"},
								optionsCase{name: "extra field", options: &openai.RequestOptions{ExtraFields: extra(`{"vendor_flag":{"on":true}}`)}, wire: `,"vendor_flag":{"on":true}}`},
								optionsCase{name: "extra field replaces stream", options: &openai.RequestOptions{ExtraFields: extra(`{"stream":true}`)}, invalid: true},
								optionsCase{name: "extra field replaces an omitted field", options: &openai.RequestOptions{ExtraFields: extra(`{"vendor_flag":1,"stream_options":{}}`)}, invalid: true},
								optionsCase{name: "extra field is not JSON", options: &openai.RequestOptions{ExtraFields: map[string]json.RawMessage{"vendor_flag": []byte(`1,"stream":true`)}}, invalid: true})
						}
						for _, test := range cases {
							t.Run(test.name, func(t *testing.T) {
								var calls atomic.Int32
								var body atomic.Pointer[string]
								server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
									calls.Add(1)
									data, _ := io.ReadAll(r.Body)
									text := string(data)
									body.Store(&text)
									w.WriteHeader(http.StatusInternalServerError)
								}))
								defer server.Close()
								client, err := adapter.new(server, []llm.Capability{llm.CapabilityStreaming})
								if err != nil {
									t.Fatal(err)
								}
								withOptions := request
								withOptions.ProviderOptions = test.options
								err = conformanceCall(t, client, context.Background(), withOptions, streaming)
								if test.wire == "" {
									want := llm.KindUnsupported
									if test.invalid {
										want = llm.KindInvalidRequest
									}
									var modelErr *llm.Error
									if !errors.As(err, &modelErr) || modelErr.Kind != want || calls.Load() != 0 {
										t.Fatalf("error = %v after %d HTTP requests, want %v before I/O", err, calls.Load(), want)
									}
									return
								}
								sent := body.Load()
								if sent == nil || !strings.Contains(*sent, test.wire) || (test.absent != "" && strings.Contains(*sent, test.absent)) {
									t.Fatalf("request body = %v (error %v), want %s", sent, err, test.wire)
								}
							})
						}
					})
					t.Run("in-flight cancellation", func(t *testing.T) {
						ctx, stop := context.WithTimeout(t.Context(), 5*time.Second)
						defer stop()
						ctx, cancel := context.WithCancelCause(ctx)
						defer cancel(nil)
						started := make(chan struct{})
						server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
							_, _ = io.Copy(io.Discard, r.Body)
							close(started)
							<-r.Context().Done()
						}))
						defer server.Close()
						client, err := adapter.new(server, []llm.Capability{llm.CapabilityStreaming})
						if err != nil {
							t.Fatal(err)
						}
						cause := fmt.Errorf("in-flight caller stopped: %w", context.Canceled)
						joined := make(chan struct{})
						go func() {
							defer close(joined)
							select {
							case <-started:
								cancel(cause)
							case <-ctx.Done():
							}
						}()
						err = conformanceCall(t, client, ctx, request, streaming)
						cancel(nil)
						<-joined
						if !errors.Is(err, cause) || !errors.Is(err, context.Canceled) {
							t.Fatalf("in-flight cancellation lost cause: %v", err)
						}
					})
				})
			}
		})
	}
}

func conformanceCall(t *testing.T, client llm.Generator, ctx context.Context, request llm.Request, streaming bool) error {
	t.Helper()
	if !streaming {
		response, err := client.Generate(ctx, request)
		if response != nil {
			t.Fatalf("Generate returned a response on a failure path: %#v, %v", response, err)
		}
		return err
	}
	stream, err := client.Stream(ctx, request)
	if stream != nil {
		defer func() { _ = stream.Close() }()
	}
	if err != nil {
		if stream != nil {
			t.Fatalf("Stream returned both a stream and an error: %v", err)
		}
		return err
	}
	if stream == nil {
		t.Fatal("Stream returned nil without an error")
	}
	for {
		if _, err := stream.Recv(); err != nil {
			return err
		}
	}
}

func conformanceErrorBody(protocol string, status int) string {
	message := "bad key"
	switch status {
	case 429:
		message = "slow down"
	case 400:
		message = "maximum context length exceeded: prompt is too long"
	}
	switch protocol {
	case "google":
		return fmt.Sprintf(`{"error":{"code":%d,"message":%q,"status":"INVALID_ARGUMENT"}}`, status, message)
	case "bedrock":
		code := "UnrecognizedClientException"
		switch status {
		case 429:
			code = "ThrottlingException"
		case 400:
			code = "ValidationException"
			message = "Input is too long for requested model."
		}
		return fmt.Sprintf(`{"__type":%q,"message":%q}`, code, message)
	case "mistral":
		return fmt.Sprintf(`{"message":%q}`, message)
	default:
		return fmt.Sprintf(`{"error":{"type":"invalid_request_error","message":%q}}`, message)
	}
}

func conformanceFailures(protocol string, eventStream bool) []adapterFailure {
	tests := []adapterFailure{
		{name: "authentication", status: 401, kind: llm.KindAuthentication},
		{name: "rate limit", status: 429, kind: llm.KindRateLimit, retryHeader: "7", retryAfter: 7 * time.Second},
		{name: "oversized retry delay stops retries", status: 429, kind: llm.KindRateLimit,
			retryHeader: "99999999999999999999", retryAfter: time.Duration(1<<63 - 1), attempts: 1},
		{name: "context limit", status: 400, kind: llm.KindContextLimit},
	}
	for i := range tests {
		tests[i].body = conformanceErrorBody(protocol, tests[i].status)
	}
	switch protocol {
	case "google":
		tests = append(tests,
			adapterFailure{name: "invalid API key", status: 400, kind: llm.KindAuthentication,
				body: `{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"API key not valid","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","domain":"googleapis.com","reason":"API_KEY_INVALID"}]}}`},
			adapterFailure{name: "unrelated error detail", status: 400, kind: llm.KindInvalidRequest,
				body: `{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"invalid request","details":[{"@type":"example.com/Other","domain":"googleapis.com","reason":"API_KEY_INVALID"}]}}`},
			adapterFailure{name: "wrong error domain", status: 400, kind: llm.KindInvalidRequest,
				body: `{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"invalid request","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","domain":"example.com","reason":"API_KEY_INVALID"}]}}`},
			adapterFailure{name: "wrong error reason", status: 400, kind: llm.KindInvalidRequest,
				body: `{"error":{"code":400,"status":"INVALID_ARGUMENT","message":"invalid request","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","domain":"googleapis.com","reason":"SOMETHING_ELSE"}]}}`},
		)
		for _, header := range []string{"", "3", "9"} {
			want := 7 * time.Second
			if header == "9" {
				want = 9 * time.Second
			}
			tests = append(tests, adapterFailure{name: "RetryInfo with header " + header, status: 429, kind: llm.KindRateLimit,
				retryHeader: header, retryAfter: want,
				body: `{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"busy","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"7s"}]}}`})
		}
		for _, test := range []struct {
			delay string
			want  time.Duration
		}{
			{"7.125s", 7125 * time.Millisecond},
			{"0.000000001s", time.Nanosecond},
			{"9223372036.854775807s", time.Duration(1<<63 - 1)},
			{"9223372036.854775808s", time.Duration(1<<63 - 1)},
			{"315576000000s", time.Duration(1<<63 - 1)},
			{"315576000001s", 0}, // Outside protobuf's 10,000-year range.
			{"-7s", 0}, {"7m", 0}, {"1m2s", 0}, {"7ms", 0},
			{"7.0000000001s", 0}, {"7e1s", 0}, {" 7s ", 0},
		} {
			tests = append(tests, adapterFailure{name: "RetryInfo duration " + test.delay, status: 429, kind: llm.KindRateLimit,
				retryAfter: test.want,
				body:       fmt.Sprintf(`{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","message":"busy","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":%q}]}}`, test.delay)})
		}
	case "bedrock":
		tests = append(tests,
			adapterFailure{name: "prompt token limit", status: 400, kind: llm.KindContextLimit,
				body: `{"__type":"ValidationException","message":"prompt is too long: 215000 tokens > 200000 maximum"}`},
			adapterFailure{name: "throttling takes precedence", status: 429, kind: llm.KindRateLimit,
				body: `{"__type":"ThrottlingException","message":"context token rate limit exceeded"}`},
			adapterFailure{name: "model error context limit", status: 424, kind: llm.KindContextLimit,
				body: `{"__type":"ModelErrorException","message":"input exceeds context length"}`},
			adapterFailure{name: "model stream error context limit", status: 424, kind: llm.KindContextLimit,
				body: `{"__type":"ModelStreamErrorException","message":"maximum context length is 8192 tokens"}`},
			adapterFailure{name: "internal error context limit", status: 500, kind: llm.KindContextLimit,
				body: `{"__type":"InternalServerException","message":"context length"}`},
			adapterFailure{name: "output parameter limit", status: 400, kind: llm.KindInvalidRequest,
				body: `{"__type":"ValidationException","message":"max_tokens exceeds the limit of 4096"}`},
			adapterFailure{name: "authentication takes precedence", status: 401, kind: llm.KindAuthentication,
				body: `{"__type":"UnrecognizedClientException","message":"context length"}`},
			adapterFailure{name: "permission takes precedence", status: 403, kind: llm.KindPermission,
				body: `{"__type":"AccessDeniedException","message":"context length"}`},
		)
	case "openai", "responses", "mistral":
		if protocol != "mistral" && eventStream {
			for _, test := range []struct {
				code string
				kind llm.ErrorKind
			}{{"authentication_error", llm.KindAuthentication}, {"rate_limit_exceeded", llm.KindRateLimit},
				{"server_authorization_cache_failed", llm.KindProvider}, {"rate_limit_probe_failed", llm.KindProvider}} {
				if protocol == "responses" {
					switch test.code {
					case "server_authorization_cache_failed":
						test.kind = llm.KindAuthentication
					case "rate_limit_probe_failed":
						test.kind = llm.KindRateLimit
					}
				}
				tests = append(tests, adapterFailure{name: "event " + test.code, status: 200, kind: test.kind, inBand: true,
					body: fmt.Sprintf("data: {\"type\":\"error\",\"error\":{\"code\":%q,\"message\":\"failed\"}}\n\n", test.code)})
			}
		}
		if protocol == "responses" {
			for _, test := range []struct {
				errorType, code string
				kind            llm.ErrorKind
			}{
				{"authentication_error", "insufficient_quota", llm.KindAuthentication},
				{"permission_denied", "unauthorized", llm.KindAuthentication},
				{"rate_limit_error", "invalid_api_key", llm.KindAuthentication},
				{"invalid_request_error", "usage_limit_reached", llm.KindRateLimit},
				{"invalid_request_error", "rate_limit_exceeded", llm.KindRateLimit},
				{"invalid_request_error", "insufficient_quota", llm.KindRateLimit},
				{"invalid_request_error", "invalid_api_key", llm.KindAuthentication},
				{"usage_limit_reached", "", llm.KindRateLimit},
				{"", "usage_limit_reached", llm.KindRateLimit},
				{"insufficient_quota", "", llm.KindRateLimit},
			} {
				attempts := 1
				if test.kind == llm.KindRateLimit {
					attempts = 2
				}
				tests = append(tests, adapterFailure{name: "Responses event " + test.errorType + "/" + test.code, status: 200,
					kind: test.kind, inBand: true, attempts: attempts,
					body: fmt.Sprintf("data: {\"type\":\"error\",\"error\":{\"type\":%q,\"code\":%q,\"message\":\"failed\"}}\n\n", test.errorType, test.code)})
			}
		}
		for _, test := range []struct {
			status int
			kind   llm.ErrorKind
		}{{401, llm.KindAuthentication}, {403, llm.KindPermission}, {429, llm.KindRateLimit},
			{500, llm.KindProvider}, {404, llm.KindInvalidRequest}, {413, llm.KindContextLimit}, {422, llm.KindContextLimit}} {
			body := `{"error":{"code":"context_length_exceeded","message":"maximum context length exceeded"}}`
			if protocol == "mistral" {
				body = `{"message":"maximum context length exceeded"}`
			}
			tests = append(tests, adapterFailure{name: fmt.Sprintf("context text HTTP %d", test.status), status: test.status, kind: test.kind,
				body: body})
		}
	}
	return tests
}
