package bedrock

import (
	"context"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

func TestSDKRetryAttemptsAndHeaders(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, attempts := range []int{1, 2} {
			t.Run(fmt.Sprintf("stream=%t/attempts=%d", streaming, attempts), func(t *testing.T) {
				var calls atomic.Int32
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					n := calls.Add(1)
					if got, want := r.Header.Get("X-Attempt"), fmt.Sprint(n); got != want {
						t.Errorf("X-Attempt = %q, want %q", got, want)
					}
					w.Header().Set("Content-Type", "application/json")
					w.Header().Set("X-Amzn-Errortype", "ThrottlingException")
					w.WriteHeader(http.StatusTooManyRequests)
					_, _ = w.Write([]byte(`{"message":"slow down"}`))
				}))
				defer server.Close()
				// No HTTPClient or static Headers: attempt metadata must work with defaults.
				client := mustClient(t, Config{Model: "model", Region: "us-east-1", BaseURL: server.URL,
					SkipAuth: true, Capabilities: []llm.Capability{llm.CapabilityStreaming}})
				retrying, err := llm.WithRetry(client, llm.RetryPolicy{MaxAttempts: attempts, InitialBackoff: time.Nanosecond,
					Hook: func(_ context.Context, attempt llm.Attempt) (http.Header, error) {
						return http.Header{"X-Attempt": {fmt.Sprint(attempt.Number)}}, nil
					}})
				if err != nil {
					t.Fatal(err)
				}
				request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
				if streaming {
					stream, streamErr := retrying.Stream(context.Background(), request)
					if streamErr != nil {
						t.Fatal(streamErr)
					}
					_, err = llm.CollectStructural(stream)
				} else {
					_, err = retrying.Generate(context.Background(), request)
				}
				var modelErr *llm.Error
				if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindRateLimit {
					t.Fatalf("error = %v, want rate limit", err)
				}
				if got := calls.Load(); int(got) != attempts {
					t.Fatalf("HTTP attempts = %d, want %d", got, attempts)
				}
			})
		}
	}
}

func TestSDKDefaultClientCustomCA(t *testing.T) {
	for _, profile := range []bool{false, true} {
		t.Run(fmt.Sprintf("profile=%t", profile), func(t *testing.T) {
			server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				writeSDKResponse(t, w)
			}))
			defer server.Close()
			dir := t.TempDir()
			bundle := filepath.Join(dir, "ca.pem")
			if err := os.WriteFile(bundle, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw}), 0600); err != nil {
				t.Fatal(err)
			}
			config := Config{Model: "model", Region: "us-east-1", BaseURL: server.URL, SkipAuth: true}
			if profile {
				path := filepath.Join(dir, "config")
				if err := os.WriteFile(path, []byte("[profile custom]\nca_bundle = "+bundle+"\n"), 0600); err != nil {
					t.Fatal(err)
				}
				t.Setenv("AWS_CA_BUNDLE", "")
				t.Setenv("AWS_CONFIG_FILE", path)
				config.Profile = "custom"
			} else {
				t.Setenv("AWS_CA_BUNDLE", bundle)
			}
			client := mustClient(t, config)
			response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil || response.Text() != "ok" {
				t.Fatalf("custom CA Generate = %v, %v", response, err)
			}
		})
	}
}

func TestSDKHeadersDoNotReachCredentials(t *testing.T) {
	for _, customClient := range []bool{false, true} {
		t.Run(fmt.Sprintf("customClient=%t", customClient), func(t *testing.T) {
			var credentialCalls atomic.Int32
			imds := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				credentialCalls.Add(1)
				if r.Header.Get("X-Static") != "" || r.Header.Get("X-Attempt") != "" {
					t.Errorf("caller headers reached credential endpoint %s", r.URL.Path)
				}
				switch r.URL.Path {
				case "/latest/api/token":
					_, _ = io.WriteString(w, "imds-token")
				case "/latest/meta-data/iam/security-credentials/":
					_, _ = io.WriteString(w, "test-role")
				case "/latest/meta-data/iam/security-credentials/test-role":
					_, _ = fmt.Fprintf(w, `{"Code":"Success","AccessKeyId":"test-key","SecretAccessKey":"test-secret","Token":"test-token","Expiration":%q}`, time.Now().Add(time.Hour).UTC().Format(time.RFC3339))
				default:
					t.Errorf("unexpected credential request %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer imds.Close()
			useTestIMDS(t, imds.URL)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("X-Static") != "static" || r.Header.Get("X-Attempt") != "attempt" {
					t.Error("caller headers missing from Bedrock request")
				}
				writeSDKResponse(t, w)
			}))
			defer server.Close()
			config := Config{Model: "model", Region: "us-east-1", BaseURL: server.URL, Headers: http.Header{"X-Static": {"static"}}}
			if customClient {
				config.HTTPClient = server.Client()
			}
			client := mustClient(t, config)
			retrying, err := llm.WithRetry(client, llm.RetryPolicy{MaxAttempts: 1, Hook: func(context.Context, llm.Attempt) (http.Header, error) {
				return http.Header{"X-Attempt": {"attempt"}}, nil
			}})
			if err != nil {
				t.Fatal(err)
			}
			response, err := retrying.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
			if err != nil || response.Text() != "ok" || credentialCalls.Load() == 0 {
				t.Fatalf("Generate = %v, %v; credential calls = %d", response, err, credentialCalls.Load())
			}
		})
	}
}

func TestSDKIMDSFailureDoesNotAcquireRetryHint(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", streaming), func(t *testing.T) {
			var imdsCalls, bedrockCalls, attempts atomic.Int32
			imds := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				imdsCalls.Add(1)
				switch r.URL.Path {
				case "/latest/api/token":
					_, _ = io.WriteString(w, "imds-token")
				case "/latest/meta-data/iam/security-credentials/":
					w.WriteHeader(http.StatusUnauthorized)
				default:
					t.Errorf("unexpected credential request %s", r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer imds.Close()
			useTestIMDS(t, imds.URL)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				bedrockCalls.Add(1)
				w.WriteHeader(http.StatusInternalServerError)
			}))
			defer server.Close()
			client := mustClient(t, Config{Model: "model", Region: "us-east-1", BaseURL: server.URL,
				Capabilities: []llm.Capability{llm.CapabilityStreaming}})
			retrying, err := llm.WithRetry(client, llm.RetryPolicy{MaxAttempts: 3, InitialBackoff: time.Nanosecond,
				Hook: func(context.Context, llm.Attempt) (http.Header, error) {
					attempts.Add(1)
					return nil, nil
				}})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
			if streaming {
				stream, openErr := retrying.Stream(ctx, request)
				if openErr != nil {
					t.Fatal(openErr)
				}
				_, err = llm.CollectStructural(stream)
			} else {
				_, err = retrying.Generate(ctx, request)
			}
			var modelErr *llm.Error
			if !errors.As(err, &modelErr) || modelErr.Retryable != nil {
				t.Errorf("credential failure acquired retry hint: %v", err)
			}
			// IMDS still owns its three token/role attempts; llm.WithRetry must
			// not multiply those attempts or reach Bedrock without credentials.
			if attempts.Load() != 1 || imdsCalls.Load() != 6 || bedrockCalls.Load() != 0 {
				t.Errorf("attempts=%d IMDS calls=%d Bedrock calls=%d, want 1/6/0", attempts.Load(), imdsCalls.Load(), bedrockCalls.Load())
			}
		})
	}
}

func useTestIMDS(t *testing.T, endpoint string) {
	t.Helper()
	for _, name := range []string{"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK", "AWS_BEDROCK_SKIP_AUTH", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"} {
		t.Setenv(name, "")
	}
	t.Setenv("AWS_CONFIG_FILE", filepath.Join(t.TempDir(), "missing-config"))
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(t.TempDir(), "missing-credentials"))
	t.Setenv("AWS_EC2_METADATA_DISABLED", "false")
	t.Setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", endpoint)
}

func TestSDKClockSkewRetryHint(t *testing.T) {
	for _, skew := range []time.Duration{0, 2 * time.Hour} {
		for _, streaming := range []bool{false, true} {
			for _, attempts := range []int{1, 2} {
				t.Run(fmt.Sprintf("skew=%s/stream=%t/attempts=%d", skew, streaming, attempts), func(t *testing.T) {
					var calls atomic.Int32
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						n := calls.Add(1)
						if r.Header.Get("X-Attempt") != fmt.Sprint(n) {
							t.Errorf("SDK made hidden attempt %d", n)
						}
						serverTime := time.Now().Add(skew)
						w.Header().Set("Date", serverTime.UTC().Format(http.TimeFormat))
						if n == 1 || skew == 0 {
							w.Header().Set("Content-Type", "application/json")
							w.Header().Set("X-Amzn-Errortype", "InvalidSignatureException")
							w.WriteHeader(http.StatusForbidden)
							_, _ = io.WriteString(w, `{"message":"bad signature"}`)
							return
						}
						signedAt, err := time.Parse("20060102T150405Z", r.Header.Get("X-Amz-Date"))
						if err != nil || serverTime.Sub(signedAt).Abs() > time.Minute {
							t.Errorf("retry signing time was not corrected: %v", err)
						}
						writeSDKResponse(t, w)
					}))
					defer server.Close()
					client := mustClient(t, Config{Model: "model", Region: "us-east-1", BaseURL: server.URL, SkipAuth: true, Capabilities: []llm.Capability{llm.CapabilityStreaming}})
					retrying, err := llm.WithRetry(client, llm.RetryPolicy{MaxAttempts: attempts, InitialBackoff: time.Nanosecond,
						Hook: func(_ context.Context, attempt llm.Attempt) (http.Header, error) {
							return http.Header{"X-Attempt": {fmt.Sprint(attempt.Number)}}, nil
						}})
					if err != nil {
						t.Fatal(err)
					}
					request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
					var response *llm.Response
					if streaming {
						stream, openErr := retrying.Stream(context.Background(), request)
						if openErr != nil {
							t.Fatal(openErr)
						}
						response, err = llm.CollectStructural(stream)
					} else {
						response, err = retrying.Generate(context.Background(), request)
					}
					if skew != 0 && attempts == 2 {
						if err != nil || response.Text() != "ok" || calls.Load() != 2 {
							t.Fatalf("skew retry = %v, %v; HTTP attempts=%d", response, err, calls.Load())
						}
					} else {
						var modelErr *llm.Error
						if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindAuthentication || calls.Load() != 1 {
							t.Fatalf("authentication error = %v; HTTP attempts=%d", err, calls.Load())
						}
						if got := modelErr.Retryable != nil && *modelErr.Retryable; got != (skew != 0) {
							t.Fatalf("retry hint=%t, want %t", got, skew != 0)
						}
						var apiErr smithy.APIError
						if !errors.As(err, &apiErr) || apiErr.ErrorCode() != "InvalidSignatureException" {
							t.Fatal("SDK error lost from error chain")
						}
					}
				})
			}
		}
	}
}

func writeSDKResponse(t *testing.T, w http.ResponseWriter) {
	t.Helper()
	w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
	encoder := eventstream.NewEncoder()
	for _, event := range []struct{ kind, payload string }{
		{"messageStart", `{"role":"assistant"}`},
		{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"ok"}}`},
		{"contentBlockStop", `{"contentBlockIndex":0}`},
		{"messageStop", `{"stopReason":"end_turn"}`},
	} {
		err := encoder.Encode(w, eventstream.Message{Headers: eventstream.Headers{
			{Name: ":message-type", Value: eventstream.StringValue("event")},
			{Name: ":event-type", Value: eventstream.StringValue(event.kind)},
			{Name: ":content-type", Value: eventstream.StringValue("application/json")},
		}, Payload: []byte(event.payload)})
		if err != nil {
			t.Error(err)
			return
		}
	}
}

func TestStreamCloseWithContextIgnoringRuntime(t *testing.T) {
	events := make(chan types.ConverseStreamOutput)
	started := make(chan struct{})
	closed := make(chan struct{})
	client := mustClient(t, Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityStreaming},
		Runtime: stalledRuntime{events: events, started: started, closed: closed}})
	stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	done := make(chan error, 1)
	go func() { done <- stream.Close() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		close(events) // Unblock the old implementation without leaking the test goroutine.
		t.Fatalf("Stream.Close blocked on a runtime that ignores cancellation; after unblocking it returned %v", <-done)
	}
	select {
	case <-closed:
	default:
		t.Fatal("Stream.Close returned before the runtime stream was closed")
	}
}

type stalledRuntime struct {
	events  chan types.ConverseStreamOutput
	started chan struct{}
	closed  chan struct{}
}

func (r stalledRuntime) ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput) (EventStream, error) {
	return r, nil
}
func (r stalledRuntime) Events() <-chan types.ConverseStreamOutput {
	close(r.started)
	return r.events
}
func (r stalledRuntime) Close() error { close(r.closed); return nil }
func (r stalledRuntime) Err() error   { return nil }

func TestGenerateTextReasoningUsageAndReplay(t *testing.T) {
	runtime := &fakeRuntime{events: []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
		deltaEvent(0, &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberText{Value: "think"}}),
		deltaEvent(0, &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberSignature{Value: "sig"}}),
		stopEvent(0),
		deltaEvent(1, &types.ContentBlockDeltaMemberText{Value: "answer"}),
		stopEvent(1),
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn}},
		&types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{Usage: &types.TokenUsage{
			InputTokens: aws.Int32(4), OutputTokens: aws.Int32(3), CacheReadInputTokens: aws.Int32(2), TotalTokens: aws.Int32(9)}}},
	}}
	client := mustClient(t, Config{Model: "anthropic.claude-sonnet-4-6", Reasoning: true, Runtime: runtime})
	response, err := client.Generate(context.Background(), llm.Request{ReasoningEffort: llm.ReasoningEffortHigh,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "question"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "answer" || response.ReasoningSummary != "think" || response.FinishReason != llm.FinishReasonStop ||
		response.Model != "anthropic.claude-sonnet-4-6" {
		t.Fatalf("response = %#v", response)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 3 ||
		response.Usage.CacheReadTokens != 2 || response.Usage.TotalTokens != 9 {
		t.Fatalf("usage = %#v", response.Usage)
	}
	data, ok := parseMessageData(response.Message.ProviderData, "anthropic.claude-sonnet-4-6")
	if !ok || len(data.Reasoning) != 1 || data.Reasoning[0].Text != "think" || data.Reasoning[0].Signature != "sig" {
		t.Fatalf("provider data = %#v, %v", data, ok)
	}

	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "question"}}}, response.Message}}
	input, err := buildInput("generate", "anthropic.claude-sonnet-4-6", true, request)
	if err != nil {
		t.Fatal(err)
	}
	assistant := input.Messages[1]
	if _, ok := assistant.Content[0].(*types.ContentBlockMemberReasoningContent); !ok {
		t.Fatalf("assistant replay = %#v", assistant.Content)
	}
	if text, ok := assistant.Content[1].(*types.ContentBlockMemberText); !ok || text.Value != "answer" {
		t.Fatalf("assistant replay order = %#v", assistant.Content)
	}
}

func TestStreamToolCallAndInputConversion(t *testing.T) {
	runtime := &fakeRuntime{events: []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
		&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(0),
			Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{ToolUseId: aws.String("call-1"), Name: aws.String("read")}}}},
		deltaEvent(0, &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(`{"path":`)}}),
		deltaEvent(0, &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(`"file"}`)}}),
		stopEvent(0),
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse}},
	}}
	client := mustClient(t, Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, Runtime: runtime})
	tools := []llm.Tool{{Name: "read", Description: "read a file", InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}}
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "read"}}}}, Tools: tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	recorded := &recordingStream{Stream: stream}
	response, err := llm.CollectStructural(recorded)
	if err != nil {
		t.Fatal(err)
	}
	for index, chunk := range recorded.chunks {
		if chunk.ID == "" && chunk.Model == "" && len(chunk.Content) == 0 && len(chunk.ToolCalls) == 0 &&
			chunk.ReasoningSummary == "" && len(chunk.ProviderData) == 0 && chunk.FinishReason == "" && chunk.Usage == nil && len(chunk.Events) == 0 {
			t.Fatalf("chunk %d is empty", index)
		}
	}
	if response.FinishReason != llm.FinishReasonToolCall || len(response.Message.ToolCalls) != 1 {
		t.Fatalf("response = %#v", response)
	}
	call := response.Message.ToolCalls[0]
	if call.ID != "call-1" || call.Name != "read" || string(call.Arguments) != `{"path":"file"}` {
		t.Fatalf("call = %#v", call)
	}
	if runtime.input == nil || runtime.input.ToolConfig == nil || len(runtime.input.ToolConfig.Tools) != 1 {
		t.Fatalf("input = %#v", runtime.input)
	}
}

func TestFilteredOutputPreservesText(t *testing.T) {
	for _, reason := range []types.StopReason{types.StopReasonContentFiltered, types.StopReasonGuardrailIntervened} {
		for _, scenario := range []string{"text", "completed tool", "open text", "open tool"} {
			for _, mode := range []string{"generate", "stream"} {
				t.Run(string(reason)+"/"+scenario+"/"+mode, func(t *testing.T) {
					events := []types.ConverseStreamOutput{
						&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
						deltaEvent(0, &types.ContentBlockDeltaMemberText{Value: "partial"}),
					}
					if scenario != "open text" {
						events = append(events, stopEvent(0))
					}
					if scenario == "completed tool" || scenario == "open tool" {
						arguments := `{"q":"x"}`
						if scenario == "open tool" {
							arguments = `{"q":`
						}
						events = append(events,
							&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(1),
								Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{ToolUseId: aws.String("call-1"), Name: aws.String("read")}}}},
							deltaEvent(1, &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(arguments)}}))
						if scenario == "completed tool" {
							events = append(events, stopEvent(1))
						}
					}
					events = append(events,
						&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: reason}},
						&types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{Usage: &types.TokenUsage{
							InputTokens: aws.Int32(4), OutputTokens: aws.Int32(3), TotalTokens: aws.Int32(7)}}})
					client := mustClient(t, Config{Model: "model", Runtime: &fakeRuntime{events: events}, Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}})
					request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}, Tools: []llm.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
					var response *llm.Response
					var err error
					if mode == "generate" {
						response, err = client.Generate(context.Background(), request)
					} else {
						stream, streamErr := client.Stream(context.Background(), request)
						if streamErr != nil {
							t.Fatal(streamErr)
						}
						recorded := &recordingStream{Stream: stream}
						response, err = llm.Collect(recorded, request.Tools)
						textEnds, toolEnds := 0, 0
						for _, chunk := range recorded.chunks {
							if len(chunk.ToolCalls) != 0 {
								t.Fatal("filtered stream emitted executable tool calls")
							}
							for _, event := range chunk.Events {
								if event.Kind == llm.StreamEventTextEnd {
									textEnds++
								}
								if event.Kind == llm.StreamEventToolCallEnd {
									toolEnds++
								}
								if event.ToolCall != nil {
									t.Fatal("filtered stream emitted executable tool event")
								}
							}
						}
						if textEnds != 1 || (scenario == "completed tool" || scenario == "open tool") && toolEnds != 1 {
							t.Fatalf("text ends = %d, tool ends = %d", textEnds, toolEnds)
						}
					}
					if err != nil {
						t.Fatal(err)
					}
					if response.Text() != "partial" || len(response.Message.ToolCalls) != 0 || response.FinishReason != llm.FinishReasonContentFilter || response.Usage == nil || response.Usage.TotalTokens != 7 {
						t.Fatalf("response = %#v", response)
					}
				})
			}
		}
	}
}

type recordingStream struct {
	llm.Stream
	chunks []llm.Chunk
}

func (s *recordingStream) Recv() (llm.Chunk, error) {
	chunk, err := s.Stream.Recv()
	if err == nil {
		s.chunks = append(s.chunks, chunk)
	}
	return chunk, err
}

func TestEmptyToolArgumentsRequireToolUse(t *testing.T) {
	for _, reason := range []types.StopReason{types.StopReasonToolUse, types.StopReasonMaxTokens} {
		for _, mode := range []string{"generate", "stream"} {
			t.Run(string(reason)+"/"+mode, func(t *testing.T) {
				runtime := &fakeRuntime{events: []types.ConverseStreamOutput{
					&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
					&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(0),
						Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{ToolUseId: aws.String("call-1"), Name: aws.String("read")}}}},
					stopEvent(0),
					&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: reason}},
				}}
				client := mustClient(t, Config{Model: "model", Runtime: runtime, Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}})
				request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "read"}}}},
					Tools: []llm.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
				var response *llm.Response
				var err error
				if mode == "generate" {
					response, err = client.Generate(context.Background(), request)
				} else {
					stream, streamErr := client.Stream(context.Background(), request)
					if streamErr != nil {
						t.Fatal(streamErr)
					}
					recorded := &recordingStream{Stream: stream}
					response, err = llm.Collect(recorded, request.Tools)
					ends := 0
					for _, chunk := range recorded.chunks {
						if reason == types.StopReasonMaxTokens && len(chunk.ToolCalls) != 0 {
							t.Fatal("truncated call became executable")
						}
						for _, event := range chunk.Events {
							if event.Kind == llm.StreamEventToolCallEnd {
								ends++
								if reason == types.StopReasonMaxTokens && event.ToolCall != nil {
									t.Fatal("truncated tool end became executable")
								}
							}
						}
					}
					if ends != 1 {
						t.Fatalf("tool ends = %d", ends)
					}
				}
				if err != nil {
					t.Fatal(err)
				}
				if reason == types.StopReasonMaxTokens {
					if response.FinishReason != llm.FinishReasonLength || len(response.Message.ToolCalls) != 0 {
						t.Fatalf("response = %#v", response)
					}
				} else if response.FinishReason != llm.FinishReasonToolCall || len(response.Message.ToolCalls) != 1 || string(response.Message.ToolCalls[0].Arguments) != "{}" {
					t.Fatalf("response = %#v", response)
				}
			})
		}
	}
}

func TestEmptyToolArguments(t *testing.T) {
	for _, arguments := range []string{"", "{}", `{"q":`} {
		for _, mode := range []string{"generate", "stream"} {
			t.Run(arguments+"/"+mode, func(t *testing.T) {
				runtime := &fakeRuntime{events: []types.ConverseStreamOutput{
					&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
					&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(0),
						Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{ToolUseId: aws.String("call-1"), Name: aws.String("read")}}}},
					deltaEvent(0, &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(arguments)}}),
					stopEvent(0),
					&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse}},
				}}
				client := mustClient(t, Config{Model: "model", Runtime: runtime, Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}})
				request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "read"}}}},
					Tools: []llm.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
				var response *llm.Response
				var err error
				if mode == "generate" {
					response, err = client.Generate(context.Background(), request)
				} else {
					stream, streamErr := client.Stream(context.Background(), request)
					if streamErr != nil {
						t.Fatal(streamErr)
					}
					response, err = llm.CollectStructural(stream)
				}
				if arguments == `{"q":` {
					var modelErr *llm.Error
					if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse {
						t.Fatalf("error = %v, want malformed response", err)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if response.FinishReason != llm.FinishReasonToolCall || len(response.Message.ToolCalls) != 1 || string(response.Message.ToolCalls[0].Arguments) != "{}" {
					t.Fatalf("response = %#v", response)
				}
			})
		}
	}
}

func TestBuildInputCachingReasoningImagesAndToolResults(t *testing.T) {
	request := llm.Request{CacheRetention: llm.CacheRetentionLong, ReasoningEffort: llm.ReasoningEffortMedium,
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.Part{{Text: "system"}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "look"}, {Kind: llm.PartImage, MediaType: "image/png", Data: []byte("png")}}},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "bad:id", Name: "read", Arguments: json.RawMessage(`{"path":"x"}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "bad:id", Name: "read", IsError: true, Content: []llm.Part{{Text: "failed"}}}}},
		}, Tools: []llm.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
	input, err := buildInput("generate", "anthropic.claude-sonnet-4-20250514-v1:0", true, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(input.System) != 2 || len(input.Messages) != 3 || input.ToolConfig == nil || input.AdditionalModelRequestFields == nil {
		t.Fatalf("input = %#v", input)
	}
	assistant := input.Messages[1].Content[0].(*types.ContentBlockMemberToolUse)
	if *assistant.Value.ToolUseId != "bad_id" {
		t.Fatalf("tool ID = %q", *assistant.Value.ToolUseId)
	}
	result := input.Messages[2].Content[0].(*types.ContentBlockMemberToolResult)
	if *result.Value.ToolUseId != "bad_id" || result.Value.Status != types.ToolResultStatusError {
		t.Fatalf("tool result = %#v", result.Value)
	}
}

func TestConsecutiveToolResultsAreMergedAndIDlessResultsPair(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{{Text: "run"}}},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"n":1}`)},
			{Name: "read", Arguments: json.RawMessage(`{"n":2}`)},
		}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{Name: "read", Content: []llm.Part{{Text: "one"}}}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{Name: "read", Content: []llm.Part{{Text: "two"}}}}},
	}}
	input, err := buildInput("generate", "model", false, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(input.Messages) != 3 || len(input.Messages[2].Content) != 2 {
		t.Fatalf("messages = %#v", input.Messages)
	}
	first := input.Messages[1].Content[0].(*types.ContentBlockMemberToolUse).Value.ToolUseId
	second := input.Messages[1].Content[1].(*types.ContentBlockMemberToolUse).Value.ToolUseId
	firstResult := input.Messages[2].Content[0].(*types.ContentBlockMemberToolResult).Value.ToolUseId
	secondResult := input.Messages[2].Content[1].(*types.ContentBlockMemberToolResult).Value.ToolUseId
	if *first == *second || *first != *firstResult || *second != *secondResult {
		t.Fatalf("tool IDs = %q/%q results %q/%q", *first, *second, *firstResult, *secondResult)
	}
}

func TestSDKErrorClassification(t *testing.T) {
	tests := []struct {
		code string
		kind llm.ErrorKind
	}{
		{"UnrecognizedClientException", llm.KindAuthentication},
		{"AccessDeniedException", llm.KindPermission},
		{"ThrottlingException", llm.KindRateLimit},
		{"ValidationException", llm.KindInvalidRequest},
		{"ServiceUnavailableException", llm.KindProvider},
	}
	for _, test := range tests {
		err := sdkError("generate", &smithy.GenericAPIError{Code: test.code, Message: "failure"})
		var modelErr *llm.Error
		if !errors.As(err, &modelErr) || modelErr.Kind != test.kind || modelErr.Op != "generate" {
			t.Errorf("%s = %v (%#v)", test.code, err, modelErr)
		}
	}
	contextErr := sdkError("generate", &smithy.GenericAPIError{Code: "ValidationException", Message: "input token count exceeds context length"})
	var modelErr *llm.Error
	if !errors.As(contextErr, &modelErr) || modelErr.Kind != llm.KindContextLimit {
		t.Fatalf("context error = %v (%#v)", contextErr, modelErr)
	}
	responseErr := &smithyhttp.ResponseError{Response: &smithyhttp.Response{Response: &http.Response{
		StatusCode: http.StatusTooManyRequests, Header: http.Header{"Retry-After": []string{"2"}},
	}}, Err: &smithy.GenericAPIError{Code: "ThrottlingException", Message: "slow down"}}
	err := sdkError("stream", responseErr)
	if !errors.As(err, &modelErr) || modelErr.HTTPStatus != http.StatusTooManyRequests || modelErr.RetryAfter != 2*time.Second {
		t.Fatalf("retry error = %v (%#v)", err, modelErr)
	}
}

func TestCachingRequiresExplicitRetention(t *testing.T) {
	t.Setenv("PI_CACHE_RETENTION", "long")
	t.Setenv("AWS_BEDROCK_FORCE_CACHE", "1")
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}}
	input, err := buildInput("generate", "application-profile-arn", false, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(input.Messages[0].Content) != 1 {
		t.Fatalf("default retention content = %#v, want no cache point", input.Messages[0].Content)
	}

	request.CacheRetention = llm.CacheRetentionLong
	input, err = buildInput("generate", "application-profile-arn", false, request)
	if err != nil {
		t.Fatal(err)
	}
	point, ok := input.Messages[0].Content[1].(*types.ContentBlockMemberCachePoint)
	if !ok || point.Value.Ttl != types.CacheTTLOneHour {
		t.Fatalf("long retention cache point = %#v", input.Messages[0].Content)
	}
}

func TestReasoningBudgetValidation(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}},
		ReasoningBudgetTokens: 512}
	if _, err := buildInput("generate", "anthropic.claude-3-7-sonnet", true, request); err == nil {
		t.Fatal("small reasoning budget succeeded")
	}
	request.ReasoningBudgetTokens, request.MaxOutputTokens = 2048, 2048
	if _, err := buildInput("generate", "anthropic.claude-3-7-sonnet", true, request); err == nil {
		t.Fatal("reasoning budget equal to max output succeeded")
	}
	request.ReasoningBudgetTokens, request.ReasoningEffort, request.MaxOutputTokens = 0, llm.ReasoningEffortHigh, 4096
	input, err := buildInput("generate", "anthropic.claude-3-7-sonnet", true, request)
	if err != nil {
		t.Fatal(err)
	}
	rawFields, err := input.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(rawFields, &fields); err != nil {
		t.Fatal(err)
	}
	thinking, ok := fields["thinking"].(map[string]any)
	if !ok || thinking["budget_tokens"] != float64(3072) {
		t.Fatalf("thinking fields = %#v", fields)
	}
	request.ReasoningBudgetTokens, request.MaxOutputTokens = 2048, 0
	if _, err := buildInput("generate", "anthropic.claude-sonnet-4-6", true, request); err == nil {
		t.Fatal("explicit adaptive reasoning budget succeeded")
	}
}

func TestCanceledContextPrecedesRuntime(t *testing.T) {
	runtime := &fakeRuntime{}
	client := mustClient(t, Config{Model: "model", Runtime: runtime})
	ctx, cancel := context.WithCancelCause(context.Background())
	cause := errors.New("stop")
	cancel(cause)
	response, err := client.Generate(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}})
	if response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || runtime.calls != 0 {
		t.Fatalf("Generate() = %#v, %v, calls=%d", response, err, runtime.calls)
	}
}

func TestPreflightAndMalformedStream(t *testing.T) {
	runtime := &fakeRuntime{}
	client := mustClient(t, Config{Model: "model", Runtime: runtime})
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}},
		ReasoningEffort: llm.ReasoningEffortHigh}
	if response, err := client.Generate(context.Background(), request); response != nil || err == nil || runtime.calls != 0 {
		t.Fatalf("Generate() = %#v, %v, calls=%d", response, err, runtime.calls)
	}

	runtime.events = []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
		deltaEvent(0, &types.ContentBlockDeltaMemberText{Value: "unterminated"}),
	}
	response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}})
	if response != nil || err == nil {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse {
		t.Fatalf("error = %v (%#v)", err, modelErr)
	}
}

func TestInfoOwnershipAndConfiguration(t *testing.T) {
	capabilities := []llm.Capability{llm.CapabilityStreaming}
	client := mustClient(t, Config{Provider: "custom-bedrock", Model: "model", Reasoning: true, Capabilities: capabilities, Runtime: &fakeRuntime{}})
	capabilities[0] = llm.CapabilityAudio
	info := client.Info()
	if info.Provider != "custom-bedrock" || info.API != llm.APIBedrockConverseStream || !info.Reasoning || info.Capabilities[1] != llm.CapabilityStreaming {
		t.Fatalf("Info() = %#v", info)
	}
	if _, err := New(Config{Runtime: &fakeRuntime{}}); err == nil {
		t.Fatal("empty model succeeded")
	}
	var nilRuntime *fakeRuntime
	if _, err := New(Config{Model: "model", Runtime: nilRuntime}); err == nil {
		t.Fatal("typed nil runtime succeeded")
	}
	if standardEndpointRegion("https://bedrock-runtime.us-west-2.amazonaws.com") != "us-west-2" ||
		shouldPinEndpoint("https://bedrock-runtime.us-east-1.amazonaws.com", "us-west-2", "") {
		t.Fatal("standard endpoint resolution is incorrect")
	}
	if got := normalizeToolID(strings.Repeat("a", 70)); len(got) != 64 {
		t.Fatalf("normalized ID length = %d", len(got))
	}
}

func mustClient(t *testing.T, config Config) *Client {
	t.Helper()
	client, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func deltaEvent(index int32, delta types.ContentBlockDelta) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(index), Delta: delta}}
}

func stopEvent(index int32) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(index)}}
}

type fakeRuntime struct {
	events []types.ConverseStreamOutput
	err    error
	input  *bedrockruntime.ConverseStreamInput
	calls  int
}

func (runtime *fakeRuntime) ConverseStream(_ context.Context, input *bedrockruntime.ConverseStreamInput) (EventStream, error) {
	runtime.calls++
	runtime.input = input
	if runtime.err != nil {
		return nil, runtime.err
	}
	channel := make(chan types.ConverseStreamOutput, len(runtime.events))
	for _, event := range runtime.events {
		channel <- event
	}
	close(channel)
	return &fakeEventStream{events: channel}, nil
}

type fakeEventStream struct {
	events chan types.ConverseStreamOutput
	err    error
	closed bool
}

func (stream *fakeEventStream) Events() <-chan types.ConverseStreamOutput { return stream.events }
func (stream *fakeEventStream) Err() error                                { return stream.err }
func (stream *fakeEventStream) Close() error {
	stream.closed = true
	return nil
}
