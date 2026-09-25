package llm_test

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/bedrock"
	"github.com/XiaoConstantine/llm-go/gemini"
	"github.com/XiaoConstantine/llm-go/mistral"
	"github.com/XiaoConstantine/llm-go/vertex"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

var errProtocolFixtureIO = errors.New("offline protocol fixture")

type protocolGuardTransport struct{ calls atomic.Int32 }

func (f *protocolGuardTransport) RoundTrip(*http.Request) (*http.Response, error) {
	f.calls.Add(1)
	return nil, errProtocolFixtureIO
}

func (f *protocolGuardTransport) ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput) (bedrock.EventStream, error) {
	f.calls.Add(1)
	return nil, errProtocolFixtureIO
}

func protocolGuardClient(t *testing.T, kind string, streaming bool) (llm.Generator, *protocolGuardTransport) {
	t.Helper()
	fixture := &protocolGuardTransport{}
	client := &http.Client{Transport: fixture}
	var capabilities []llm.Capability
	if streaming {
		capabilities = []llm.Capability{llm.CapabilityStreaming}
	}
	var generator llm.Generator
	var err error
	switch kind {
	case "gemini":
		generator, err = gemini.New(gemini.Config{Model: "fixture", APIKey: "fixture", Reasoning: true, BaseURL: "https://fixture.invalid", HTTPClient: client, Capabilities: capabilities})
	case "vertex":
		generator, err = vertex.New(vertex.Config{Model: "fixture", APIKey: "fixture", Reasoning: true, BaseURL: "https://fixture.invalid", HTTPClient: client, Capabilities: capabilities})
	case "bedrock":
		generator, err = bedrock.New(bedrock.Config{Model: "anthropic.claude-sonnet-4", Reasoning: true, Runtime: fixture, Capabilities: capabilities})
	case "mistral":
		generator, err = mistral.New(mistral.Config{Model: "mistral-small-latest", APIKey: "fixture", Reasoning: true, BaseURL: "https://fixture.invalid", HTTPClient: client, Capabilities: capabilities})
	default:
		t.Fatalf("unknown protocol fixture %q", kind)
	}
	if err != nil {
		t.Fatal(err)
	}
	return generator, fixture
}

func protocolGuardCall(t *testing.T, ctx context.Context, client llm.Generator, op string, request llm.Request) error {
	t.Helper()
	if op == "generate" {
		_, err := client.Generate(ctx, request)
		return err
	}
	stream, err := client.Stream(ctx, request)
	if err != nil {
		if stream != nil {
			t.Fatal("Stream returned a non-nil stream with an error")
		}
		return err
	}
	defer func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	}()
	_, err = stream.Recv()
	return err
}

func TestProtocolOptionsRejectUnsupportedAdaptersBeforeIO(t *testing.T) {
	for _, kind := range []string{"gemini", "vertex", "bedrock", "mistral"} {
		for _, op := range []string{"generate", "stream"} {
			for _, change := range []struct {
				name string
				set  func(*llm.Request)
			}{
				{"top_k_zero", func(r *llm.Request) { r.TopK = new(0) }},
				{"parallel_false", func(r *llm.Request) { r.ParallelToolCalls = new(false) }},
				{"parallel_true", func(r *llm.Request) { r.ParallelToolCalls = new(true) }},
				{"chat_options", func(r *llm.Request) { r.OpenAIChat = &llm.OpenAIChatOptions{} }},
				{"responses_options", func(r *llm.Request) { r.OpenAIResponses = &llm.OpenAIResponsesOptions{} }},
				{"anthropic_options", func(r *llm.Request) { r.Anthropic = &llm.AnthropicOptions{} }},
				{"exact_default", func(r *llm.Request) { r.ReasoningPolicy = llm.ReasoningPolicyExact }},
				{"exact_xhigh", func(r *llm.Request) {
					r.ReasoningPolicy, r.ReasoningEffort = llm.ReasoningPolicyExact, llm.ReasoningEffortXHigh
				}},
				{"exact_budget", func(r *llm.Request) {
					r.ReasoningPolicy, r.ReasoningBudgetTokens = llm.ReasoningPolicyExact, 2048
				}},
			} {
				t.Run(kind+"/"+op+"/"+change.name, func(t *testing.T) {
					client, fixture := protocolGuardClient(t, kind, true)
					request := protocolRequest()
					change.set(&request)
					err := protocolGuardCall(t, context.Background(), client, op, request)
					var modelErr *llm.Error
					if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported || modelErr.Op != op || fixture.calls.Load() != 0 {
						t.Fatalf("error=%v calls=%d", err, fixture.calls.Load())
					}
				})
			}
		}
	}
}

func TestProtocolOptionGuardsPreserveValidationOrder(t *testing.T) {
	for _, kind := range []string{"gemini", "vertex", "bedrock", "mistral"} {
		for _, op := range []string{"generate", "stream"} {
			t.Run(kind+"/"+op, func(t *testing.T) {
				client, fixture := protocolGuardClient(t, kind, false)
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				request := protocolRequest()
				request.TopK = new(1)
				invalid := request
				invalid.Messages = nil
				var modelErr *llm.Error
				if err := protocolGuardCall(t, ctx, client, op, invalid); !errors.As(err, &modelErr) || modelErr.Kind != llm.KindInvalidRequest || modelErr.Op != "validate" {
					t.Fatalf("validation did not precede cancellation: %v", err)
				}
				if err := protocolGuardCall(t, ctx, client, op, request); !errors.Is(err, context.Canceled) {
					t.Fatalf("cancellation did not precede protocol rejection: %v", err)
				}
				request.Tools = []llm.Tool{{Name: "fixture", InputSchema: []byte(`{"type":"object"}`)}}
				if err := protocolGuardCall(t, context.Background(), client, op, request); !errors.As(err, &modelErr) || modelErr.Kind != llm.KindUnsupported || !strings.Contains(modelErr.Error(), "capability") {
					t.Fatalf("capability did not precede protocol rejection: %v", err)
				}
				if fixture.calls.Load() != 0 {
					t.Fatal("validation performed I/O")
				}
			})
		}
	}
}

func TestProtocolOptionDefaultsRetainExistingAdapterBehavior(t *testing.T) {
	for _, kind := range []string{"gemini", "vertex", "bedrock", "mistral"} {
		for _, op := range []string{"generate", "stream"} {
			t.Run(kind+"/"+op, func(t *testing.T) {
				client, fixture := protocolGuardClient(t, kind, true)
				err := protocolGuardCall(t, context.Background(), client, op, protocolRequest())
				if !errors.Is(err, errProtocolFixtureIO) || fixture.calls.Load() != 1 {
					t.Fatalf("default request rejected or transport error lost: %v, calls=%d", err, fixture.calls.Load())
				}
			})
		}
	}
}
