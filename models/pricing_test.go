package models

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/bedrock"
	"github.com/XiaoConstantine/llm-go/mistral"
	"github.com/XiaoConstantine/llm-go/openai"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func TestPricedGeneratorPricesGenerateAndStreamUsage(t *testing.T) {
	base := &pricingGeneratorStub{
		info: llm.ModelInfo{Provider: "provider", Model: "model", Capabilities: []llm.Capability{llm.CapabilityGeneration, llm.CapabilityStreaming}},
		response: &llm.Response{Usage: &llm.Usage{
			InputTokens: 500, OutputTokens: 100, CacheReadTokens: 400, CacheWriteTokens: 100, TotalTokens: 1_100,
		}},
		chunks: []llm.Chunk{
			{Usage: &llm.Usage{InputTokens: 250, OutputTokens: 50, CacheReadTokens: 200, TotalTokens: 500}},
			{FinishReason: llm.FinishReasonStop, Usage: &llm.Usage{InputTokens: 500, OutputTokens: 100, CacheReadTokens: 400, CacheWriteTokens: 100, TotalTokens: 1_100}},
		},
	}
	info := base.info
	info.Cost = &llm.ModelCost{Input: 2, Output: 8, CacheRead: 0.2, CacheWrite: 2.5}
	generator := withPricing(base, info)

	response, err := generator.Generate(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if response.Usage.Cost == nil || response.Usage.Cost.Total == 0 {
		t.Fatalf("Generate().Usage.Cost = %#v", response.Usage.Cost)
	}
	if base.response.Usage.Cost != nil {
		t.Fatal("pricing wrapper mutated underlying response usage")
	}

	stream, err := generator.Stream(context.Background(), llm.Request{})
	if err != nil {
		t.Fatalf("Stream() error = %v", err)
	}
	for index := range base.chunks {
		chunk, err := stream.Recv()
		if err != nil {
			t.Fatalf("Recv(%d) error = %v", index, err)
		}
		if chunk.Usage == nil || chunk.Usage.Cost == nil || chunk.Usage.Cost.Total == 0 {
			t.Fatalf("Recv(%d).Usage = %#v", index, chunk.Usage)
		}
	}
	if _, err := stream.Recv(); err != io.EOF {
		t.Fatalf("terminal Recv() error = %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
}

func TestUnpriceableOutputIsPreserved(t *testing.T) {
	usage := llm.Usage{InputTokens: 2_000_000, OutputTokens: 3_000_000, TotalTokens: 5_000_000}
	base := &pricingGeneratorStub{
		response: &llm.Response{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{{Text: "answer"}}}, FinishReason: llm.FinishReasonStop, Usage: &usage},
		chunks: []llm.Chunk{
			{Content: []llm.Part{{Text: "answer"}}, Usage: &usage},
			{FinishReason: llm.FinishReasonStop},
		},
	}
	// The table is valid, but multiplying its finite rate by this usage overflows.
	generator := withPricing(base, llm.ModelInfo{Cost: &llm.ModelCost{Input: math.MaxFloat64, Output: 2}})
	for _, mode := range []string{"generate", "stream"} {
		t.Run(mode, func(t *testing.T) {
			var response *llm.Response
			var err error
			if mode == "generate" {
				response, err = generator.Generate(context.Background(), llm.Request{})
			} else {
				stream, streamErr := generator.Stream(context.Background(), llm.Request{})
				if streamErr != nil {
					t.Fatal(streamErr)
				}
				response, err = llm.Collect(stream, nil)
			}
			if err != nil {
				t.Fatal(err)
			}
			if response.Text() != "answer" || response.FinishReason != llm.FinishReasonStop || response.Usage == nil || *response.Usage != usage {
				t.Fatalf("response = %#v, usage = %#v", response, response.Usage)
			}
		})
	}
}

func TestPricingPreservesCollectorValidation(t *testing.T) {
	usage := llm.Usage{InputTokens: 2, OutputTokens: 3, TotalTokens: 99}
	base := &pricingGeneratorStub{
		response: &llm.Response{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{{Text: "answer"}}}, FinishReason: llm.FinishReasonStop, Usage: &usage},
		chunks: []llm.Chunk{
			{Content: []llm.Part{{Text: "answer"}}},
			{FinishReason: llm.FinishReasonStop, Usage: &usage},
		},
	}
	generator := withPricing(base, llm.ModelInfo{Cost: &llm.ModelCost{Input: 1, Output: 2}})
	response, err := generator.Generate(context.Background(), llm.Request{})
	if err != nil || response == nil || response.Text() != "answer" || response.Usage == nil || *response.Usage != usage {
		t.Fatalf("Generate() = (%#v, %v), want original usage with nil cost", response, err)
	}
	stream, err := generator.Stream(context.Background(), llm.Request{})
	if err != nil {
		t.Fatal(err)
	}
	response, err = llm.Collect(stream, nil)
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse || modelErr.Op != "collect" || response.Text() != "answer" {
		t.Fatalf("Collect() = (%#v, %v), want collector rejection with prior content", response, err)
	}
}

func TestPricedAdaptersHandleUsageTotals(t *testing.T) {
	for _, provider := range []string{"bedrock", "mistral", "openai"} {
		for _, test := range []struct {
			name        string
			total       *int32
			onlyTotal   bool
			wantNoUsage bool
		}{
			{name: "missing"},
			{name: "null"},
			{name: "zero", total: aws.Int32(0)},
			{name: "equal", total: aws.Int32(18)},
			{name: "below", total: aws.Int32(1), wantNoUsage: true},
			{name: "above", total: aws.Int32(99), wantNoUsage: true},
			{name: "only total", total: aws.Int32(50), onlyTotal: true, wantNoUsage: true},
			{name: "negative", total: aws.Int32(-1)},
		} {
			t.Run(provider+"/"+test.name, func(t *testing.T) {
				caps := []llm.Capability{llm.CapabilityStreaming}
				var adapter llm.Generator
				var err error
				wantUsage := llm.Usage{InputTokens: 6, OutputTokens: 7, CacheReadTokens: 3, CacheWriteTokens: 2, TotalTokens: 18}
				wantCost := 0.000044
				if provider == "bedrock" {
					usage := &types.TokenUsage{InputTokens: aws.Int32(6), OutputTokens: aws.Int32(7), CacheReadInputTokens: aws.Int32(3), CacheWriteInputTokens: aws.Int32(2)}
					if test.onlyTotal {
						usage = &types.TokenUsage{}
					}
					usage.TotalTokens = test.total
					adapter, err = bedrock.New(bedrock.Config{Model: "model", Capabilities: caps, Runtime: pricingBedrockRuntime{usage: usage}})
				} else {
					if provider == "mistral" {
						wantUsage = llm.Usage{InputTokens: 11, OutputTokens: 7, TotalTokens: 18}
						wantCost = 0.000043
					}
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						var request struct{ Stream bool }
						if decodeErr := json.NewDecoder(r.Body).Decode(&request); decodeErr != nil {
							t.Errorf("decode request: %v", decodeErr)
							return
						}
						usage := `{"prompt_tokens":11,"completion_tokens":7`
						if test.onlyTotal {
							usage = `{`
						}
						if test.total != nil {
							if !test.onlyTotal {
								usage += `,`
							}
							usage += fmt.Sprintf(`"total_tokens":%d`, *test.total)
						} else if test.name == "null" {
							usage += `,"total_tokens":null`
						}
						if provider == "openai" && !test.onlyTotal {
							usage += `,"prompt_tokens_details":{"cached_tokens":3,"cache_write_tokens":2}`
						}
						usage += `}`
						if request.Stream {
							w.Header().Set("Content-Type", "text/event-stream")
							_, _ = fmt.Fprintf(w, "data: "+`{"choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]}`+"\n\ndata: "+`{"choices":[],"usage":%s}`+"\n\ndata: [DONE]\n\n", usage)
						} else {
							w.Header().Set("Content-Type", "application/json")
							_, _ = fmt.Fprintf(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"usage":%s}`, usage)
						}
					}))
					defer server.Close()
					if provider == "mistral" {
						adapter, err = mistral.New(mistral.Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: caps})
					} else {
						adapter, err = openai.New(openai.Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: caps})
					}
				}
				if err != nil {
					t.Fatal(err)
				}
				info := adapter.Info()
				info.Cost = &llm.ModelCost{Input: 2, Output: 3, CacheRead: 1, CacheWrite: 4}
				generator := withPricing(adapter, info)
				request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
				for _, mode := range []string{"generate", "stream"} {
					t.Run(mode, func(t *testing.T) {
						var response *llm.Response
						if mode == "generate" {
							response, err = generator.Generate(context.Background(), request)
						} else {
							stream, streamErr := generator.Stream(context.Background(), request)
							if streamErr != nil {
								t.Fatal(streamErr)
							}
							response, err = llm.Collect(stream, nil)
						}
						if test.total != nil && *test.total < 0 {
							var modelErr *llm.Error
							if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse {
								t.Fatalf("error = %v, want malformed response", err)
							}
							return
						}
						if err != nil {
							t.Fatal(err)
						}
						if response.Text() != "answer" || response.FinishReason != llm.FinishReasonStop {
							t.Fatalf("response = %#v", response)
						}
						if test.wantNoUsage {
							if response.Usage != nil {
								t.Fatalf("usage = %#v, want nil for inconsistent accounting", response.Usage)
							}
							return
						}
						if response.Usage == nil || response.Usage.Cost == nil {
							t.Fatalf("usage = %#v, want priced usage", response.Usage)
						}
						usage := *response.Usage
						cost := usage.Cost.Total
						usage.Cost = nil
						if usage != wantUsage || math.Abs(cost-wantCost) > 1e-15 {
							t.Fatalf("usage = %#v, cost = %g; want %#v, %g", usage, cost, wantUsage, wantCost)
						}
					})
				}
			})
		}
	}
}

type pricingBedrockRuntime struct{ usage *types.TokenUsage }

func (r pricingBedrockRuntime) ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput) (bedrock.EventStream, error) {
	events := make(pricingBedrockEvents, 5)
	events <- &types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}}
	events <- &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(0), Delta: &types.ContentBlockDeltaMemberText{Value: "answer"}}}
	events <- &types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(0)}}
	events <- &types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn}}
	events <- &types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{Usage: r.usage}}
	close(events)
	return events, nil
}

type pricingBedrockEvents chan types.ConverseStreamOutput

func (s pricingBedrockEvents) Events() <-chan types.ConverseStreamOutput { return s }
func (pricingBedrockEvents) Err() error                                  { return nil }
func (pricingBedrockEvents) Close() error                                { return nil }

func TestPricedGeneratorInfoOwnsPricing(t *testing.T) {
	cost := llm.ModelCost{Input: 1, Tiers: []llm.ModelCostTier{{InputTokensAbove: 100, Input: 2}}}
	generator := withPricing(&pricingGeneratorStub{}, llm.ModelInfo{Provider: "provider", Model: "model", Cost: &cost})
	cost.Input = 99
	cost.Tiers[0].Input = 99
	first := generator.Info()
	first.Cost.Input = 98
	first.Cost.Tiers[0].Input = 98
	second := generator.Info()
	if second.Cost.Input != 1 || second.Cost.Tiers[0].Input != 2 {
		t.Fatalf("Info().Cost = %#v", second.Cost)
	}
}

func TestPricedGeneratorPreservesBackgroundCapability(t *testing.T) {
	usage := &llm.Usage{InputTokens: 500, OutputTokens: 100, TotalTokens: 600}
	base := &pricingBackgroundStub{pricingGeneratorStub: &pricingGeneratorStub{info: llm.ModelInfo{
		Provider: "provider", Model: "model", API: llm.APIOpenAIResponses,
	}}}
	info := base.info
	info.Cost = &llm.ModelCost{Input: 2, Output: 8}
	generator := withPricing(base, info)
	background, ok := generator.(llm.BackgroundGenerator)
	if !ok {
		t.Fatalf("priced generator type = %T, want BackgroundGenerator", generator)
	}
	base.result = &llm.BackgroundResult{Status: llm.BackgroundCompleted,
		Response: &llm.Response{Usage: usage}}
	result, err := background.StartBackground(context.Background(), llm.Request{})
	if err != nil || result.Response.Usage.Cost == nil || result.Response.Usage.Cost.Total == 0 {
		t.Fatalf("StartBackground() = (%#v, %v)", result, err)
	}
	if usage.Cost != nil {
		t.Fatal("pricing wrapper mutated underlying background usage")
	}
	providerErr := errors.New("context ended after response")
	base.err = providerErr
	result, err = background.FetchBackground(context.Background(), llm.BackgroundHandle{})
	if result == nil || result.Response.Usage.Cost == nil || !errors.Is(err, providerErr) {
		t.Fatalf("FetchBackground() with observed response = (%#v, %v)", result, err)
	}
	base.err = nil
	base.result = &llm.BackgroundResult{Status: llm.BackgroundCompleted,
		Response: &llm.Response{Usage: &llm.Usage{InputTokens: 2, TotalTokens: 1}}}
	result, err = background.FetchBackground(context.Background(), llm.BackgroundHandle{})
	if result == nil || err != nil || result.Response.Usage.Cost != nil || result.Response.Usage.TotalTokens != 1 {
		t.Fatalf("FetchBackground() = (%#v, %v), want observed result without cost", result, err)
	}
	base.err = providerErr
	result, err = background.FetchBackground(context.Background(), llm.BackgroundHandle{})
	if result == nil || err != providerErr || result.Response.Usage.Cost != nil {
		t.Fatalf("FetchBackground() = (%#v, %v), want original provider error", result, err)
	}
}

type pricingGeneratorStub struct {
	info     llm.ModelInfo
	response *llm.Response
	chunks   []llm.Chunk
}

func (g *pricingGeneratorStub) Info() llm.ModelInfo { return g.info }

func (g *pricingGeneratorStub) Generate(context.Context, llm.Request) (*llm.Response, error) {
	if g.response == nil {
		return &llm.Response{}, nil
	}
	response := *g.response
	if g.response.Usage != nil {
		usage := *g.response.Usage
		response.Usage = &usage
	}
	return &response, nil
}

func (g *pricingGeneratorStub) Stream(context.Context, llm.Request) (llm.Stream, error) {
	chunks := make([]llm.Chunk, len(g.chunks))
	for index, chunk := range g.chunks {
		chunks[index] = chunk
		if chunk.Usage != nil {
			usage := *chunk.Usage
			chunks[index].Usage = &usage
		}
	}
	return &pricingStreamStub{chunks: chunks}, nil
}

type pricingBackgroundStub struct {
	*pricingGeneratorStub
	result *llm.BackgroundResult
	err    error
}

func (g *pricingBackgroundStub) StartBackground(context.Context, llm.Request) (*llm.BackgroundResult, error) {
	result := *g.result
	response := *g.result.Response
	usage := *g.result.Response.Usage
	response.Usage = &usage
	result.Response = &response
	return &result, g.err
}

func (g *pricingBackgroundStub) FetchBackground(context.Context, llm.BackgroundHandle) (*llm.BackgroundResult, error) {
	return g.StartBackground(context.Background(), llm.Request{})
}

func (g *pricingBackgroundStub) CancelBackground(context.Context, llm.BackgroundHandle) (*llm.BackgroundResult, error) {
	return &llm.BackgroundResult{Status: llm.BackgroundCancelled}, nil
}

type pricingStreamStub struct {
	chunks []llm.Chunk
	index  int
}

func (s *pricingStreamStub) Recv() (llm.Chunk, error) {
	if s.index == len(s.chunks) {
		return llm.Chunk{}, io.EOF
	}
	chunk := s.chunks[s.index]
	s.index++
	return chunk, nil
}
func (*pricingStreamStub) Close() error { return nil }
