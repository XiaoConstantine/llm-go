package models

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
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

func TestPricedStreamPricingErrorAndCloseFirstTerminalWins(t *testing.T) {
	generator := &pricedGenerator{info: llm.ModelInfo{Provider: "provider"}, cost: &llm.ModelCost{Input: 1}}
	invalid := llm.Chunk{Usage: &llm.Usage{InputTokens: 1, TotalTokens: 2}}

	t.Run("pricing error first", func(t *testing.T) {
		underlying := &pricingStreamStub{chunks: []llm.Chunk{invalid}}
		stream := &pricedStream{Stream: underlying, generator: generator}
		_, first := stream.Recv()
		if first == nil || first == io.EOF {
			t.Fatalf("first Recv() error = %v", first)
		}
		if err := stream.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		_, second := stream.Recv()
		if second != first {
			t.Fatalf("second Recv() error = %v, want identical %v", second, first)
		}
	})

	t.Run("close first", func(t *testing.T) {
		underlying := newBlockingPricingStream(invalid)
		stream := &pricedStream{Stream: underlying, generator: generator}
		result := make(chan error, 1)
		go func() {
			_, err := stream.Recv()
			result <- err
		}()
		<-underlying.entered
		if err := stream.Close(); err != nil {
			t.Fatalf("Close() error = %v", err)
		}
		if err := <-result; err != io.EOF {
			t.Fatalf("concurrent Recv() error = %v, want EOF", err)
		}
		if _, err := stream.Recv(); err != io.EOF {
			t.Fatalf("later Recv() error = %v, want EOF", err)
		}
	})
}

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
	if result == nil || err == nil {
		t.Fatalf("FetchBackground() = (%#v, %v), want observed result and pricing error", result, err)
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

type blockingPricingStream struct {
	chunk     llm.Chunk
	entered   chan struct{}
	release   chan struct{}
	enterOnce sync.Once
	closeOnce sync.Once
}

func newBlockingPricingStream(chunk llm.Chunk) *blockingPricingStream {
	return &blockingPricingStream{chunk: chunk, entered: make(chan struct{}), release: make(chan struct{})}
}

func (s *blockingPricingStream) Recv() (llm.Chunk, error) {
	s.enterOnce.Do(func() { close(s.entered) })
	<-s.release
	return s.chunk, nil
}

func (s *blockingPricingStream) Close() error {
	s.closeOnce.Do(func() { close(s.release) })
	return nil
}
