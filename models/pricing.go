package models

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"

	llm "github.com/XiaoConstantine/llm-go"
)

type pricedGenerator struct {
	generator llm.Generator
	info      llm.ModelInfo
	cost      *llm.ModelCost
}

func withPricing(generator llm.Generator, info llm.ModelInfo) llm.Generator {
	if info.Cost == nil && info.Compatibility == nil && !info.Reasoning && info.API == "" && info.ContextWindow == 0 && info.MaxOutputTokens == 0 {
		return generator
	}
	return withPricingInfo(generator, info, generator.Info())
}

func withPricingInfo(generator llm.Generator, info, actual llm.ModelInfo) llm.Generator {
	var cost *llm.ModelCost
	if info.Cost != nil {
		value := *info.Cost
		value.Tiers = append([]llm.ModelCostTier(nil), info.Cost.Tiers...)
		cost = &value
	}
	configured := cloneModelInfo(actual)
	configured.Capabilities = append([]llm.Capability(nil), info.Capabilities...)
	if info.API != "" {
		configured.API = info.API
	}
	configured.ContextWindow = info.ContextWindow
	configured.MaxOutputTokens = info.MaxOutputTokens
	configured.Reasoning = info.Reasoning
	configured.Cost = cost
	configured.Compatibility = cloneCompatibility(info.Compatibility)
	priced := &pricedGenerator{generator: generator, info: configured, cost: cost}
	if background, ok := generator.(llm.BackgroundGenerator); ok {
		return &pricedBackgroundGenerator{pricedGenerator: priced, background: background}
	}
	return priced
}

func (g *pricedGenerator) Info() llm.ModelInfo {
	info := g.info
	info.Capabilities = append([]llm.Capability(nil), info.Capabilities...)
	if g.cost != nil {
		cost := *g.cost
		cost.Tiers = append([]llm.ModelCostTier(nil), g.cost.Tiers...)
		info.Cost = &cost
	}
	info.Compatibility = cloneCompatibility(g.info.Compatibility)
	return info
}

func (g *pricedGenerator) Generate(ctx context.Context, request llm.Request) (*llm.Response, error) {
	response, err := g.generator.Generate(ctx, request)
	if err != nil || response == nil || response.Usage == nil || g.cost == nil {
		return response, err
	}
	usage, err := g.price(*response.Usage)
	if err != nil {
		return nil, err
	}
	response.Usage = &usage
	return response, nil
}

func (g *pricedGenerator) Stream(ctx context.Context, request llm.Request) (llm.Stream, error) {
	stream, err := g.generator.Stream(ctx, request)
	if err != nil {
		return nil, err
	}
	if g.cost == nil {
		return stream, nil
	}
	return &pricedStream{Stream: stream, generator: g}, nil
}

func (g *pricedGenerator) price(usage llm.Usage) (llm.Usage, error) {
	cost, err := llm.CalculateCost(*g.cost, usage)
	if err != nil {
		return llm.Usage{}, &llm.Error{
			Kind:     llm.KindMalformedResponse,
			Op:       "price",
			Provider: g.info.Provider,
			Err:      fmt.Errorf("calculate usage cost: %w", err),
		}
	}
	usage.Cost = &cost
	return usage, nil
}

type pricedBackgroundGenerator struct {
	*pricedGenerator
	background llm.BackgroundGenerator
}

func (g *pricedBackgroundGenerator) StartBackground(ctx context.Context, request llm.Request) (*llm.BackgroundResult, error) {
	result, err := g.background.StartBackground(ctx, request)
	return g.priceBackground(result, err)
}

func (g *pricedBackgroundGenerator) FetchBackground(ctx context.Context, handle llm.BackgroundHandle) (*llm.BackgroundResult, error) {
	result, err := g.background.FetchBackground(ctx, handle)
	return g.priceBackground(result, err)
}

func (g *pricedBackgroundGenerator) CancelBackground(ctx context.Context, handle llm.BackgroundHandle) (*llm.BackgroundResult, error) {
	result, err := g.background.CancelBackground(ctx, handle)
	return g.priceBackground(result, err)
}

func (g *pricedBackgroundGenerator) priceBackground(result *llm.BackgroundResult, err error) (*llm.BackgroundResult, error) {
	if result == nil || result.Response == nil || result.Response.Usage == nil || g.cost == nil {
		return result, err
	}
	usage, priceErr := g.price(*result.Response.Usage)
	if priceErr != nil {
		return result, errors.Join(err, priceErr)
	}
	result.Response.Usage = &usage
	return result, err
}

type pricedStream struct {
	llm.Stream
	generator    *pricedGenerator
	mu           sync.Mutex
	terminalErr  error
	closeStarted bool
}

func (s *pricedStream) Recv() (llm.Chunk, error) {
	s.mu.Lock()
	if s.terminalErr != nil {
		err := s.terminalErr
		s.mu.Unlock()
		return llm.Chunk{}, err
	}
	s.mu.Unlock()
	chunk, err := s.Stream.Recv()
	if err != nil || chunk.Usage == nil {
		return chunk, err
	}
	usage, priceErr := s.generator.price(*chunk.Usage)
	if priceErr != nil {
		s.mu.Lock()
		if s.terminalErr == nil {
			if s.closeStarted {
				s.terminalErr = io.EOF
			} else {
				s.terminalErr = priceErr
			}
		}
		terminalErr := s.terminalErr
		s.mu.Unlock()
		_ = s.Stream.Close()
		return llm.Chunk{}, terminalErr
	}
	chunk.Usage = &usage
	return chunk, nil
}

func (s *pricedStream) Close() error {
	s.mu.Lock()
	s.closeStarted = true
	s.mu.Unlock()
	return s.Stream.Close()
}

var (
	_ llm.Generator           = (*pricedGenerator)(nil)
	_ llm.BackgroundGenerator = (*pricedBackgroundGenerator)(nil)
)
var _ llm.Stream = (*pricedStream)(nil)
