package models

import (
	"context"
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
	if info.Cost == nil && info.Compatibility == nil && !info.Reasoning {
		return generator
	}
	var cost *llm.ModelCost
	if info.Cost != nil {
		value := *info.Cost
		value.Tiers = append([]llm.ModelCostTier(nil), info.Cost.Tiers...)
		cost = &value
	}
	configured := generator.Info()
	configured.Capabilities = append([]llm.Capability(nil), configured.Capabilities...)
	configured.Reasoning = info.Reasoning
	configured.Cost = cost
	configured.Compatibility = cloneCompatibility(info.Compatibility)
	return &pricedGenerator{generator: generator, info: configured, cost: cost}
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

var _ llm.Generator = (*pricedGenerator)(nil)
var _ llm.Stream = (*pricedStream)(nil)
