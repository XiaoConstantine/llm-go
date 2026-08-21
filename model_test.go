package llm

import (
	"context"
	"io"
)

type generatorStub struct{}

func (generatorStub) Info() ModelInfo {
	return ModelInfo{
		Provider:     "test",
		Model:        "generate",
		Capabilities: []Capability{CapabilityGeneration},
	}
}

func (generatorStub) Generate(context.Context, Request) (*Response, error) {
	return &Response{Message: Message{Role: RoleAssistant}}, nil
}

func (generatorStub) Stream(context.Context, Request) (Stream, error) {
	return nil, &Error{Kind: KindUnsupported}
}

type streamStub struct{}

func (streamStub) Recv() (Chunk, error) { return Chunk{}, io.EOF }
func (streamStub) Close() error         { return nil }

var (
	_ Generator = generatorStub{}
	_ Stream    = streamStub{}
)
