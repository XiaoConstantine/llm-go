package llm_test

import (
	"context"
	"net/http"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

type attemptHeaderGenerator struct {
	seen http.Header
}

func (g *attemptHeaderGenerator) Info() llm.ModelInfo {
	return llm.ModelInfo{Provider: "custom", Model: "model", Capabilities: []llm.Capability{llm.CapabilityGeneration}}
}

func (g *attemptHeaderGenerator) Generate(ctx context.Context, _ llm.Request) (*llm.Response, error) {
	g.seen = llm.AttemptHeaders(ctx)
	g.seen.Set("X-Trace", "custom-mutated")
	return &llm.Response{Message: llm.Message{Role: llm.RoleAssistant}}, nil
}

func (*attemptHeaderGenerator) Stream(context.Context, llm.Request) (llm.Stream, error) {
	panic("not used")
}

func TestCustomGeneratorCanReadOwnedAttemptHeaders(t *testing.T) {
	generator := &attemptHeaderGenerator{}
	returned := http.Header{"X-Trace": {"hook-value"}}
	retrying, err := llm.WithRetry(generator, llm.RetryPolicy{MaxAttempts: 1, Hook: func(context.Context, llm.Attempt) (http.Header, error) {
		return returned, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	if _, err := retrying.Generate(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if generator.seen.Get("X-Trace") != "custom-mutated" {
		t.Fatalf("custom generator headers = %#v", generator.seen)
	}
	if returned.Get("X-Trace") != "hook-value" {
		t.Fatalf("hook-owned headers were mutated: %#v", returned)
	}
	generator.seen.Set("X-Trace", "after-return")
	if got := llm.AttemptHeaders(context.Background()); got != nil {
		t.Fatalf("headers escaped into unrelated context: %#v", got)
	}
}
