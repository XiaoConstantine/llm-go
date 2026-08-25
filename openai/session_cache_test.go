package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestChatSessionAffinityFormatsAndCacheIndependence(t *testing.T) {
	type observed struct {
		headers http.Header
		body    string
	}
	seen := make(chan observed, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		seen <- observed{headers: request.Header.Clone(), body: string(body)}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"response","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	openAICompatibility := &llm.OpenAIChatCompatibility{MaxTokensField: llm.MaxTokensFieldCompletion, LongCacheRetention: llm.CompatibilityEnabled, SessionAffinity: llm.CompatibilityEnabled, SessionAffinityFormat: llm.SessionAffinityOpenAI}
	openAI, err := NewWithCompatibility(Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client()}, openAICompatibility)
	if err != nil {
		t.Fatal(err)
	}
	base := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	both := base
	both.CacheRetention, both.CacheKey, both.SessionID = llm.CacheRetentionShort, "cache", "session"
	if _, err := openAI.Generate(context.Background(), both); err != nil {
		t.Fatal(err)
	}
	affinityOnly := base
	affinityOnly.CacheRetention, affinityOnly.SessionID = llm.CacheRetentionNone, "affinity"
	if _, err := openAI.Generate(context.Background(), affinityOnly); err != nil {
		t.Fatal(err)
	}
	openrouterCompatibility := *openAICompatibility
	openrouterCompatibility.SessionAffinityFormat = llm.SessionAffinityOpenRouter
	openrouter, err := NewWithCompatibility(Config{Provider: "openrouter", Model: "model", BaseURL: server.URL, HTTPClient: server.Client()}, &openrouterCompatibility)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := openrouter.Generate(context.Background(), affinityOnly); err != nil {
		t.Fatal(err)
	}
	ungated, err := NewWithCompatibility(Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client()}, &llm.OpenAIChatCompatibility{SessionAffinityFormat: llm.SessionAffinityOpenAI})
	if err != nil {
		t.Fatal(err)
	}
	_, err = ungated.Generate(context.Background(), affinityOnly)
	if err == nil || !strings.Contains(err.Error(), "session affinity") {
		t.Fatalf("ungated SessionID error = %v", err)
	}
	requireModelError(t, err, llm.KindUnsupported, "generate")
	if got := len(seen); got != 3 {
		t.Fatalf("HTTP requests before ungated SessionID rejection = %d, want 3", got)
	}
	first, second, third := <-seen, <-seen, <-seen
	if first.headers.Get("session_id") != "session" || first.headers.Get("X-Client-Request-Id") != "session" || first.headers.Get("X-Session-Affinity") != "session" || !strings.Contains(first.body, `"prompt_cache_key":"cache"`) {
		t.Fatalf("OpenAI cache+session request = %#v", first)
	}
	if second.headers.Get("session_id") != "affinity" || second.headers.Get("X-Client-Request-Id") != "affinity" || strings.Contains(second.body, "prompt_cache_key") {
		t.Fatalf("OpenAI affinity-only request = %#v", second)
	}
	if third.headers.Get("X-Session-Id") != "affinity" || third.headers.Get("session_id") != "" || third.headers.Get("X-Client-Request-Id") != "" {
		t.Fatalf("OpenRouter affinity request = %#v", third)
	}
}

func TestPromptCacheKeyCharacterLimit(t *testing.T) {
	base := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	for _, field := range []string{"key", "session"} {
		request := base
		if field == "key" {
			request.CacheKey = strings.Repeat("a", 65)
		} else {
			request.SessionID = strings.Repeat("a", 65)
		}
		if err := checkRequest("generate", request); err == nil || !strings.Contains(err.Error(), "64 characters") {
			t.Errorf("%s error = %v", field, err)
		}
	}
	base.CacheKey = strings.Repeat("界", 64)
	if err := checkRequest("generate", base); err != nil {
		t.Fatalf("64 multibyte characters: %v", err)
	}
	affinityOnly := llm.Request{Messages: base.Messages, CacheRetention: llm.CacheRetentionNone, SessionID: strings.Repeat("a", 65)}
	if err := checkRequest("generate", affinityOnly); err != nil {
		t.Fatalf("65-character affinity-only session: %v", err)
	}
}

func TestPromptCacheKeyIsGatedToVerifiedEndpoints(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, CacheKey: "stable"}
	native, err := New(Config{Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	wire, err := newChatRequestFor("generate", "model", request, native.compatibility)
	if err != nil || wire.PromptCacheKey != "stable" {
		t.Fatalf("native wire = %#v, %v", wire, err)
	}
	custom, err := New(Config{Model: "model", BaseURL: "https://local.example/v1"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := newChatRequestFor("generate", "model", request, custom.compatibility); err == nil || !strings.Contains(err.Error(), "prompt cache keys") {
		t.Fatalf("custom endpoint error = %v", err)
	}
	verified := custom.compatibility
	verified.LongCacheRetention = llm.CompatibilityEnabled
	wire, err = newChatRequestFor("generate", "model", request, verified)
	if err != nil || wire.PromptCacheKey != "stable" {
		t.Fatalf("verified wire = %#v, %v", wire, err)
	}
}
