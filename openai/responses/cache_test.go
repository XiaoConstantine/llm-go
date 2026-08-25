package responses

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestSessionAffinityIsIndependentFromPromptCache(t *testing.T) {
	type observed struct {
		headers http.Header
		body    string
	}
	seen := make(chan observed, 3)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		seen <- observed{headers: request.Header.Clone(), body: string(body)}
		w.Header().Set("Content-Type", "text/event-stream")
		writeSSE(t, w, `{"type":"response.completed","response":{"id":"resp","status":"completed"}}`)
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	base := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}
	both := base
	both.CacheRetention, both.CacheKey, both.SessionID = llm.CacheRetentionShort, "cache", "session"
	if _, err := client.Generate(context.Background(), both); err != nil {
		t.Fatal(err)
	}
	affinityOnly := base
	affinityOnly.CacheRetention, affinityOnly.SessionID = llm.CacheRetentionNone, "affinity"
	if _, err := client.Generate(context.Background(), affinityOnly); err != nil {
		t.Fatal(err)
	}
	openrouter, err := NewWithCompatibility(Config{Provider: "openrouter", Model: "model", APIKey: "key", BaseURL: server.URL, HTTPClient: server.Client()}, &llm.OpenAIResponsesCompatibility{SessionAffinityFormat: llm.SessionAffinityOpenRouter})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := openrouter.Generate(context.Background(), affinityOnly); err != nil {
		t.Fatal(err)
	}
	first, second, third := <-seen, <-seen, <-seen
	if first.headers.Get("session_id") != "session" || first.headers.Get("X-Client-Request-Id") != "session" || !strings.Contains(first.body, `"prompt_cache_key":"cache"`) || strings.Contains(first.body, `"prompt_cache_key":"session"`) {
		t.Fatalf("cache+session request = %#v", first)
	}
	if second.headers.Get("session_id") != "affinity" || second.headers.Get("X-Client-Request-Id") != "affinity" || strings.Contains(second.body, "prompt_cache_key") {
		t.Fatalf("affinity-only request = %#v", second)
	}
	if third.headers.Get("X-Session-Id") != "affinity" || third.headers.Get("session_id") != "" || third.headers.Get("X-Client-Request-Id") != "" || strings.Contains(third.body, "prompt_cache_key") {
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
		if err := checkRequest("openai", "generate", request); err == nil || !strings.Contains(err.Error(), "64 characters") {
			t.Errorf("%s error = %v", field, err)
		}
	}
	base.SessionID = strings.Repeat("界", 64)
	if err := checkRequest("openai", "generate", base); err != nil {
		t.Fatalf("64 multibyte characters: %v", err)
	}
	affinityOnly := llm.Request{Messages: base.Messages, CacheRetention: llm.CacheRetentionNone, SessionID: strings.Repeat("a", 65)}
	if err := checkRequest("openai", "generate", affinityOnly); err != nil {
		t.Fatalf("65-character affinity-only session: %v", err)
	}
}
