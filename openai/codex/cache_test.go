package codex

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestCacheKeyAndSessionHeadersAreIndependent(t *testing.T) {
	type observed struct{ session, clientRequestID, body string }
	seen := make(chan observed, 4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, _ := io.ReadAll(request.Body)
		seen <- observed{session: request.Header.Get("Session-Id"), clientRequestID: request.Header.Get("X-Client-Request-Id"), body: string(body)}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp\",\"status\":\"completed\"}}\n\n")
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", AccessToken: testAccessToken("account"), BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	base := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}, CacheRetention: llm.CacheRetentionShort}
	cacheOnly := base
	cacheOnly.CacheKey = "cache-key"
	if _, err := client.Generate(context.Background(), cacheOnly); err != nil {
		t.Fatal(err)
	}
	session := base
	session.SessionID = "session-id"
	if _, err := client.Generate(context.Background(), session); err != nil {
		t.Fatal(err)
	}
	both := base
	both.CacheKey, both.SessionID = "cache-key", "session-id"
	if _, err := client.Generate(context.Background(), both); err != nil {
		t.Fatal(err)
	}
	affinityOnly := base
	affinityOnly.CacheRetention, affinityOnly.SessionID = llm.CacheRetentionNone, "affinity-only"
	if _, err := client.Generate(context.Background(), affinityOnly); err != nil {
		t.Fatal(err)
	}
	first, second, third, fourth := <-seen, <-seen, <-seen, <-seen
	if first.session != "" || first.clientRequestID != "" || !strings.Contains(first.body, `"prompt_cache_key":"cache-key"`) {
		t.Fatalf("cache-key-only request = %#v", first)
	}
	if second.session != "session-id" || second.clientRequestID != "session-id" || !strings.Contains(second.body, `"prompt_cache_key":"session-id"`) {
		t.Fatalf("session request = %#v", second)
	}
	if third.session != "session-id" || !strings.Contains(third.body, `"prompt_cache_key":"cache-key"`) || strings.Contains(third.body, `"prompt_cache_key":"session-id"`) {
		t.Fatalf("cache+session request = %#v", third)
	}
	if fourth.session != "affinity-only" || strings.Contains(fourth.body, "prompt_cache_key") {
		t.Fatalf("affinity-only request = %#v", fourth)
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
