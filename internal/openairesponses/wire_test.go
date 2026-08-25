package openairesponses

import (
	jsonv2 "encoding/json/v2"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestExplicitPromptCacheWireSemantics(t *testing.T) {
	codec := Codec{Provider: "openai", MaxProviderDataBytes: 1 << 20}
	base := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
	for _, test := range []struct {
		name      string
		retention llm.CacheRetention
		key       string
		want      []string
		doNotWant []string
	}{
		{name: "default leaves policy unchanged", want: []string{`"content":"hello"`}, doNotWant: []string{"prompt_cache_options", "prompt_cache_breakpoint", "prompt_cache_key"}},
		{name: "none disables implicit caching", retention: llm.CacheRetentionNone, key: "ignored", want: []string{`"prompt_cache_options":{"mode":"explicit"}`}, doNotWant: []string{"prompt_cache_breakpoint", "prompt_cache_key"}},
		{name: "short uses key and retention only", retention: llm.CacheRetentionShort, key: "stable", want: []string{`"prompt_cache_key":"stable"`, `"prompt_cache_retention":"in_memory"`}, doNotWant: []string{"prompt_cache_options", "prompt_cache_breakpoint"}},
		{name: "long uses key and retention only", retention: llm.CacheRetentionLong, key: "stable", want: []string{`"prompt_cache_key":"stable"`, `"prompt_cache_retention":"24h"`}, doNotWant: []string{"prompt_cache_options", "prompt_cache_breakpoint"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			request := base
			request.CacheRetention = test.retention
			request.CacheKey = test.key
			params, err := codec.Request("generate", "model", request, RequestOptions{ExplicitPromptCache: true})
			if err != nil {
				t.Fatal(err)
			}
			body, err := jsonv2.Marshal(params)
			if err != nil {
				t.Fatal(err)
			}
			wire := string(body)
			for _, want := range test.want {
				if !strings.Contains(wire, want) {
					t.Errorf("wire = %s, want %s", wire, want)
				}
			}
			for _, unwanted := range test.doNotWant {
				if strings.Contains(wire, unwanted) {
					t.Errorf("wire = %s, unexpectedly contains %s", wire, unwanted)
				}
			}
		})
	}
}
