package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestGeneratedCatalogIsCurrent(t *testing.T) {
	generated, err := generate(filepath.Join("..", "..", "catalogsource", "catalog.json"))
	if err != nil {
		t.Fatalf("generate() error = %v", err)
	}
	current, err := os.ReadFile(filepath.Join("..", "..", "catalog_generated.go"))
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if !bytes.Equal(generated, current) {
		t.Fatal("catalog_generated.go is stale; run go generate ./models")
	}
}

func TestGenerateIsDeterministic(t *testing.T) {
	path := filepath.Join("..", "..", "catalogsource", "catalog.json")
	first, err := generate(path)
	if err != nil {
		t.Fatalf("first generate() error = %v", err)
	}
	second, err := generate(path)
	if err != nil {
		t.Fatalf("second generate() error = %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("generate() output changed without source changes")
	}
}

func TestJSONCapabilitySupportMatchesProtocols(t *testing.T) {
	for _, api := range []llm.API{llm.APIOpenAIChatCompletions, llm.APIOpenAIResponses, llm.APIGeminiGenerateContent} {
		if !supportsCapability(api, llm.CapabilityJSON) {
			t.Errorf("supportsCapability(%q, json) = false", api)
		}
	}
	for _, api := range []llm.API{llm.APIOpenAICodexResponses, llm.APIAnthropicMessages} {
		if supportsCapability(api, llm.CapabilityJSON) {
			t.Errorf("supportsCapability(%q, json) = true", api)
		}
	}
}

func TestAudioCapabilitySupportMatchesProtocols(t *testing.T) {
	for _, api := range []llm.API{llm.APIOpenAIChatCompletions, llm.APIOpenAICodexResponses, llm.APIGeminiGenerateContent} {
		if !supportsCapability(api, llm.CapabilityAudio) {
			t.Errorf("supportsCapability(%q, audio) = false", api)
		}
	}
	for _, api := range []llm.API{llm.APIOpenAIResponses, llm.APIAnthropicMessages} {
		if supportsCapability(api, llm.CapabilityAudio) {
			t.Errorf("supportsCapability(%q, audio) = true", api)
		}
	}
}

func TestLoadRejectsInvalidSources(t *testing.T) {
	const required = `"reasoning":false,"capabilities":[],"context_window":1,"max_output_tokens":1`
	tests := []struct {
		name     string
		source   string
		contains string
	}{
		{"schema", `{"schema_version":2,"revision":1,"models":[]}`, "schema_version"},
		{"revision", `{"schema_version":1,"revision":0,"models":[]}`, "revision"},
		{"unknown field", `{"schema_version":1,"revision":1,"models":[],"extra":true}`, "unknown object member"},
		{"duplicate field", `{"schema_version":1,"schema_version":1,"revision":1,"models":[]}`, "duplicate"},
		{"invalid UTF-8", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"` + string([]byte{0xff}) + `","api":"openai-responses"}]}`, "invalid UTF-8"},
		{"surrounding whitespace", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"M ","api":"openai-responses"}]}`, "surrounding whitespace"},
		{"missing required model field", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses"}]}`, "are required"},
		{"unknown provider", `{"schema_version":1,"revision":1,"models":[{"provider":"unknown","id":"m","name":"M","api":"openai-responses",` + required + `}]}`, "unsupported built-in provider"},
		{"provider API mismatch", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"anthropic-messages",` + required + `}]}`, "requires API"},
		{"missing cost field", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses",` + required + `,"cost":{"input":1}}]}`, "are required"},
		{"null cost field", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses",` + required + `,"cost":{"input":1,"output":null,"cache_read":0,"cache_write":0}}]}`, "must not be null"},
		{"missing tier field", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses","reasoning":false,"capabilities":[],"context_window":100,"max_output_tokens":1,"cost":{"input":1,"output":1,"cache_read":0,"cache_write":0,"tiers":[{"input_tokens_above":50,"input":2}]}}]}`, "tier 0"},
		{"unreachable cost tier", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses","reasoning":false,"capabilities":[],"context_window":100,"max_output_tokens":1,"cost":{"input":1,"output":1,"cache_read":0,"cache_write":0,"tiers":[{"input_tokens_above":100,"input":2,"output":2,"cache_read":0,"cache_write":0}]}}]}`, "below context window"},
		{"protocol capability", `{"schema_version":1,"revision":1,"models":[{"provider":"anthropic","id":"m","name":"M","api":"anthropic-messages","reasoning":false,"capabilities":["audio"],"context_window":1,"max_output_tokens":1}]}`, "does not support capability"},
		{"duplicate", `{"schema_version":1,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses",` + required + `},{"provider":"openai","id":"m","name":"M","api":"openai-responses",` + required + `}]}`, "duplicate model"},
		{"wrong compatibility API", `{"schema_version":1,"revision":1,"models":[{"provider":"anthropic","id":"m","name":"M","api":"anthropic-messages",` + required + `,"compatibility":{"openai_chat":{}}}]}`, "compatibility"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "catalog.json")
			if err := os.WriteFile(path, []byte(test.source), 0o600); err != nil {
				t.Fatal(err)
			}
			_, _, err := load(path)
			if err == nil || !strings.Contains(err.Error(), test.contains) {
				t.Fatalf("load() error = %v, want containing %q", err, test.contains)
			}
		})
	}
}
