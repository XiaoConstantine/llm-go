package main

import (
	"bytes"
	jsonv2 "encoding/json/v2"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/catalogmeta"
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
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var source sourceCatalog
	if err := jsonv2.Unmarshal(data, &source); err != nil {
		t.Fatal(err)
	}
	slices.Reverse(source.Models)
	for i := range source.Models {
		// Ownership is build-time provenance, not runtime model metadata.
		source.Models[i].Source = ""
	}
	data, err = jsonv2.Marshal(source)
	if err != nil {
		t.Fatal(err)
	}
	path = filepath.Join(t.TempDir(), "catalog.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	reordered, err := generate(path)
	if err != nil || !bytes.Equal(first, reordered) {
		t.Fatalf("source order/ownership changed generated output: %v", err)
	}
}

func TestJSONCapabilitySupportMatchesProtocols(t *testing.T) {
	for _, api := range []llm.API{llm.APIOpenAIChatCompletions, llm.APIOpenAIResponses, llm.APIGeminiGenerateContent} {
		if !catalogmeta.SupportsCapability(api, llm.CapabilityJSON) {
			t.Errorf("supportsCapability(%q, json) = false", api)
		}
	}
	for _, api := range []llm.API{llm.APIOpenAICodexResponses, llm.APIAnthropicMessages} {
		if catalogmeta.SupportsCapability(api, llm.CapabilityJSON) {
			t.Errorf("supportsCapability(%q, json) = true", api)
		}
	}
}

func TestAudioCapabilitySupportMatchesProtocols(t *testing.T) {
	for _, api := range []llm.API{llm.APIOpenAIChatCompletions, llm.APIOpenAICodexResponses, llm.APIGeminiGenerateContent} {
		if !catalogmeta.SupportsCapability(api, llm.CapabilityAudio) {
			t.Errorf("supportsCapability(%q, audio) = false", api)
		}
	}
	for _, api := range []llm.API{llm.APIOpenAIResponses, llm.APIAnthropicMessages} {
		if catalogmeta.SupportsCapability(api, llm.CapabilityAudio) {
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
		{"schema", `{"schema_version":99,"revision":1,"models":[]}`, "schema_version"},
		{"revision", `{"schema_version":2,"revision":0,"models":[]}`, "revision"},
		{"unknown source", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","source":"unknown","name":"M","api":"openai-responses",` + required + `}]}`, "unknown source"},
		{"managed Codex", `{"schema_version":2,"revision":1,"models":[{"provider":"openai-codex","id":"m","source":"models.dev","name":"M","api":"openai-codex-responses",` + required + `}]}`, "must be curated"},
		{"unknown field", `{"schema_version":2,"revision":1,"models":[],"extra":true}`, "unknown object member"},
		{"duplicate field", `{"schema_version":2,"schema_version":2,"revision":1,"models":[]}`, "duplicate"},
		{"invalid UTF-8", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"` + string([]byte{0xff}) + `","api":"openai-responses"}]}`, "invalid UTF-8"},
		{"surrounding whitespace", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M ","api":"openai-responses"}]}`, "surrounding whitespace"},
		{"missing required model field", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses"}]}`, "are required"},
		{"unknown provider", `{"schema_version":2,"revision":1,"models":[{"provider":"unknown","id":"m","name":"M","api":"openai-responses",` + required + `}]}`, "unsupported built-in provider"},
		{"provider API mismatch", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"anthropic-messages",` + required + `}]}`, "requires API"},
		{"missing cost field", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses",` + required + `,"cost":{"input":1}}]}`, "are required"},
		{"null cost field", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses",` + required + `,"cost":{"input":1,"output":null,"cache_read":0,"cache_write":0}}]}`, "must not be null"},
		{"missing tier field", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses","reasoning":false,"capabilities":[],"context_window":100,"max_output_tokens":1,"cost":{"input":1,"output":1,"cache_read":0,"cache_write":0,"tiers":[{"input_tokens_above":50,"input":2}]}}]}`, "tier 0"},
		{"unreachable cost tier", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses","reasoning":false,"capabilities":[],"context_window":100,"max_output_tokens":1,"cost":{"input":1,"output":1,"cache_read":0,"cache_write":0,"tiers":[{"input_tokens_above":100,"input":2,"output":2,"cache_read":0,"cache_write":0}]}}]}`, "below context window"},
		{"protocol capability", `{"schema_version":2,"revision":1,"models":[{"provider":"anthropic","id":"m","name":"M","api":"anthropic-messages","reasoning":false,"capabilities":["audio"],"context_window":1,"max_output_tokens":1}]}`, "does not support capability"},
		{"duplicate", `{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses",` + required + `},{"provider":"openai","id":"m","name":"M","api":"openai-responses",` + required + `}]}`, "duplicate model"},
		{"wrong compatibility API", `{"schema_version":2,"revision":1,"models":[{"provider":"anthropic","id":"m","name":"M","api":"anthropic-messages",` + required + `,"compatibility":{"openai_chat":{}}}]}`, "compatibility"},
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

func TestLoadCacheReadPriceRelationships(t *testing.T) {
	for _, test := range []struct {
		name     string
		baseRead float64
		tierRead float64
		wantErr  bool
	}{
		{"discounted", 0.1, 0.2, false},
		{"equal", 1, 2, false},
		{"base above input", 1.01, 0.2, true},
		{"tier above input", 0.1, 2.01, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			// Cheap cache writes are allowed: some providers price storage separately.
			source := fmt.Sprintf(`{"schema_version":2,"revision":1,"models":[{"provider":"openai","id":"m","name":"M","api":"openai-responses","reasoning":false,"capabilities":[],"context_window":100,"max_output_tokens":1,"cost":{"input":1,"output":0.5,"cache_read":%g,"cache_write":0.01,"tiers":[{"input_tokens_above":50,"input":2,"output":1,"cache_read":%g,"cache_write":0}]}}]}`, test.baseRead, test.tierRead)
			path := filepath.Join(t.TempDir(), "catalog.json")
			if err := os.WriteFile(path, []byte(source), 0o600); err != nil {
				t.Fatal(err)
			}
			_, _, err := load(path)
			if test.wantErr {
				if err == nil || !strings.Contains(err.Error(), "cache_read must not exceed input") {
					t.Fatalf("load() error = %v, want cache_read price error", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
}
