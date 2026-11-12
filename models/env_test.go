package models

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestFindEnvAPIKeysAndEnvAPIKey(t *testing.T) {
	for _, variable := range []string{"COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"} {
		t.Setenv(variable, "")
	}
	t.Setenv("GH_TOKEN", "gh")
	t.Setenv("GITHUB_TOKEN", "github")
	if got := FindEnvAPIKeys("github-copilot"); !reflect.DeepEqual(got, []string{"GH_TOKEN", "GITHUB_TOKEN"}) {
		t.Fatalf("FindEnvAPIKeys() = %#v", got)
	}
	key, ok := EnvAPIKey("github-copilot")
	if !ok || key != "gh" {
		t.Fatalf("EnvAPIKey() = %q, %v", key, ok)
	}
	keys := FindEnvAPIKeys("github-copilot")
	keys[0] = "changed"
	if got := FindEnvAPIKeys("github-copilot")[0]; got != "GH_TOKEN" {
		t.Fatalf("environment key storage was mutated: %q", got)
	}
	if keys := FindEnvAPIKeys("unknown"); keys != nil {
		t.Fatalf("unknown provider keys = %#v", keys)
	}
}

func TestEnvAnthropicAPIKeyDoesNotSelectOAuthToken(t *testing.T) {
	t.Setenv("ANTHROPIC_OAUTH_TOKEN", "oauth-token")
	t.Setenv("ANTHROPIC_API_KEY", "api-key")
	if got := FindEnvAPIKeys("anthropic"); !reflect.DeepEqual(got, []string{"ANTHROPIC_API_KEY"}) {
		t.Errorf("FindEnvAPIKeys(anthropic) = %#v, want only ANTHROPIC_API_KEY", got)
	}
	if key, ok := EnvAPIKey("anthropic"); !ok || key != "api-key" {
		t.Errorf("EnvAPIKey(anthropic) = %q, %t, want API key", key, ok)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Api-Key") != "api-key" || request.Header.Get("Authorization") != "" {
			t.Errorf("Anthropic headers = %#v, want API-key authentication", request.Header)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg","type":"message","role":"assistant","content":[],"model":"model","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":0}}`)
	}))
	defer server.Close()
	key, _ := EnvAPIKey("anthropic")
	collection, err := New(ProviderConfig{ID: "anthropic", API: AnthropicMessages, APIKey: key, BaseURL: server.URL, HTTPClient: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	generator, err := collection.Generator(llm.ModelInfo{Provider: "anthropic", Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := generator.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}}); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ANTHROPIC_API_KEY", "")
	if key, ok := EnvAPIKey("anthropic"); ok || key != "" {
		t.Errorf("EnvAPIKey(anthropic) = %q, %t, want no API key", key, ok)
	}
}

func TestHasAmbientBedrockCredentials(t *testing.T) {
	variables := []string{"AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BEARER_TOKEN_BEDROCK",
		"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_WEB_IDENTITY_TOKEN_FILE"}
	for _, variable := range variables {
		t.Setenv(variable, "")
	}
	if HasAmbientCredentials(ProviderAmazonBedrock) {
		t.Fatal("empty environment reported Bedrock credentials")
	}
	t.Setenv("AWS_ACCESS_KEY_ID", "id")
	if HasAmbientCredentials(ProviderAmazonBedrock) {
		t.Fatal("incomplete IAM key pair reported Bedrock credentials")
	}
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secret")
	if !HasAmbientCredentials(ProviderAmazonBedrock) {
		t.Fatal("IAM key pair was not detected")
	}
}

func TestHasAmbientVertexCredentials(t *testing.T) {
	for _, variable := range []string{"GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"} {
		t.Setenv(variable, "")
	}
	credentials := filepath.Join(t.TempDir(), "credentials.json")
	if err := os.WriteFile(credentials, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", credentials)
	t.Setenv("GCLOUD_PROJECT", "project")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
	if !HasAmbientCredentials(ProviderGoogleVertex) {
		t.Fatal("Vertex ADC was not detected")
	}
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	if HasAmbientCredentials(ProviderGoogleVertex) {
		t.Fatal("Vertex ADC without location was detected")
	}
	if HasAmbientCredentials("openai") {
		t.Fatal("unsupported ambient provider was detected")
	}
}
