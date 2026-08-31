package models

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
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
