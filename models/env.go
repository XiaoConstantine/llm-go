package models

import (
	"os"
	"path/filepath"
)

var providerAPIKeyEnvironment = map[string][]string{
	"github-copilot":         {"COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"},
	"anthropic":              {"ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"},
	"openai":                 {"OPENAI_API_KEY"},
	"azure-openai-responses": {"AZURE_OPENAI_API_KEY"},
	"deepseek":               {"DEEPSEEK_API_KEY"},
	"google":                 {"GEMINI_API_KEY"},
	"google-vertex":          {"GOOGLE_CLOUD_API_KEY"},
	"groq":                   {"GROQ_API_KEY"},
	"cerebras":               {"CEREBRAS_API_KEY"},
	"xai":                    {"XAI_API_KEY"},
	"openrouter":             {"OPENROUTER_API_KEY"},
	"vercel-ai-gateway":      {"AI_GATEWAY_API_KEY"},
	"zai":                    {"ZAI_API_KEY"},
	"mistral":                {"MISTRAL_API_KEY"},
	"minimax":                {"MINIMAX_API_KEY"},
	"minimax-cn":             {"MINIMAX_CN_API_KEY"},
	"moonshotai":             {"MOONSHOT_API_KEY"},
	"moonshotai-cn":          {"MOONSHOT_API_KEY"},
	"huggingface":            {"HF_TOKEN"},
	"fireworks":              {"FIREWORKS_API_KEY"},
	"opencode":               {"OPENCODE_API_KEY"},
	"opencode-go":            {"OPENCODE_API_KEY"},
	"kimi-coding":            {"KIMI_API_KEY"},
	"cloudflare-workers-ai":  {"CLOUDFLARE_API_KEY"},
	"cloudflare-ai-gateway":  {"CLOUDFLARE_API_KEY"},
	"xiaomi":                 {"XIAOMI_API_KEY"},
	"xiaomi-token-plan-cn":   {"XIAOMI_TOKEN_PLAN_CN_API_KEY"},
	"xiaomi-token-plan-ams":  {"XIAOMI_TOKEN_PLAN_AMS_API_KEY"},
	"xiaomi-token-plan-sgp":  {"XIAOMI_TOKEN_PLAN_SGP_API_KEY"},
}

// FindEnvAPIKeys returns the configured environment-variable names that can
// supply an API key or OAuth token for provider, in precedence order. It does
// not report ambient AWS or Google credentials. The returned slice is owned by
// the caller and is nil when no known variable is configured.
func FindEnvAPIKeys(provider string) []string {
	variables := providerAPIKeyEnvironment[provider]
	configured := make([]string, 0, len(variables))
	for _, variable := range variables {
		if os.Getenv(variable) != "" {
			configured = append(configured, variable)
		}
	}
	if len(configured) == 0 {
		return nil
	}
	return configured
}

// EnvAPIKey returns the highest-precedence configured API key or OAuth token
// for provider. Ambient AWS and Google credentials are intentionally reported
// by HasAmbientCredentials instead of being represented as a sentinel key.
func EnvAPIKey(provider string) (string, bool) {
	variables := FindEnvAPIKeys(provider)
	if len(variables) == 0 {
		return "", false
	}
	return os.Getenv(variables[0]), true
}

// HasAmbientCredentials reports whether the process has a discoverable
// non-API-key credential source for Google Vertex AI or Amazon Bedrock. It is a
// local availability hint, not proof that the credentials are valid or
// authorized for a particular model.
func HasAmbientCredentials(provider string) bool {
	switch provider {
	case ProviderGoogleVertex:
		return hasVertexADCCredentials()
	case ProviderAmazonBedrock:
		return os.Getenv("AWS_PROFILE") != "" ||
			os.Getenv("AWS_ACCESS_KEY_ID") != "" && os.Getenv("AWS_SECRET_ACCESS_KEY") != "" ||
			os.Getenv("AWS_BEARER_TOKEN_BEDROCK") != "" ||
			os.Getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") != "" ||
			os.Getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI") != "" ||
			os.Getenv("AWS_WEB_IDENTITY_TOKEN_FILE") != ""
	default:
		return false
	}
}

func hasVertexADCCredentials() bool {
	if os.Getenv("GOOGLE_CLOUD_PROJECT") == "" && os.Getenv("GCLOUD_PROJECT") == "" || os.Getenv("GOOGLE_CLOUD_LOCATION") == "" {
		return false
	}
	credentials := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if credentials == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return false
		}
		credentials = filepath.Join(home, ".config", "gcloud", "application_default_credentials.json")
	}
	info, err := os.Stat(credentials)
	return err == nil && !info.IsDir()
}
