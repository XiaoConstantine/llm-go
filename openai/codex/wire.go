package codex

import (
	"encoding/base64"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	internalresponses "github.com/XiaoConstantine/llm-go/internal/openairesponses"
	openairesponses "github.com/openai/openai-go/v3/responses"
)

const (
	accountClaim        = "https://api.openai.com/auth"
	providerDataAPI     = "openai-codex-responses"
	providerDataVersion = 1
	maxAudioInputBytes  = 50 << 20
)

type providerDataEnvelope struct {
	API     string           `json:"api"`
	Version int              `json:"version"`
	Model   string           `json:"model"`
	Output  []jsontext.Value `json:"output"`
}

// AccountIDFromToken extracts the ChatGPT account identifier from an OpenAI ID
// or access JWT without verifying its signature. The server verifies the token
// when it is used; this helper only reads the routing claim.
func AccountIDFromToken(token string) (string, error) {
	parts := strings.Split(strings.TrimSpace(token), ".")
	if len(parts) != 3 {
		return "", errors.New("access token is not a JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return "", fmt.Errorf("decode access token: %w", err)
	}
	var claims map[string]jsontext.Value
	if err := jsonv2.Unmarshal(payload, &claims); err != nil {
		return "", fmt.Errorf("decode access token claims: %w", err)
	}
	rawAuth, exists := claims[accountClaim]
	if !exists {
		return "", errors.New("access token is missing OpenAI auth claims")
	}
	var auth struct {
		AccountID string `json:"chatgpt_account_id"`
	}
	if err := jsonv2.Unmarshal(rawAuth, &auth); err != nil {
		return "", fmt.Errorf("decode OpenAI auth claims: %w", err)
	}
	accountID := strings.TrimSpace(auth.AccountID)
	if accountID == "" {
		return "", errors.New("access token is missing chatgpt_account_id")
	}
	return accountID, nil
}

func responseCodec() internalresponses.Codec {
	return internalresponses.Codec{
		Provider:             defaultProvider,
		ProviderDataAPI:      providerDataAPI,
		MaxProviderDataBytes: maxProviderDataBytes,
	}
}

func requestToWire(op, model string, request llm.Request) (openairesponses.ResponseNewParams, error) {
	return requestToWireWithCompatibility(op, model, request, llm.OpenAIResponsesCompatibility{})
}

func requestToWireWithCompatibility(op, model string, request llm.Request, compatibility llm.OpenAIResponsesCompatibility) (openairesponses.ResponseNewParams, error) {
	encryptedReasoning := compatibility.EncryptedReasoning != llm.CompatibilityDisabled
	return responseCodec().Request(op, model, request, internalresponses.RequestOptions{
		Subscription:       true,
		ReasoningSummary:   true,
		EncryptedReasoning: encryptedReasoning,
		StrictTools:        compatibility.StrictTools != llm.CompatibilityDisabled,
		AdditionalTools:    compatibility.AdditionalTools == llm.CompatibilityEnabled,
		ToolSearch:         compatibility.ToolSearch == llm.CompatibilityEnabled,
		InputAudio: &internalresponses.InputAudioOptions{
			MaxDecodedBytes: maxAudioInputBytes,
		},
	})
}
