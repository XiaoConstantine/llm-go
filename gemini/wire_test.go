package gemini

import (
	"bufio"
	"errors"
	"testing"

	"encoding/json"

	llm "github.com/XiaoConstantine/llm-go"
	"google.golang.org/genai"
)

func TestResponseFromSDKRepresentsBlockedOutputWithoutContent(t *testing.T) {
	tests := []struct {
		name     string
		response *genai.GenerateContentResponse
	}{
		{
			name: "prompt blocked",
			response: &genai.GenerateContentResponse{
				PromptFeedback: &genai.GenerateContentResponsePromptFeedback{BlockReason: genai.BlockedReasonSafety},
			},
		},
		{
			name: "candidate blocked",
			response: &genai.GenerateContentResponse{Candidates: []*genai.Candidate{{
				FinishReason: genai.FinishReasonSafety,
			}}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			response, err := responseFromSDK("model", textRequest("hello"), test.response)
			if err != nil {
				t.Fatalf("responseFromSDK() error = %v", err)
			}
			if response.FinishReason != llm.FinishReasonContentFilter || response.Message.Role != llm.RoleAssistant || response.Text() != "" {
				t.Fatalf("responseFromSDK() = %#v", response)
			}
		})
	}
}

func TestResponseFromSDKRejectsBlockedPromptWithCandidate(t *testing.T) {
	for _, candidate := range []*genai.Candidate{
		{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "visible"}}}, FinishReason: genai.FinishReasonStop},
		{FinishReason: genai.FinishReasonSafety},
	} {
		response, err := responseFromSDK("model", textRequest("hello"), &genai.GenerateContentResponse{
			PromptFeedback: &genai.GenerateContentResponsePromptFeedback{BlockReason: genai.BlockedReasonSafety},
			Candidates:     []*genai.Candidate{candidate},
		})
		if response != nil {
			t.Fatalf("responseFromSDK() response = %#v, want nil", response)
		}
		requireModelError(t, err, llm.KindMalformedResponse, "generate", defaultProvider)
	}
}

func TestResponseFromSDKTreatsUnspecifiedPromptFeedbackAsNonBlocking(t *testing.T) {
	response, err := responseFromSDK("model", textRequest("hello"), &genai.GenerateContentResponse{
		PromptFeedback: &genai.GenerateContentResponsePromptFeedback{BlockReason: genai.BlockedReasonUnspecified},
		Candidates: []*genai.Candidate{{
			Content:      &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "visible"}}},
			FinishReason: genai.FinishReasonStop,
		}},
	})
	if err != nil || response.Text() != "visible" || response.FinishReason != llm.FinishReasonStop {
		t.Fatalf("responseFromSDK() = (%#v, %v)", response, err)
	}
}

func TestResponseFromSDKRejectsMissingUnblockedContent(t *testing.T) {
	response, err := responseFromSDK("model", textRequest("hello"), &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{FinishReason: genai.FinishReasonStop}},
	})
	if response != nil {
		t.Fatalf("responseFromSDK() response = %#v, want nil", response)
	}
	requireModelError(t, err, llm.KindMalformedResponse, "generate", defaultProvider)
}

func TestAssistantProviderDataRecognition(t *testing.T) {
	message := llm.Message{
		Role:    llm.RoleAssistant,
		Content: []llm.Part{{Text: "answer"}},
	}

	t.Run("other provider is ignored", func(t *testing.T) {
		message.ProviderData = json.RawMessage(`{"provider":"other","version":1,"data":null}`)
		parts, err := assistantPartsToSDK("generate", message)
		if err != nil {
			t.Fatalf("assistantPartsToSDK() error = %v", err)
		}
		if len(parts) != 1 || parts[0].Text != "answer" {
			t.Fatalf("assistantPartsToSDK() = %#v", parts)
		}
	})

	t.Run("future Gemini version is ignored", func(t *testing.T) {
		message.ProviderData = json.RawMessage(`{"provider":"gemini","version":2,"data":null}`)
		parts, err := assistantPartsToSDK("generate", message)
		if err != nil {
			t.Fatalf("assistantPartsToSDK() error = %v", err)
		}
		if len(parts) != 1 || parts[0].Text != "answer" {
			t.Fatalf("assistantPartsToSDK() = %#v", parts)
		}
	})

	t.Run("corrupt recognized data is rejected", func(t *testing.T) {
		message.ProviderData = json.RawMessage(`{"provider":"gemini","version":1,"data":null}`)
		_, err := assistantPartsToSDK("generate", message)
		requireModelError(t, err, llm.KindInvalidRequest, "generate", defaultProvider)
	})
}

func TestThoughtOnlyAssistantRoundTrip(t *testing.T) {
	request := textRequest("hello")
	response, err := responseFromSDK("model", request, &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{
				Text: "hidden", Thought: true, ThoughtSignature: []byte("signature"),
			}}},
			FinishReason: genai.FinishReasonStop,
		}},
	})
	if err != nil {
		t.Fatalf("responseFromSDK() error = %v", err)
	}
	if len(response.Message.Content) != 0 || len(response.Message.ToolCalls) != 0 || len(response.Message.ProviderData) == 0 {
		t.Fatalf("response message = %#v", response.Message)
	}
	roundTrip := llm.Request{Messages: []llm.Message{
		request.Messages[0],
		response.Message,
		{Role: llm.RoleUser, Content: []llm.Part{{Text: "continue"}}},
	}}
	if err := checkRequest("generate", roundTrip); err != nil {
		t.Fatalf("checkRequest() error = %v", err)
	}
	contents, _, err := requestToSDK("generate", roundTrip)
	if err != nil {
		t.Fatalf("requestToSDK() error = %v", err)
	}
	parts := contents[1].Parts
	if len(parts) != 1 || !parts[0].Thought || parts[0].Text != "hidden" || string(parts[0].ThoughtSignature) != "signature" {
		t.Fatalf("round-tripped thought parts = %#v", parts)
	}
}

func TestSDKErrorHandlesWrappedAPIErrors(t *testing.T) {
	apiErr := genai.APIError{Code: 429, Status: "RESOURCE_EXHAUSTED", Message: "slow down"}
	err := sdkError("generate", errors.Join(errors.New("gateway"), apiErr))
	modelErr := requireModelError(t, err, llm.KindRateLimit, "generate", defaultProvider)
	var gotAPIError genai.APIError
	if modelErr.HTTPStatus != 429 || !errors.As(err, &gotAPIError) || gotAPIError.Code != 429 {
		t.Fatalf("sdkError() = %#v", modelErr)
	}
}

func TestClassifyAPIErrorPrefersAuthenticationStatus(t *testing.T) {
	apiErr := genai.APIError{
		Code:    401,
		Status:  "UNAUTHENTICATED",
		Message: "invalid input token",
	}
	if got := classifyAPIError(apiErr); got != llm.KindAuthentication {
		t.Fatalf("classifyAPIError() = %v, want %v", got, llm.KindAuthentication)
	}
}

func TestSDKErrorClassifiesOversizedStreamEvent(t *testing.T) {
	err := sdkError("stream", bufio.ErrTooLong)
	requireModelError(t, err, llm.KindMalformedResponse, "stream", defaultProvider)
}

func TestToolChoiceBudgetAndImageToolResultWire(t *testing.T) {
	request := llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "go"}}},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "inspect", Arguments: []byte(`{}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", Content: []llm.Part{{Text: "screen"}, {Kind: llm.PartImage, Data: []byte{3}, MediaType: "image/webp"}}}}},
		},
		Tools:                 []llm.Tool{{Name: "inspect", InputSchema: []byte(`{"type":"object"}`)}},
		ToolChoice:            llm.ToolChoice{Mode: llm.ToolChoiceNamed, Name: "inspect"},
		ReasoningBudgetTokens: 2048,
	}
	contents, config, err := requestToSDK("generate", request)
	if err != nil {
		t.Fatal(err)
	}
	calling := config.ToolConfig.FunctionCallingConfig
	if calling.Mode != genai.FunctionCallingConfigModeAny || len(calling.AllowedFunctionNames) != 1 || calling.AllowedFunctionNames[0] != "inspect" {
		t.Fatalf("function calling config = %#v", calling)
	}
	if config.ThinkingConfig == nil || config.ThinkingConfig.ThinkingBudget == nil || *config.ThinkingConfig.ThinkingBudget != 2048 {
		t.Fatalf("thinking config = %#v", config.ThinkingConfig)
	}
	response := contents[2].Parts[0].FunctionResponse
	if response == nil || len(response.Parts) != 1 || response.Parts[0].InlineData == nil || response.Parts[0].InlineData.MIMEType != "image/webp" || response.Parts[0].InlineData.Data[0] != 3 {
		t.Fatalf("function response = %#v", response)
	}
	request.CacheRetention = llm.CacheRetentionShort
	if _, _, err := requestToSDK("generate", request); err == nil {
		t.Fatal("Gemini cache control unexpectedly succeeded")
	}
}
