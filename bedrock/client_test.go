package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

func TestGenerateTextReasoningUsageAndReplay(t *testing.T) {
	runtime := &fakeRuntime{events: []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
		deltaEvent(0, &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberText{Value: "think"}}),
		deltaEvent(0, &types.ContentBlockDeltaMemberReasoningContent{Value: &types.ReasoningContentBlockDeltaMemberSignature{Value: "sig"}}),
		stopEvent(0),
		deltaEvent(1, &types.ContentBlockDeltaMemberText{Value: "answer"}),
		stopEvent(1),
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn}},
		&types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{Usage: &types.TokenUsage{
			InputTokens: aws.Int32(4), OutputTokens: aws.Int32(3), CacheReadInputTokens: aws.Int32(2), TotalTokens: aws.Int32(9)}}},
	}}
	client := mustClient(t, Config{Model: "anthropic.claude-sonnet-4-6", Reasoning: true, Runtime: runtime})
	response, err := client.Generate(context.Background(), llm.Request{ReasoningEffort: llm.ReasoningEffortHigh,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "question"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	if response.Text() != "answer" || response.ReasoningSummary != "think" || response.FinishReason != llm.FinishReasonStop ||
		response.Model != "anthropic.claude-sonnet-4-6" {
		t.Fatalf("response = %#v", response)
	}
	if response.Usage == nil || response.Usage.InputTokens != 4 || response.Usage.OutputTokens != 3 ||
		response.Usage.CacheReadTokens != 2 || response.Usage.TotalTokens != 9 {
		t.Fatalf("usage = %#v", response.Usage)
	}
	data, ok := parseMessageData(response.Message.ProviderData, "anthropic.claude-sonnet-4-6")
	if !ok || len(data.Reasoning) != 1 || data.Reasoning[0].Text != "think" || data.Reasoning[0].Signature != "sig" {
		t.Fatalf("provider data = %#v, %v", data, ok)
	}

	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "question"}}}, response.Message}}
	input, err := buildInput("generate", "anthropic.claude-sonnet-4-6", true, request)
	if err != nil {
		t.Fatal(err)
	}
	assistant := input.Messages[1]
	if _, ok := assistant.Content[0].(*types.ContentBlockMemberReasoningContent); !ok {
		t.Fatalf("assistant replay = %#v", assistant.Content)
	}
	if text, ok := assistant.Content[1].(*types.ContentBlockMemberText); !ok || text.Value != "answer" {
		t.Fatalf("assistant replay order = %#v", assistant.Content)
	}
}

func TestStreamToolCallAndInputConversion(t *testing.T) {
	runtime := &fakeRuntime{events: []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
		&types.ConverseStreamOutputMemberContentBlockStart{Value: types.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(0),
			Start: &types.ContentBlockStartMemberToolUse{Value: types.ToolUseBlockStart{ToolUseId: aws.String("call-1"), Name: aws.String("read")}}}},
		deltaEvent(0, &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(`{"path":`)}}),
		deltaEvent(0, &types.ContentBlockDeltaMemberToolUse{Value: types.ToolUseBlockDelta{Input: aws.String(`"file"}`)}}),
		stopEvent(0),
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonToolUse}},
	}}
	client := mustClient(t, Config{Model: "model", Capabilities: []llm.Capability{llm.CapabilityStreaming, llm.CapabilityTools}, Runtime: runtime})
	tools := []llm.Tool{{Name: "read", Description: "read a file", InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`)}}
	stream, err := client.Stream(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "read"}}}}, Tools: tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	response, err := llm.Collect(stream, tools)
	if err != nil {
		t.Fatal(err)
	}
	if response.FinishReason != llm.FinishReasonToolCall || len(response.Message.ToolCalls) != 1 {
		t.Fatalf("response = %#v", response)
	}
	call := response.Message.ToolCalls[0]
	if call.ID != "call-1" || call.Name != "read" || string(call.Arguments) != `{"path":"file"}` {
		t.Fatalf("call = %#v", call)
	}
	if runtime.input == nil || runtime.input.ToolConfig == nil || len(runtime.input.ToolConfig.Tools) != 1 {
		t.Fatalf("input = %#v", runtime.input)
	}
}

func TestBuildInputCachingReasoningImagesAndToolResults(t *testing.T) {
	request := llm.Request{CacheRetention: llm.CacheRetentionLong, ReasoningEffort: llm.ReasoningEffortMedium,
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: []llm.Part{{Text: "system"}}},
			{Role: llm.RoleUser, Content: []llm.Part{{Text: "look"}, {Kind: llm.PartImage, MediaType: "image/png", Data: []byte("png")}}},
			{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "bad:id", Name: "read", Arguments: json.RawMessage(`{"path":"x"}`)}}},
			{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "bad:id", Name: "read", IsError: true, Content: []llm.Part{{Text: "failed"}}}}},
		}, Tools: []llm.Tool{{Name: "read", InputSchema: json.RawMessage(`{"type":"object"}`)}}}
	input, err := buildInput("generate", "anthropic.claude-sonnet-4-20250514-v1:0", true, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(input.System) != 2 || len(input.Messages) != 3 || input.ToolConfig == nil || input.AdditionalModelRequestFields == nil {
		t.Fatalf("input = %#v", input)
	}
	assistant := input.Messages[1].Content[0].(*types.ContentBlockMemberToolUse)
	if *assistant.Value.ToolUseId != "bad_id" {
		t.Fatalf("tool ID = %q", *assistant.Value.ToolUseId)
	}
	result := input.Messages[2].Content[0].(*types.ContentBlockMemberToolResult)
	if *result.Value.ToolUseId != "bad_id" || result.Value.Status != types.ToolResultStatusError {
		t.Fatalf("tool result = %#v", result.Value)
	}
}

func TestConsecutiveToolResultsAreMergedAndIDlessResultsPair(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{{Text: "run"}}},
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"n":1}`)},
			{Name: "read", Arguments: json.RawMessage(`{"n":2}`)},
		}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{Name: "read", Content: []llm.Part{{Text: "one"}}}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{Name: "read", Content: []llm.Part{{Text: "two"}}}}},
	}}
	input, err := buildInput("generate", "model", false, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(input.Messages) != 3 || len(input.Messages[2].Content) != 2 {
		t.Fatalf("messages = %#v", input.Messages)
	}
	first := input.Messages[1].Content[0].(*types.ContentBlockMemberToolUse).Value.ToolUseId
	second := input.Messages[1].Content[1].(*types.ContentBlockMemberToolUse).Value.ToolUseId
	firstResult := input.Messages[2].Content[0].(*types.ContentBlockMemberToolResult).Value.ToolUseId
	secondResult := input.Messages[2].Content[1].(*types.ContentBlockMemberToolResult).Value.ToolUseId
	if *first == *second || *first != *firstResult || *second != *secondResult {
		t.Fatalf("tool IDs = %q/%q results %q/%q", *first, *second, *firstResult, *secondResult)
	}
}

func TestSDKErrorClassification(t *testing.T) {
	tests := []struct {
		code string
		kind llm.ErrorKind
	}{
		{"UnrecognizedClientException", llm.KindAuthentication},
		{"AccessDeniedException", llm.KindPermission},
		{"ThrottlingException", llm.KindRateLimit},
		{"ValidationException", llm.KindInvalidRequest},
		{"ServiceUnavailableException", llm.KindProvider},
	}
	for _, test := range tests {
		err := sdkError("generate", &smithy.GenericAPIError{Code: test.code, Message: "failure"})
		var modelErr *llm.Error
		if !errors.As(err, &modelErr) || modelErr.Kind != test.kind || modelErr.Op != "generate" {
			t.Errorf("%s = %v (%#v)", test.code, err, modelErr)
		}
	}
	contextErr := sdkError("generate", &smithy.GenericAPIError{Code: "ValidationException", Message: "input token count exceeds context length"})
	var modelErr *llm.Error
	if !errors.As(contextErr, &modelErr) || modelErr.Kind != llm.KindContextLimit {
		t.Fatalf("context error = %v (%#v)", contextErr, modelErr)
	}
	responseErr := &smithyhttp.ResponseError{Response: &smithyhttp.Response{Response: &http.Response{
		StatusCode: http.StatusTooManyRequests, Header: http.Header{"Retry-After": []string{"2"}},
	}}, Err: &smithy.GenericAPIError{Code: "ThrottlingException", Message: "slow down"}}
	err := sdkError("stream", responseErr)
	if !errors.As(err, &modelErr) || modelErr.HTTPStatus != http.StatusTooManyRequests || modelErr.RetryAfter != 2*time.Second {
		t.Fatalf("retry error = %v (%#v)", err, modelErr)
	}
}

func TestCachingRequiresExplicitRetention(t *testing.T) {
	t.Setenv("PI_CACHE_RETENTION", "long")
	t.Setenv("AWS_BEDROCK_FORCE_CACHE", "1")
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}}
	input, err := buildInput("generate", "application-profile-arn", false, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(input.Messages[0].Content) != 1 {
		t.Fatalf("default retention content = %#v, want no cache point", input.Messages[0].Content)
	}

	request.CacheRetention = llm.CacheRetentionLong
	input, err = buildInput("generate", "application-profile-arn", false, request)
	if err != nil {
		t.Fatal(err)
	}
	point, ok := input.Messages[0].Content[1].(*types.ContentBlockMemberCachePoint)
	if !ok || point.Value.Ttl != types.CacheTTLOneHour {
		t.Fatalf("long retention cache point = %#v", input.Messages[0].Content)
	}
}

func TestReasoningBudgetValidation(t *testing.T) {
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}},
		ReasoningBudgetTokens: 512}
	if _, err := buildInput("generate", "anthropic.claude-3-7-sonnet", true, request); err == nil {
		t.Fatal("small reasoning budget succeeded")
	}
	request.ReasoningBudgetTokens, request.MaxOutputTokens = 2048, 2048
	if _, err := buildInput("generate", "anthropic.claude-3-7-sonnet", true, request); err == nil {
		t.Fatal("reasoning budget equal to max output succeeded")
	}
	request.ReasoningBudgetTokens, request.ReasoningEffort, request.MaxOutputTokens = 0, llm.ReasoningEffortHigh, 4096
	input, err := buildInput("generate", "anthropic.claude-3-7-sonnet", true, request)
	if err != nil {
		t.Fatal(err)
	}
	rawFields, err := input.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(rawFields, &fields); err != nil {
		t.Fatal(err)
	}
	thinking, ok := fields["thinking"].(map[string]any)
	if !ok || thinking["budget_tokens"] != float64(3072) {
		t.Fatalf("thinking fields = %#v", fields)
	}
	request.ReasoningBudgetTokens, request.MaxOutputTokens = 2048, 0
	if _, err := buildInput("generate", "anthropic.claude-sonnet-4-6", true, request); err == nil {
		t.Fatal("explicit adaptive reasoning budget succeeded")
	}
}

func TestCanceledContextPrecedesRuntime(t *testing.T) {
	runtime := &fakeRuntime{}
	client := mustClient(t, Config{Model: "model", Runtime: runtime})
	ctx, cancel := context.WithCancelCause(context.Background())
	cause := errors.New("stop")
	cancel(cause)
	response, err := client.Generate(ctx, llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}})
	if response != nil || !errors.Is(err, context.Canceled) || !errors.Is(err, cause) || runtime.calls != 0 {
		t.Fatalf("Generate() = %#v, %v, calls=%d", response, err, runtime.calls)
	}
}

func TestPreflightAndMalformedStream(t *testing.T) {
	runtime := &fakeRuntime{}
	client := mustClient(t, Config{Model: "model", Runtime: runtime})
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}},
		ReasoningEffort: llm.ReasoningEffortHigh}
	if response, err := client.Generate(context.Background(), request); response != nil || err == nil || runtime.calls != 0 {
		t.Fatalf("Generate() = %#v, %v, calls=%d", response, err, runtime.calls)
	}

	runtime.events = []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
		deltaEvent(0, &types.ContentBlockDeltaMemberText{Value: "unterminated"}),
	}
	response, err := client.Generate(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hi"}}}}})
	if response != nil || err == nil {
		t.Fatalf("Generate() = %#v, %v", response, err)
	}
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindMalformedResponse {
		t.Fatalf("error = %v (%#v)", err, modelErr)
	}
}

func TestInfoOwnershipAndConfiguration(t *testing.T) {
	capabilities := []llm.Capability{llm.CapabilityStreaming}
	client := mustClient(t, Config{Provider: "custom-bedrock", Model: "model", Reasoning: true, Capabilities: capabilities, Runtime: &fakeRuntime{}})
	capabilities[0] = llm.CapabilityAudio
	info := client.Info()
	if info.Provider != "custom-bedrock" || info.API != llm.APIBedrockConverseStream || !info.Reasoning || info.Capabilities[1] != llm.CapabilityStreaming {
		t.Fatalf("Info() = %#v", info)
	}
	if _, err := New(Config{Runtime: &fakeRuntime{}}); err == nil {
		t.Fatal("empty model succeeded")
	}
	var nilRuntime *fakeRuntime
	if _, err := New(Config{Model: "model", Runtime: nilRuntime}); err == nil {
		t.Fatal("typed nil runtime succeeded")
	}
	if standardEndpointRegion("https://bedrock-runtime.us-west-2.amazonaws.com") != "us-west-2" ||
		shouldPinEndpoint("https://bedrock-runtime.us-east-1.amazonaws.com", "us-west-2", "") {
		t.Fatal("standard endpoint resolution is incorrect")
	}
	if got := normalizeToolID(strings.Repeat("a", 70)); len(got) != 64 {
		t.Fatalf("normalized ID length = %d", len(got))
	}
}

func mustClient(t *testing.T, config Config) *Client {
	t.Helper()
	client, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func deltaEvent(index int32, delta types.ContentBlockDelta) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(index), Delta: delta}}
}

func stopEvent(index int32) types.ConverseStreamOutput {
	return &types.ConverseStreamOutputMemberContentBlockStop{Value: types.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(index)}}
}

type fakeRuntime struct {
	events []types.ConverseStreamOutput
	err    error
	input  *bedrockruntime.ConverseStreamInput
	calls  int
}

func (runtime *fakeRuntime) ConverseStream(_ context.Context, input *bedrockruntime.ConverseStreamInput) (EventStream, error) {
	runtime.calls++
	runtime.input = input
	if runtime.err != nil {
		return nil, runtime.err
	}
	channel := make(chan types.ConverseStreamOutput, len(runtime.events))
	for _, event := range runtime.events {
		channel <- event
	}
	close(channel)
	return &fakeEventStream{events: channel}, nil
}

type fakeEventStream struct {
	events chan types.ConverseStreamOutput
	err    error
	closed bool
}

func (stream *fakeEventStream) Events() <-chan types.ConverseStreamOutput { return stream.events }
func (stream *fakeEventStream) Err() error                                { return stream.err }
func (stream *fakeEventStream) Close() error {
	stream.closed = true
	return nil
}
