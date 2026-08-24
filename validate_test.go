package llm

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestRequestValidate(t *testing.T) {
	zero := 0.0
	one := 1.0
	request := Request{
		Messages: []Message{
			{Role: RoleSystem, Content: []Part{{Text: "be useful"}}},
			{Role: RoleUser, Content: []Part{
				{Text: "inspect "},
				{Kind: PartImage, Data: []byte("image"), MediaType: "image/png"},
			}},
			{Role: RoleAssistant, ToolCalls: []ToolCall{
				{ID: "call-1", Name: "lookup", Arguments: []byte(`{"query":"first"}`)},
				{Name: "lookup", Arguments: []byte(`{"query":"second"}`)},
			}},
			{Role: RoleTool, ToolResults: []ToolResult{
				{CallID: "call-1", Name: "lookup", Content: []Part{{Text: "first result"}}},
				{Name: "lookup", Content: []Part{{Text: "second result"}}},
			}},
			{Role: RoleUser, Content: []Part{{Text: "continue"}}, ProviderData: []byte(`{"state":1}`)},
		},
		Tools: []Tool{{
			Name:        "lookup",
			Description: "looks something up",
			InputSchema: []byte(`{"type":"object"}`),
			Strict:      true,
		}},
		ResponseFormat:   ResponseFormatJSON,
		MaxOutputTokens:  100,
		Temperature:      &zero,
		TopP:             &one,
		PresencePenalty:  &zero,
		FrequencyPenalty: &zero,
		Stop:             []string{"done"},
	}

	if err := request.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestRequestValidateAllowsUnansweredToolCall(t *testing.T) {
	request := Request{Messages: []Message{
		{Role: RoleUser, Content: []Part{{Text: "look it up"}}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(`{}`)}}},
	}}

	if err := request.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestRequestValidateReasoningEffort(t *testing.T) {
	request := Request{
		Messages:        []Message{{Role: RoleUser, Content: []Part{{Text: "hello"}}}},
		ReasoningEffort: ReasoningEffort("extreme"),
	}
	if err := request.Validate(); err == nil || !strings.Contains(err.Error(), "reasoning effort") {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestRequestValidateToolHistory(t *testing.T) {
	tests := []struct {
		name     string
		messages []Message
		want     string
	}{
		{
			name: "identified results out of order",
			messages: []Message{
				{Role: RoleAssistant, ToolCalls: []ToolCall{
					{ID: "call-1", Name: "first", Arguments: []byte(`{}`)},
					{ID: "call-2", Name: "second", Arguments: []byte(`{}`)},
				}},
				{Role: RoleUser, Content: []Part{{Text: "intervening turn"}}},
				{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call-2"}}},
				{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call-1"}}},
			},
		},
		{
			name: "ID-less results by name across turns",
			messages: []Message{
				{Role: RoleAssistant, ToolCalls: []ToolCall{
					{Name: "first", Arguments: []byte(`{}`)},
					{Name: "second", Arguments: []byte(`{}`)},
				}},
				{Role: RoleUser, Content: []Part{{Text: "intervening turn"}}},
				{Role: RoleTool, ToolResults: []ToolResult{{Name: "second"}}},
				{Role: RoleTool, ToolResults: []ToolResult{{Name: "first"}}},
			},
		},
		{
			name: "identified call ID reused after result",
			messages: []Message{
				{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "first", Arguments: []byte(`{}`)}}},
				{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call-1"}}},
				{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "second", Arguments: []byte(`{}`)}}},
				{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call-1"}}},
			},
		},
		{
			name: "duplicate pending ID across turns",
			messages: []Message{
				{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "first", Arguments: []byte(`{}`)}}},
				{Role: RoleUser, Content: []Part{{Text: "intervening turn"}}},
				{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "second", Arguments: []byte(`{}`)}}},
			},
			want: `duplicate pending ID "call-1"`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := (Request{Messages: test.messages}).Validate()
			if test.want == "" {
				if err != nil {
					t.Fatalf("Validate() error = %v", err)
				}
				return
			}
			requireInvalidRequest(t, err, test.want)
		})
	}
}

func TestPendingToolCallsConsumesSameNameInOrder(t *testing.T) {
	var pending pendingToolCalls
	pending.add(ToolCall{Name: "lookup", Arguments: []byte(`{"order":1}`)})
	pending.add(ToolCall{Name: "lookup", Arguments: []byte(`{"order":2}`)})

	for _, want := range []string{`{"order":1}`, `{"order":2}`} {
		call, ok := pending.consume(ToolResult{Name: "lookup"})
		if !ok {
			t.Fatal("consume() found no matching call")
		}
		if got := string(call.Arguments); got != want {
			t.Fatalf("consume() arguments = %s, want %s", got, want)
		}
	}
}

func TestRequestValidateRejectsInvalidRequests(t *testing.T) {
	nan := math.NaN()
	negative := -1.0
	tooLarge := 1.1
	invalidUTF8 := string([]byte{0xff})

	tests := []struct {
		name    string
		request Request
		want    string
	}{
		{name: "empty messages", request: Request{}, want: "messages must not be empty"},
		{
			name:    "invalid role",
			request: Request{Messages: []Message{{Role: Role("critic")}}},
			want:    `role "critic" is invalid`,
		},
		{
			name: "tools on user message",
			request: Request{Messages: []Message{{
				Role:      RoleUser,
				ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(`{}`)}},
			}}},
			want: "user message must not contain tool calls or results",
		},
		{
			name: "results on assistant message",
			request: Request{Messages: []Message{{
				Role:        RoleAssistant,
				ToolResults: []ToolResult{{Name: "lookup"}},
			}}},
			want: "assistant message must not contain tool results",
		},
		{
			name: "content on tool message",
			request: Request{Messages: []Message{{
				Role:        RoleTool,
				Content:     []Part{{Text: "result"}},
				ToolResults: []ToolResult{{Name: "lookup"}},
			}}},
			want: "tool message content must be empty",
		},
		{
			name:    "empty tool message",
			request: Request{Messages: []Message{{Role: RoleTool}}},
			want:    "tool message must contain at least one result",
		},
		{
			name: "invalid text part",
			request: Request{Messages: []Message{{
				Role:    RoleUser,
				Content: []Part{{Text: "text", Data: []byte("data")}},
			}}},
			want: "text part must not contain data or media type",
		},
		{
			name: "empty image data",
			request: Request{Messages: []Message{{
				Role:    RoleUser,
				Content: []Part{{Kind: PartImage, MediaType: "image/png"}},
			}}},
			want: "binary part data must not be empty",
		},
		{
			name: "invalid UTF-8 text",
			request: Request{Messages: []Message{{
				Role:    RoleUser,
				Content: []Part{{Text: invalidUTF8}},
			}}},
			want: "text must be valid UTF-8",
		},
		{
			name: "invalid provider data",
			request: Request{Messages: []Message{{
				Role:         RoleUser,
				ProviderData: []byte(`{"duplicate":1,"duplicate":2}`),
			}}},
			want: "provider data: must contain strict JSON",
		},
		{
			name: "empty tool name",
			request: Request{
				Messages: []Message{{Role: RoleUser}},
				Tools:    []Tool{{InputSchema: []byte(`{}`)}},
			},
			want: "tools[0].name must not be empty",
		},
		{
			name: "duplicate tool name",
			request: Request{
				Messages: []Message{{Role: RoleUser}},
				Tools: []Tool{
					{Name: "lookup", InputSchema: []byte(`{}`)},
					{Name: "lookup", InputSchema: []byte(`{}`)},
				},
			},
			want: `tools[1].name "lookup" is duplicated`,
		},
		{
			name: "invalid tool schema",
			request: Request{
				Messages: []Message{{Role: RoleUser}},
				Tools:    []Tool{{Name: "lookup", InputSchema: []byte(`{`)}},
			},
			want: "input schema: must contain strict JSON",
		},
		{
			name: "invalid tool schema kind",
			request: Request{
				Messages: []Message{{Role: RoleUser}},
				Tools:    []Tool{{Name: "lookup", InputSchema: []byte(`null`)}},
			},
			want: "input schema: must contain a JSON Schema object or boolean",
		},
		{
			name:    "invalid response format",
			request: Request{Messages: []Message{{Role: RoleUser}}, ResponseFormat: ResponseFormat(255)},
			want:    "response format 255 is invalid",
		},
		{
			name:    "negative max output tokens",
			request: Request{Messages: []Message{{Role: RoleUser}}, MaxOutputTokens: -1},
			want:    "max output tokens must not be negative",
		},
		{
			name:    "non-finite temperature",
			request: Request{Messages: []Message{{Role: RoleUser}}, Temperature: &nan},
			want:    "temperature must be finite",
		},
		{
			name:    "negative temperature",
			request: Request{Messages: []Message{{Role: RoleUser}}, Temperature: &negative},
			want:    "temperature must not be negative",
		},
		{
			name:    "top-p out of range",
			request: Request{Messages: []Message{{Role: RoleUser}}, TopP: &tooLarge},
			want:    "top-p must be between 0 and 1",
		},
		{
			name:    "empty stop",
			request: Request{Messages: []Message{{Role: RoleUser}}, Stop: []string{""}},
			want:    "stop[0] must not be empty",
		},
		{
			name: "invalid tool call arguments",
			request: Request{Messages: []Message{{
				Role:      RoleAssistant,
				ToolCalls: []ToolCall{{Name: "lookup", Arguments: []byte(`{`)}},
			}}},
			want: "arguments: must contain strict JSON",
		},
		{
			name: "duplicate pending call ID",
			request: Request{Messages: []Message{{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{ID: "call-1", Name: "one", Arguments: []byte(`{}`)},
					{ID: "call-1", Name: "two", Arguments: []byte(`{}`)},
				},
			}}},
			want: `duplicate pending ID "call-1"`,
		},
		{
			name: "unmatched tool result",
			request: Request{Messages: []Message{{
				Role:        RoleTool,
				ToolResults: []ToolResult{{Name: "lookup"}},
			}}},
			want: "no matching preceding tool call",
		},
		{
			name: "name-only result cannot match ID call",
			request: Request{Messages: []Message{
				{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "lookup", Arguments: []byte(`{}`)}}},
				{Role: RoleTool, ToolResults: []ToolResult{{Name: "lookup"}}},
			}},
			want: "no matching preceding tool call",
		},
		{
			name: "mismatched result name",
			request: Request{Messages: []Message{
				{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call-1", Name: "lookup", Arguments: []byte(`{}`)}}},
				{Role: RoleTool, ToolResults: []ToolResult{{CallID: "call-1", Name: "other"}}},
			}},
			want: `name "other" does not match call name "lookup"`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requireInvalidRequest(t, test.request.Validate(), test.want)
		})
	}
}

func requireInvalidRequest(t *testing.T, err error, contains string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), contains) {
		t.Fatalf("Validate() error = %v, want containing %q", err, contains)
	}

	var modelErr *Error
	if !errors.As(err, &modelErr) {
		t.Fatalf("Validate() error type = %T, want *Error", err)
	}
	if modelErr.Kind != KindInvalidRequest {
		t.Fatalf("error kind = %v, want %v", modelErr.Kind, KindInvalidRequest)
	}
	if modelErr.Op != "validate" {
		t.Fatalf("error operation = %q, want %q", modelErr.Op, "validate")
	}
}
