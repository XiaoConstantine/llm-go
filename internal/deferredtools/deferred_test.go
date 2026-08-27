package deferredtools

import (
	"fmt"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestBuildPartitionsToolsAtFirstEligibleMarker(t *testing.T) {
	tools := []llm.Tool{{Name: "used"}, {Name: "later"}, {Name: "last"}}
	request := llm.Request{Tools: tools, Messages: []llm.Message{
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "call", Name: "used"}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{CallID: "call", AddedToolNames: []string{"used", "later"}}}},
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{Name: "unused", AddedToolNames: []string{"later", "last"}}}},
	}}
	plan := Build(request, true, false)
	if got := toolNames(plan.Immediate); fmt.Sprint(got) != "[used]" {
		t.Fatalf("Immediate = %v", got)
	}
	if got := toolNames(plan.Deferred); fmt.Sprint(got) != "[later last]" {
		t.Fatalf("Deferred = %v", got)
	}
	if got := toolNames(plan.After(1, 0)); fmt.Sprint(got) != "[later]" {
		t.Fatalf("After(1, 0) = %v", got)
	}
	if got := toolNames(plan.After(2, 0)); fmt.Sprint(got) != "[last]" {
		t.Fatalf("After(2, 0) = %v", got)
	}
}

func TestBuildUnsupportedAndAllDeferredFallback(t *testing.T) {
	request := llm.Request{Tools: []llm.Tool{{Name: "one"}}, Messages: []llm.Message{
		{Role: llm.RoleTool, ToolResults: []llm.ToolResult{{AddedToolNames: []string{"one"}}}},
	}}
	for _, plan := range []Plan{Build(request, false, false), Build(request, true, true)} {
		if got := toolNames(plan.Immediate); fmt.Sprint(got) != "[one]" || len(plan.Deferred) != 0 || len(plan.After(0, 0)) != 0 {
			t.Fatalf("fallback plan = %#v", plan)
		}
	}
}

func toolNames(tools []llm.Tool) []string {
	names := make([]string, len(tools))
	for index, tool := range tools {
		names[index] = tool.Name
	}
	return names
}
