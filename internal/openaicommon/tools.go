package openaicommon

import (
	"fmt"

	llm "github.com/XiaoConstantine/llm-go"
)

// CheckToolHistory requires tool results immediately after their calls.
// Request.Validate must already have checked call/result identity.
func CheckToolHistory(provider, op string, messages []llm.Message) error {
	pending := 0
	for i, message := range messages {
		if pending != 0 && message.Role != llm.RoleTool {
			return &llm.Error{Kind: llm.KindInvalidRequest, Provider: provider, Op: op,
				Err: fmt.Errorf("messages[%d] must contain tool results for %d pending calls", i, pending)}
		}
		switch message.Role {
		case llm.RoleAssistant:
			pending = len(message.ToolCalls)
		case llm.RoleTool:
			pending -= len(message.ToolResults)
		}
	}
	if pending != 0 {
		return &llm.Error{Kind: llm.KindInvalidRequest, Provider: provider, Op: op,
			Err: fmt.Errorf("messages end with %d pending tool calls", pending)}
	}
	return nil
}

// ValidFunctionName implements OpenAI's 1-64 ASCII function-name syntax.
func ValidFunctionName(name string) bool {
	if len(name) == 0 || len(name) > 64 {
		return false
	}
	for i := range len(name) {
		character := name[i]
		if character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || character == '_' || character == '-' {
			continue
		}
		return false
	}
	return true
}
