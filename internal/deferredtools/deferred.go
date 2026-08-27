// Package deferredtools derives provider wire placements from portable tool
// introduction markers in a validated request.
package deferredtools

import llm "github.com/XiaoConstantine/llm-go"

// Position identifies a tool result within Request.Messages.
type Position struct {
	Message int
	Result  int
}

// Plan partitions request tools and records where deferred definitions become
// available while replaying history.
type Plan struct {
	Immediate  []llm.Tool
	Deferred   []llm.Tool
	placements map[Position][]llm.Tool
}

// Build creates a plan for a validated request. When supported is false, every
// tool is immediate. When requireImmediate is true, a plan that would defer all
// tools is changed to make every tool immediate.
func Build(request llm.Request, supported, requireImmediate bool) Plan {
	if !supported {
		return Plan{Immediate: append([]llm.Tool(nil), request.Tools...)}
	}
	byName := make(map[string]llm.Tool, len(request.Tools))
	for _, tool := range request.Tools {
		byName[tool.Name] = tool
	}
	used := make(map[string]struct{})
	deferred := make(map[string]struct{})
	placements := make(map[Position][]llm.Tool)
	for messageIndex, message := range request.Messages {
		for _, call := range message.ToolCalls {
			used[call.Name] = struct{}{}
		}
		for resultIndex, result := range message.ToolResults {
			position := Position{Message: messageIndex, Result: resultIndex}
			for _, name := range result.AddedToolNames {
				tool, exists := byName[name]
				if !exists {
					continue
				}
				if _, wasUsed := used[name]; wasUsed {
					continue
				}
				if _, alreadyDeferred := deferred[name]; alreadyDeferred {
					continue
				}
				deferred[name] = struct{}{}
				placements[position] = append(placements[position], tool)
			}
		}
	}
	if requireImmediate && len(deferred) == len(request.Tools) && len(deferred) != 0 {
		return Plan{Immediate: append([]llm.Tool(nil), request.Tools...)}
	}
	plan := Plan{placements: placements}
	for _, tool := range request.Tools {
		if _, ok := deferred[tool.Name]; ok {
			plan.Deferred = append(plan.Deferred, tool)
		} else {
			plan.Immediate = append(plan.Immediate, tool)
		}
	}
	return plan
}

// After returns definitions introduced immediately after one tool result. The
// returned slice aliases only the plan and must be treated as read-only.
func (p Plan) After(messageIndex, resultIndex int) []llm.Tool {
	return p.placements[Position{Message: messageIndex, Result: resultIndex}]
}
