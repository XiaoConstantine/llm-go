package llm

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"slices"
	"strings"
)

// TransformWarningCode identifies one explicit history transformation or repair.
type TransformWarningCode string

const (
	TransformRemovedProviderData     TransformWarningCode = "removed_provider_data"
	TransformReplacedImage           TransformWarningCode = "replaced_image"
	TransformReplacedAudio           TransformWarningCode = "replaced_audio"
	TransformNormalizedToolCallID    TransformWarningCode = "normalized_tool_call_id"
	TransformAddedMissingToolResult  TransformWarningCode = "added_missing_tool_result"
	TransformRemovedOrphanToolResult TransformWarningCode = "removed_orphan_tool_result"
	TransformCorrectedToolResultName TransformWarningCode = "corrected_tool_result_name"
)

// TransformWarning describes one transformed or repaired history item.
type TransformWarning struct {
	Code    TransformWarningCode
	Path    string
	Message string
}

// TransformHistory creates a caller-owned history for target. ProviderData is
// preserved only for the exact same provider, protocol, and model because it may
// contain model-bound signatures or encrypted state. Unsupported image and audio
// parts are replaced with visible text placeholders and individually reported.
// OpenAI Chat tool-result images are preserved only when its explicit image
// fallback compatibility is enabled; tool-result audio is always converted.
// Tool history targeting a model without tool capability fails rather than being
// silently omitted. Unless source and target have the same complete identity,
// tool-call IDs are normalized to a target-compatible form; duplicate IDs are
// normalized in all cases. Missing tool results are inserted before the next
// non-tool turn and at the end of the transcript, orphan results are removed,
// and result names that conflict with their matched call are corrected. Every
// repair is reported. Input messages and nested storage are never modified.
func TransformHistory(source, target ModelInfo, history []Message) ([]Message, []TransformWarning, error) {
	if err := validateTransformHistory(history); err != nil {
		return nil, nil, &Error{Kind: KindInvalidRequest, Op: "transform", Err: fmt.Errorf("source history: %w", err)}
	}
	sameModel := strings.TrimSpace(source.Provider) != "" && strings.TrimSpace(string(source.API)) != "" && strings.TrimSpace(source.Model) != "" &&
		strings.TrimSpace(target.Provider) != "" && strings.TrimSpace(string(target.API)) != "" && strings.TrimSpace(target.Model) != "" &&
		source.Provider == target.Provider && source.API == target.API && source.Model == target.Model
	targetVision := modelHasCapability(target, CapabilityVision)
	targetAudio := modelHasCapability(target, CapabilityAudio)
	toolResultVision := targetVision && supportsImageToolResults(target)
	targetTools := modelHasCapability(target, CapabilityTools)

	output := make([]Message, 0, len(history))
	var warnings []TransformWarning
	var pending transformedToolCalls
	usedToolCallIDs := make(map[string]struct{})
	nextToolCallIDSequence := make(map[string]int)
	for messageIndex, message := range history {
		if !targetTools && (len(message.ToolCalls) != 0 || len(message.ToolResults) != 0) {
			return nil, warnings, &Error{Kind: KindUnsupported, Op: "transform", Provider: target.Provider,
				Err: fmt.Errorf("messages[%d] contains tool history unsupported by target model", messageIndex)}
		}
		if message.Role != RoleTool {
			output, warnings = appendMissingToolResults(output, pending.takeAll(), messageIndex, warnings)
		}
		warningCount := len(warnings)
		converted := Message{Role: message.Role,
			Content: make([]Part, len(message.Content))}
		for partIndex, part := range message.Content {
			converted.Content[partIndex], warnings = transformPart(part, targetVision, targetAudio,
				fmt.Sprintf("messages[%d].content[%d]", messageIndex, partIndex), warnings)
		}
		converted.ToolCalls = make([]ToolCall, len(message.ToolCalls))
		messageChanged := false
		for callIndex, call := range message.ToolCalls {
			converted.ToolCalls[callIndex] = cloneToolCall(call)
			if call.ID != "" && !sameModel {
				converted.ToolCalls[callIndex].ID = portableToolCallID(call.ID, target, usedToolCallIDs, nextToolCallIDSequence)
			} else if call.ID != "" {
				if _, duplicate := usedToolCallIDs[call.ID]; duplicate {
					converted.ToolCalls[callIndex].ID = portableToolCallID(call.ID, target, usedToolCallIDs, nextToolCallIDSequence)
				} else {
					usedToolCallIDs[call.ID] = struct{}{}
				}
			} else if !sameModel && target.API != APIGeminiGenerateContent {
				converted.ToolCalls[callIndex].ID = portableToolCallID(generatedToolCallID(messageIndex, callIndex, call), target,
					usedToolCallIDs, nextToolCallIDSequence)
			}
			if converted.ToolCalls[callIndex].ID != call.ID {
				messageChanged = true
				warnings = append(warnings, TransformWarning{Code: TransformNormalizedToolCallID,
					Path:    fmt.Sprintf("messages[%d].tool calls[%d].ID", messageIndex, callIndex),
					Message: fmt.Sprintf("normalized tool-call ID %q to %q", call.ID, converted.ToolCalls[callIndex].ID)})
			}
			pending.add(call, converted.ToolCalls[callIndex])
		}
		converted.ToolResults = make([]ToolResult, 0, len(message.ToolResults))
		for resultIndex, result := range message.ToolResults {
			convertedResult := ToolResult{CallID: result.CallID, Name: result.Name, IsError: result.IsError,
				Content: make([]Part, len(result.Content)), AddedToolNames: append([]string(nil), result.AddedToolNames...)}
			call, ok := pending.consume(result)
			if !ok {
				messageChanged = true
				warnings = append(warnings, TransformWarning{Code: TransformRemovedOrphanToolResult,
					Path:    fmt.Sprintf("messages[%d].tool results[%d]", messageIndex, resultIndex),
					Message: "removed tool result because it has no pending matching call"})
				continue
			}
			if call.converted.ID != "" {
				convertedResult.CallID = call.converted.ID
			}
			messageChanged = messageChanged || convertedResult.CallID != result.CallID
			if convertedResult.Name != "" && convertedResult.Name != call.converted.Name {
				messageChanged = true
				warnings = append(warnings, TransformWarning{Code: TransformCorrectedToolResultName,
					Path:    fmt.Sprintf("messages[%d].tool results[%d].name", messageIndex, resultIndex),
					Message: fmt.Sprintf("corrected tool-result name %q to %q", convertedResult.Name, call.converted.Name)})
				convertedResult.Name = call.converted.Name
			}
			for partIndex, part := range result.Content {
				convertedResult.Content[partIndex], warnings = transformPart(part, toolResultVision, false,
					fmt.Sprintf("messages[%d].tool results[%d].content[%d]", messageIndex, resultIndex, partIndex), warnings)
			}
			converted.ToolResults = append(converted.ToolResults, convertedResult)
		}
		messageChanged = messageChanged || len(warnings) != warningCount
		if len(message.ProviderData) != 0 {
			if sameModel && !messageChanged {
				converted.ProviderData = append([]byte(nil), message.ProviderData...)
			} else {
				warnings = append(warnings, TransformWarning{Code: TransformRemovedProviderData,
					Path:    fmt.Sprintf("messages[%d].provider data", messageIndex),
					Message: "removed provider data because model identity differs or the message was transformed"})
			}
		}
		if message.Role == RoleTool && len(converted.ToolResults) == 0 {
			continue
		}
		output = append(output, converted)
	}
	output, warnings = appendMissingToolResults(output, pending.takeAll(), len(history), warnings)
	if err := validateRepairedToolHistory(output); err != nil {
		return nil, warnings, &Error{Kind: KindInvalidRequest, Op: "transform", Provider: target.Provider,
			Err: fmt.Errorf("transformed history: %w", err)}
	}
	return output, warnings, nil
}

type transformedToolCall struct {
	converted ToolCall
	consumed  bool
}

type transformedToolCalls struct {
	ordered []*transformedToolCall
	byID    map[string][]*transformedToolCall
	byName  map[string][]*transformedToolCall
	pending int
}

func (p *transformedToolCalls) add(source, converted ToolCall) {
	call := &transformedToolCall{converted: converted}
	p.ordered = append(p.ordered, call)
	p.pending++
	if source.ID != "" {
		if p.byID == nil {
			p.byID = make(map[string][]*transformedToolCall)
		}
		p.byID[source.ID] = append(p.byID[source.ID], call)
		return
	}
	if p.byName == nil {
		p.byName = make(map[string][]*transformedToolCall)
	}
	p.byName[source.Name] = append(p.byName[source.Name], call)
}

func (p *transformedToolCalls) consume(result ToolResult) (*transformedToolCall, bool) {
	var call *transformedToolCall
	if result.CallID != "" {
		queue := p.byID[result.CallID]
		if len(queue) != 0 {
			call = queue[0]
			if len(queue) == 1 {
				delete(p.byID, result.CallID)
			} else {
				p.byID[result.CallID] = queue[1:]
			}
		}
	} else {
		queue := p.byName[result.Name]
		if len(queue) != 0 {
			call = queue[0]
			if len(queue) == 1 {
				delete(p.byName, result.Name)
			} else {
				p.byName[result.Name] = queue[1:]
			}
		}
	}
	if call == nil {
		return nil, false
	}
	call.consumed = true
	p.pending--
	return call, true
}

func (p *transformedToolCalls) takeAll() []*transformedToolCall {
	if p.pending == 0 {
		p.ordered = nil
		clear(p.byID)
		clear(p.byName)
		return nil
	}
	calls := make([]*transformedToolCall, 0, p.pending)
	for _, call := range p.ordered {
		if !call.consumed {
			calls = append(calls, call)
		}
	}
	p.ordered = nil
	clear(p.byID)
	clear(p.byName)
	p.pending = 0
	return calls
}

func appendMissingToolResults(output []Message, calls []*transformedToolCall, beforeMessage int, warnings []TransformWarning) ([]Message, []TransformWarning) {
	if len(calls) == 0 {
		return output, warnings
	}
	results := make([]ToolResult, len(calls))
	for index, call := range calls {
		results[index] = ToolResult{CallID: call.converted.ID, Name: call.converted.Name, IsError: true,
			Content: []Part{{Text: "No result provided"}}}
		warnings = append(warnings, TransformWarning{Code: TransformAddedMissingToolResult,
			Path:    fmt.Sprintf("messages[%d]", beforeMessage),
			Message: fmt.Sprintf("added missing result for tool call %q", call.converted.Name)})
	}
	return append(output, Message{Role: RoleTool, ToolResults: results}), warnings
}

func portableToolCallID(value string, target ModelInfo, used map[string]struct{}, nextSequence map[string]int) string {
	limit := 64
	if target.Provider == "openai" && target.API == APIOpenAIChatCompletions {
		limit = 40
	}
	var base strings.Builder
	base.Grow(len(value))
	for _, character := range value {
		if character >= 'A' && character <= 'Z' || character >= 'a' && character <= 'z' ||
			character >= '0' && character <= '9' || character == '_' || character == '-' {
			base.WriteRune(character)
		} else {
			base.WriteByte('_')
		}
	}
	normalized := base.String()
	if len(normalized) <= limit {
		if _, exists := used[normalized]; !exists {
			used[normalized] = struct{}{}
			return normalized
		}
	}
	for sequence := nextSequence[value]; ; sequence++ {
		seed := value
		if sequence != 0 {
			seed = fmt.Sprintf("%s#%d", value, sequence)
		}
		digest := sha256.Sum256([]byte(seed))
		suffix := "_" + hex.EncodeToString(digest[:6])
		prefixLength := limit - len(suffix)
		prefix := normalized
		if len(prefix) > prefixLength {
			prefix = prefix[:prefixLength]
		}
		candidate := prefix + suffix
		if _, exists := used[candidate]; !exists {
			used[candidate] = struct{}{}
			nextSequence[value] = sequence + 1
			return candidate
		}
	}
}

func generatedToolCallID(messageIndex, callIndex int, call ToolCall) string {
	digest := sha256.Sum256([]byte(fmt.Sprintf("%d:%d:%s:%s", messageIndex, callIndex, call.Name, call.Arguments)))
	return "call_llm_go_" + hex.EncodeToString(digest[:8])
}

func validateTransformHistory(history []Message) error {
	if len(history) == 0 {
		return fmt.Errorf("messages must not be empty")
	}
	for index, message := range history {
		if err := validateMessage(message); err != nil {
			return fmt.Errorf("messages[%d]: %w", index, err)
		}
	}
	return nil
}

func validateRepairedToolHistory(history []Message) error {
	if err := validateTransformHistory(history); err != nil {
		return err
	}
	var pending transformedToolCalls
	seenIDs := make(map[string]struct{})
	for messageIndex, message := range history {
		if message.Role != RoleTool && pending.pending != 0 {
			return fmt.Errorf("messages[%d] follows %d tool calls without results", messageIndex, pending.pending)
		}
		for callIndex, call := range message.ToolCalls {
			if call.ID != "" {
				if _, duplicate := seenIDs[call.ID]; duplicate {
					return fmt.Errorf("messages[%d].tool calls[%d] repeats ID %q", messageIndex, callIndex, call.ID)
				}
				seenIDs[call.ID] = struct{}{}
			}
			pending.add(call, call)
		}
		for resultIndex, result := range message.ToolResults {
			call, ok := pending.consume(result)
			if !ok {
				return fmt.Errorf("messages[%d].tool results[%d] has no pending matching call", messageIndex, resultIndex)
			}
			if result.Name != "" && result.Name != call.converted.Name {
				return fmt.Errorf("messages[%d].tool results[%d] name %q does not match call name %q",
					messageIndex, resultIndex, result.Name, call.converted.Name)
			}
		}
	}
	if pending.pending != 0 {
		return fmt.Errorf("messages end with %d tool calls without results", pending.pending)
	}
	return nil
}

func transformPart(part Part, vision, audio bool, path string, warnings []TransformWarning) (Part, []TransformWarning) {
	switch part.Kind {
	case PartImage:
		if !vision {
			warnings = append(warnings, TransformWarning{Code: TransformReplacedImage, Path: path,
				Message: "replaced image with text because target model does not support images"})
			return Part{Kind: PartText, Text: "(image omitted: target model does not support images)"}, warnings
		}
	case PartAudio:
		if !audio {
			warnings = append(warnings, TransformWarning{Code: TransformReplacedAudio, Path: path,
				Message: "replaced audio with text because target model does not support audio"})
			return Part{Kind: PartText, Text: "(audio omitted: target model does not support audio)"}, warnings
		}
	}
	return clonePart(part), warnings
}

func supportsImageToolResults(info ModelInfo) bool {
	if info.API != APIOpenAIChatCompletions {
		return true
	}
	return info.Compatibility != nil && info.Compatibility.OpenAIChat != nil &&
		info.Compatibility.OpenAIChat.ToolResultImageFallback == CompatibilityEnabled
}

func modelHasCapability(info ModelInfo, target Capability) bool {
	return slices.Contains(info.Capabilities, target)
}
