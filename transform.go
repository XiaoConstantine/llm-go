package llm

import (
	"fmt"
	"strings"
)

// TransformWarningCode identifies one explicit lossy history transformation.
type TransformWarningCode string

const (
	TransformRemovedProviderData TransformWarningCode = "removed_provider_data"
	TransformReplacedImage       TransformWarningCode = "replaced_image"
	TransformReplacedAudio       TransformWarningCode = "replaced_audio"
)

// TransformWarning describes one removed or replaced history item.
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
// silently omitted. Input messages and nested storage are never modified.
func TransformHistory(source, target ModelInfo, history []Message) ([]Message, []TransformWarning, error) {
	if err := (Request{Messages: history}).Validate(); err != nil {
		return nil, nil, &Error{Kind: KindInvalidRequest, Op: "transform", Err: fmt.Errorf("source history: %w", err)}
	}
	sameModel := strings.TrimSpace(source.Provider) != "" && strings.TrimSpace(string(source.API)) != "" && strings.TrimSpace(source.Model) != "" &&
		strings.TrimSpace(target.Provider) != "" && strings.TrimSpace(string(target.API)) != "" && strings.TrimSpace(target.Model) != "" &&
		source.Provider == target.Provider && source.API == target.API && source.Model == target.Model
	targetVision := modelHasCapability(target, CapabilityVision)
	targetAudio := modelHasCapability(target, CapabilityAudio)
	toolResultVision := targetVision && supportsImageToolResults(target)
	targetTools := modelHasCapability(target, CapabilityTools)

	output := make([]Message, len(history))
	var warnings []TransformWarning
	for messageIndex, message := range history {
		if !targetTools && (len(message.ToolCalls) != 0 || len(message.ToolResults) != 0) {
			return nil, warnings, &Error{Kind: KindUnsupported, Op: "transform", Provider: target.Provider,
				Err: fmt.Errorf("messages[%d] contains tool history unsupported by target model", messageIndex)}
		}
		converted := Message{Role: message.Role}
		converted.Content = make([]Part, len(message.Content))
		for partIndex, part := range message.Content {
			converted.Content[partIndex], warnings = transformPart(part, targetVision, targetAudio,
				fmt.Sprintf("messages[%d].content[%d]", messageIndex, partIndex), warnings)
		}
		converted.ToolCalls = make([]ToolCall, len(message.ToolCalls))
		for callIndex, call := range message.ToolCalls {
			converted.ToolCalls[callIndex] = cloneToolCall(call)
		}
		converted.ToolResults = make([]ToolResult, len(message.ToolResults))
		for resultIndex, result := range message.ToolResults {
			convertedResult := ToolResult{CallID: result.CallID, Name: result.Name, IsError: result.IsError, Content: make([]Part, len(result.Content))}
			for partIndex, part := range result.Content {
				convertedResult.Content[partIndex], warnings = transformPart(part, toolResultVision, false,
					fmt.Sprintf("messages[%d].tool results[%d].content[%d]", messageIndex, resultIndex, partIndex), warnings)
			}
			converted.ToolResults[resultIndex] = convertedResult
		}
		if len(message.ProviderData) != 0 {
			if sameModel {
				converted.ProviderData = append([]byte(nil), message.ProviderData...)
			} else {
				warnings = append(warnings, TransformWarning{Code: TransformRemovedProviderData,
					Path:    fmt.Sprintf("messages[%d].provider data", messageIndex),
					Message: "removed provider data because source and target provider/protocol/model differ"})
			}
		}
		output[messageIndex] = converted
	}
	if err := (Request{Messages: output}).Validate(); err != nil {
		return nil, warnings, &Error{Kind: KindInvalidRequest, Op: "transform", Provider: target.Provider,
			Err: fmt.Errorf("transformed history: %w", err)}
	}
	return output, warnings, nil
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
	for _, capability := range info.Capabilities {
		if capability == target {
			return true
		}
	}
	return false
}
