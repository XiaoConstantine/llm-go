package openairesponses

import (
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"fmt"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	sdkresponses "github.com/openai/openai-go/v3/responses"
)

// FullResponse converts one completed or incomplete non-streaming Responses
// object with the same output validation and replay envelope used by streaming.
func (c Codec) FullResponse(op, configuredModel string, tools []llm.Tool, format llm.ResponseFormat, response sdkresponses.Response) (*llm.Response, error) {
	if response.ID == "" {
		return nil, c.malformedResponse(op, "response has no ID")
	}
	if !response.JSON.Status.Valid() {
		return nil, c.malformedResponse(op, "response has no status")
	}
	if !response.JSON.Output.Valid() {
		return nil, c.malformedResponse(op, "response has no output")
	}
	reason := llm.FinishReasonStop
	switch response.Status {
	case sdkresponses.ResponseStatusCompleted:
	case sdkresponses.ResponseStatusIncomplete:
		switch response.IncompleteDetails.Reason {
		case "max_output_tokens":
			reason = llm.FinishReasonLength
		case "content_filter":
			reason = llm.FinishReasonContentFilter
		default:
			return nil, &llm.Error{Kind: llm.KindProvider, Op: op, Provider: c.Provider,
				Err: fmt.Errorf("response incomplete: %s", response.IncompleteDetails.Reason)}
		}
	default:
		return nil, c.malformedResponse(op, "response has nonterminal status %q", response.Status)
	}

	declared := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		declared[tool.Name] = struct{}{}
	}
	message := llm.Message{Role: llm.RoleAssistant}
	var reasoning strings.Builder
	var outputText strings.Builder
	refusalSeen := false
	seenCallIDs := make(map[string]struct{})
	seenItemIDs := make(map[string]struct{})
	outputItems := make(map[int]jsontext.Value, len(response.Output))
	providerBytes := 0
	replayable := true
	for index, item := range response.Output {
		raw := jsontext.Value(item.RawJSON())
		if len(raw) == 0 || !raw.IsValid() || raw.Kind() != jsontext.KindBeginObject {
			return nil, c.malformedResponse(op, "output item %d is not a strict JSON object", index)
		}
		raw = append(jsontext.Value(nil), raw...)
		if providerBytes > c.MaxProviderDataBytes-len(raw) {
			return nil, c.malformedResponse(op, "provider data exceeds %d bytes", c.MaxProviderDataBytes)
		}
		providerBytes += len(raw)
		outputItems[index] = raw
		if item.Status == "in_progress" || item.Status == "incomplete" {
			replayable = false
		}
		switch item.Type {
		case "message", "function_call", "reasoning":
			if item.ID == "" {
				return nil, c.malformedResponse(op, "output item %d of type %q has no ID", index, item.Type)
			}
			if _, duplicate := seenItemIDs[item.ID]; duplicate {
				return nil, c.malformedResponse(op, "output item ID %q is repeated", item.ID)
			}
			seenItemIDs[item.ID] = struct{}{}
		}

		switch item.Type {
		case "message":
			if !item.JSON.Content.Valid() || !item.JSON.Role.Valid() || item.Role != "assistant" || !item.JSON.Status.Valid() {
				return nil, c.malformedResponse(op, "message output item %d has incomplete identity", index)
			}
			if err := validateOutputItemStatus(response.Status, item.Status); err != nil {
				return nil, c.malformedResponse(op, "message output item %d status: %v", index, err)
			}
			for contentIndex, content := range item.Content {
				switch content.Type {
				case "output_text":
					if !content.JSON.Text.Valid() {
						return nil, c.malformedResponse(op, "message output item %d content %d has no text", index, contentIndex)
					}
					message.Content = append(message.Content, llm.Part{Text: content.Text})
					outputText.WriteString(content.Text)
				case "refusal":
					if !validRefusalContent(content) {
						return nil, c.malformedResponse(op, "message output item %d content %d has an invalid refusal", index, contentIndex)
					}
					refusalSeen = true
				default:
					return nil, c.malformedResponse(op, "message output item %d content %d has unsupported type %q", index, contentIndex, content.Type)
				}
			}
		case "function_call":
			if !item.JSON.Status.Valid() {
				return nil, c.malformedResponse(op, "function call at output index %d has incomplete identity", index)
			}
			if err := validateOutputItemStatus(response.Status, item.Status); err != nil {
				return nil, c.malformedResponse(op, "function call at output index %d status: %v", index, err)
			}
			if item.Status != "completed" {
				continue
			}
			if item.CallID == "" || item.Name == "" {
				return nil, c.malformedResponse(op, "function call at output index %d is missing call_id or name", index)
			}
			if _, duplicate := seenCallIDs[item.CallID]; duplicate {
				return nil, c.malformedResponse(op, "function call ID %q is repeated", item.CallID)
			}
			seenCallIDs[item.CallID] = struct{}{}
			if _, ok := declared[item.Name]; !ok {
				return nil, c.malformedResponse(op, "function call at output index %d names undeclared tool %q", index, item.Name)
			}
			var encodedArguments jsontext.Value
			if err := jsonv2.Unmarshal([]byte(item.RawJSON()), &struct {
				Arguments *jsontext.Value `json:"arguments"`
			}{Arguments: &encodedArguments}); err != nil {
				return nil, c.malformedResponse(op, "decode function call at output index %d: %v", index, err)
			}
			arguments, err := functionArguments(encodedArguments)
			if err != nil {
				return nil, c.malformedResponse(op, "function call at output index %d arguments: %v", index, err)
			}
			message.ToolCalls = append(message.ToolCalls, llm.ToolCall{ID: item.CallID, Name: item.Name, Arguments: arguments})
		case "reasoning":
			if !item.JSON.Summary.Valid() {
				return nil, c.malformedResponse(op, "reasoning output item %d has incomplete identity", index)
			}
			if item.JSON.Status.Valid() {
				if err := validateOutputItemStatus(response.Status, item.Status); err != nil {
					return nil, c.malformedResponse(op, "reasoning output item %d status: %v", index, err)
				}
			}
			for summaryIndex, summary := range item.Summary {
				if !summary.JSON.Text.Valid() {
					return nil, c.malformedResponse(op, "reasoning output item %d summary %d has no text", index, summaryIndex)
				}
				reasoning.WriteString(summary.Text)
			}
		}
	}
	if len(message.ToolCalls) != 0 && reason == llm.FinishReasonStop {
		reason = llm.FinishReasonToolCall
	}
	if format == llm.ResponseFormatJSON && reason == llm.FinishReasonStop && !(outputText.Len() == 0 && refusalSeen) &&
		!jsontext.Value(outputText.String()).IsValid() {
		return nil, c.malformedResponse(op, "completed JSON response is not strict JSON")
	}
	usage, err := responseUsage(response)
	if err != nil {
		return nil, c.malformedResponse(op, "usage: %v", err)
	}
	var providerData jsontext.Value
	if replayable {
		providerData, err = c.marshalProviderData(configuredModel, outputItems)
		if err != nil {
			return nil, c.malformedResponse(op, "encode provider data: %v", err)
		}
	}
	message.ProviderData = providerData
	model := string(response.Model)
	if model == "" {
		model = configuredModel
	}
	return &llm.Response{ID: response.ID, Model: model, Message: message, ReasoningSummary: reasoning.String(),
		FinishReason: reason, Usage: usage}, nil
}

func validateOutputItemStatus(responseStatus sdkresponses.ResponseStatus, itemStatus string) error {
	switch itemStatus {
	case "completed":
		return nil
	case "in_progress", "incomplete":
		if responseStatus == sdkresponses.ResponseStatusIncomplete {
			return nil
		}
		return fmt.Errorf("nonterminal item status %q in completed response", itemStatus)
	default:
		return fmt.Errorf("unknown item status %q", itemStatus)
	}
}

// FailedResponse converts a failed non-streaming Responses object into a
// provider error.
func (c Codec) FailedResponse(op string, response sdkresponses.Response) error {
	if !response.JSON.Error.Valid() || response.Error.Message == "" {
		return c.malformedResponse(op, "failed response is missing error details")
	}
	apiErr := &APIError{Code: string(response.Error.Code), Message: response.Error.Message}
	return &llm.Error{Kind: classifyAPIError(apiErr), Op: op, Provider: c.Provider, Err: apiErr}
}
