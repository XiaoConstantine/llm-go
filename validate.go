package llm

import (
	"encoding/json/jsontext"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
)

// Validate reports whether r satisfies the provider-neutral Request contract.
// JSON values must conform to RFC 7493, including valid UTF-8 and unique object
// names. Provider implementations may apply additional model-specific
// constraints. On failure, Validate returns an *Error with KindInvalidRequest
// and Op set to "validate".
func (r Request) Validate() error {
	if err := r.validate(); err != nil {
		return &Error{Kind: KindInvalidRequest, Op: "validate", Err: err}
	}
	return nil
}

func (r Request) validate() error {
	if len(r.Messages) == 0 {
		return fmt.Errorf("messages must not be empty")
	}

	if err := validateTools(r.Tools); err != nil {
		return err
	}
	if err := validateToolChoice(r.ToolChoice, r.Tools); err != nil {
		return err
	}
	if err := validateCacheControls(r.CacheRetention, r.CacheKey, r.SessionID); err != nil {
		return err
	}
	if err := validateResponseFormat(r.ResponseFormat); err != nil {
		return err
	}
	if err := validateReasoningEffort(r.ReasoningEffort); err != nil {
		return err
	}
	if r.ReasoningBudgetTokens < 0 {
		return fmt.Errorf("reasoning budget tokens must not be negative")
	}
	if r.ReasoningBudgetTokens != 0 && r.ReasoningEffort == ReasoningEffortNone {
		return fmt.Errorf("reasoning budget tokens cannot be set when reasoning effort is none")
	}
	if r.MaxOutputTokens < 0 {
		return fmt.Errorf("max output tokens must not be negative")
	}
	if err := validateSampling(r); err != nil {
		return err
	}
	for i, stop := range r.Stop {
		if stop == "" {
			return fmt.Errorf("stop[%d] must not be empty", i)
		}
		if !utf8.ValidString(stop) {
			return fmt.Errorf("stop[%d] must be valid UTF-8", i)
		}
	}

	var pending pendingToolCalls
	for i, message := range r.Messages {
		if err := validateMessage(message); err != nil {
			return fmt.Errorf("messages[%d]: %w", i, err)
		}

		for j, call := range message.ToolCalls {
			if !pending.add(call) {
				return fmt.Errorf("messages[%d].tool calls[%d]: duplicate pending ID %q", i, j, call.ID)
			}
		}
		for j, result := range message.ToolResults {
			call, ok := pending.consume(result)
			if !ok {
				return fmt.Errorf("messages[%d].tool results[%d]: no matching preceding tool call", i, j)
			}
			if result.Name != "" && result.Name != call.Name {
				return fmt.Errorf("messages[%d].tool results[%d]: name %q does not match call name %q", i, j, result.Name, call.Name)
			}
		}
	}

	return nil
}

func validateReasoningEffort(effort ReasoningEffort) error {
	switch effort {
	case ReasoningEffortDefault, ReasoningEffortNone, ReasoningEffortMinimal,
		ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh,
		ReasoningEffortXHigh, ReasoningEffortMax:
		return nil
	default:
		return fmt.Errorf("reasoning effort %q is invalid", effort)
	}
}

func validateTools(tools []Tool) error {
	names := make(map[string]struct{}, len(tools))
	for i, tool := range tools {
		if tool.Name == "" {
			return fmt.Errorf("tools[%d].name must not be empty", i)
		}
		if !utf8.ValidString(tool.Name) {
			return fmt.Errorf("tools[%d].name must be valid UTF-8", i)
		}
		if !utf8.ValidString(tool.Description) {
			return fmt.Errorf("tools[%d].description must be valid UTF-8", i)
		}
		if _, exists := names[tool.Name]; exists {
			return fmt.Errorf("tools[%d].name %q is duplicated", i, tool.Name)
		}
		names[tool.Name] = struct{}{}
		switch tool.Strictness {
		case ToolStrictDefault, ToolStrictPrefer, ToolStrictRequire:
		default:
			return fmt.Errorf("tools[%d].strictness %q is invalid", i, tool.Strictness)
		}
		if tool.Strict && tool.Strictness != ToolStrictDefault {
			return fmt.Errorf("tools[%d] cannot set both strict and strictness", i)
		}
		if err := validateJSONSchema(tool.InputSchema); err != nil {
			return fmt.Errorf("tools[%d].input schema: %w", i, err)
		}
	}
	return nil
}

func validateToolChoice(choice ToolChoice, tools []Tool) error {
	switch choice.Mode {
	case ToolChoiceAuto, ToolChoiceNone:
		if choice.Name != "" {
			return fmt.Errorf("tool choice name is valid only for named choice")
		}
	case ToolChoiceRequired:
		if choice.Name != "" {
			return fmt.Errorf("tool choice name is valid only for named choice")
		}
		if len(tools) == 0 {
			return fmt.Errorf("required tool choice requires declared tools")
		}
	case ToolChoiceNamed:
		if choice.Name == "" {
			return fmt.Errorf("named tool choice requires a name")
		}
		for _, tool := range tools {
			if tool.Name == choice.Name {
				return nil
			}
		}
		return fmt.Errorf("named tool choice %q is not declared", choice.Name)
	default:
		return fmt.Errorf("tool choice mode %q is invalid", choice.Mode)
	}
	return nil
}

func validateCacheControls(retention CacheRetention, cacheKey, sessionID string) error {
	switch retention {
	case CacheRetentionDefault, CacheRetentionNone, CacheRetentionShort, CacheRetentionLong:
	default:
		return fmt.Errorf("cache retention %q is invalid", retention)
	}
	for _, field := range []struct{ name, value string }{{"cache key", cacheKey}, {"session ID", sessionID}} {
		if !utf8.ValidString(field.value) {
			return fmt.Errorf("%s must be valid UTF-8", field.name)
		}
		if field.value != strings.TrimSpace(field.value) {
			return fmt.Errorf("%s must not contain surrounding whitespace", field.name)
		}
		if len(field.value) > 256 {
			return fmt.Errorf("%s exceeds 256 bytes", field.name)
		}
	}
	if retention == CacheRetentionNone && cacheKey != "" {
		return fmt.Errorf("cache key cannot be set when cache retention is none")
	}
	return nil
}

func validateResponseFormat(format ResponseFormat) error {
	switch format {
	case ResponseFormatText, ResponseFormatJSON:
		return nil
	default:
		return fmt.Errorf("response format %d is invalid", format)
	}
}

func validateSampling(r Request) error {
	if err := validateFinite("temperature", r.Temperature); err != nil {
		return err
	}
	if r.Temperature != nil && *r.Temperature < 0 {
		return fmt.Errorf("temperature must not be negative")
	}
	if err := validateFinite("top-p", r.TopP); err != nil {
		return err
	}
	if r.TopP != nil && (*r.TopP < 0 || *r.TopP > 1) {
		return fmt.Errorf("top-p must be between 0 and 1")
	}
	if err := validateFinite("presence penalty", r.PresencePenalty); err != nil {
		return err
	}
	return validateFinite("frequency penalty", r.FrequencyPenalty)
}

func validateFinite(name string, value *float64) error {
	if value != nil && (math.IsNaN(*value) || math.IsInf(*value, 0)) {
		return fmt.Errorf("%s must be finite", name)
	}
	return nil
}

func validateMessage(message Message) error {
	switch message.Role {
	case RoleSystem, RoleUser:
		if len(message.ToolCalls) != 0 || len(message.ToolResults) != 0 {
			return fmt.Errorf("%s message must not contain tool calls or results", message.Role)
		}
	case RoleAssistant:
		if len(message.ToolResults) != 0 {
			return fmt.Errorf("assistant message must not contain tool results")
		}
	case RoleTool:
		if len(message.Content) != 0 {
			return fmt.Errorf("tool message content must be empty")
		}
		if len(message.ToolCalls) != 0 {
			return fmt.Errorf("tool message must not contain tool calls")
		}
		if len(message.ToolResults) == 0 {
			return fmt.Errorf("tool message must contain at least one result")
		}
	default:
		return fmt.Errorf("role %q is invalid", message.Role)
	}

	for i, part := range message.Content {
		if err := validatePart(part); err != nil {
			return fmt.Errorf("content[%d]: %w", i, err)
		}
	}
	for i, call := range message.ToolCalls {
		if err := validateToolCall(call); err != nil {
			return fmt.Errorf("tool calls[%d]: %w", i, err)
		}
	}
	for i, result := range message.ToolResults {
		if err := validateToolResult(result); err != nil {
			return fmt.Errorf("tool results[%d]: %w", i, err)
		}
	}
	if len(message.ProviderData) != 0 {
		if err := validateJSON(message.ProviderData); err != nil {
			return fmt.Errorf("provider data: %w", err)
		}
	}
	return nil
}

func validatePart(part Part) error {
	switch part.Kind {
	case PartText:
		if len(part.Data) != 0 || part.MediaType != "" {
			return fmt.Errorf("text part must not contain data or media type")
		}
		if !utf8.ValidString(part.Text) {
			return fmt.Errorf("text must be valid UTF-8")
		}
	case PartImage, PartAudio:
		if part.Text != "" {
			return fmt.Errorf("binary part must not contain text")
		}
		if len(part.Data) == 0 {
			return fmt.Errorf("binary part data must not be empty")
		}
		if part.MediaType == "" {
			return fmt.Errorf("binary part media type must not be empty")
		}
		if !utf8.ValidString(part.MediaType) {
			return fmt.Errorf("binary part media type must be valid UTF-8")
		}
	default:
		return fmt.Errorf("kind %d is invalid", part.Kind)
	}
	return nil
}

func validateToolCall(call ToolCall) error {
	if call.ID != "" && !utf8.ValidString(call.ID) {
		return fmt.Errorf("ID must be valid UTF-8")
	}
	if call.Name == "" {
		return fmt.Errorf("name must not be empty")
	}
	if !utf8.ValidString(call.Name) {
		return fmt.Errorf("name must be valid UTF-8")
	}
	if err := validateJSON(call.Arguments); err != nil {
		return fmt.Errorf("arguments: %w", err)
	}
	return nil
}

func validateToolResult(result ToolResult) error {
	if result.CallID == "" && result.Name == "" {
		return fmt.Errorf("call ID or name must be set")
	}
	if !utf8.ValidString(result.CallID) {
		return fmt.Errorf("call ID must be valid UTF-8")
	}
	if !utf8.ValidString(result.Name) {
		return fmt.Errorf("name must be valid UTF-8")
	}
	for i, part := range result.Content {
		if err := validatePart(part); err != nil {
			return fmt.Errorf("content[%d]: %w", i, err)
		}
	}
	return nil
}

func validateJSON(value []byte) error {
	if len(value) == 0 {
		return fmt.Errorf("must contain one JSON value")
	}
	if !jsontext.Value(value).IsValid() {
		return fmt.Errorf("must contain strict JSON")
	}
	return nil
}

type pendingToolCalls struct {
	byID   map[string]ToolCall
	byName map[string][]ToolCall
}

func (p *pendingToolCalls) add(call ToolCall) bool {
	if call.ID != "" {
		if p.byID == nil {
			p.byID = make(map[string]ToolCall)
		}
		if _, exists := p.byID[call.ID]; exists {
			return false
		}
		p.byID[call.ID] = call
		return true
	}
	if p.byName == nil {
		p.byName = make(map[string][]ToolCall)
	}
	p.byName[call.Name] = append(p.byName[call.Name], call)
	return true
}

func (p *pendingToolCalls) consume(result ToolResult) (ToolCall, bool) {
	if result.CallID != "" {
		call, ok := p.byID[result.CallID]
		if ok {
			delete(p.byID, result.CallID)
		}
		return call, ok
	}

	queue := p.byName[result.Name]
	if len(queue) == 0 {
		return ToolCall{}, false
	}
	call := queue[0]
	if len(queue) == 1 {
		delete(p.byName, result.Name)
	} else {
		p.byName[result.Name] = queue[1:]
	}
	return call, true
}
