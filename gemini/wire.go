package gemini

import (
	"encoding/json"
	"fmt"
	"math"
	"mime"
	"strings"

	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	llm "github.com/XiaoConstantine/llm-go"
	"google.golang.org/genai"
)

const (
	maxTools            = 512
	messageDataVersion  = 1
	messageDataProvider = "gemini"
)

type messageDataEnvelope struct {
	Provider string       `json:"provider"`
	Version  int          `json:"version"`
	Data     *messageData `json:"data"`
}

type messageData struct {
	Parts []messageDataPart `json:"parts"`
}

type messageDataPart struct {
	Kind             string  `json:"kind"`
	Index            int     `json:"index,omitzero"`
	Text             *string `json:"text,omitempty"`
	Data             []byte  `json:"data,omitempty"`
	MediaType        string  `json:"media_type,omitempty"`
	ThoughtSignature []byte  `json:"thought_signature,omitempty"`
}

func checkRequest(op string, request llm.Request) error {
	if request.MaxOutputTokens > math.MaxInt32 {
		return requestError(op, "max output tokens must not exceed %d", math.MaxInt32)
	}
	if request.ReasoningBudgetTokens > math.MaxInt32 {
		return requestError(op, "reasoning budget tokens must not exceed %d", math.MaxInt32)
	}
	if request.CacheRetention != llm.CacheRetentionDefault || request.CacheKey != "" || request.SessionID != "" {
		return unsupported(op, "Gemini GenerateContent does not support portable prompt-cache or session controls")
	}
	numericValues := []struct {
		name  string
		value *float64
	}{
		{name: "temperature", value: request.Temperature},
		{name: "top-p", value: request.TopP},
		{name: "presence penalty", value: request.PresencePenalty},
		{name: "frequency penalty", value: request.FrequencyPenalty},
	}
	for _, value := range numericValues {
		if value.value != nil && math.IsInf(float64(float32(*value.value)), 0) {
			return requestError(op, "%s exceeds Gemini's numeric range", value.name)
		}
	}
	if len(request.Tools) > maxTools {
		return requestError(op, "tools must not contain more than %d definitions", maxTools)
	}

	declared := make(map[string]struct{}, len(request.Tools))
	for index, tool := range request.Tools {
		if !validFunctionName(tool.Name) {
			return requestError(op, "tools[%d].name %q is not a valid Gemini function name", index, tool.Name)
		}
		if jsontext.Value(tool.InputSchema).Kind() != jsontext.KindBeginObject {
			return requestError(op, "tools[%d].input schema must be a JSON object", index)
		}
		declared[tool.Name] = struct{}{}
	}

	conversationMessages := 0
	seenCallIDs := make(map[string]struct{})
	for messageIndex, message := range request.Messages {
		if message.Role == llm.RoleSystem {
			if conversationMessages != 0 {
				return requestError(op, "messages[%d]: system messages must precede conversation messages", messageIndex)
			}
			if len(message.Content) == 0 {
				return requestError(op, "messages[%d]: system message content must not be empty", messageIndex)
			}
			for _, part := range message.Content {
				if part.Kind != llm.PartText {
					return unsupported(op, "Gemini system instructions support only text")
				}
			}
			continue
		}

		conversationMessages++
		switch message.Role {
		case llm.RoleUser:
			if len(message.Content) == 0 {
				return requestError(op, "messages[%d]: user message content must not be empty", messageIndex)
			}
		case llm.RoleTool:
			for _, result := range message.ToolResults {
				for _, part := range result.Content {
					if part.Kind == llm.PartAudio {
						return unsupported(op, "audio tool results are not supported")
					}
				}
			}
		}

		for callIndex, call := range message.ToolCalls {
			if !validFunctionName(call.Name) {
				return requestError(op, "messages[%d].tool calls[%d].name %q is not a valid Gemini function name", messageIndex, callIndex, call.Name)
			}
			if _, ok := declared[call.Name]; !ok {
				return requestError(op, "messages[%d].tool calls[%d] names undeclared tool %q", messageIndex, callIndex, call.Name)
			}
			if jsontext.Value(call.Arguments).Kind() != jsontext.KindBeginObject {
				return requestError(op, "messages[%d].tool calls[%d].arguments must be a JSON object", messageIndex, callIndex)
			}
			if call.ID != "" {
				if _, exists := seenCallIDs[call.ID]; exists {
					return requestError(op, "messages[%d].tool calls[%d] repeats ID %q", messageIndex, callIndex, call.ID)
				}
				seenCallIDs[call.ID] = struct{}{}
			}
		}
	}
	if conversationMessages == 0 {
		return requestError(op, "messages must contain a user, assistant, or tool message")
	}
	return checkToolHistory(op, request.Messages)
}

func checkToolHistory(op string, messages []llm.Message) error {
	pending := 0
	for index, message := range messages {
		if pending != 0 && message.Role != llm.RoleTool {
			return requestError(op, "messages[%d] must contain tool results for %d pending calls", index, pending)
		}
		switch message.Role {
		case llm.RoleAssistant:
			pending = len(message.ToolCalls)
		case llm.RoleTool:
			pending -= len(message.ToolResults)
			if pending < 0 {
				return requestError(op, "messages[%d] contains too many tool results", index)
			}
		}
	}
	if pending != 0 {
		return requestError(op, "messages end with %d pending tool calls", pending)
	}
	return nil
}

func validFunctionName(name string) bool {
	if len(name) == 0 || len(name) > 128 {
		return false
	}
	for index := range len(name) {
		character := name[index]
		letter := character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z'
		if index == 0 && !letter && character != '_' {
			return false
		}
		if !letter && (character < '0' || character > '9') &&
			character != '_' && character != '-' && character != '.' && character != ':' {
			return false
		}
	}
	return true
}

func requestToSDK(op string, request llm.Request) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	if request.CacheRetention != llm.CacheRetentionDefault || request.CacheKey != "" || request.SessionID != "" {
		return nil, nil, unsupported(op, "Gemini GenerateContent does not support portable prompt-cache or session controls")
	}
	generationConfig := &genai.GenerateContentConfig{
		MaxOutputTokens: int32(request.MaxOutputTokens),
		StopSequences:   append([]string(nil), request.Stop...),
	}
	var err error
	if generationConfig.Temperature, err = float32Value(op, "temperature", request.Temperature); err != nil {
		return nil, nil, err
	}
	if generationConfig.TopP, err = float32Value(op, "top-p", request.TopP); err != nil {
		return nil, nil, err
	}
	if generationConfig.PresencePenalty, err = float32Value(op, "presence penalty", request.PresencePenalty); err != nil {
		return nil, nil, err
	}
	if generationConfig.FrequencyPenalty, err = float32Value(op, "frequency penalty", request.FrequencyPenalty); err != nil {
		return nil, nil, err
	}
	if request.ResponseFormat == llm.ResponseFormatJSON {
		generationConfig.ResponseMIMEType = "application/json"
	}
	if request.ReasoningBudgetTokens != 0 {
		budget := int32(request.ReasoningBudgetTokens)
		generationConfig.ThinkingConfig = &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: &budget}
	} else if request.ReasoningEffort == llm.ReasoningEffortNone {
		budget := int32(0)
		generationConfig.ThinkingConfig = &genai.ThinkingConfig{ThinkingBudget: &budget}
	} else if request.ReasoningEffort != llm.ReasoningEffortDefault {
		level := genai.ThinkingLevel(strings.ToUpper(string(request.ReasoningEffort)))
		if request.ReasoningEffort == llm.ReasoningEffortXHigh || request.ReasoningEffort == llm.ReasoningEffortMax {
			level = genai.ThinkingLevelHigh
		}
		generationConfig.ThinkingConfig = &genai.ThinkingConfig{IncludeThoughts: true, ThinkingLevel: level}
	}

	if len(request.Tools) != 0 {
		declarations := make([]*genai.FunctionDeclaration, len(request.Tools))
		strict := false
		for index, tool := range request.Tools {
			var schema any
			if err := jsonv2.Unmarshal(tool.InputSchema, &schema); err != nil {
				return nil, nil, requestError(op, "decode tools[%d].input schema: %w", index, err)
			}
			declarations[index] = &genai.FunctionDeclaration{
				Name:                 tool.Name,
				Description:          tool.Description,
				ParametersJsonSchema: schema,
			}
			strict = strict || tool.StrictEnabled(true)
		}
		generationConfig.Tools = []*genai.Tool{{FunctionDeclarations: declarations}}
		if strict {
			generationConfig.ToolConfig = &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{Mode: genai.FunctionCallingConfigModeValidated},
			}
		}
	}
	if request.ToolChoice.Mode != llm.ToolChoiceAuto {
		config := &genai.FunctionCallingConfig{}
		switch request.ToolChoice.Mode {
		case llm.ToolChoiceNone:
			config.Mode = genai.FunctionCallingConfigModeNone
		case llm.ToolChoiceRequired:
			config.Mode = genai.FunctionCallingConfigModeAny
		case llm.ToolChoiceNamed:
			config.Mode = genai.FunctionCallingConfigModeAny
			config.AllowedFunctionNames = []string{request.ToolChoice.Name}
		}
		generationConfig.ToolConfig = &genai.ToolConfig{FunctionCallingConfig: config}
	}

	encoder := newMessageEncoder(op)
	contents := make([]*genai.Content, 0, len(request.Messages))
	var systemParts []*genai.Part
	for _, message := range request.Messages {
		if message.Role == llm.RoleSystem {
			for _, part := range message.Content {
				systemParts = append(systemParts, &genai.Part{Text: part.Text})
			}
			continue
		}
		content, err := encoder.convert(message)
		if err != nil {
			return nil, nil, err
		}
		contents = append(contents, content)
	}
	if len(systemParts) != 0 {
		generationConfig.SystemInstruction = &genai.Content{Parts: systemParts}
	}
	return contents, generationConfig, nil
}

func float32Value(op, name string, value *float64) (*float32, error) {
	if value == nil {
		return nil, nil
	}
	converted := float32(*value)
	if math.IsInf(float64(converted), 0) {
		return nil, requestError(op, "%s exceeds Gemini's numeric range", name)
	}
	return &converted, nil
}

type messageEncoder struct {
	op            string
	pendingByID   map[string]llm.ToolCall
	pendingByName map[string][]llm.ToolCall
}

func newMessageEncoder(op string) *messageEncoder {
	return &messageEncoder{
		op:            op,
		pendingByID:   make(map[string]llm.ToolCall),
		pendingByName: make(map[string][]llm.ToolCall),
	}
}

func (encoder *messageEncoder) convert(message llm.Message) (*genai.Content, error) {
	switch message.Role {
	case llm.RoleUser:
		parts, err := contentPartsToSDK(encoder.op, message.Content)
		if err != nil {
			return nil, err
		}
		return &genai.Content{Role: genai.RoleUser, Parts: parts}, nil
	case llm.RoleAssistant:
		parts, err := assistantPartsToSDK(encoder.op, message)
		if err != nil {
			return nil, err
		}
		if len(parts) == 0 {
			return nil, requestError(encoder.op, "assistant message must contain content, tool calls, or recognized provider data")
		}
		for _, call := range message.ToolCalls {
			if call.ID != "" {
				encoder.pendingByID[call.ID] = call
			} else {
				encoder.pendingByName[call.Name] = append(encoder.pendingByName[call.Name], call)
			}
		}
		return &genai.Content{Role: genai.RoleModel, Parts: parts}, nil
	case llm.RoleTool:
		parts := make([]*genai.Part, len(message.ToolResults))
		for index, result := range message.ToolResults {
			call, err := encoder.consume(result)
			if err != nil {
				return nil, err
			}
			key := "output"
			if result.IsError {
				key = "error"
			}
			response := &genai.FunctionResponse{ID: call.ID, Name: call.Name, Response: map[string]any{key: resultText(result.Content)}}
			for partIndex, part := range result.Content {
				if part.Kind != llm.PartImage {
					continue
				}
				mediaType, _, err := mime.ParseMediaType(part.MediaType)
				if err != nil || !strings.HasPrefix(strings.ToLower(mediaType), "image/") {
					return nil, requestError(encoder.op, "tool result %d content[%d] has invalid image media type %q", index, partIndex, part.MediaType)
				}
				response.Parts = append(response.Parts, genai.NewFunctionResponsePartFromBytes(append([]byte(nil), part.Data...), mediaType))
			}
			parts[index] = &genai.Part{FunctionResponse: response}
		}
		return &genai.Content{Role: genai.RoleUser, Parts: parts}, nil
	default:
		panic("gemini: validated message has invalid role")
	}
}

func (encoder *messageEncoder) consume(result llm.ToolResult) (llm.ToolCall, error) {
	if result.CallID != "" {
		call, ok := encoder.pendingByID[result.CallID]
		if !ok {
			return llm.ToolCall{}, requestError(encoder.op, "tool result refers to unknown call ID %q", result.CallID)
		}
		delete(encoder.pendingByID, result.CallID)
		return call, nil
	}
	queue := encoder.pendingByName[result.Name]
	if len(queue) == 0 {
		return llm.ToolCall{}, requestError(encoder.op, "tool result refers to unknown call name %q", result.Name)
	}
	call := queue[0]
	if len(queue) == 1 {
		delete(encoder.pendingByName, result.Name)
	} else {
		encoder.pendingByName[result.Name] = queue[1:]
	}
	return call, nil
}

func resultText(parts []llm.Part) string {
	var builder strings.Builder
	for _, part := range parts {
		builder.WriteString(part.Text)
	}
	return builder.String()
}

func contentPartsToSDK(op string, parts []llm.Part) ([]*genai.Part, error) {
	converted := make([]*genai.Part, len(parts))
	for index, part := range parts {
		value, err := contentPartToSDK(op, part)
		if err != nil {
			return nil, err
		}
		converted[index] = value
	}
	return converted, nil
}

func contentPartToSDK(op string, part llm.Part) (*genai.Part, error) {
	switch part.Kind {
	case llm.PartText:
		return &genai.Part{Text: part.Text}, nil
	case llm.PartImage, llm.PartAudio:
		mediaType, _, err := mime.ParseMediaType(part.MediaType)
		if err != nil {
			return nil, requestError(op, "invalid media type %q", part.MediaType)
		}
		prefix := "image/"
		if part.Kind == llm.PartAudio {
			prefix = "audio/"
		}
		if !strings.HasPrefix(strings.ToLower(mediaType), prefix) {
			return nil, requestError(op, "media type %q does not match content kind", part.MediaType)
		}
		return &genai.Part{InlineData: &genai.Blob{
			Data:     append([]byte(nil), part.Data...),
			MIMEType: mediaType,
		}}, nil
	default:
		panic("gemini: validated content has invalid kind")
	}
}

func assistantPartsToSDK(op string, message llm.Message) ([]*genai.Part, error) {
	data, recognized, err := parseMessageData(message.ProviderData)
	if err != nil {
		return nil, requestError(op, "decode assistant provider data: %w", err)
	}
	if !recognized {
		parts, err := contentPartsToSDK(op, message.Content)
		if err != nil {
			return nil, err
		}
		for _, call := range message.ToolCalls {
			part, err := toolCallToSDK(op, call)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part)
		}
		return parts, nil
	}

	parts := make([]*genai.Part, 0, len(data.Parts))
	contentUsed := make([]bool, len(message.Content))
	callUsed := make([]bool, len(message.ToolCalls))
	for partIndex, metadata := range data.Parts {
		var part *genai.Part
		switch metadata.Kind {
		case "content":
			if metadata.Index < 0 || metadata.Index >= len(message.Content) || contentUsed[metadata.Index] {
				return nil, requestError(op, "assistant provider data part %d has invalid content index %d", partIndex, metadata.Index)
			}
			if metadata.Text != nil || len(metadata.Data) != 0 || metadata.MediaType != "" {
				return nil, requestError(op, "assistant provider data part %d mixes content metadata with thought content", partIndex)
			}
			part, err = contentPartToSDK(op, message.Content[metadata.Index])
			if err != nil {
				return nil, err
			}
			contentUsed[metadata.Index] = true
		case "tool_call":
			if metadata.Index < 0 || metadata.Index >= len(message.ToolCalls) || callUsed[metadata.Index] {
				return nil, requestError(op, "assistant provider data part %d has invalid tool call index %d", partIndex, metadata.Index)
			}
			if metadata.Text != nil || len(metadata.Data) != 0 || metadata.MediaType != "" {
				return nil, requestError(op, "assistant provider data part %d mixes tool metadata with thought content", partIndex)
			}
			part, err = toolCallToSDK(op, message.ToolCalls[metadata.Index])
			if err != nil {
				return nil, err
			}
			callUsed[metadata.Index] = true
		case "thought":
			if metadata.Text != nil && (len(metadata.Data) != 0 || metadata.MediaType != "") {
				return nil, requestError(op, "assistant provider data part %d contains both text and binary thought content", partIndex)
			}
			if metadata.Text != nil {
				part = &genai.Part{Text: *metadata.Text, Thought: true}
			} else if len(metadata.Data) != 0 && metadata.MediaType != "" {
				part = &genai.Part{InlineData: &genai.Blob{
					Data:     append([]byte(nil), metadata.Data...),
					MIMEType: metadata.MediaType,
				}, Thought: true}
			} else {
				return nil, requestError(op, "assistant provider data part %d has no thought content", partIndex)
			}
		default:
			return nil, requestError(op, "assistant provider data part %d has unknown kind %q", partIndex, metadata.Kind)
		}
		part.ThoughtSignature = append([]byte(nil), metadata.ThoughtSignature...)
		parts = append(parts, part)
	}
	for index, used := range contentUsed {
		if !used {
			return nil, requestError(op, "assistant provider data does not reference content part %d", index)
		}
	}
	for index, used := range callUsed {
		if !used {
			return nil, requestError(op, "assistant provider data does not reference tool call %d", index)
		}
	}
	return parts, nil
}

func toolCallToSDK(op string, call llm.ToolCall) (*genai.Part, error) {
	var arguments map[string]any
	if err := jsonv2.Unmarshal(call.Arguments, &arguments); err != nil {
		return nil, requestError(op, "decode tool call %q arguments: %w", call.Name, err)
	}
	return &genai.Part{FunctionCall: &genai.FunctionCall{
		ID:   call.ID,
		Name: call.Name,
		Args: arguments,
	}}, nil
}

func parseMessageData(raw json.RawMessage) (messageData, bool, error) {
	if len(raw) == 0 {
		return messageData{}, false, nil
	}
	value := jsontext.Value(raw)
	if !value.IsValid() {
		return messageData{}, false, fmt.Errorf("invalid JSON")
	}
	if value.Kind() != jsontext.KindBeginObject {
		return messageData{}, false, nil
	}
	var identity struct {
		Provider string `json:"provider"`
	}
	if err := jsonv2.Unmarshal(raw, &identity); err != nil || identity.Provider != messageDataProvider {
		return messageData{}, false, nil
	}
	var header struct {
		Version int `json:"version"`
	}
	if err := jsonv2.Unmarshal(raw, &header); err != nil {
		return messageData{}, false, fmt.Errorf("decode header: %w", err)
	}
	if header.Version != messageDataVersion {
		return messageData{}, false, nil
	}
	var envelope messageDataEnvelope
	if err := jsonv2.Unmarshal(raw, &envelope); err != nil {
		return messageData{}, false, fmt.Errorf("decode data: %w", err)
	}
	if envelope.Data == nil {
		return messageData{}, false, fmt.Errorf("missing data")
	}
	return *envelope.Data, true, nil
}

func marshalMessageData(data messageData) (json.RawMessage, error) {
	raw, err := jsonv2.Marshal(&messageDataEnvelope{
		Provider: messageDataProvider,
		Version:  messageDataVersion,
		Data:     &data,
	})
	if err != nil {
		return nil, err
	}
	return json.RawMessage(raw), nil
}

type convertedParts struct {
	content []llm.Part
	calls   []llm.ToolCall
	data    []messageDataPart
	order   []convertedPartRef
}

type convertedPartRef struct {
	tool  bool
	index int
}

func convertSDKParts(op string, parts []*genai.Part, declared, seenIDs map[string]struct{}, contentOffset, callOffset int) (convertedParts, error) {
	converted := convertedParts{
		content: make([]llm.Part, 0, len(parts)),
		calls:   make([]llm.ToolCall, 0),
		data:    make([]messageDataPart, 0, len(parts)),
		order:   make([]convertedPartRef, 0, len(parts)),
	}
	for index, part := range parts {
		if part == nil {
			return convertedParts{}, malformedResponseFor(op, "response part %d is null", index)
		}
		if part.CodeExecutionResult != nil || part.ExecutableCode != nil || part.FileData != nil ||
			part.FunctionResponse != nil || part.ToolCall != nil || part.ToolResponse != nil {
			return convertedParts{}, unsupported(op, fmt.Sprintf("response part %d uses an unsupported Gemini content type", index))
		}
		if part.Text != "" && (part.InlineData != nil || part.FunctionCall != nil) ||
			part.InlineData != nil && part.FunctionCall != nil {
			return convertedParts{}, malformedResponseFor(op, "response part %d contains multiple content types", index)
		}

		metadata := messageDataPart{ThoughtSignature: append([]byte(nil), part.ThoughtSignature...)}
		if part.Thought {
			if part.FunctionCall != nil {
				return convertedParts{}, malformedResponseFor(op, "response part %d marks a function call as thought content", index)
			}
			metadata.Kind = "thought"
			if part.InlineData != nil {
				if len(part.InlineData.Data) == 0 || part.InlineData.MIMEType == "" {
					return convertedParts{}, malformedResponseFor(op, "response thought part %d has incomplete inline data", index)
				}
				metadata.Data = append([]byte(nil), part.InlineData.Data...)
				metadata.MediaType = part.InlineData.MIMEType
			} else {
				text := part.Text
				metadata.Text = &text
			}
			converted.data = append(converted.data, metadata)
			continue
		}

		switch {
		case part.FunctionCall != nil:
			call, err := toolCallFromSDK(op, index, part.FunctionCall, declared, seenIDs)
			if err != nil {
				return convertedParts{}, err
			}
			metadata.Kind = "tool_call"
			metadata.Index = callOffset + len(converted.calls)
			converted.calls = append(converted.calls, call)
			converted.order = append(converted.order, convertedPartRef{tool: true, index: len(converted.calls) - 1})
		case part.InlineData != nil:
			content, err := inlineDataFromSDK(op, index, part.InlineData)
			if err != nil {
				return convertedParts{}, err
			}
			metadata.Kind = "content"
			metadata.Index = contentOffset + len(converted.content)
			converted.content = append(converted.content, content)
			converted.order = append(converted.order, convertedPartRef{index: len(converted.content) - 1})
		default:
			metadata.Kind = "content"
			metadata.Index = contentOffset + len(converted.content)
			converted.content = append(converted.content, llm.Part{Kind: llm.PartText, Text: part.Text})
			converted.order = append(converted.order, convertedPartRef{index: len(converted.content) - 1})
		}
		converted.data = append(converted.data, metadata)
	}
	return converted, nil
}

func inlineDataFromSDK(op string, index int, blob *genai.Blob) (llm.Part, error) {
	if blob == nil || len(blob.Data) == 0 || blob.MIMEType == "" {
		return llm.Part{}, malformedResponseFor(op, "response inline data part %d is incomplete", index)
	}
	mediaType, _, err := mime.ParseMediaType(blob.MIMEType)
	if err != nil {
		return llm.Part{}, malformedResponseFor(op, "response inline data part %d has invalid media type %q", index, blob.MIMEType)
	}
	kind := llm.PartKind(255)
	switch {
	case strings.HasPrefix(strings.ToLower(mediaType), "image/"):
		kind = llm.PartImage
	case strings.HasPrefix(strings.ToLower(mediaType), "audio/"):
		kind = llm.PartAudio
	default:
		return llm.Part{}, unsupported(op, fmt.Sprintf("response inline data part %d has unsupported media type %q", index, mediaType))
	}
	return llm.Part{Kind: kind, Data: append([]byte(nil), blob.Data...), MediaType: mediaType}, nil
}

func toolCallFromSDK(op string, index int, call *genai.FunctionCall, declared, seenIDs map[string]struct{}) (llm.ToolCall, error) {
	if call == nil || !validFunctionName(call.Name) {
		return llm.ToolCall{}, malformedResponseFor(op, "response function call %d has invalid name", index)
	}
	if _, ok := declared[call.Name]; !ok {
		return llm.ToolCall{}, malformedResponseFor(op, "response function call %d names undeclared tool %q", index, call.Name)
	}
	if call.ID != "" {
		if _, exists := seenIDs[call.ID]; exists {
			return llm.ToolCall{}, malformedResponseFor(op, "response function call %d repeats ID %q", index, call.ID)
		}
		seenIDs[call.ID] = struct{}{}
	}
	arguments := call.Args
	if arguments == nil {
		arguments = map[string]any{}
	}
	raw, err := jsonv2.Marshal(arguments)
	if err != nil {
		return llm.ToolCall{}, malformedResponseFor(op, "encode response function call %d arguments: %w", index, err)
	}
	if jsontext.Value(raw).Kind() != jsontext.KindBeginObject {
		return llm.ToolCall{}, malformedResponseFor(op, "response function call %d arguments are not a JSON object", index)
	}
	return llm.ToolCall{
		ID:        call.ID,
		Name:      call.Name,
		Arguments: append(json.RawMessage(nil), raw...),
	}, nil
}

func responseFromSDK(configuredModel string, request llm.Request, response *genai.GenerateContentResponse) (*llm.Response, error) {
	if response == nil {
		return nil, malformedResponseFor("generate", "SDK returned no response")
	}
	usage, err := usageFromSDK("generate", response.UsageMetadata)
	if err != nil {
		return nil, err
	}
	model := response.ModelVersion
	if model == "" {
		model = configuredModel
	}

	if promptWasBlocked(response.PromptFeedback) {
		if len(response.Candidates) != 0 {
			return nil, malformedResponseFor("generate", "blocked prompt feedback must contain no candidates")
		}
		return &llm.Response{
			ID:           response.ResponseID,
			Model:        model,
			Message:      llm.Message{Role: llm.RoleAssistant},
			FinishReason: llm.FinishReasonContentFilter,
			Usage:        usage,
		}, nil
	}
	if len(response.Candidates) == 0 {
		return nil, malformedResponseFor("generate", "response contains no candidates")
	}
	if len(response.Candidates) != 1 {
		return nil, malformedResponseFor("generate", "response contains %d candidates, want exactly one", len(response.Candidates))
	}
	candidate := response.Candidates[0]
	if candidate == nil {
		return nil, malformedResponseFor("generate", "response candidate is null")
	}
	if candidate.Content == nil {
		finish, err := finishReasonFor("generate", candidate.FinishReason, 0)
		if err != nil {
			return nil, err
		}
		if finish != llm.FinishReasonContentFilter {
			return nil, malformedResponseFor("generate", "response candidate has no content")
		}
		return &llm.Response{
			ID:           response.ResponseID,
			Model:        model,
			Message:      llm.Message{Role: llm.RoleAssistant},
			FinishReason: finish,
			Usage:        usage,
		}, nil
	}
	if candidate.Content.Role != "" && candidate.Content.Role != genai.RoleModel {
		return nil, malformedResponseFor("generate", "response content has role %q, want model", candidate.Content.Role)
	}

	declared := declaredToolNames(request.Tools)
	seenIDs := priorToolCallIDs(request.Messages)
	parts, err := convertSDKParts("generate", candidate.Content.Parts, declared, seenIDs, 0, 0)
	if err != nil {
		return nil, err
	}
	finish, err := finishReasonFor("generate", candidate.FinishReason, len(parts.calls))
	if err != nil {
		return nil, err
	}
	providerData, err := marshalMessageData(messageData{Parts: parts.data})
	if err != nil {
		return nil, malformedResponseFor("generate", "encode Gemini message state: %w", err)
	}
	message := llm.Message{
		Role:         llm.RoleAssistant,
		Content:      parts.content,
		ToolCalls:    parts.calls,
		ProviderData: providerData,
	}
	if err := validateJSONResponse("generate", request.ResponseFormat, finish, message); err != nil {
		return nil, err
	}
	return &llm.Response{
		ID:           response.ResponseID,
		Model:        model,
		Message:      message,
		FinishReason: finish,
		Usage:        usage,
	}, nil
}

func declaredToolNames(tools []llm.Tool) map[string]struct{} {
	declared := make(map[string]struct{}, len(tools))
	for _, tool := range tools {
		declared[tool.Name] = struct{}{}
	}
	return declared
}

func priorToolCallIDs(messages []llm.Message) map[string]struct{} {
	ids := make(map[string]struct{})
	for _, message := range messages {
		for _, call := range message.ToolCalls {
			if call.ID != "" {
				ids[call.ID] = struct{}{}
			}
		}
	}
	return ids
}

func promptWasBlocked(feedback *genai.GenerateContentResponsePromptFeedback) bool {
	return feedback != nil && feedback.BlockReason != "" && feedback.BlockReason != genai.BlockedReasonUnspecified
}

func finishReasonFor(op string, reason genai.FinishReason, toolCalls int) (llm.FinishReason, error) {
	switch reason {
	case genai.FinishReasonStop:
		if toolCalls != 0 {
			return llm.FinishReasonToolCall, nil
		}
		return llm.FinishReasonStop, nil
	case genai.FinishReasonMaxTokens:
		if toolCalls != 0 {
			return "", unsupported(op, "truncated Gemini function calls are not representable")
		}
		return llm.FinishReasonLength, nil
	case genai.FinishReasonSafety, genai.FinishReasonRecitation, genai.FinishReasonLanguage,
		genai.FinishReasonBlocklist, genai.FinishReasonProhibitedContent, genai.FinishReasonSPII,
		genai.FinishReasonImageSafety, genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		return llm.FinishReasonContentFilter, nil
	case "", genai.FinishReasonUnspecified:
		return "", malformedResponseFor(op, "response has no finish reason")
	default:
		return "", unsupported(op, fmt.Sprintf("Gemini finish reason %q is not representable", reason))
	}
}

func usageFromSDK(op string, usage *genai.GenerateContentResponseUsageMetadata) (*llm.Usage, error) {
	if usage == nil {
		return nil, nil
	}
	values := []int32{
		usage.PromptTokenCount,
		usage.ToolUsePromptTokenCount,
		usage.CandidatesTokenCount,
		usage.ThoughtsTokenCount,
		usage.CachedContentTokenCount,
		usage.TotalTokenCount,
	}
	for _, value := range values {
		if value < 0 {
			return nil, malformedResponseFor(op, "response has negative token usage")
		}
	}
	input := int64(values[0]) + int64(values[1])
	output := int64(values[2]) + int64(values[3])
	cacheRead := int64(values[4])
	if cacheRead > int64(values[0]) || input+output != int64(values[5]) {
		return nil, malformedResponseFor(op, "response has inconsistent token usage")
	}
	return &llm.Usage{
		InputTokens:     int(input - cacheRead),
		OutputTokens:    int(output),
		CacheReadTokens: int(cacheRead),
		ReasoningTokens: int(values[3]),
		TotalTokens:     int(values[5]),
	}, nil
}

func validateJSONResponse(op string, format llm.ResponseFormat, finish llm.FinishReason, message llm.Message) error {
	if format != llm.ResponseFormatJSON || finish != llm.FinishReasonStop {
		return nil
	}
	for _, part := range message.Content {
		if part.Kind != llm.PartText {
			return malformedResponseFor(op, "JSON-mode response contains non-text content")
		}
	}
	if !jsontext.Value(message.Text()).IsValid() {
		return malformedResponseFor(op, "JSON-mode response content is not strict JSON")
	}
	return nil
}
