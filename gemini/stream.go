package gemini

import (
	"context"
	"strings"

	"encoding/json/jsontext"

	llm "github.com/XiaoConstantine/llm-go"
	internalstream "github.com/XiaoConstantine/llm-go/internal/stream"
	"google.golang.org/genai"
)

const maxStreamJSONBytes = 16 << 20

func (c *Client) produceStream(
	ctx context.Context,
	request llm.Request,
	contents []*genai.Content,
	generationConfig *genai.GenerateContentConfig,
	emit internalstream.Emit,
) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = sdkError("stream", &sdkPanicError{value: recovered})
		}
		err = relabelProviderError(err, c.provider)
	}()

	accumulator := newStreamAccumulator(c.model, request)
	for response, streamErr := range c.sdkClient.Models.GenerateContentStream(ctx, c.model, contents, generationConfig) {
		if contextErr := contextErr(ctx); contextErr != nil {
			return contextErr
		}
		if streamErr != nil {
			return sdkError("stream", streamErr)
		}
		chunk, err := accumulator.consume(response)
		if err != nil {
			return err
		}
		if len(chunk.Content) != 0 || len(chunk.ToolCalls) != 0 {
			if !emit(chunk) {
				return nil
			}
		}
	}
	if contextErr := contextErr(ctx); contextErr != nil {
		return contextErr
	}
	final, err := accumulator.finalChunk()
	if err != nil {
		return err
	}
	if !emit(final) {
		return nil
	}
	return nil
}

type streamAccumulator struct {
	configuredModel string
	format          llm.ResponseFormat
	declared        map[string]struct{}
	seenIDs         map[string]struct{}

	id             string
	model          string
	providerFinish genai.FinishReason
	blocked        bool
	responseCount  int
	contentCount   int
	toolCallCount  int
	data           []messageDataPart
	usage          *llm.Usage
	jsonContent    strings.Builder
}

func newStreamAccumulator(configuredModel string, request llm.Request) *streamAccumulator {
	return &streamAccumulator{
		configuredModel: configuredModel,
		format:          request.ResponseFormat,
		declared:        declaredToolNames(request.Tools),
		seenIDs:         priorToolCallIDs(request.Messages),
	}
}

func (accumulator *streamAccumulator) consume(response *genai.GenerateContentResponse) (llm.Chunk, error) {
	if response == nil {
		return llm.Chunk{}, malformedResponseFor("stream", "SDK returned a null response event")
	}
	blocked := promptWasBlocked(response.PromptFeedback)
	if blocked && (accumulator.responseCount != 0 || len(response.Candidates) != 0 || accumulator.blocked) {
		return llm.Chunk{}, malformedResponseFor("stream", "blocked prompt feedback must be the first event and contain no candidates")
	}
	accumulator.responseCount++
	if err := accumulator.observeMetadata(response.ResponseID, response.ModelVersion); err != nil {
		return llm.Chunk{}, err
	}
	if response.UsageMetadata != nil {
		usage, err := usageFromSDK("stream", response.UsageMetadata)
		if err != nil {
			return llm.Chunk{}, err
		}
		if err := accumulator.observeUsage(usage); err != nil {
			return llm.Chunk{}, err
		}
	}
	if blocked {
		accumulator.blocked = true
		return llm.Chunk{}, nil
	}

	if len(response.Candidates) == 0 {
		if response.UsageMetadata == nil && response.ResponseID == "" && response.ModelVersion == "" && response.PromptFeedback == nil {
			return llm.Chunk{}, malformedResponseFor("stream", "response event contains no candidate or metadata")
		}
		return llm.Chunk{}, nil
	}
	if len(response.Candidates) != 1 {
		return llm.Chunk{}, malformedResponseFor("stream", "response event contains %d candidates, want exactly one", len(response.Candidates))
	}
	if accumulator.blocked {
		return llm.Chunk{}, malformedResponseFor("stream", "response event contains a candidate after prompt blocking")
	}
	if accumulator.providerFinish != "" {
		return llm.Chunk{}, malformedResponseFor("stream", "response event contains a candidate after the finish reason")
	}

	candidate := response.Candidates[0]
	if candidate == nil {
		return llm.Chunk{}, malformedResponseFor("stream", "response candidate is null")
	}
	if candidate.Content == nil {
		finish, err := finishReasonFor("stream", candidate.FinishReason, 0)
		if err != nil {
			return llm.Chunk{}, err
		}
		if finish != llm.FinishReasonContentFilter {
			return llm.Chunk{}, malformedResponseFor("stream", "response candidate has no content")
		}
		accumulator.providerFinish = candidate.FinishReason
		return llm.Chunk{}, nil
	}

	chunk := llm.Chunk{}
	if candidate.Content != nil {
		if candidate.Content.Role != "" && candidate.Content.Role != genai.RoleModel {
			return llm.Chunk{}, malformedResponseFor("stream", "response content has role %q, want model", candidate.Content.Role)
		}
		parts, err := convertSDKParts(
			"stream",
			candidate.Content.Parts,
			accumulator.declared,
			accumulator.seenIDs,
			accumulator.contentCount,
			accumulator.toolCallCount,
		)
		if err != nil {
			return llm.Chunk{}, err
		}
		accumulator.contentCount += len(parts.content)
		accumulator.toolCallCount += len(parts.calls)
		accumulator.data = append(accumulator.data, parts.data...)
		chunk.Content = parts.content
		chunk.ToolCalls = parts.calls
		if accumulator.format == llm.ResponseFormatJSON {
			for _, part := range parts.content {
				if part.Kind != llm.PartText {
					return llm.Chunk{}, malformedResponseFor("stream", "JSON-mode stream contains non-text content")
				}
				if accumulator.jsonContent.Len() > maxStreamJSONBytes-len(part.Text) {
					return llm.Chunk{}, malformedResponseFor("stream", "JSON-mode response exceeds %d buffered bytes", maxStreamJSONBytes)
				}
				accumulator.jsonContent.WriteString(part.Text)
			}
		}
	}

	if candidate.FinishReason != "" && candidate.FinishReason != genai.FinishReasonUnspecified {
		accumulator.providerFinish = candidate.FinishReason
	}
	return chunk, nil
}

func (accumulator *streamAccumulator) observeMetadata(id, model string) error {
	if id != "" {
		if accumulator.id != "" && accumulator.id != id {
			return malformedResponseFor("stream", "response ID changed from %q to %q", accumulator.id, id)
		}
		accumulator.id = id
	}
	if model != "" {
		if accumulator.model != "" && accumulator.model != model {
			return malformedResponseFor("stream", "response model changed from %q to %q", accumulator.model, model)
		}
		accumulator.model = model
	}
	return nil
}

func (accumulator *streamAccumulator) observeUsage(usage *llm.Usage) error {
	if usage == nil {
		return nil
	}
	if accumulator.usage != nil &&
		(usage.InputTokens < accumulator.usage.InputTokens ||
			usage.OutputTokens < accumulator.usage.OutputTokens ||
			usage.TotalTokens < accumulator.usage.TotalTokens) {
		return malformedResponseFor("stream", "response token usage decreased")
	}
	clone := *usage
	accumulator.usage = &clone
	return nil
}

func (accumulator *streamAccumulator) finalChunk() (llm.Chunk, error) {
	finish := llm.FinishReasonContentFilter
	if !accumulator.blocked {
		var err error
		finish, err = finishReasonFor("stream", accumulator.providerFinish, accumulator.toolCallCount)
		if err != nil {
			return llm.Chunk{}, err
		}
	}
	if accumulator.format == llm.ResponseFormatJSON && finish == llm.FinishReasonStop &&
		!jsontext.Value(accumulator.jsonContent.String()).IsValid() {
		return llm.Chunk{}, malformedResponseFor("stream", "completed JSON response is not strict JSON")
	}

	var providerData []byte
	if len(accumulator.data) != 0 {
		var err error
		providerData, err = marshalMessageData(messageData{Parts: accumulator.data})
		if err != nil {
			return llm.Chunk{}, malformedResponseFor("stream", "encode Gemini message state: %w", err)
		}
	}
	model := accumulator.model
	if model == "" {
		model = accumulator.configuredModel
	}
	return llm.Chunk{
		ID:           accumulator.id,
		Model:        model,
		ProviderData: providerData,
		FinishReason: finish,
		Usage:        accumulator.usage,
	}, nil
}
