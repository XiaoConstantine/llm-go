package llm

import (
	"encoding/json"
	"encoding/json/jsontext"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// ReasoningPolicy controls whether provider convenience conversions are allowed.
type ReasoningPolicy string

const (
	ReasoningPolicyDefault ReasoningPolicy = ""
	ReasoningPolicyExact   ReasoningPolicy = "exact"
)

// OpenAIChatOptions contains options specific to Chat Completions.
type OpenAIChatOptions struct {
	LogitBias        map[string]int64
	LogProbs         *bool
	TopLogProbs      *int
	User             *string
	Verbosity        *string
	Prediction       *ChatPrediction
	Store            *bool
	Metadata         map[string]string
	SafetyIdentifier *string
	ServiceTier      *string
	ExtraFields      ChatExtraFields
}

// ChatPrediction is the supported textual predicted output shape.
type ChatPrediction struct{ Content string }

// OpenAIResponsesOptions contains options shared by Responses endpoints.
type OpenAIResponsesOptions struct {
	Verbosity   *string
	ServiceTier *string
}

// AnthropicOptions contains options specific to the Messages protocol.
type AnthropicOptions struct {
	ThinkingDisplay *string
	ExtraFields     AnthropicExtraFields
}

// Extra envelopes own a validated JSON copy. Their zero values are empty.
type ChatExtraFields struct{ fields map[string]json.RawMessage }
type AnthropicExtraFields struct{ fields map[string]json.RawMessage }

// NewChatExtraFields accepts literal top-level additions, never JSON paths.
func NewChatExtraFields(fields map[string]any) (ChatExtraFields, error) {
	copy, err := newExtraFields(fields, false)
	return ChatExtraFields{fields: copy}, err
}

// NewAnthropicExtraFields accepts plain top-level names only.
func NewAnthropicExtraFields(fields map[string]any) (AnthropicExtraFields, error) {
	copy, err := newExtraFields(fields, true)
	return AnthropicExtraFields{fields: copy}, err
}

// Fields returns an independent copy for a protocol serializer.
func (e ChatExtraFields) Fields() map[string]json.RawMessage      { return cloneExtraFields(e.fields) }
func (e AnthropicExtraFields) Fields() map[string]json.RawMessage { return cloneExtraFields(e.fields) }

func cloneExtraFields(fields map[string]json.RawMessage) map[string]json.RawMessage {
	copy := make(map[string]json.RawMessage, len(fields))
	for key, value := range fields {
		copy[key] = append(json.RawMessage(nil), value...)
	}
	return copy
}

// Reserve aliases as well as fields already emitted by the typed serializers.
const ownedExtraFields = " model messages input instructions system tools tool_choice toolChoice stream stream_options temperature top_p top_k presence_penalty frequency_penalty stop stop_sequences max_tokens max_completion_tokens max_output_tokens maxOutputTokens reasoning reasoning_effort reasoning_budget_tokens thinking enable_thinking output_config response_format text parallel_tool_calls disable_parallel_tool_use logit_bias logprobs log_probs top_logprobs top_log_probs user verbosity prediction store safety_identifier service_tier prompt_cache_key prompt_cache_retention prompt_cache_options cache_control session_id include background endpoint base_url headers authorization api_key "

func newExtraFields(fields map[string]any, anthropic bool) (map[string]json.RawMessage, error) {
	result := make(map[string]json.RawMessage, len(fields))
	for key, value := range fields {
		if key == "" || !utf8.ValidString(key) {
			return nil, fmt.Errorf("extra field name must be nonempty UTF-8")
		}
		if anthropic {
			for _, r := range key {
				if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-') {
					return nil, fmt.Errorf("extra field %q must be a plain top-level name", key)
				}
			}
		}
		if strings.Contains(ownedExtraFields, " "+key+" ") {
			return nil, fmt.Errorf("extra field %q is owned by the request", key)
		}
		data, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("extra field %q must be JSON: %w", key, err)
		}
		if !jsontext.Value(data).IsValid() {
			return nil, fmt.Errorf("extra field %q must be valid I-JSON", key)
		}
		result[key] = data
	}
	return result, nil
}

func (r Request) validateProtocolOptions() error {
	if r.ReasoningPolicy != ReasoningPolicyDefault && r.ReasoningPolicy != ReasoningPolicyExact {
		return fmt.Errorf("reasoning policy %q is invalid", r.ReasoningPolicy)
	}
	if r.ReasoningPolicy == ReasoningPolicyExact && r.ReasoningBudgetTokens != 0 && r.ReasoningEffort != ReasoningEffortDefault {
		return fmt.Errorf("exact reasoning cannot combine an effort and an explicit budget")
	}
	if r.TopK != nil && *r.TopK < 0 {
		return fmt.Errorf("top-k must not be negative")
	}
	protocols := 0
	if r.OpenAIChat != nil {
		protocols++
	}
	if r.OpenAIResponses != nil {
		protocols++
	}
	if r.Anthropic != nil {
		protocols++
	}
	if protocols > 1 {
		return fmt.Errorf("request cannot combine different protocol options")
	}
	if options := r.OpenAIChat; options != nil {
		if err := validateVerbosity(options.Verbosity); err != nil {
			return err
		}
		if err := validateServiceTier(options.ServiceTier); err != nil {
			return err
		}
		if options.TopLogProbs != nil {
			if *options.TopLogProbs < 0 || *options.TopLogProbs > 20 {
				return fmt.Errorf("top logprobs must be between 0 and 20")
			}
			if options.LogProbs != nil && !*options.LogProbs {
				return fmt.Errorf("top logprobs conflicts with logprobs false")
			}
		}
		for token, bias := range options.LogitBias {
			id, err := strconv.ParseUint(token, 10, 64)
			if err != nil || strconv.FormatUint(id, 10) != token {
				return fmt.Errorf("logit bias key %q must be a token ID", token)
			}
			if bias < -100 || bias > 100 {
				return fmt.Errorf("logit bias for token %q must be between -100 and 100", token)
			}
		}
		if len(options.Metadata) > 16 {
			return fmt.Errorf("metadata must not contain more than 16 entries")
		}
		for key, value := range options.Metadata {
			if !utf8.ValidString(key) || !utf8.ValidString(value) || utf8.RuneCountInString(key) > 64 || utf8.RuneCountInString(value) > 512 {
				return fmt.Errorf("metadata key %q or value is invalid", key)
			}
		}
		if len(options.Metadata) != 0 {
			if _, ok := options.ExtraFields.fields["metadata"]; ok {
				return fmt.Errorf("extra metadata conflicts with typed metadata")
			}
		}
		for name, value := range map[string]*string{"user": options.User, "safety identifier": options.SafetyIdentifier} {
			if value != nil && !utf8.ValidString(*value) {
				return fmt.Errorf("%s must be valid UTF-8", name)
			}
		}
		if options.Prediction != nil && !utf8.ValidString(options.Prediction.Content) {
			return fmt.Errorf("prediction content must be valid UTF-8")
		}
	}
	if options := r.OpenAIResponses; options != nil {
		if err := validateVerbosity(options.Verbosity); err != nil {
			return err
		}
		if err := validateServiceTier(options.ServiceTier); err != nil {
			return err
		}
	}
	if options := r.Anthropic; options != nil && options.ThinkingDisplay != nil {
		if *options.ThinkingDisplay != "summarized" && *options.ThinkingDisplay != "omitted" {
			return fmt.Errorf("thinking display must be summarized or omitted")
		}
		if r.ReasoningEffort == ReasoningEffortNone || r.ReasoningEffort == ReasoningEffortDefault && r.ReasoningBudgetTokens == 0 {
			return fmt.Errorf("thinking display requires explicit thinking")
		}
	}
	return nil
}

func validateVerbosity(value *string) error {
	if value != nil && *value != "low" && *value != "medium" && *value != "high" {
		return fmt.Errorf("verbosity must be low, medium or high")
	}
	return nil
}

func validateServiceTier(value *string) error {
	if value != nil {
		switch *value {
		case "auto", "default", "flex", "scale", "priority", "fast", "ultrafast":
		default:
			return fmt.Errorf("service tier %q is invalid", *value)
		}
	}
	return nil
}
