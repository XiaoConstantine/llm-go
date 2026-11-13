package openai

import (
	"fmt"

	llm "github.com/XiaoConstantine/llm-go"
)

// RequestOptions holds request settings specific to the Chat Completions protocol.
// Set a pointer to it as llm.Request.ProviderOptions.
type RequestOptions struct {
	// ServiceTier is sent as service_tier, for example "auto", "default",
	// "flex", or "priority" on OpenAI. Compatible endpoints define their own
	// values, so it is not checked locally. Empty omits the field.
	ServiceTier string
}

// CloneProviderOptions implements llm.ProviderOptions.
func (o *RequestOptions) CloneProviderOptions() llm.ProviderOptions {
	if o == nil {
		return o
	}
	clone := *o
	return &clone
}

// requestOptions returns the request's Chat Completions options, or nil when
// it has none. Options that belong to another adapter are unsupported.
func requestOptions(op string, request llm.Request) (*RequestOptions, error) {
	switch options := request.ProviderOptions.(type) {
	case nil:
		return nil, nil
	case *RequestOptions:
		return options, nil
	default:
		return nil, unsupported(op, fmt.Sprintf("provider options of type %T belong to another adapter", options))
	}
}
