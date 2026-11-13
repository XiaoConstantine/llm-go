package anthropic

import (
	"encoding/json"
	jsonv2 "encoding/json/v2"
	"fmt"
	"reflect"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/adapterutil"
)

// reservedFields are the request body members the adapter declares.
var reservedFields = adapterutil.WireFields(reflect.TypeFor[messageRequest]())

// RequestOptions holds request settings specific to the Messages protocol. Set a
// pointer to it as llm.Request.ProviderOptions.
type RequestOptions struct {
	// ServiceTier is sent as service_tier. Anthropic documents "auto" and
	// "standard_only". Empty omits the field.
	ServiceTier string
	// ExtraFields adds top-level members to the request body for settings
	// this package does not model. Each value is one JSON value, sent as is.
	// A name the adapter itself can send, such as "model", "messages", or
	// "stream", is an invalid request whether or not this request sets it.
	ExtraFields map[string]json.RawMessage
}

// CloneProviderOptions implements llm.ProviderOptions.
func (o *RequestOptions) CloneProviderOptions() llm.ProviderOptions {
	if o == nil {
		return o
	}
	clone := *o
	clone.ExtraFields = adapterutil.CloneExtraFields(o.ExtraFields)
	return &clone
}

// requestOptions returns the request's Messages options, or nil when it has
// none. Options that belong to another adapter are unsupported, and invalid
// extra fields are an invalid request.
func requestOptions(op string, request llm.Request) (*RequestOptions, error) {
	switch options := request.ProviderOptions.(type) {
	case nil:
		return nil, nil
	case *RequestOptions:
		if options == nil {
			return nil, nil
		}
		if err := adapterutil.CheckExtraFields(options.ExtraFields, reservedFields); err != nil {
			return nil, requestError(op, "provider options: %w", err)
		}
		return options, nil
	default:
		return nil, unsupported(op, fmt.Sprintf("provider options of type %T belong to another adapter", options))
	}
}

// encode serializes the request body and appends the checked extra fields.
func (r *messageRequest) encode() ([]byte, error) {
	body, err := jsonv2.Marshal(r)
	if err != nil {
		return nil, err
	}
	return adapterutil.AppendExtraFields(body, r.extra)
}
