package gemini

import (
	"context"
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/internal/adapterutil"
	"google.golang.org/genai"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/types/known/durationpb"
)

type errorMetadataKey struct{}

// The SDK's APIError retains Details but drops response headers. Capture only
// Retry-After, per request, without consuming or replacing the response body.
type errorMetadata struct{ retryAfter atomic.Int64 }

func withErrorMetadata(ctx context.Context) (context.Context, *errorMetadata) {
	metadata := new(errorMetadata)
	return context.WithValue(ctx, errorMetadataKey{}, metadata), metadata
}

type errorMetadataTransport struct{ base http.RoundTripper }

func (t errorMetadataTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := t.base.RoundTrip(request)
	if response != nil {
		if metadata, ok := request.Context().Value(errorMetadataKey{}).(*errorMetadata); ok {
			metadata.retryAfter.Store(int64(adapterutil.RetryAfter(response.Header.Get("Retry-After"))))
		}
	}
	return response, err
}

func (m *errorMetadata) sdkError(op string, err error) error {
	converted := sdkError(op, err)
	if modelErr, ok := converted.(*llm.Error); ok && modelErr.HTTPStatus != 0 {
		modelErr.RetryAfter = max(modelErr.RetryAfter, time.Duration(m.retryAfter.Load()))
	}
	return converted
}

func retryInfoDelay(apiErr genai.APIError) time.Duration {
	var delay time.Duration
	for _, detail := range apiErr.Details {
		if detail["@type"] != "type.googleapis.com/google.rpc.RetryInfo" {
			continue
		}
		value, ok := detail["retryDelay"].(string)
		if !ok {
			continue
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			continue
		}
		var parsed durationpb.Duration
		if err := protojson.Unmarshal(encoded, &parsed); err == nil {
			delay = max(delay, parsed.AsDuration())
		}
	}
	return delay
}

func invalidAPIKey(apiErr genai.APIError) bool {
	for _, detail := range apiErr.Details {
		if detail["@type"] == "type.googleapis.com/google.rpc.ErrorInfo" &&
			detail["domain"] == "googleapis.com" && detail["reason"] == "API_KEY_INVALID" {
			return true
		}
	}
	return false
}
