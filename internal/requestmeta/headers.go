package requestmeta

import (
	"net/http"

	llm "github.com/XiaoConstantine/llm-go"
)

// WrapClient clones client and installs the safe per-attempt header transport.
func WrapClient(client *http.Client) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}
	clone := *client
	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	clone.Transport = headerTransport{base: base}
	return &clone
}

type headerTransport struct {
	base http.RoundTripper
}

func (t headerTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	headers := llm.AttemptHeaders(request.Context())
	if len(headers) == 0 {
		return t.base.RoundTrip(request)
	}
	clone := request.Clone(request.Context())
	clone.Header = request.Header.Clone()
	for name, values := range headers {
		clone.Header.Del(name)
		for _, value := range values {
			clone.Header.Add(name, value)
		}
	}
	return t.base.RoundTrip(clone)
}
