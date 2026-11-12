package llm_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/gemini"
	"github.com/XiaoConstantine/llm-go/openai/azure"
	"github.com/XiaoConstantine/llm-go/openai/codex"
	"github.com/XiaoConstantine/llm-go/openai/responses"
	"github.com/XiaoConstantine/llm-go/vertex"
)

// Exercise the real SDK parsers, including the adapters that delegate to them.
func TestAdapterResponseReadLimits(t *testing.T) {
	for _, adapter := range []string{"responses", "azure", "codex", "gemini", "vertex"} {
		for _, streaming := range []bool{false, true} {
			google := adapter == "gemini" || adapter == "vertex"
			scenarios := []string{"error"}
			// Google success-body limits are exercised with a small injected
			// transport budget in gemini, not 64 MiB SDK allocations per case.
			if !google {
				scenarios = append(scenarios, "line", "event")
			}
			for _, scenario := range scenarios {
				t.Run(fmt.Sprintf("%s/stream=%t/%s", adapter, streaming, scenario), func(t *testing.T) {
					limit := 17 << 20
					status := http.StatusOK
					kind := llm.KindMalformedResponse
					contentType := "text/event-stream"
					var wire string
					switch scenario {
					case "error":
						limit, status, kind, contentType = 1<<20, http.StatusTooManyRequests, llm.KindRateLimit, "application/json"
						wire = `{"error":{"message":"` + strings.Repeat("x", limit*2) + `"}}`
						if google {
							wire = `{"error":{"code":429,"message":"` + strings.Repeat("x", limit*2) + `"}}`
						}
					case "line":
						wire = "data: " + strings.Repeat("x", limit*2) // No newline: must stop before SDK's line limit.
					case "event":
						wire = strings.Repeat("data: x\n", limit/4) // Many short lines, no blank dispatch line.
					}
					body := &countedResponseBody{Reader: strings.NewReader(wire)}
					client := resourceClient(t, adapter, &http.Client{Transport: responseTransport(func(r *http.Request) (*http.Response, error) {
						return &http.Response{StatusCode: status, Status: fmt.Sprintf("%d test", status),
							Header: http.Header{"Content-Type": {contentType}}, Body: body, Request: r}, nil
					})})
					request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
					var err error
					if streaming {
						stream, openErr := client.Stream(context.Background(), request)
						if openErr != nil {
							t.Fatal(openErr)
						}
						_, err = llm.CollectStructural(stream)
					} else {
						_, err = client.Generate(context.Background(), request)
					}
					if body.read > limit+1 || scenario != "error" && body.read != limit+1 {
						t.Errorf("read %d bytes, want budget %d plus one overflow byte", body.read, limit)
					}
					if !body.closed {
						t.Error("response body was not closed")
					}
					var modelErr *llm.Error
					if !errors.As(err, &modelErr) || modelErr.Kind != kind {
						t.Fatalf("error kind = %v, want %s", errorKind(err), kind)
					}
					if scenario == "error" && (modelErr.HTTPStatus != status || len(err.Error()) > limit+1024) {
						t.Errorf("error status=%d length=%d, want %d and bounded detail", modelErr.HTTPStatus, len(err.Error()), status)
					}
					if scenario != "error" && len(err.Error()) > 1024 {
						t.Errorf("over-limit error length = %d, want <= 1024 without payload", len(err.Error()))
					}
				})
			}
		}
	}
}

func TestAdapterStreamLimitIsPerEvent(t *testing.T) {
	for _, adapter := range []string{"responses", "azure", "codex"} {
		for _, newline := range []string{"\n", "\r\n"} {
			t.Run(fmt.Sprintf("%s/newline=%q", adapter, newline), func(t *testing.T) {
				first := `{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m","role":"assistant","content":[]}}`
				tail := []string{
					`{"type":"response.output_text.done","output_index":0,"content_index":0,"item_id":"m","text":"hello"}`,
					`{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"m","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`,
					`{"type":"response.completed","response":{"id":"r","status":"completed","output":[]}}`,
				}
				// The first event is exactly 17 MiB; the total stream exceeds that limit.
				wire := "data: " + first + strings.Repeat(" ", (17<<20)-len("data: ")-len(first)-2*len(newline)) + newline + newline
				wire += "data: " + strings.Join(tail, newline+newline+"data: ") + newline + newline
				body := &countedResponseBody{Reader: strings.NewReader(wire)}
				client := resourceClient(t, adapter, &http.Client{Transport: responseTransport(func(r *http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: body, Request: r}, nil
				})})
				stream, err := client.Stream(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}})
				if err != nil {
					t.Fatal(err)
				}
				response, err := llm.CollectStructural(stream)
				if err != nil || response.Text() != "hello" || !body.closed {
					t.Fatalf("response=%v error=%v closed=%t", response, err, body.closed)
				}
			})
		}
	}
}

func TestLargeResponsesEvents(t *testing.T) {
	for _, adapter := range []string{"responses", "azure", "codex"} {
		for _, items := range []int{1, 2} {
			for _, streaming := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/items=%d/stream=%t", adapter, items, streaming), func(t *testing.T) {
					var wire strings.Builder
					var output []string
					var want strings.Builder
					for i := range items {
						text := strings.Repeat(string(rune('a'+i)), (5<<20)/items)
						want.WriteString(text)
						item := fmt.Sprintf(`{"type":"message","id":"m%d","role":"assistant","status":"completed","content":[{"type":"output_text","text":%q,"annotations":[]}]}`, i, text)
						fmt.Fprintf(&wire, "data: {\"type\":\"response.output_item.added\",\"output_index\":%d,\"item\":{\"type\":\"message\",\"id\":\"m%d\",\"role\":\"assistant\",\"content\":[]}}\n\n", i, i)
						fmt.Fprintf(&wire, "data: {\"type\":\"response.output_text.delta\",\"output_index\":%d,\"content_index\":0,\"item_id\":\"m%d\",\"delta\":%q}\n\n", i, i, text)
						fmt.Fprintf(&wire, "data: {\"type\":\"response.output_item.done\",\"output_index\":%d,\"item\":%s}\n\n", i, item)
						output = append(output, item)
					}
					// The terminal event repeats output already delivered by earlier events.
					fmt.Fprintf(&wire, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"status\":\"completed\",\"output\":[%s]}}\n\n", strings.Join(output, ","))
					response := resourceResponse(t, adapter, wire.String(), streaming)
					if response.Text() != want.String() {
						t.Fatalf("text length = %d, want %d with matching content", len(response.Text()), want.Len())
					}
				})
			}
		}
	}
}

func TestGoogleInlineMediaResponseLimits(t *testing.T) {
	for _, adapter := range []string{"gemini", "vertex"} {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", adapter, streaming), func(t *testing.T) {
				size, mediaType, kind := 13<<20, "audio/wav", llm.PartAudio
				if streaming {
					size, mediaType, kind = 14<<20, "image/png", llm.PartImage
				}
				data := bytes.Repeat([]byte{0x3a, 0x7d}, size/2)
				wire := `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"` + mediaType + `","data":"` + base64.StdEncoding.EncodeToString(data) + `"}}]},"finishReason":"STOP"}]}`
				if streaming {
					wire = "data: " + wire + "\n\n"
				}
				response := resourceResponse(t, adapter, wire, streaming)
				if len(response.Message.Content) != 1 {
					t.Fatalf("content parts = %d, want one", len(response.Message.Content))
				}
				part := response.Message.Content[0]
				if part.Kind != kind || part.MediaType != mediaType || !bytes.Equal(part.Data, data) {
					t.Fatalf("media kind=%v type=%s bytes=%d, want %v %s %d with matching data", part.Kind, part.MediaType, len(part.Data), kind, mediaType, len(data))
				}
			})
		}
	}
}

func resourceResponse(t *testing.T, adapter, wire string, streaming bool) *llm.Response {
	t.Helper()
	contentType := "text/event-stream"
	if (adapter == "gemini" || adapter == "vertex") && !streaming {
		contentType = "application/json"
	}
	body := &countedResponseBody{Reader: strings.NewReader(wire)}
	client := resourceClient(t, adapter, &http.Client{Transport: responseTransport(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {contentType}}, Body: body, Request: r}, nil
	})})
	request := llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{{Text: "hello"}}}}}
	var response *llm.Response
	var err error
	if streaming {
		stream, openErr := client.Stream(context.Background(), request)
		if openErr != nil {
			t.Fatal(openErr)
		}
		response, err = llm.CollectStructural(stream)
	} else {
		response, err = client.Generate(context.Background(), request)
	}
	if !body.closed {
		t.Error("response body was not closed")
	}
	if err != nil {
		t.Fatalf("response error kind=%v detail=%.200s", errorKind(err), err)
	}
	return response
}

func TestResponsesBackgroundErrorBodyLimit(t *testing.T) {
	body := &countedResponseBody{Reader: strings.NewReader(strings.Repeat("x", 2<<20))}
	client := resourceClient(t, "responses", &http.Client{Transport: responseTransport(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{"Content-Type": {"application/json"}}, Body: body, Request: r}, nil
	})})
	_, err := client.(llm.BackgroundGenerator).StartBackground(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.RoleUser}}})
	var modelErr *llm.Error
	if !errors.As(err, &modelErr) || modelErr.Kind != llm.KindRateLimit || modelErr.HTTPStatus != 429 || body.read > 1<<20 || !body.closed {
		t.Fatalf("error kind=%v read=%d closed=%t", errorKind(err), body.read, body.closed)
	}
}

func resourceClient(t *testing.T, adapter string, httpClient *http.Client) llm.Generator {
	t.Helper()
	capabilities := []llm.Capability{llm.CapabilityStreaming}
	var client llm.Generator
	var err error
	switch adapter {
	case "responses":
		client, err = responses.New(responses.Config{Model: "model", APIKey: "key", HTTPClient: httpClient, Capabilities: capabilities})
	case "azure":
		client, err = azure.New(azure.Config{Model: "model", APIKey: "key", ResourceName: "test", HTTPClient: httpClient, Capabilities: capabilities})
	case "codex":
		client, err = codex.New(codex.Config{Model: "model", AccessToken: "token", AccountID: "account", HTTPClient: httpClient, Capabilities: capabilities})
	case "gemini":
		client, err = gemini.New(gemini.Config{Model: "model", APIKey: "key", HTTPClient: httpClient, Capabilities: capabilities})
	case "vertex":
		client, err = vertex.New(vertex.Config{Model: "model", APIKey: "key", HTTPClient: httpClient, Capabilities: capabilities})
	}
	if err != nil {
		t.Fatal(err)
	}
	return client
}

type responseTransport func(*http.Request) (*http.Response, error)

func (f responseTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type countedResponseBody struct {
	io.Reader
	read   int
	closed bool
}

func (b *countedResponseBody) Read(p []byte) (int, error) {
	n, err := b.Reader.Read(p)
	b.read += n
	return n, err
}
func (b *countedResponseBody) Close() error { b.closed = true; return nil }

func errorKind(err error) llm.ErrorKind {
	var modelErr *llm.Error
	if errors.As(err, &modelErr) {
		return modelErr.Kind
	}
	return 0
}
