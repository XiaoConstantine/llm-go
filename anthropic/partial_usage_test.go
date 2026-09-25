package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestPartialUsageSurvivesErrorAndTruncation(t *testing.T) {
	for _, ending := range []string{"error", "truncated"} {
		t.Run(ending, func(t *testing.T) {
			body := streamEvent("message_start", `{"type":"message_start","message":{"id":"partial","type":"message","role":"assistant","model":"model","content":[],"stop_reason":null,"usage":{"input_tokens":8,"output_tokens":0,"cache_creation_input_tokens":1,"cache_read_input_tokens":2}}}`) +
				streamEvent("message_delta", `{"type":"message_delta","delta":{},"usage":{"output_tokens":3}}`)
			if ending == "error" {
				body += streamEvent("error", `{"type":"error","error":{"type":"overloaded_error","message":"fixture failure"}}`)
			}
			client := newStaticStreamClient(t, http.StatusOK, "text/event-stream", body)
			stream, err := client.Stream(context.Background(), textRequest("hello"))
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			chunks, terminal := receiveAll(stream)
			if terminal == nil || errors.Is(terminal, io.EOF) {
				t.Fatalf("terminal=%v", terminal)
			}
			if len(chunks) != 2 || chunks[0].Usage == nil || chunks[1].Usage == nil {
				t.Fatalf("chunks=%+v", chunks)
			}
			first, last := chunks[0].Usage, chunks[1].Usage
			if first.InputTokens != 8 || first.OutputTokens != 0 || first.CacheReadTokens != 2 || first.CacheWriteTokens != 1 {
				t.Fatalf("first usage=%+v", first)
			}
			if last.InputTokens != 8 || last.OutputTokens != 3 || last.CacheReadTokens != 2 || last.CacheWriteTokens != 1 {
				t.Fatalf("last usage=%+v", last)
			}
			if chunks[0].FinishReason != "" || chunks[1].FinishReason != "" {
				t.Fatal("partial usage declared success")
			}
		})
	}
}

func TestPartialUsageSurvivesCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, streamEvent("message_start", validStreamStart))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer server.Close()
	client, err := New(Config{Model: "model", BaseURL: server.URL, HTTPClient: server.Client(), Capabilities: []llm.Capability{llm.CapabilityStreaming}})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := client.Stream(ctx, textRequest("hello"))
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	chunk, err := stream.Recv()
	if err != nil || chunk.Usage == nil {
		t.Fatalf("chunk=%+v err=%v", chunk, err)
	}
	cancel()
	_, terminal := stream.Recv()
	if !errors.Is(terminal, context.Canceled) {
		t.Fatalf("terminal=%v", terminal)
	}
	if chunk.Usage.InputTokens != 1 || chunk.Usage.OutputTokens != 1 {
		t.Fatalf("reported usage=%+v", chunk.Usage)
	}
}
