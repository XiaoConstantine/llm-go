package responsebody_test

import (
	"bufio"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/iotest"

	"github.com/XiaoConstantine/llm-go/internal/responsebody"
)

func TestLimitsAcrossReadBoundaries(t *testing.T) {
	for _, test := range []struct {
		name     string
		wrap     func(io.ReadCloser, int64) io.ReadCloser
		wire     string
		limit    int64
		tooLarge bool
	}{
		{"body below", responsebody.Limit, "1234567", 8, false},
		{"body exactly", responsebody.Limit, "12345678", 8, false},
		{"body over", responsebody.Limit, "123456789", 8, true},
		{"LF events exactly", responsebody.SSE, "data:x\n\ndata:y\n\n", 8, false},
		{"LF event over", responsebody.SSE, "data:xx\n\n", 8, true},
		{"CRLF events exactly", responsebody.SSE, "data:x\r\n\r\ndata:y\r\n\r\n", 10, false},
		{"mixed SSE lines", responsebody.SSE, "data:x\n\r\ndata:y\n\r\n", 9, false},
		{"multiline event", responsebody.SSE, "data:x\ndata:y\n\n", 8, true},
		{"comments count", responsebody.SSE, ":12345\n:67890\n\n", 8, true},
		{"ignored fields count", responsebody.SSE, "id:1234\nid:5678\n\n", 8, true},
		{"CRLF chunks", responsebody.SSEChunks, "data:x\r\n\r\ndata:y\r\n\r\n", 10, false},
		{"LF chunks", responsebody.SSEChunks, "data:x\n\ndata:y\n\n", 8, false},
		{"LF chunk over", responsebody.SSEChunks, "data:xx\n\n", 8, true},
		{"CRLF chunk over", responsebody.SSEChunks, "data:xx\r\n\r\n", 10, true},
		{"mixed chunk boundary not recognized", responsebody.SSEChunks, "data:x\n\r\ndata:y\n\r\n", 9, true},
	} {
		for _, fragmented := range []bool{false, true} {
			t.Run(test.name+map[bool]string{false: "/whole", true: "/one-byte"}[fragmented], func(t *testing.T) {
				var source io.Reader = strings.NewReader(test.wire)
				if fragmented {
					source = iotest.OneByteReader(source)
				}
				reader := test.wrap(io.NopCloser(source), test.limit)
				defer func() { _ = reader.Close() }()
				got, err := io.ReadAll(reader)
				if test.tooLarge {
					if !errors.Is(err, bufio.ErrTooLong) || !errors.Is(responsebody.Err(reader), bufio.ErrTooLong) {
						t.Fatalf("read error = %v, captured = %v, want ErrTooLong", err, responsebody.Err(reader))
					}
					if n, nextErr := reader.Read(make([]byte, 1)); n != 0 || !errors.Is(nextErr, bufio.ErrTooLong) {
						t.Fatalf("subsequent read = %d, %v", n, nextErr)
					}
				} else if err != nil || string(got) != test.wire {
					t.Fatalf("read = %q, %v; want %q", got, err, test.wire)
				}
			})
		}
	}
}
