// Package responsebody bounds SDK reads without buffering whole responses.
package responsebody

import (
	"bufio"
	"fmt"
	"io"
)

type body struct {
	io.Reader
	io.Closer
}

// Truncate exposes at most limit bytes, retaining ownership of the original body.
// It is for error responses, whose HTTP status remains useful without full detail.
func Truncate(source io.ReadCloser, limit int64) io.ReadCloser {
	return body{Reader: io.LimitReader(source, limit), Closer: source}
}

// Limit rejects a response larger than limit, reading at most one extra byte.
// bufio.ErrTooLong lets existing SDK error adapters classify oversized input.
func Limit(source io.ReadCloser, limit int64) io.ReadCloser {
	return body{Reader: &limitReader{source: source, limit: limit}, Closer: source}
}

// SSE bounds each wire event, including data, comments, and ignored fields.
// An empty LF or CRLF line resets the budget, matching the SDK scanners. It does
// not limit the total stream length or allocate storage proportional to an event.
func SSE(source io.ReadCloser, limit int64) io.ReadCloser {
	return body{Reader: &limitReader{source: source, limit: limit, events: true}, Closer: source}
}

// SSEChunks matches scanners that split only on LF-LF or CRLF-CRLF, rather
// than individual lines. A mixed LF-CRLF boundary must not reset the budget
// when the downstream scanner will keep accumulating the same chunk.
func SSEChunks(source io.ReadCloser, limit int64) io.ReadCloser {
	return body{Reader: &limitReader{source: source, limit: limit, events: true, chunks: true}, Closer: source}
}

// Err returns the sticky size-limit error from a body returned by Limit, SSE,
// or SSEChunks. Some SDK scanners decode a partial token before reporting the
// reader error; callers can use this to prefer the short size-limit error.
// Call only after the SDK has finished reading (or from its synchronous yield).
func Err(source io.ReadCloser) error {
	if b, ok := source.(body); ok {
		if r, ok := b.Reader.(*limitReader); ok {
			return r.err
		}
	}
	return nil
}

type limitReader struct {
	source       io.Reader
	limit        int64
	used         int64
	events       bool
	chunks       bool
	line         int64
	lastCR       bool
	previousCRLF bool
	err          error
}

func (r *limitReader) Read(p []byte) (int, error) {
	if r.err != nil {
		return 0, r.err
	}
	p = p[:min(int64(len(p)), r.limit-r.used+1)]
	n, err := r.source.Read(p)
	for i, b := range p[:n] {
		r.used++
		if r.used > r.limit {
			r.err = fmt.Errorf("response is too large: exceeds %d bytes: %w", r.limit, bufio.ErrTooLong)
			return i, r.err
		}
		if r.events {
			if b == '\n' {
				if r.line == 0 || r.line == 1 && r.lastCR && (!r.chunks || r.previousCRLF) {
					r.used = 0
				}
				r.previousCRLF = r.lastCR
				r.line = 0
			} else {
				r.line++
			}
			r.lastCR = b == '\r'
		}
	}
	return n, err
}
