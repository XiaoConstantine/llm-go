package modelsdev

import (
	"context"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/models"
)

// DefaultURL is the standard models.dev API endpoint.
const DefaultURL = "https://models.dev/api.json"

// maxResponseSize limits downloaded models.dev dataset size to 32MB.
const maxResponseSize = 32 << 20

// Option configures a Source.
type Option func(*Source)

// WithHTTPClient sets a custom HTTP client for models.dev requests.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Source) {
		if client != nil {
			s.httpClient = client
		}
	}
}

// WithURL sets a custom endpoint URL.
func WithURL(url string) Option {
	return func(s *Source) {
		if url != "" {
			s.url = url
		}
	}
}

// WithProviderAlias configures an alias mapping from an llm-go provider name to
// a models.dev provider name.
func WithProviderAlias(llmProvider, modelsDevProvider string) Option {
	return func(s *Source) {
		llmProvider = strings.TrimSpace(llmProvider)
		modelsDevProvider = strings.TrimSpace(modelsDevProvider)
		if llmProvider != "" && modelsDevProvider != "" {
			s.providerAliases[llmProvider] = modelsDevProvider
		}
	}
}

// WithAPIResolver overrides default protocol assignment for models. The resolver
// may be invoked concurrently by multiple goroutines and must not retain or mutate
// its ModelEntry argument.
func WithAPIResolver(fn func(provider string, model ModelEntry) llm.API) Option {
	return func(s *Source) {
		if fn != nil {
			s.apiResolver = fn
		}
	}
}

// WithCompatibilityResolver overrides default compatibility assignment. The resolver
// may be invoked concurrently by multiple goroutines and must not retain or mutate
// its ModelEntry argument.
func WithCompatibilityResolver(fn func(provider string, model ModelEntry, api llm.API) *llm.ModelCompatibility) Option {
	return func(s *Source) {
		if fn != nil {
			s.compatResolver = fn
		}
	}
}

// WithFilter sets a predicate to include or exclude models. The filter may be
// invoked concurrently by multiple goroutines and must not retain or mutate its
// ModelEntry argument.
func WithFilter(fn func(provider string, model ModelEntry) bool) Option {
	return func(s *Source) {
		s.filter = fn
	}
}

type inflightFetch struct {
	done chan struct{}
	err  error
}

// Source implements models.CatalogSource by fetching model definitions from
// models.dev. It is safe for concurrent use.
type Source struct {
	httpClient      *http.Client
	url             string
	providerAliases map[string]string
	apiResolver     func(provider string, model ModelEntry) llm.API
	compatResolver  func(provider string, model ModelEntry, api llm.API) *llm.ModelCompatibility
	filter          func(provider string, model ModelEntry) bool

	mu       sync.Mutex
	lastETag string
	lastMod  time.Time
	dataset  Dataset
	inflight *inflightFetch
}

// NewSource creates a Source with default provider aliases and settings.
func NewSource(opts ...Option) *Source {
	s := &Source{
		httpClient: http.DefaultClient,
		url:        DefaultURL,
		providerAliases: map[string]string{
			"fireworks": "fireworks-ai",
			"together":  "togetherai",
			"moonshot":  "moonshotai",
		},
		apiResolver:    func(p string, _ ModelEntry) llm.API { return DefaultProviderAPI(p) },
		compatResolver: DefaultProviderCompatibility,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Fetch implements models.CatalogSource.
func (s *Source) Fetch(ctx context.Context, req models.CatalogFetchRequest) (models.CatalogFetchResponse, error) {
	if err := ctx.Err(); err != nil {
		return models.CatalogFetchResponse{}, err
	}

	provider := strings.TrimSpace(req.Provider)
	if provider == "" {
		return models.CatalogFetchResponse{}, &llm.Error{
			Kind:     llm.KindInvalidRequest,
			Op:       "modelsdev.Fetch",
			Provider: req.Provider,
			Err:      fmt.Errorf("provider must not be empty"),
		}
	}

	dataset, etag, lastMod, notModified, err := s.getDataset(ctx, req)
	if err != nil {
		return models.CatalogFetchResponse{}, err
	}
	if notModified {
		return models.CatalogFetchResponse{
			NotModified:  true,
			ETag:         etag,
			LastModified: lastMod,
		}, nil
	}

	targetProvider := s.resolveProvider(provider)
	providerEntry, found := dataset[targetProvider]
	if !found {
		providerEntry, found = dataset[strings.ToLower(targetProvider)]
	}
	if !found {
		return models.CatalogFetchResponse{
			Models:       nil,
			ETag:         etag,
			LastModified: lastMod,
		}, nil
	}

	modelsList := make([]llm.Model, 0, len(providerEntry.Models))
	for modelID, rawModel := range providerEntry.Models {
		if s.filter != nil && !s.filter(provider, rawModel) {
			continue
		}
		api := s.apiResolver(provider, rawModel)
		compat := s.compatResolver(provider, rawModel, api)
		converted, err := ConvertModel(provider, rawModel, api, compat)
		if err != nil {
			return models.CatalogFetchResponse{}, &llm.Error{
				Kind:     llm.KindMalformedResponse,
				Op:       "modelsdev.Fetch",
				Provider: provider,
				Err:      fmt.Errorf("model %q: %w", modelID, err),
			}
		}
		modelsList = append(modelsList, converted)
	}

	sort.Slice(modelsList, func(i, j int) bool {
		return modelsList[i].ID < modelsList[j].ID
	})

	return models.CatalogFetchResponse{
		Models:       modelsList,
		ETag:         etag,
		LastModified: lastMod,
	}, nil
}

func (s *Source) resolveProvider(provider string) string {
	s.mu.Lock()
	alias, ok := s.providerAliases[provider]
	s.mu.Unlock()
	if ok {
		return alias
	}
	return provider
}

func (s *Source) getDataset(ctx context.Context, req models.CatalogFetchRequest) (Dataset, string, time.Time, bool, error) {
	for {
		s.mu.Lock()
		if s.inflight != nil {
			flight := s.inflight
			s.mu.Unlock()

			select {
			case <-ctx.Done():
				return nil, "", time.Time{}, false, ctx.Err()
			case <-flight.done:
				if flight.err != nil {
					if errors.Is(flight.err, context.Canceled) || errors.Is(flight.err, context.DeadlineExceeded) {
						// Leader context canceled; retry as leader or next waiter
						continue
					}
					return nil, "", time.Time{}, false, flight.err
				}

				s.mu.Lock()
				dataset := s.dataset
				etag := s.lastETag
				lastMod := s.lastMod
				s.mu.Unlock()

				if !req.Force {
					if req.ETag != "" {
						if req.ETag == etag {
							return nil, etag, lastMod, true, nil
						}
					} else if !req.LastModified.IsZero() && !lastMod.IsZero() {
						if req.LastModified.Equal(lastMod) || req.LastModified.After(lastMod) {
							return nil, etag, lastMod, true, nil
						}
					}
				}
				if dataset == nil {
					// Inflight returned 304 without cached data; waiter must perform full fetch
					continue
				}
				return dataset, etag, lastMod, false, nil
			}
		}

		flight := &inflightFetch{done: make(chan struct{})}
		s.inflight = flight
		currentETag := s.lastETag
		currentLastMod := s.lastMod
		s.mu.Unlock()

		dataset, etag, lastMod, err := s.doFetch(ctx, req, currentETag, currentLastMod)

		s.mu.Lock()
		flight.err = err
		if err == nil && dataset != nil {
			s.dataset = dataset
			s.lastETag = etag
			s.lastMod = lastMod
		}
		s.inflight = nil
		close(flight.done)
		s.mu.Unlock()

		if err != nil {
			return nil, "", time.Time{}, false, err
		}

		if dataset == nil {
			// Server returned 304 Not Modified
			if !req.Force {
				if req.ETag != "" {
					return nil, etag, lastMod, true, nil
				}
				if !req.LastModified.IsZero() {
					return nil, etag, lastMod, true, nil
				}
			}
			s.mu.Lock()
			cached := s.dataset
			s.mu.Unlock()
			if cached == nil {
				// No cached dataset; perform full unconditional fetch
				reqCopy := req
				reqCopy.Force = true
				dataset, etag, lastMod, err = s.doFetch(ctx, reqCopy, "", time.Time{})
				if err != nil {
					return nil, "", time.Time{}, false, err
				}
				s.mu.Lock()
				s.dataset = dataset
				s.lastETag = etag
				s.lastMod = lastMod
				s.mu.Unlock()
				return dataset, etag, lastMod, false, nil
			}
			return cached, etag, lastMod, false, nil
		}

		return dataset, etag, lastMod, false, nil
	}
}

func (s *Source) doFetch(ctx context.Context, req models.CatalogFetchRequest, cachedETag string, cachedLastMod time.Time) (Dataset, string, time.Time, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url, nil)
	if err != nil {
		return nil, "", time.Time{}, &llm.Error{
			Kind:     llm.KindInvalidRequest,
			Op:       "modelsdev.doFetch",
			Provider: req.Provider,
			Err:      err,
		}
	}
	httpReq.Header.Set("Accept", "application/json")
	if !req.Force {
		if req.ETag != "" {
			httpReq.Header.Set("If-None-Match", req.ETag)
		} else if cachedETag != "" {
			httpReq.Header.Set("If-None-Match", cachedETag)
		}
		if !req.LastModified.IsZero() {
			httpReq.Header.Set("If-Modified-Since", req.LastModified.UTC().Format(http.TimeFormat))
		} else if !cachedLastMod.IsZero() {
			httpReq.Header.Set("If-Modified-Since", cachedLastMod.UTC().Format(http.TimeFormat))
		}
	}

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, "", time.Time{}, &llm.Error{
			Kind:     llm.KindTransport,
			Op:       "modelsdev.doFetch",
			Provider: req.Provider,
			Err:      err,
		}
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode == http.StatusNotModified {
		etag := resp.Header.Get("ETag")
		if etag == "" {
			etag = cachedETag
		}
		return nil, etag, cachedLastMod, nil
	}

	if resp.StatusCode != http.StatusOK {
		return nil, "", time.Time{}, &llm.Error{
			Kind:       llm.KindTransport,
			Op:         "modelsdev.doFetch",
			Provider:   req.Provider,
			HTTPStatus: resp.StatusCode,
			Err:        fmt.Errorf("models.dev returned unexpected status %d", resp.StatusCode),
		}
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseSize+1))
	if err != nil {
		return nil, "", time.Time{}, &llm.Error{
			Kind:     llm.KindTransport,
			Op:       "modelsdev.doFetch",
			Provider: req.Provider,
			Err:      err,
		}
	}
	if len(body) > maxResponseSize {
		return nil, "", time.Time{}, &llm.Error{
			Kind:     llm.KindMalformedResponse,
			Op:       "modelsdev.doFetch",
			Provider: req.Provider,
			Err:      fmt.Errorf("models.dev payload exceeded maximum allowed size of %d bytes", maxResponseSize),
		}
	}

	var dataset Dataset
	if err := jsonv2.Unmarshal(body, &dataset); err != nil {
		return nil, "", time.Time{}, &llm.Error{
			Kind:     llm.KindMalformedResponse,
			Op:       "modelsdev.doFetch",
			Provider: req.Provider,
			Err:      fmt.Errorf("failed to decode models.dev response: %w", err),
		}
	}

	etag := resp.Header.Get("ETag")
	var lastMod time.Time
	if lastModStr := resp.Header.Get("Last-Modified"); lastModStr != "" {
		lastMod, _ = http.ParseTime(lastModStr)
	}

	return dataset, etag, lastMod, nil
}
