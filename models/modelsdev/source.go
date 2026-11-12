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

// WithAPIResolver overrides protocol assignment. Setting a non-nil resolver
// disables the default baseline unless WithBaseline is also specified. An explicit
// baseline takes precedence for matching models, regardless of option order.
// The resolver may be invoked concurrently and must not retain or mutate its argument.
func WithAPIResolver(fn func(provider string, model ModelEntry) llm.API) Option {
	return func(s *Source) {
		if fn != nil {
			s.apiResolver = fn
		}
	}
}

// WithCompatibilityResolver overrides compatibility. Setting a non-nil resolver
// disables the default baseline unless WithBaseline is also specified. An explicit
// baseline takes precedence for matching models, regardless of option order.
// The resolver may be invoked concurrently and must not retain or mutate its argument.
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

// WithBaseline preserves curated metadata for matching provider/model IDs.
// Only known limits and cost from models.dev replace baseline fields; API,
// compatibility, capabilities, name, and reasoning remain authoritative in the
// baseline. Absent cost rates retain baseline values, except an omitted cache-read
// rate is capped at the merged input rate. A non-empty list of surviving upstream
// tiers defines the set of thresholds, with matching thresholds inheriting baseline
// rates. Baseline tiers are retained only when upstream omits tiers or all upstream
// tiers are filtered out. Retained tiers at or above the merged, known context
// window are dropped. Resolvers apply only to models absent from an explicit
// baseline. Pass nil to opt out and convert all upstream metadata.
// Use the same catalog as CatalogManagerConfig.Baseline. Catalog is immutable
// and may be shared with the source.
func WithBaseline(baseline *models.Catalog) Option {
	return func(s *Source) {
		s.baseline = baseline
		s.baselineExplicit = true
	}
}

type inflightFetch struct {
	done chan struct{}
	err  error
}

// Source implements models.CatalogSource by fetching model definitions from
// models.dev. Records that fail model conversion are skipped unless every
// unfiltered record fails, in which case Fetch returns KindMalformedResponse.
// Transport and dataset decoding failures also fail the fetch.
// It is safe for concurrent use.
type Source struct {
	httpClient       *http.Client
	url              string
	providerAliases  map[string]string
	apiResolver      func(provider string, model ModelEntry) llm.API
	compatResolver   func(provider string, model ModelEntry, api llm.API) *llm.ModelCompatibility
	filter           func(provider string, model ModelEntry) bool
	baseline         *models.Catalog
	baselineExplicit bool

	mu       sync.Mutex
	lastETag string
	lastMod  time.Time
	dataset  Dataset
	inflight *inflightFetch
}

// NewSource creates a Source with default provider aliases and settings.
// It preserves curated fields from models.BuiltinCatalog() by default, updating
// only known limits and cost rates for matching models. A custom API or compatibility
// resolver disables that default. WithBaseline explicitly selects a baseline and
// takes precedence over resolvers for known models, independent of option order.
// Use WithBaseline(nil) to disable preservation.
//
// CatalogStore snapshots contain the curated fields captured at fetch time.
// After a library or baseline upgrade, offline restore or a 304 can restore older
// metadata over the new baseline. Clear the store or force a network refresh
// with a source constructed against the new baseline to recover.
func NewSource(opts ...Option) *Source {
	s := &Source{
		httpClient: http.DefaultClient,
		url:        DefaultURL,
		providerAliases: map[string]string{
			"fireworks": "fireworks-ai",
			"together":  "togetherai",
			"moonshot":  "moonshotai",
		},
	}
	for _, opt := range opts {
		opt(s)
	}
	if !s.baselineExplicit && s.apiResolver == nil && s.compatResolver == nil {
		s.baseline = models.BuiltinCatalog()
	}
	if s.apiResolver == nil {
		s.apiResolver = func(p string, _ ModelEntry) llm.API { return DefaultProviderAPI(p) }
	}
	if s.compatResolver == nil {
		s.compatResolver = DefaultProviderCompatibility
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
	var rejected []string
	var firstRejectedID string
	var firstCause error
	for modelID, rawModel := range providerEntry.Models {
		if s.filter != nil && !s.filter(provider, rawModel) {
			continue
		}
		curated, known := s.baseline.Model(provider, rawModel.ID)
		var api llm.API
		var compat *llm.ModelCompatibility
		if known {
			api, compat = curated.API, curated.Compatibility
			// Fill absent limits from the baseline before validating the pair.
			limits := Limit{Context: curated.ContextWindow, Output: curated.MaxOutputTokens}
			if rawModel.Limit != nil {
				if rawModel.Limit.Context != 0 {
					limits.Context = rawModel.Limit.Context
				}
				if rawModel.Limit.Output != 0 {
					limits.Output = rawModel.Limit.Output
				}
			}
			rawModel.Limit = &limits
		} else {
			api = s.apiResolver(provider, rawModel)
			compat = s.compatResolver(provider, rawModel, api)
		}
		converted, err := convertModel(provider, rawModel, api, compat, curated.Cost)
		if err != nil {
			rejected = append(rejected, modelID)
			if firstCause == nil || modelID < firstRejectedID {
				firstRejectedID, firstCause = modelID, err
			}
			continue
		}
		if known {
			curated.ContextWindow = converted.ContextWindow
			curated.MaxOutputTokens = converted.MaxOutputTokens
			if converted.Cost != nil {
				curated.Cost = converted.Cost
			}
			converted = curated
		}
		modelsList = append(modelsList, converted)
	}

	if len(modelsList) == 0 && len(rejected) > 0 {
		sort.Strings(rejected)
		const maxReported = 10
		ids := make([]string, min(len(rejected), maxReported))
		for i := range ids {
			ids[i] = truncateDiagnostic(rejected[i], 128)
		}
		summary := fmt.Sprintf("%q", ids)
		if len(rejected) > maxReported {
			summary += fmt.Sprintf(" and %d more", len(rejected)-maxReported)
		}
		return models.CatalogFetchResponse{}, &llm.Error{
			Kind: llm.KindMalformedResponse, Op: "modelsdev.Fetch", Provider: provider,
			Err: fmt.Errorf("no valid models (%d rejected): %s; first cause: %s", len(rejected), summary, truncateDiagnostic(firstCause.Error(), 256)),
		}
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

func truncateDiagnostic(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return strings.ToValidUTF8(value[:limit], "") + "..."
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
