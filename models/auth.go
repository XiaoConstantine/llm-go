package models

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// CredentialRefresher exchanges or renews an OAuth credential. It must honor
// ctx and must not call any method on a CredentialManager or CredentialStore.
// The returned credential must have type CredentialOAuth and a usable access
// token. Once invoked by a manager, ctx is independent of Resolve callers and
// has the configured RefreshTimeout. A successful result is persisted even if
// callers have canceled or the refresh deadline has just elapsed.
type CredentialRefresher func(ctx context.Context, current StoredCredential) (StoredCredential, error)

// CredentialRefreshConfig configures OAuth refresh for one provider.
type CredentialRefreshConfig struct {
	Provider string
	// RefreshBefore is the expiry skew. Zero refreshes only at expiry (or when
	// the access token is absent/rejected); no default skew is added. Set a
	// positive value to allow for clock skew and request transit time.
	RefreshBefore time.Duration
	// RefreshTimeout bounds each Refresh callback, not the subsequent store
	// commit. Zero defaults to 30 seconds. Negative values are invalid.
	RefreshTimeout time.Duration
	Refresh        CredentialRefresher
}

// CredentialManager resolves credentials and refreshes OAuth credentials under
// the store's serialized provider update. It is safe for concurrent use when
// its store and refreshers are safe for concurrent use. A started refresh owns
// its lifetime until the callback and store commit finish; it is not canceled
// when its last caller leaves. Refresh callbacks must honor RefreshTimeout.
// Store commits remain independent of caller cancellation so a rotated refresh
// token is not discarded. Storage failures still require application recovery.
type CredentialManager struct {
	store      CredentialStore
	refreshers map[string]credentialRefreshConfig
	now        func() time.Time

	mu       sync.Mutex
	inflight map[string]*credentialResolution
	rejected map[string]rejectedRefresh
}

type credentialResolution struct {
	provider     string
	done         chan struct{}
	cancel       context.CancelFunc
	credential   StoredCredential
	found        bool
	err          error
	panicValue   any
	participants int
	refreshing   bool
}

type credentialRefreshConfig struct {
	before  time.Duration
	timeout time.Duration
	refresh CredentialRefresher
}

type rejectedRefresh struct {
	token string
	err   error
}

// NewCredentialManager constructs a manager. Refresh configuration is optional;
// providers without one can resolve API keys and unexpired OAuth credentials.
// Provider IDs must be unique and canonical. RefreshBefore and RefreshTimeout
// must not be negative.
func NewCredentialManager(store CredentialStore, configs ...CredentialRefreshConfig) (*CredentialManager, error) {
	if store == nil {
		return nil, fmt.Errorf("credential store must not be nil")
	}
	manager := &CredentialManager{
		store:      store,
		refreshers: make(map[string]credentialRefreshConfig, len(configs)),
		now:        time.Now,
		inflight:   make(map[string]*credentialResolution),
		rejected:   make(map[string]rejectedRefresh),
	}
	for index, config := range configs {
		provider, err := normalizeProviderID(config.Provider)
		if err != nil {
			return nil, fmt.Errorf("refreshers[%d]: %w", index, err)
		}
		if provider != config.Provider {
			return nil, fmt.Errorf("refreshers[%d]: provider %q contains surrounding whitespace", index, config.Provider)
		}
		if config.RefreshBefore < 0 {
			return nil, fmt.Errorf("refreshers[%d]: refresh before must not be negative", index)
		}
		if config.RefreshTimeout < 0 {
			return nil, fmt.Errorf("refreshers[%d]: refresh timeout must not be negative", index)
		}
		if config.RefreshTimeout == 0 {
			config.RefreshTimeout = 30 * time.Second
		}
		if config.Refresh == nil {
			return nil, fmt.Errorf("refreshers[%d]: refresh callback must not be nil", index)
		}
		if _, exists := manager.refreshers[provider]; exists {
			return nil, fmt.Errorf("refreshers[%d]: provider %q is configured more than once", index, provider)
		}
		manager.refreshers[provider] = credentialRefreshConfig{before: config.RefreshBefore, timeout: config.RefreshTimeout, refresh: config.Refresh}
	}
	return manager, nil
}

// Resolve returns the current credential for provider. It refreshes an OAuth
// credential when its access token is empty or its nonzero expiry is within the
// configured RefreshBefore interval. Concurrent Resolve calls through one
// manager are coalesced, including refresh failures. A refresh failure leaves
// the stored credential unchanged. A caller can stop waiting by canceling ctx;
// once Refresh has started, its successful result is still committed.
//
// Built-in OAuth refreshers' invalid_grant responses (HTTP 400 or 401) are
// remembered for the current refresh token, avoiding repeated endpoint calls.
// Replace that token in the store to recover. This cache is local to the
// manager, keeps at most one rejection per provider, and is cleared on a
// successful refresh. Other errors are not cached beyond concurrent callers.
func (m *CredentialManager) Resolve(ctx context.Context, provider string) (StoredCredential, bool, error) {
	return m.resolveRequest(ctx, provider, "")
}

// resolveRejectedOAuth refreshes provider only while its current OAuth access
// token still matches rejectedAccessToken. Concurrent calls for the same
// rejected token are coalesced; a token already rotated by another caller is
// returned without another refresh.
func (m *CredentialManager) resolveRejectedOAuth(ctx context.Context, provider, rejectedAccessToken string) (StoredCredential, bool, error) {
	if strings.TrimSpace(rejectedAccessToken) == "" {
		return m.resolveRequest(ctx, provider, "")
	}
	return m.resolveRequest(ctx, provider, rejectedAccessToken)
}

func (m *CredentialManager) resolveRequest(ctx context.Context, provider, rejectedAccessToken string) (StoredCredential, bool, error) {
	provider, err := normalizeProviderID(provider)
	if err != nil {
		return StoredCredential{}, false, err
	}
	if m == nil || m.store == nil || m.now == nil || m.inflight == nil {
		return StoredCredential{}, false, fmt.Errorf("credential manager is nil or uninitialized")
	}
	if ctx == nil {
		return StoredCredential{}, false, fmt.Errorf("credential manager context must not be nil")
	}
	if err := ctx.Err(); err != nil {
		return StoredCredential{}, false, err
	}

	resolutionKey := provider
	if rejectedAccessToken != "" {
		resolutionKey += "\x00" + rejectedAccessToken
	}
	m.mu.Lock()
	inflight := m.inflight[resolutionKey]
	if inflight == nil {
		operationContext, cancel := context.WithCancel(context.WithoutCancel(ctx))
		inflight = &credentialResolution{
			provider:     resolutionKey,
			done:         make(chan struct{}),
			cancel:       cancel,
			participants: 1,
		}
		m.inflight[resolutionKey] = inflight
		go m.runResolution(operationContext, resolutionKey, provider, rejectedAccessToken, inflight)
	} else {
		inflight.participants++
	}
	m.mu.Unlock()
	return m.awaitResolution(ctx, inflight)
}

func (m *CredentialManager) runResolution(ctx context.Context, resolutionKey, provider, rejectedAccessToken string, inflight *credentialResolution) {
	defer func() {
		if value := recover(); value != nil {
			inflight.panicValue = value
		}
		m.mu.Lock()
		if m.inflight[resolutionKey] == inflight {
			delete(m.inflight, resolutionKey)
		}
		inflight.cancel()
		close(inflight.done)
		m.mu.Unlock()
	}()
	credential, found, err := m.resolve(ctx, provider, rejectedAccessToken, inflight)
	inflight.credential = cloneStoredCredential(credential)
	inflight.found = found
	inflight.err = err
}

func (m *CredentialManager) awaitResolution(ctx context.Context, inflight *credentialResolution) (StoredCredential, bool, error) {
	select {
	case <-inflight.done:
		if inflight.panicValue != nil {
			panic(inflight.panicValue)
		}
		return cloneStoredCredential(inflight.credential), inflight.found, inflight.err
	case <-ctx.Done():
		m.mu.Lock()
		inflight.participants--
		if inflight.participants == 0 && !inflight.refreshing {
			inflight.cancel()
			if m.inflight[inflight.provider] == inflight {
				delete(m.inflight, inflight.provider)
			}
		}
		m.mu.Unlock()
		return StoredCredential{}, false, ctx.Err()
	}
}

func (m *CredentialManager) resolve(ctx context.Context, provider, rejectedAccessToken string, inflight *credentialResolution) (StoredCredential, bool, error) {
	credential, found, err := m.store.Read(ctx, provider)
	if err != nil {
		return StoredCredential{}, false, err
	}
	if !found {
		return StoredCredential{}, false, nil
	}
	if err := credential.Validate(); err != nil {
		return StoredCredential{}, false, fmt.Errorf("resolve credential for provider %q: stored credential is invalid: %w", provider, err)
	}
	config, configured := m.refreshers[provider]
	if rejectedAccessToken != "" && (credential.Type != CredentialOAuth || credential.AccessToken != rejectedAccessToken) {
		return credential, true, nil
	}
	if credential.Type != CredentialOAuth || !credentialNeedsRefresh(credential, config, m.now()) && rejectedAccessToken == "" {
		return credential, true, nil
	}
	if !configured {
		return StoredCredential{}, false, fmt.Errorf("refresh OAuth credential for provider %q: no refresher is configured", provider)
	}

	var resolved StoredCredential
	err = m.store.Modify(ctx, provider, func(current StoredCredential, found bool) (*StoredCredential, error) {
		if !found {
			return nil, fmt.Errorf("credential was deleted while resolving")
		}
		if err := current.Validate(); err != nil {
			return nil, fmt.Errorf("stored credential is invalid: %w", err)
		}
		if rejectedAccessToken != "" && (current.Type != CredentialOAuth || current.AccessToken != rejectedAccessToken) {
			resolved = cloneStoredCredential(current)
			return &current, nil
		}
		if current.Type != CredentialOAuth || !credentialNeedsRefresh(current, config, m.now()) && rejectedAccessToken == "" {
			resolved = cloneStoredCredential(current)
			return &current, nil
		}
		m.mu.Lock()
		if err := ctx.Err(); err != nil {
			m.mu.Unlock()
			return nil, err
		}
		rejected := m.rejected[provider]
		if rejected.err != nil && rejected.token == current.RefreshToken {
			m.mu.Unlock()
			return nil, fmt.Errorf("refresh OAuth credential: %w", rejected.err)
		}
		// Serialize this transition with the last waiter's departure. Until
		// now store access is cancelable; after this point rotation can occur.
		inflight.refreshing = true
		m.mu.Unlock()
		refreshContext, cancel := context.WithTimeout(ctx, config.timeout)
		defer cancel()
		refreshed, err := config.refresh(refreshContext, cloneStoredCredential(current))
		cancel()
		if err != nil {
			var endpointError *oauthTokenError
			if errors.As(err, &endpointError) && endpointError.code == "invalid_grant" &&
				(endpointError.status == http.StatusBadRequest || endpointError.status == http.StatusUnauthorized) {
				m.mu.Lock()
				m.rejected[provider] = rejectedRefresh{token: current.RefreshToken, err: err}
				m.mu.Unlock()
			}
			return nil, fmt.Errorf("refresh OAuth credential: %w", err)
		}
		if refreshed.Type != CredentialOAuth {
			return nil, fmt.Errorf("refresh returned credential type %q, want %q", refreshed.Type, CredentialOAuth)
		}
		if refreshed.RefreshToken == "" {
			refreshed.RefreshToken = current.RefreshToken
		}
		if err := refreshed.Validate(); err != nil {
			return nil, fmt.Errorf("refresh returned invalid credential: %w", err)
		}
		if strings.TrimSpace(refreshed.AccessToken) == "" {
			return nil, fmt.Errorf("refresh returned a credential without an access token")
		}
		if credentialNeedsRefresh(refreshed, config, m.now()) {
			return nil, fmt.Errorf("refresh returned a credential that still requires refresh")
		}
		m.mu.Lock()
		delete(m.rejected, provider)
		m.mu.Unlock()
		resolved = cloneStoredCredential(refreshed)
		return &refreshed, nil
	})
	if err != nil {
		return StoredCredential{}, false, fmt.Errorf("resolve credential for provider %q: %w", provider, err)
	}
	return resolved, true, nil
}

func credentialNeedsRefresh(credential StoredCredential, config credentialRefreshConfig, now time.Time) bool {
	if strings.TrimSpace(credential.AccessToken) == "" {
		return true
	}
	return !credential.ExpiresAt.IsZero() && !credential.ExpiresAt.After(now.Add(config.before))
}
