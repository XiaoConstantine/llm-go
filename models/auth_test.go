package models

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestNewCredentialManagerRejectsInvalidConfig(t *testing.T) {
	store, err := NewMemoryCredentialStore(nil)
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	refresh := func(context.Context, StoredCredential) (StoredCredential, error) { return StoredCredential{}, nil }
	tests := []struct {
		name    string
		store   CredentialStore
		configs []CredentialRefreshConfig
		want    string
	}{
		{name: "nil store", want: "store"},
		{name: "empty provider", store: store, configs: []CredentialRefreshConfig{{Refresh: refresh}}, want: "provider"},
		{name: "noncanonical provider", store: store, configs: []CredentialRefreshConfig{{Provider: " openai ", Refresh: refresh}}, want: "whitespace"},
		{name: "negative refresh before", store: store, configs: []CredentialRefreshConfig{{Provider: "openai", RefreshBefore: -1, Refresh: refresh}}, want: "must not be negative"},
		{name: "nil refresher", store: store, configs: []CredentialRefreshConfig{{Provider: "openai"}}, want: "must not be nil"},
		{name: "duplicate provider", store: store, configs: []CredentialRefreshConfig{{Provider: "openai", Refresh: refresh}, {Provider: "openai", Refresh: refresh}}, want: "more than once"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manager, err := NewCredentialManager(test.store, test.configs...)
			if manager != nil {
				t.Fatalf("NewCredentialManager() = %#v, want nil", manager)
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("NewCredentialManager() error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestCredentialManagerResolvesWithoutRefresh(t *testing.T) {
	future := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai":    {Type: CredentialAPIKey, APIKey: "key"},
		"anthropic": {Type: CredentialOAuth, AccessToken: "token", ExpiresAt: future},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	var refreshes atomic.Int32
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider: "anthropic",
		Refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			refreshes.Add(1)
			return StoredCredential{}, errors.New("unexpected refresh")
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	manager.now = func() time.Time { return future.Add(-time.Hour) }

	for _, provider := range []string{"openai", "anthropic"} {
		credential, found, err := manager.Resolve(context.Background(), provider)
		if err != nil || !found {
			t.Fatalf("Resolve(%q) = (%#v, %t, %v)", provider, credential, found, err)
		}
	}
	if refreshes.Load() != 0 {
		t.Fatalf("refresh calls = %d, want 0", refreshes.Load())
	}
	if credential, found, err := manager.Resolve(context.Background(), "missing"); err != nil || found || credential.Type != "" {
		t.Fatalf("Resolve(missing) = (%#v, %t, %v)", credential, found, err)
	}
}

func TestCredentialManagerRefreshesAndPreservesRotatingState(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai-codex": {
			Type:         CredentialOAuth,
			AccessToken:  "old-access",
			RefreshToken: "old-refresh",
			AccountID:    "account",
			ExpiresAt:    now.Add(time.Minute),
			Attributes:   map[string]string{"tenant": "original"},
		},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider:      "openai-codex",
		RefreshBefore: 2 * time.Minute,
		Refresh: func(_ context.Context, current StoredCredential) (StoredCredential, error) {
			if current.AccessToken != "old-access" || current.RefreshToken != "old-refresh" || current.Attributes["tenant"] != "original" {
				return StoredCredential{}, errors.New("refresh received unexpected current credential")
			}
			current.Attributes["tenant"] = "callback mutation"
			return StoredCredential{
				Type:        CredentialOAuth,
				AccessToken: "new-access",
				AccountID:   current.AccountID,
				ExpiresAt:   now.Add(time.Hour),
				Attributes:  map[string]string{"tenant": "refreshed"},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	manager.now = func() time.Time { return now }

	credential, found, err := manager.Resolve(context.Background(), "openai-codex")
	if err != nil || !found {
		t.Fatalf("Resolve() = (%#v, %t, %v)", credential, found, err)
	}
	if credential.AccessToken != "new-access" || credential.RefreshToken != "old-refresh" || credential.Attributes["tenant"] != "refreshed" {
		t.Fatalf("Resolve() credential = %#v", credential)
	}
	credential.Attributes["tenant"] = "caller mutation"
	stored, found, err := store.Read(context.Background(), "openai-codex")
	if err != nil || !found || stored.AccessToken != "new-access" || stored.RefreshToken != "old-refresh" || stored.Attributes["tenant"] != "refreshed" {
		t.Fatalf("stored credential = (%#v, %t, %v)", stored, found, err)
	}
}

func TestCredentialManagerRotatesRefreshToken(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialOAuth, AccessToken: "old-access", RefreshToken: "old-refresh", ExpiresAt: now.Add(-time.Minute)},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider: "openai",
		Refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			return StoredCredential{
				Type:         CredentialOAuth,
				AccessToken:  "new-access",
				RefreshToken: "new-refresh",
				ExpiresAt:    now.Add(time.Hour),
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	manager.now = func() time.Time { return now }
	credential, found, err := manager.Resolve(context.Background(), "openai")
	if err != nil || !found || credential.RefreshToken != "new-refresh" {
		t.Fatalf("Resolve() = (%#v, %t, %v), want rotated refresh token", credential, found, err)
	}
	stored, found, err := store.Read(context.Background(), "openai")
	if err != nil || !found || stored.RefreshToken != "new-refresh" {
		t.Fatalf("Read() = (%#v, %t, %v), want rotated refresh token", stored, found, err)
	}
}

func TestCredentialManagerCoalescesConcurrentRefresh(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialOAuth, RefreshToken: "refresh"},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	const callers = 50
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseRefresh := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseRefresh)
	var refreshes atomic.Int32
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider: "openai",
		Refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			refreshes.Add(1)
			close(entered)
			<-release
			return StoredCredential{Type: CredentialOAuth, AccessToken: "access", RefreshToken: "refresh", ExpiresAt: now.Add(time.Hour)}, nil
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	manager.now = func() time.Time { return now }

	var group sync.WaitGroup
	results := make(chan error, callers)
	for range callers {
		group.Add(1)
		go func() {
			defer group.Done()
			credential, found, err := manager.Resolve(context.Background(), "openai")
			if err == nil && (!found || credential.AccessToken != "access") {
				err = errors.New("resolved credential is missing refreshed access token")
			}
			results <- err
		}()
	}
	waitSignal(t, entered, "credential refresher entry")
	waitCredentialManagerWaiters(t, manager, "openai", callers-1)
	releaseRefresh()
	done := make(chan struct{})
	go func() {
		group.Wait()
		close(done)
	}()
	waitSignal(t, done, "concurrent credential resolution")
	close(results)
	for err := range results {
		if err != nil {
			t.Fatalf("Resolve() error = %v", err)
		}
	}
	if refreshes.Load() != 1 {
		t.Fatalf("refresh calls = %d, want 1", refreshes.Load())
	}
}

func TestCredentialManagerCoalescesConcurrentRefreshFailure(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialOAuth, RefreshToken: "refresh"},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	const callers = 50
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseRefresh := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseRefresh)
	refreshErr := errors.New("refresh failed")
	var refreshes atomic.Int32
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider: "openai",
		Refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			refreshes.Add(1)
			close(entered)
			<-release
			return StoredCredential{}, refreshErr
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	manager.now = func() time.Time { return now }

	var group sync.WaitGroup
	results := make(chan error, callers)
	for range callers {
		group.Add(1)
		go func() {
			defer group.Done()
			_, _, err := manager.Resolve(context.Background(), "openai")
			results <- err
		}()
	}
	waitSignal(t, entered, "failing credential refresher entry")
	waitCredentialManagerWaiters(t, manager, "openai", callers-1)
	releaseRefresh()
	done := make(chan struct{})
	go func() {
		group.Wait()
		close(done)
	}()
	waitSignal(t, done, "concurrent failed credential resolution")
	close(results)
	for err := range results {
		if !errors.Is(err, refreshErr) {
			t.Fatalf("Resolve() error = %v, want refreshErr", err)
		}
	}
	if refreshes.Load() != 1 {
		t.Fatalf("refresh calls = %d, want 1", refreshes.Load())
	}
}

func TestCredentialManagerRefreshFailuresPreserveCredential(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	original := StoredCredential{Type: CredentialOAuth, AccessToken: "old", RefreshToken: "refresh", ExpiresAt: now.Add(-time.Minute)}
	refreshErr := errors.New("provider unavailable")
	tests := []struct {
		name          string
		refresh       CredentialRefresher
		refreshBefore time.Duration
		want          string
	}{
		{name: "provider error", refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			return StoredCredential{}, refreshErr
		}, want: "provider unavailable"},
		{name: "wrong type", refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			return StoredCredential{Type: CredentialAPIKey, APIKey: "key"}, nil
		}, want: "credential type"},
		{name: "invalid credential", refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			return StoredCredential{Type: CredentialOAuth, AccessToken: " "}, nil
		}, want: "invalid credential"},
		{name: "missing access token", refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			return StoredCredential{Type: CredentialOAuth, RefreshToken: "refresh"}, nil
		}, want: "without an access token"},
		{name: "expired credential", refresh: func(context.Context, StoredCredential) (StoredCredential, error) { return original, nil }, want: "still requires refresh"},
		{name: "credential inside refresh window", refreshBefore: 2 * time.Minute, refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			return StoredCredential{Type: CredentialOAuth, AccessToken: "new", RefreshToken: "refresh", ExpiresAt: now.Add(time.Minute)}, nil
		}, want: "still requires refresh"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store, err := NewMemoryCredentialStore(map[string]StoredCredential{"openai": original})
			if err != nil {
				t.Fatalf("NewMemoryCredentialStore() error = %v", err)
			}
			manager, err := NewCredentialManager(store, CredentialRefreshConfig{
				Provider:      "openai",
				RefreshBefore: test.refreshBefore,
				Refresh:       test.refresh,
			})
			if err != nil {
				t.Fatalf("NewCredentialManager() error = %v", err)
			}
			manager.now = func() time.Time { return now }
			credential, found, err := manager.Resolve(context.Background(), "openai")
			if found || credential.Type != "" || err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Resolve() = (%#v, %t, %v), want error containing %q", credential, found, err, test.want)
			}
			if test.name == "provider error" && !errors.Is(err, refreshErr) {
				t.Fatalf("errors.Is(Resolve() error, refreshErr) = false: %v", err)
			}
			stored, found, readErr := store.Read(context.Background(), "openai")
			if readErr != nil || !found || stored.AccessToken != original.AccessToken || stored.ExpiresAt != original.ExpiresAt {
				t.Fatalf("stored credential after failure = (%#v, %t, %v)", stored, found, readErr)
			}
		})
	}
}

func TestCredentialManagerRequiresRefresherForExpiredOAuth(t *testing.T) {
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialOAuth, RefreshToken: "refresh"},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	manager, err := NewCredentialManager(store)
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	credential, found, err := manager.Resolve(context.Background(), "openai")
	if found || credential.Type != "" || err == nil || !strings.Contains(err.Error(), "no refresher") {
		t.Fatalf("Resolve() = (%#v, %t, %v)", credential, found, err)
	}
}

func TestCredentialManagerHonorsCancellation(t *testing.T) {
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialOAuth, RefreshToken: "refresh"},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	entered := make(chan struct{})
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider: "openai",
		Refresh: func(ctx context.Context, _ StoredCredential) (StoredCredential, error) {
			close(entered)
			<-ctx.Done()
			return StoredCredential{}, ctx.Err()
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, _, err := manager.Resolve(ctx, "openai")
		done <- err
	}()
	waitSignal(t, entered, "credential refresher entry")
	cancel()
	if err := waitError(t, done, "canceled credential resolution"); !errors.Is(err, context.Canceled) {
		t.Fatalf("Resolve() error = %v, want context canceled", err)
	}
	stored, found, err := store.Read(context.Background(), "openai")
	if err != nil || !found || stored.AccessToken != "" || stored.RefreshToken != "refresh" {
		t.Fatalf("stored credential after cancellation = (%#v, %t, %v)", stored, found, err)
	}
}

func TestCredentialManagerLeaderCancellationDoesNotCancelWaiter(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialOAuth, RefreshToken: "refresh"},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseRefresh := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseRefresh)
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider: "openai",
		Refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			close(entered)
			<-release
			return StoredCredential{Type: CredentialOAuth, AccessToken: "access", RefreshToken: "refresh", ExpiresAt: now.Add(time.Hour)}, nil
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	manager.now = func() time.Time { return now }

	leaderContext, cancelLeader := context.WithCancel(context.Background())
	leaderDone := make(chan error, 1)
	go func() {
		_, _, err := manager.Resolve(leaderContext, "openai")
		leaderDone <- err
	}()
	waitSignal(t, entered, "leader credential refresher entry")
	waiterDone := make(chan error, 1)
	go func() {
		credential, found, err := manager.Resolve(context.Background(), "openai")
		if err == nil && (!found || credential.AccessToken != "access") {
			err = errors.New("waiter did not receive refreshed credential")
		}
		waiterDone <- err
	}()
	waitCredentialManagerWaiters(t, manager, "openai", 1)
	cancelLeader()
	if err := waitError(t, leaderDone, "canceled leader completion"); !errors.Is(err, context.Canceled) {
		t.Fatalf("leader Resolve() error = %v, want context canceled", err)
	}
	releaseRefresh()
	if err := waitError(t, waiterDone, "live waiter completion"); err != nil {
		t.Fatalf("waiter Resolve() error = %v", err)
	}
}

func TestCredentialManagerNewCallerReplacesCanceledResolution(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialOAuth, RefreshToken: "refresh"},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	firstEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	var releaseOnce sync.Once
	releaseRefresh := func() { releaseOnce.Do(func() { close(releaseFirst) }) }
	t.Cleanup(releaseRefresh)
	var refreshes atomic.Int32
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider: "openai",
		Refresh: func(ctx context.Context, _ StoredCredential) (StoredCredential, error) {
			if refreshes.Add(1) == 1 {
				close(firstEntered)
				<-releaseFirst
				return StoredCredential{}, ctx.Err()
			}
			return StoredCredential{Type: CredentialOAuth, AccessToken: "access", RefreshToken: "refresh", ExpiresAt: now.Add(time.Hour)}, nil
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	manager.now = func() time.Time { return now }

	firstContext, cancelFirst := context.WithCancel(context.Background())
	firstDone := make(chan error, 1)
	go func() {
		_, _, err := manager.Resolve(firstContext, "openai")
		firstDone <- err
	}()
	waitSignal(t, firstEntered, "first credential refresher entry")
	cancelFirst()
	if err := waitError(t, firstDone, "first canceled resolution"); !errors.Is(err, context.Canceled) {
		t.Fatalf("first Resolve() error = %v, want context canceled", err)
	}

	secondDone := make(chan error, 1)
	go func() {
		credential, found, err := manager.Resolve(context.Background(), "openai")
		if err == nil && (!found || credential.AccessToken != "access") {
			err = errors.New("replacement resolution did not return access token")
		}
		secondDone <- err
	}()
	waitCredentialManagerParticipants(t, manager, "openai", 1)
	releaseRefresh()
	if err := waitError(t, secondDone, "replacement credential resolution"); err != nil {
		t.Fatalf("second Resolve() error = %v", err)
	}
	if refreshes.Load() != 2 {
		t.Fatalf("refresh calls = %d, want 2", refreshes.Load())
	}
}

func TestCredentialManagerSanitizesAndValidatesStoreResults(t *testing.T) {
	secret := StoredCredential{Type: CredentialAPIKey, APIKey: "secret"}
	storeErr := errors.New("store unavailable")
	tests := []struct {
		name  string
		store CredentialStore
		want  string
	}{
		{
			name: "miss with stale value",
			store: credentialStoreStub{read: func(context.Context, string) (StoredCredential, bool, error) {
				return secret, false, nil
			}},
		},
		{
			name: "error with stale value",
			store: credentialStoreStub{read: func(context.Context, string) (StoredCredential, bool, error) {
				return secret, true, storeErr
			}},
			want: "store unavailable",
		},
		{
			name: "invalid found credential",
			store: credentialStoreStub{read: func(context.Context, string) (StoredCredential, bool, error) {
				return StoredCredential{Type: CredentialAPIKey}, true, nil
			}},
			want: "stored credential is invalid",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manager, err := NewCredentialManager(test.store)
			if err != nil {
				t.Fatalf("NewCredentialManager() error = %v", err)
			}
			credential, found, err := manager.Resolve(context.Background(), "openai")
			if !isZeroCredential(credential) || found {
				t.Fatalf("Resolve() = (%#v, %t, %v), want zero, false", credential, found, err)
			}
			if test.want == "" && err != nil {
				t.Fatalf("Resolve() error = %v, want nil", err)
			}
			if test.want != "" && (err == nil || !strings.Contains(err.Error(), test.want)) {
				t.Fatalf("Resolve() error = %v, want substring %q", err, test.want)
			}
			if test.name == "error with stale value" && !errors.Is(err, storeErr) {
				t.Fatalf("errors.Is(Resolve() error, storeErr) = false: %v", err)
			}
		})
	}
}

func TestCredentialManagerValidatesCredentialReplacedBeforeRefresh(t *testing.T) {
	now := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	valid := StoredCredential{Type: CredentialOAuth, AccessToken: "expired", RefreshToken: "refresh", ExpiresAt: now.Add(-time.Minute)}
	store := credentialStoreStub{
		read: func(context.Context, string) (StoredCredential, bool, error) { return valid, true, nil },
		modify: func(ctx context.Context, _ string, update func(StoredCredential, bool) (*StoredCredential, error)) error {
			_, err := update(StoredCredential{Type: CredentialAPIKey}, true)
			return err
		},
	}
	var refresherCalled atomic.Bool
	manager, err := NewCredentialManager(store, CredentialRefreshConfig{
		Provider: "openai",
		Refresh: func(context.Context, StoredCredential) (StoredCredential, error) {
			refresherCalled.Store(true)
			return StoredCredential{}, errors.New("refresher called with invalid replacement credential")
		},
	})
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	manager.now = func() time.Time { return now }
	credential, found, err := manager.Resolve(context.Background(), "openai")
	if !isZeroCredential(credential) || found || err == nil || !strings.Contains(err.Error(), "stored credential is invalid") {
		t.Fatalf("Resolve() = (%#v, %t, %v)", credential, found, err)
	}
	if refresherCalled.Load() {
		t.Fatal("refresher called with invalid replacement credential")
	}
}

func TestCredentialManagerPanicDoesNotWedgeProvider(t *testing.T) {
	var reads atomic.Int32
	store := credentialStoreStub{read: func(context.Context, string) (StoredCredential, bool, error) {
		if reads.Add(1) == 1 {
			panic("store panic")
		}
		return StoredCredential{Type: CredentialAPIKey, APIKey: "key"}, true, nil
	}}
	manager, err := NewCredentialManager(store)
	if err != nil {
		t.Fatalf("NewCredentialManager() error = %v", err)
	}
	var panicValue any
	func() {
		defer func() { panicValue = recover() }()
		_, _, _ = manager.Resolve(context.Background(), "openai")
	}()
	if panicValue != "store panic" {
		t.Fatalf("Resolve() panic = %#v, want %q", panicValue, "store panic")
	}
	credential, found, err := manager.Resolve(context.Background(), "openai")
	if err != nil || !found || credential.APIKey != "key" {
		t.Fatalf("Resolve() after panic = (%#v, %t, %v)", credential, found, err)
	}
}

func TestNilCredentialManagerReturnsError(t *testing.T) {
	var manager *CredentialManager
	if _, _, err := manager.Resolve(context.Background(), "openai"); err == nil {
		t.Fatal("nil manager Resolve() error = nil")
	}
}

type credentialStoreStub struct {
	read   func(context.Context, string) (StoredCredential, bool, error)
	modify func(context.Context, string, func(StoredCredential, bool) (*StoredCredential, error)) error
}

func (s credentialStoreStub) Read(ctx context.Context, provider string) (StoredCredential, bool, error) {
	return s.read(ctx, provider)
}

func (credentialStoreStub) List(context.Context) ([]CredentialMetadata, error) {
	return nil, errors.New("unexpected List call")
}

func (s credentialStoreStub) Modify(ctx context.Context, provider string, update func(StoredCredential, bool) (*StoredCredential, error)) error {
	if s.modify == nil {
		return errors.New("unexpected Modify call")
	}
	return s.modify(ctx, provider, update)
}

func (credentialStoreStub) Delete(context.Context, string) error {
	return errors.New("unexpected Delete call")
}

func waitCredentialManagerWaiters(t *testing.T, manager *CredentialManager, provider string, want int) {
	t.Helper()
	waitCredentialManagerParticipants(t, manager, provider, want+1)
}

func waitCredentialManagerParticipants(t *testing.T, manager *CredentialManager, provider string, want int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		manager.mu.Lock()
		inflight := manager.inflight[provider]
		got := 0
		if inflight != nil {
			got = inflight.participants
		}
		manager.mu.Unlock()
		if inflight != nil && got == want {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("credential manager participants = %d, want %d", got, want)
		}
		time.Sleep(time.Millisecond)
	}
}

func isZeroCredential(credential StoredCredential) bool {
	return credential.Type == "" && credential.APIKey == "" && credential.AccessToken == "" &&
		credential.RefreshToken == "" && credential.AccountID == "" && credential.ExpiresAt.IsZero() &&
		credential.Attributes == nil
}
