package models

import (
	"context"
	"errors"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestStoredCredentialValidate(t *testing.T) {
	tests := []struct {
		name       string
		credential StoredCredential
		want       string
	}{
		{name: "API key", credential: StoredCredential{Type: CredentialAPIKey, APIKey: "key"}},
		{name: "OAuth access token", credential: StoredCredential{Type: CredentialOAuth, AccessToken: "token"}},
		{name: "OAuth refresh token", credential: StoredCredential{Type: CredentialOAuth, RefreshToken: "refresh"}},
		{name: "unknown type", credential: StoredCredential{Type: "future"}, want: "type"},
		{name: "empty API key", credential: StoredCredential{Type: CredentialAPIKey}, want: "must not be empty"},
		{name: "API key with OAuth data", credential: StoredCredential{Type: CredentialAPIKey, APIKey: "key", AccountID: "account"}, want: "OAuth fields"},
		{name: "OAuth with API key", credential: StoredCredential{Type: CredentialOAuth, APIKey: "key", AccessToken: "token"}, want: "must not contain"},
		{name: "OAuth without token", credential: StoredCredential{Type: CredentialOAuth}, want: "is required"},
		{name: "OAuth whitespace access token", credential: StoredCredential{Type: CredentialOAuth, AccessToken: " ", RefreshToken: "refresh"}, want: "access token"},
		{name: "OAuth whitespace refresh token", credential: StoredCredential{Type: CredentialOAuth, AccessToken: "access", RefreshToken: "\t"}, want: "refresh token"},
		{name: "empty attribute", credential: StoredCredential{Type: CredentialAPIKey, APIKey: "key", Attributes: map[string]string{" ": "value"}}, want: "attribute name"},
		{name: "invalid secret UTF-8", credential: StoredCredential{Type: CredentialAPIKey, APIKey: string([]byte{0xff})}, want: "valid UTF-8"},
		{name: "invalid attribute UTF-8", credential: StoredCredential{Type: CredentialAPIKey, APIKey: "key", Attributes: map[string]string{"key": string([]byte{0xff})}}, want: "valid UTF-8"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.credential.Validate()
			if test.want == "" {
				if err != nil {
					t.Fatalf("Validate() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Validate() error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestMemoryCredentialStoreLifecycleAndOwnership(t *testing.T) {
	attributes := map[string]string{"tenant": "original"}
	initial := map[string]StoredCredential{
		"openai": {Type: CredentialAPIKey, APIKey: "key", Attributes: attributes},
	}
	store, err := NewMemoryCredentialStore(initial)
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	attributes["tenant"] = "mutated"
	initial["openai"] = StoredCredential{Type: CredentialAPIKey, APIKey: "changed"}

	credential, found, err := store.Read(context.Background(), " openai ")
	if err != nil || !found {
		t.Fatalf("Read() = (%#v, %t, %v)", credential, found, err)
	}
	if credential.APIKey != "key" || credential.Attributes["tenant"] != "original" {
		t.Fatalf("Read() credential = %#v", credential)
	}
	credential.Attributes["tenant"] = "caller mutation"

	err = store.Modify(context.Background(), "openai", func(current StoredCredential, found bool) (*StoredCredential, error) {
		if !found || current.Attributes["tenant"] != "original" {
			t.Fatalf("Modify() current = (%#v, %t)", current, found)
		}
		current.APIKey = "rotated"
		current.Attributes["tenant"] = "updated"
		return &current, nil
	})
	if err != nil {
		t.Fatalf("Modify() error = %v", err)
	}

	credential, found, err = store.Read(context.Background(), "openai")
	if err != nil || !found || credential.APIKey != "rotated" || credential.Attributes["tenant"] != "updated" {
		t.Fatalf("Read() after Modify = (%#v, %t, %v)", credential, found, err)
	}
	metadata, err := store.List(context.Background())
	if err != nil || !reflect.DeepEqual(metadata, []CredentialMetadata{{Provider: "openai", Type: CredentialAPIKey}}) {
		t.Fatalf("List() = (%#v, %v)", metadata, err)
	}
	if err := store.Delete(context.Background(), "openai"); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}
	if credential, found, err := store.Read(context.Background(), "openai"); err != nil || found || credential.Type != "" {
		t.Fatalf("Read() after Delete = (%#v, %t, %v)", credential, found, err)
	}
}

func TestMemoryCredentialStoreModifyFailureIsAtomic(t *testing.T) {
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialAPIKey, APIKey: "original", Attributes: map[string]string{"state": "original"}},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	updateErr := errors.New("update failed")
	err = store.Modify(context.Background(), "openai", func(current StoredCredential, found bool) (*StoredCredential, error) {
		current.APIKey = "changed"
		current.Attributes["state"] = "changed"
		return &current, updateErr
	})
	if !errors.Is(err, updateErr) {
		t.Fatalf("Modify() error = %v, want %v", err, updateErr)
	}
	credential, found, err := store.Read(context.Background(), "openai")
	if err != nil || !found || credential.APIKey != "original" || credential.Attributes["state"] != "original" {
		t.Fatalf("Read() after failed Modify = (%#v, %t, %v)", credential, found, err)
	}

	var returned *StoredCredential
	err = store.Modify(context.Background(), "openai", func(current StoredCredential, _ bool) (*StoredCredential, error) {
		current.APIKey = "committed"
		current.Attributes["state"] = "committed"
		returned = &current
		return returned, nil
	})
	if err != nil {
		t.Fatalf("Modify(success) error = %v", err)
	}
	returned.APIKey = "mutated after return"
	returned.Attributes["state"] = "mutated after return"
	credential, found, err = store.Read(context.Background(), "openai")
	if err != nil || !found || credential.APIKey != "committed" || credential.Attributes["state"] != "committed" {
		t.Fatalf("Read() after returned mutation = (%#v, %t, %v)", credential, found, err)
	}

	err = store.Modify(context.Background(), "openai", func(StoredCredential, bool) (*StoredCredential, error) {
		invalid := StoredCredential{Type: CredentialAPIKey}
		return &invalid, nil
	})
	if err == nil || !strings.Contains(err.Error(), "API key") {
		t.Fatalf("Modify(invalid) error = %v", err)
	}
	credential, found, err = store.Read(context.Background(), "openai")
	if err != nil || !found || credential.APIKey != "committed" || credential.Attributes["state"] != "committed" {
		t.Fatalf("Read() after invalid Modify = (%#v, %t, %v)", credential, found, err)
	}
}

func TestMemoryCredentialStoreSerializesConcurrentUpdates(t *testing.T) {
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialAPIKey, APIKey: "key", Attributes: map[string]string{"updates": "0"}},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	const updates = 100
	var group sync.WaitGroup
	errors := make(chan error, updates)
	for range updates {
		group.Go(func() {
			errors <- store.Modify(context.Background(), "openai", func(current StoredCredential, found bool) (*StoredCredential, error) {
				value, err := strconv.Atoi(current.Attributes["updates"])
				if err != nil {
					return nil, err
				}
				current.Attributes["updates"] = strconv.Itoa(value + 1)
				return &current, nil
			})
		})
	}
	updatesDone := make(chan struct{})
	go func() {
		group.Wait()
		close(updatesDone)
	}()
	waitSignal(t, updatesDone, "concurrent credential updates")
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatalf("Modify() error = %v", err)
		}
	}
	credential, found, err := store.Read(context.Background(), "openai")
	if err != nil || !found || credential.Attributes["updates"] != strconv.Itoa(updates) {
		t.Fatalf("Read() after concurrent updates = (%#v, %t, %v)", credential, found, err)
	}
}

func TestMemoryCredentialStoreListIsSortedAndSecretFree(t *testing.T) {
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"zeta":  {Type: CredentialOAuth, AccessToken: "secret"},
		"alpha": {Type: CredentialAPIKey, APIKey: "also-secret"},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	metadata, err := store.List(context.Background())
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	want := []CredentialMetadata{
		{Provider: "alpha", Type: CredentialAPIKey},
		{Provider: "zeta", Type: CredentialOAuth},
	}
	if !reflect.DeepEqual(metadata, want) {
		t.Fatalf("List() = %#v, want %#v", metadata, want)
	}
}

func TestMemoryCredentialStoreDoesNotCommitAfterCancellation(t *testing.T) {
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai": {Type: CredentialAPIKey, APIKey: "original"},
	})
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	err = store.Modify(ctx, "openai", func(current StoredCredential, _ bool) (*StoredCredential, error) {
		current.APIKey = "changed"
		cancel()
		return &current, nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Modify() error = %v, want context canceled", err)
	}
	credential, found, err := store.Read(context.Background(), "openai")
	if err != nil || !found || credential.APIKey != "original" {
		t.Fatalf("Read() after canceled Modify = (%#v, %t, %v)", credential, found, err)
	}
}

func TestMemoryCredentialStoreUpdatesDifferentProvidersIndependently(t *testing.T) {
	store, err := NewMemoryCredentialStore(nil)
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseAll := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseAll)
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- store.Modify(context.Background(), "openai", func(StoredCredential, bool) (*StoredCredential, error) {
			close(entered)
			<-release
			credential := StoredCredential{Type: CredentialAPIKey, APIKey: "openai-key"}
			return &credential, nil
		})
	}()
	waitSignal(t, entered, "openai callback entry")

	secondDone := make(chan error, 1)
	go func() {
		secondDone <- store.Modify(context.Background(), "anthropic", func(StoredCredential, bool) (*StoredCredential, error) {
			credential := StoredCredential{Type: CredentialAPIKey, APIKey: "anthropic-key"}
			return &credential, nil
		})
	}()
	if err := waitError(t, secondDone, "anthropic update completion"); err != nil {
		t.Fatalf("Modify(anthropic) error = %v", err)
	}
	if metadata, err := store.List(context.Background()); err != nil || len(metadata) != 1 || metadata[0].Provider != "anthropic" {
		t.Fatalf("List() during openai callback = (%#v, %v)", metadata, err)
	}
	releaseAll()
	if err := waitError(t, firstDone, "openai update completion"); err != nil {
		t.Fatalf("Modify(openai) error = %v", err)
	}
}

func TestMemoryCredentialStoreHonorsCancellationWhileWaiting(t *testing.T) {
	store, err := NewMemoryCredentialStore(nil)
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseAll := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseAll)
	done := make(chan error, 1)
	go func() {
		done <- store.Modify(context.Background(), "openai", func(StoredCredential, bool) (*StoredCredential, error) {
			close(entered)
			<-release
			credential := StoredCredential{Type: CredentialAPIKey, APIKey: "key"}
			return &credential, nil
		})
	}()
	waitSignal(t, entered, "blocking callback entry")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	waitingDone := make(chan error, 1)
	go func() {
		waitingDone <- store.Modify(ctx, "openai", func(StoredCredential, bool) (*StoredCredential, error) {
			t.Error("waiting Modify callback was called")
			return nil, nil
		})
	}()
	if err := waitError(t, waitingDone, "canceled update completion"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Modify() error = %v, want deadline exceeded", err)
	}
	releaseAll()
	if err := waitError(t, done, "blocking update completion"); err != nil {
		t.Fatalf("blocking Modify() error = %v", err)
	}

	cancelled, cancelNow := context.WithCancel(context.Background())
	cancelNow()
	if _, err := store.List(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("List(cancelled) error = %v, want context canceled", err)
	}
}

func TestMemoryCredentialStoreDoesNotRunCallbackWhenCancellationRacesRelease(t *testing.T) {
	store, err := NewMemoryCredentialStore(nil)
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	entered := make(chan struct{})
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseAll := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseAll)
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- store.Modify(context.Background(), "openai", func(StoredCredential, bool) (*StoredCredential, error) {
			close(entered)
			<-release
			credential := StoredCredential{Type: CredentialAPIKey, APIKey: "first"}
			return &credential, nil
		})
	}()
	waitSignal(t, entered, "first callback entry")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var callbackCalled atomic.Bool
	secondDone := make(chan error, 1)
	go func() {
		secondDone <- store.Modify(ctx, "openai", func(StoredCredential, bool) (*StoredCredential, error) {
			callbackCalled.Store(true)
			credential := StoredCredential{Type: CredentialAPIKey, APIKey: "second"}
			return &credential, nil
		})
	}()
	deadline := time.Now().Add(time.Second)
	for {
		if err := store.lockState(context.Background()); err != nil {
			t.Fatalf("lockState() error = %v", err)
		}
		refs := store.gates["openai"].refs
		store.unlockState()
		if refs == 2 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("second update did not begin waiting")
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	releaseAll()
	if err := waitError(t, secondDone, "second update completion"); !errors.Is(err, context.Canceled) {
		t.Fatalf("second Modify() error = %v, want context canceled", err)
	}
	if callbackCalled.Load() {
		t.Fatal("canceled Modify callback was called")
	}
	if err := waitError(t, firstDone, "first update completion"); err != nil {
		t.Fatalf("first Modify() error = %v", err)
	}
}

func TestMemoryCredentialStoreRejectsInvalidInputs(t *testing.T) {
	if store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		" openai ": {Type: CredentialAPIKey, APIKey: "key"},
	}); store != nil || err == nil {
		t.Fatalf("NewMemoryCredentialStore(whitespace) = (%#v, %v), want nil, error", store, err)
	}
	store, err := NewMemoryCredentialStore(nil)
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	if _, _, err := store.Read(context.Background(), " "); err == nil {
		t.Fatal("Read(empty provider) error = nil")
	}
	if err := store.Modify(context.Background(), "openai", nil); err == nil {
		t.Fatal("Modify(nil callback) error = nil")
	}
	var nilStore *MemoryCredentialStore
	if _, _, err := nilStore.Read(context.Background(), "openai"); err == nil {
		t.Fatal("nil store Read() error = nil")
	}
	var nilCtx context.Context
	if _, _, err := store.Read(nilCtx, "openai"); err == nil {
		t.Fatal("Read(nil context) error = nil")
	}
}

func TestMemoryCredentialStoreReleasesProviderGates(t *testing.T) {
	store, err := NewMemoryCredentialStore(nil)
	if err != nil {
		t.Fatalf("NewMemoryCredentialStore() error = %v", err)
	}
	for index := range 1_000 {
		provider := "provider-" + strconv.Itoa(index)
		err := store.Modify(context.Background(), provider, func(StoredCredential, bool) (*StoredCredential, error) {
			credential := StoredCredential{Type: CredentialAPIKey, APIKey: "key"}
			return &credential, nil
		})
		if err != nil {
			t.Fatalf("Modify(%q) error = %v", provider, err)
		}
	}
	if len(store.gates) != 0 {
		t.Fatalf("provider gate count = %d, want 0", len(store.gates))
	}
}

func waitSignal(t *testing.T, signal <-chan struct{}, operation string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for %s", operation)
	}
}

func waitError(t *testing.T, result <-chan error, operation string) error {
	t.Helper()
	select {
	case err := <-result:
		return err
	case <-time.After(time.Second):
		t.Fatalf("timed out waiting for %s", operation)
		return nil
	}
}
