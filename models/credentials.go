package models

import (
	"context"
	"fmt"
	"maps"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

// CredentialType identifies the representation of a stored credential.
type CredentialType string

const (
	CredentialAPIKey CredentialType = "api_key"
	CredentialOAuth  CredentialType = "oauth"
)

// StoredCredential contains provider credentials. API-key credentials use
// APIKey. OAuth credentials require either AccessToken or RefreshToken and may
// include AccountID and ExpiresAt. Attributes holds provider-specific values.
// Stores must copy Attributes on input and output.
type StoredCredential struct {
	Type         CredentialType
	APIKey       string
	AccessToken  string
	RefreshToken string
	AccountID    string
	ExpiresAt    time.Time
	Attributes   map[string]string
}

// Validate reports whether a stored credential is internally consistent.
func (c StoredCredential) Validate() error {
	textFields := [...]struct {
		name  string
		value string
	}{
		{name: "credential type", value: string(c.Type)},
		{name: "API key", value: c.APIKey},
		{name: "access token", value: c.AccessToken},
		{name: "refresh token", value: c.RefreshToken},
		{name: "account ID", value: c.AccountID},
	}
	for _, field := range textFields {
		if !utf8.ValidString(field.value) {
			return fmt.Errorf("%s must be valid UTF-8", field.name)
		}
	}
	switch c.Type {
	case CredentialAPIKey:
		if strings.TrimSpace(c.APIKey) == "" {
			return fmt.Errorf("API key must not be empty")
		}
		if c.AccessToken != "" || c.RefreshToken != "" || c.AccountID != "" || !c.ExpiresAt.IsZero() {
			return fmt.Errorf("API-key credential must not contain OAuth fields")
		}
	case CredentialOAuth:
		if c.APIKey != "" {
			return fmt.Errorf("OAuth credential must not contain an API key")
		}
		if c.AccessToken != "" && strings.TrimSpace(c.AccessToken) == "" {
			return fmt.Errorf("OAuth access token must not contain only whitespace")
		}
		if c.RefreshToken != "" && strings.TrimSpace(c.RefreshToken) == "" {
			return fmt.Errorf("OAuth refresh token must not contain only whitespace")
		}
		if strings.TrimSpace(c.AccessToken) == "" && strings.TrimSpace(c.RefreshToken) == "" {
			return fmt.Errorf("OAuth access token or refresh token is required")
		}
	default:
		return fmt.Errorf("credential type %q is invalid", c.Type)
	}
	for key, value := range c.Attributes {
		if strings.TrimSpace(key) == "" {
			return fmt.Errorf("credential attribute name must not be empty")
		}
		if !utf8.ValidString(key) || !utf8.ValidString(value) {
			return fmt.Errorf("credential attributes must be valid UTF-8")
		}
	}
	return nil
}

// CredentialMetadata describes a stored credential without exposing secrets.
type CredentialMetadata struct {
	Provider string
	Type     CredentialType
}

// CredentialStore persists one credential per provider. Modify must serialize
// updates for a provider, including across store instances backed by the same
// storage. Implementations must honor context cancellation while waiting for
// storage and must not commit an update after its context is done. A Modify
// callback must not call Modify or Delete on any CredentialStore.
type CredentialStore interface {
	Read(ctx context.Context, provider string) (StoredCredential, bool, error)
	List(ctx context.Context) ([]CredentialMetadata, error)
	Modify(ctx context.Context, provider string, update func(StoredCredential, bool) (*StoredCredential, error)) error
	Delete(ctx context.Context, provider string) error
}

// MemoryCredentialStore is an in-memory CredentialStore. It is safe for
// concurrent use. Its zero value is not usable; use NewMemoryCredentialStore.
type MemoryCredentialStore struct {
	state       chan struct{}
	gates       map[string]*credentialProviderGate
	credentials map[string]StoredCredential
}

type credentialProviderGate struct {
	token chan struct{}
	refs  int
}

// NewMemoryCredentialStore constructs an in-memory store from initial values.
// The input map and credential attributes are copied. A nil map creates an
// empty store.
func NewMemoryCredentialStore(initial map[string]StoredCredential) (*MemoryCredentialStore, error) {
	store := &MemoryCredentialStore{
		state:       make(chan struct{}, 1),
		gates:       make(map[string]*credentialProviderGate),
		credentials: make(map[string]StoredCredential, len(initial)),
	}
	for provider, credential := range initial {
		normalized, err := normalizeProviderID(provider)
		if err != nil {
			return nil, err
		}
		if normalized != provider {
			return nil, fmt.Errorf("credential provider %q contains surrounding whitespace", provider)
		}
		if err := credential.Validate(); err != nil {
			return nil, fmt.Errorf("credential for provider %q: %w", provider, err)
		}
		store.credentials[provider] = cloneStoredCredential(credential)
	}
	return store, nil
}

// Read implements CredentialStore.
func (s *MemoryCredentialStore) Read(ctx context.Context, provider string) (StoredCredential, bool, error) {
	provider, err := normalizeProviderID(provider)
	if err != nil {
		return StoredCredential{}, false, err
	}
	if err := s.lockState(ctx); err != nil {
		return StoredCredential{}, false, err
	}
	credential, found := s.credentials[provider]
	s.unlockState()
	credential = cloneStoredCredential(credential)
	if err := ctx.Err(); err != nil {
		return StoredCredential{}, false, err
	}
	return credential, found, nil
}

// List implements CredentialStore. Results are sorted by provider ID.
func (s *MemoryCredentialStore) List(ctx context.Context) ([]CredentialMetadata, error) {
	if err := s.lockState(ctx); err != nil {
		return nil, err
	}
	metadata := make([]CredentialMetadata, 0, len(s.credentials))
	for provider, credential := range s.credentials {
		metadata = append(metadata, CredentialMetadata{Provider: provider, Type: credential.Type})
	}
	s.unlockState()
	sort.Slice(metadata, func(i, j int) bool { return metadata[i].Provider < metadata[j].Provider })
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return metadata, nil
}

// Modify implements CredentialStore. Returning nil from update deletes the
// credential. If update returns an error, the stored value is unchanged.
func (s *MemoryCredentialStore) Modify(ctx context.Context, provider string, update func(StoredCredential, bool) (*StoredCredential, error)) error {
	provider, err := normalizeProviderID(provider)
	if err != nil {
		return err
	}
	if update == nil {
		return fmt.Errorf("credential update callback must not be nil")
	}
	unlock, err := s.lockProvider(ctx, provider)
	if err != nil {
		return err
	}
	defer unlock()
	if err := s.lockState(ctx); err != nil {
		return err
	}
	current, found := s.credentials[provider]
	s.unlockState()
	next, err := update(cloneStoredCredential(current), found)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if next != nil {
		if err := next.Validate(); err != nil {
			return fmt.Errorf("credential for provider %q: %w", provider, err)
		}
	}
	if err := s.lockState(ctx); err != nil {
		return err
	}
	defer s.unlockState()
	if next == nil {
		delete(s.credentials, provider)
		return nil
	}
	s.credentials[provider] = cloneStoredCredential(*next)
	return nil
}

// Delete implements CredentialStore. Deleting an unknown provider succeeds.
func (s *MemoryCredentialStore) Delete(ctx context.Context, provider string) error {
	return s.Modify(ctx, provider, func(StoredCredential, bool) (*StoredCredential, error) {
		return nil, nil
	})
}

func (s *MemoryCredentialStore) lockProvider(ctx context.Context, provider string) (func(), error) {
	if err := s.lockState(ctx); err != nil {
		return nil, err
	}
	gate := s.gates[provider]
	if gate == nil {
		gate = &credentialProviderGate{token: make(chan struct{}, 1)}
		s.gates[provider] = gate
	}
	gate.refs++
	s.unlockState()

	select {
	case gate.token <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-gate.token
			s.releaseProviderGate(provider, gate)
			return nil, err
		}
		return func() {
			<-gate.token
			s.releaseProviderGate(provider, gate)
		}, nil
	case <-ctx.Done():
		s.releaseProviderGate(provider, gate)
		return nil, ctx.Err()
	}
}

func (s *MemoryCredentialStore) releaseProviderGate(provider string, gate *credentialProviderGate) {
	s.state <- struct{}{}
	gate.refs--
	if gate.refs == 0 && s.gates[provider] == gate {
		delete(s.gates, provider)
	}
	s.unlockState()
}

func (s *MemoryCredentialStore) lockState(ctx context.Context) error {
	if s == nil || s.state == nil || s.gates == nil || s.credentials == nil {
		return fmt.Errorf("credential store is nil or uninitialized")
	}
	if ctx == nil {
		return fmt.Errorf("credential store context must not be nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case s.state <- struct{}{}:
		if err := ctx.Err(); err != nil {
			s.unlockState()
			return err
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *MemoryCredentialStore) unlockState() {
	<-s.state
}

func normalizeProviderID(provider string) (string, error) {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return "", fmt.Errorf("credential provider must not be empty")
	}
	if !utf8.ValidString(provider) {
		return "", fmt.Errorf("credential provider must be valid UTF-8")
	}
	return provider, nil
}

func cloneStoredCredential(credential StoredCredential) StoredCredential {
	if credential.Attributes == nil {
		return credential
	}
	attributes := credential.Attributes
	credential.Attributes = make(map[string]string, len(attributes))
	maps.Copy(credential.Attributes, attributes)
	return credential
}
