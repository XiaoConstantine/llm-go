package models

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	llm "github.com/XiaoConstantine/llm-go"
)

func TestConcreteOAuthPKCEExchangeAndRefresh(t *testing.T) {
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		requests++
		body, _ := io.ReadAll(request.Body)
		if request.URL.Path == "/anthropic" && request.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Anthropic content type = %q", request.Header.Get("Content-Type"))
		}
		if request.URL.Path == "/codex" && !strings.Contains(request.Header.Get("Content-Type"), "application/x-www-form-urlencoded") {
			t.Errorf("Codex content type = %q", request.Header.Get("Content-Type"))
		}
		if !strings.Contains(string(body), "client") {
			t.Errorf("token body = %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"access_token":"access","refresh_token":"refresh","expires_in":3600}`)
	}))
	defer server.Close()
	now := time.Unix(1_000, 0)
	anthropic := AnthropicOAuth{HTTPClient: server.Client(), AuthorizeURL: server.URL + "/authorize", TokenURL: server.URL + "/anthropic", ClientID: "client", Now: func() time.Time { return now }}
	auth, err := anthropic.Begin("http://localhost/callback")
	if err != nil {
		t.Fatal(err)
	}
	parsed, _ := url.Parse(auth.URL)
	if parsed.Query().Get("code_challenge") == "" || parsed.Query().Get("state") != auth.State || auth.State == auth.Verifier {
		t.Fatalf("authorization = %#v", auth)
	}
	credential, err := anthropic.Exchange(context.Background(), auth, "code", auth.State)
	if err != nil || credential.AccessToken != "access" || !credential.ExpiresAt.Equal(now.Add(time.Hour)) {
		t.Fatalf("exchange = %#v, %v", credential, err)
	}
	if _, err := anthropic.Refresh(context.Background(), credential); err != nil {
		t.Fatal(err)
	}

	codex := CodexOAuth{HTTPClient: server.Client(), AuthorizeURL: server.URL + "/authorize", TokenURL: server.URL + "/codex", ClientID: "client", Now: func() time.Time { return now }}
	codexAuth, err := codex.Begin("http://localhost/callback")
	if err != nil {
		t.Fatal(err)
	}
	codexURL, err := url.Parse(codexAuth.URL)
	if err != nil {
		t.Fatal(err)
	}
	query := codexURL.Query()
	if query.Get("id_token_add_organizations") != "true" || query.Get("codex_cli_simplified_flow") != "true" || query.Get("originator") != "llm-go" {
		t.Fatalf("Codex authorization query = %v", query)
	}
	customCodex := codex
	customCodex.Originator = "assigned-originator"
	customAuth, err := customCodex.Begin("http://localhost/callback")
	if err != nil {
		t.Fatal(err)
	}
	customURL, err := url.Parse(customAuth.URL)
	if err != nil {
		t.Fatal(err)
	}
	if got := customURL.Query().Get("originator"); got != "assigned-originator" {
		t.Fatalf("custom Codex originator = %q", got)
	}
	if _, err := codex.Exchange(context.Background(), codexAuth, "code", codexAuth.State); err != nil {
		t.Fatal(err)
	}
	if _, err := codex.Refresh(context.Background(), credential); err != nil {
		t.Fatal(err)
	}
	if requests != 4 {
		t.Fatalf("token requests = %d, want 4", requests)
	}
}

func TestOAuthRejectsInvalidExchangeWithoutIO(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { requests++ }))
	defer server.Close()
	anthropic := AnthropicOAuth{AuthorizeURL: server.URL, TokenURL: server.URL, HTTPClient: server.Client()}
	authorization, err := anthropic.Begin("http://localhost/callback")
	if err != nil {
		t.Fatal(err)
	}
	credential := StoredCredential{Type: CredentialOAuth, AccessToken: "access", RefreshToken: "refresh", ExpiresAt: time.Now().Add(time.Hour)}
	codex := CodexOAuth{TokenURL: server.URL, HTTPClient: server.Client()}
	for name, call := range map[string]func() error{
		"Anthropic Exchange": func() error {
			_, err := anthropic.Exchange(nil, authorization, "code", authorization.State) //nolint:staticcheck // Verify nil-context rejection.
			return err
		},
		"Anthropic Refresh": func() error {
			_, err := anthropic.Refresh(nil, credential) //nolint:staticcheck // Verify nil-context rejection.
			return err
		},
		"Codex Exchange": func() error {
			_, err := codex.Exchange(nil, authorization, "code", authorization.State) //nolint:staticcheck // Verify nil-context rejection.
			return err
		},
		"Codex Refresh": func() error {
			_, err := codex.Refresh(nil, credential) //nolint:staticcheck // Verify nil-context rejection.
			return err
		},
	} {
		if err := call(); err == nil {
			t.Errorf("%s(nil context) succeeded", name)
		}
	}
	exchanges := []struct {
		name string
		call func(OAuthAuthorization, string, string) error
	}{
		{name: "Anthropic", call: func(authorization OAuthAuthorization, code, state string) error {
			_, err := anthropic.Exchange(context.Background(), authorization, code, state)
			return err
		}},
		{name: "Codex", call: func(authorization OAuthAuthorization, code, state string) error {
			_, err := codex.Exchange(context.Background(), authorization, code, state)
			return err
		}},
	}
	invalid := []struct {
		name  string
		alter func(*OAuthAuthorization, *string, *string)
	}{
		{name: "empty expected state", alter: func(authorization *OAuthAuthorization, _, _ *string) { authorization.State = "" }},
		{name: "empty callback state", alter: func(_ *OAuthAuthorization, _, state *string) { *state = "" }},
		{name: "state mismatch", alter: func(_ *OAuthAuthorization, _, state *string) { *state = "wrong" }},
		{name: "empty code", alter: func(_ *OAuthAuthorization, code, _ *string) { *code = "" }},
		{name: "empty verifier", alter: func(authorization *OAuthAuthorization, _, _ *string) { authorization.Verifier = "" }},
		{name: "empty redirect URI", alter: func(authorization *OAuthAuthorization, _, _ *string) { authorization.RedirectURI = "" }},
	}
	for _, exchange := range exchanges {
		for _, test := range invalid {
			t.Run(exchange.name+"/"+test.name, func(t *testing.T) {
				candidate := authorization
				code, state := "code", authorization.State
				test.alter(&candidate, &code, &state)
				if err := exchange.call(candidate, code, state); err == nil {
					t.Fatal("Exchange() succeeded")
				}
			})
		}
	}
	if requests != 0 {
		t.Fatalf("token endpoint requests = %d, want zero", requests)
	}
}

func TestOAuthTokenRequestsRejectRedirectAndSanitizeErrors(t *testing.T) {
	destinationRequests := 0
	destination := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { destinationRequests++ }))
	defer destination.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", destination.URL)
		w.WriteHeader(http.StatusTemporaryRedirect)
	}))
	defer redirect.Close()
	originalRedirects := 0
	client := *redirect.Client()
	client.CheckRedirect = func(*http.Request, []*http.Request) error { originalRedirects++; return nil }
	oauth := AnthropicOAuth{TokenURL: redirect.URL, HTTPClient: &client}
	authorization := OAuthAuthorization{State: "state", Verifier: "verifier", RedirectURI: "http://localhost/callback"}
	if _, err := oauth.Exchange(context.Background(), authorization, "secret-code", "state"); err == nil {
		t.Fatal("redirecting exchange succeeded")
	}
	if destinationRequests != 0 || originalRedirects != 0 {
		t.Fatalf("redirect destination/original policy calls = %d/%d", destinationRequests, originalRedirects)
	}

	const secret = "known-refresh-secret"
	errorRequests := 0
	errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		errorRequests++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		if errorRequests == 1 {
			_, _ = io.WriteString(w, `{"error":"invalid_grant","error_description":"`+secret+`","echo":"`+secret+`"}`)
		} else {
			_, _ = io.WriteString(w, `{"error":"`+secret+`"}`)
		}
	}))
	defer errorServer.Close()
	oauth.TokenURL = errorServer.URL
	oauth.HTTPClient = errorServer.Client()
	_, err := oauth.Refresh(context.Background(), StoredCredential{Type: CredentialOAuth, AccessToken: "old", RefreshToken: secret, ExpiresAt: time.Now()})
	if err == nil || strings.Contains(err.Error(), secret) || !strings.Contains(err.Error(), "invalid_grant") {
		t.Fatalf("sanitized OAuth error = %v", err)
	}
	_, err = oauth.Refresh(context.Background(), StoredCredential{Type: CredentialOAuth, AccessToken: "old", RefreshToken: secret, ExpiresAt: time.Now()})
	if err == nil || strings.Contains(err.Error(), secret) {
		t.Fatalf("echoed OAuth error code = %v", err)
	}
}

func TestOAuthExpiresInBoundsAndRefreshTokenFallback(t *testing.T) {
	const maxExpiresIn = int64((1<<63 - 1) / time.Second)
	expiresIn := maxExpiresIn
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"access_token":"next","expires_in":`+fmt.Sprint(expiresIn)+`}`)
	}))
	defer server.Close()
	now := time.Unix(0, 0)
	oauth := AnthropicOAuth{TokenURL: server.URL, HTTPClient: server.Client(), Now: func() time.Time { return now }}
	current := StoredCredential{Type: CredentialOAuth, AccessToken: "old", RefreshToken: "preserved", ExpiresAt: now.Add(time.Second)}
	credential, err := oauth.Refresh(context.Background(), current)
	if err != nil || credential.RefreshToken != "preserved" || !credential.ExpiresAt.Equal(now.Add(time.Duration(maxExpiresIn)*time.Second)) {
		t.Fatalf("boundary refresh = %#v, %v", credential, err)
	}
	expiresIn = maxExpiresIn + 1
	if _, err := oauth.Refresh(context.Background(), current); err == nil || !strings.Contains(err.Error(), "supported range") {
		t.Fatalf("overflow refresh error = %v", err)
	}
}

func TestCollectionResolvesManagedCredentials(t *testing.T) {
	store, err := NewMemoryCredentialStore(map[string]StoredCredential{
		"openai":    {Type: CredentialAPIKey, APIKey: "managed-key"},
		"anthropic": {Type: CredentialOAuth, AccessToken: "sk-ant-oat-test", RefreshToken: "refresh", ExpiresAt: time.Now().Add(time.Hour)},
	})
	if err != nil {
		t.Fatal(err)
	}
	manager, err := NewCredentialManager(store)
	if err != nil {
		t.Fatal(err)
	}
	collection, err := NewWithCredentialManager(manager,
		ProviderConfig{ID: "openai", API: OpenAIResponses},
		ProviderConfig{ID: "anthropic", API: AnthropicMessages},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, info := range []llm.ModelInfo{
		{Provider: "openai", Model: "model"},
		{Provider: "anthropic", Model: "model"},
	} {
		generator, err := collection.GeneratorContext(context.Background(), info)
		if err != nil || generator == nil {
			t.Fatalf("GeneratorContext(%s) = %#v, %v", info.Provider, generator, err)
		}
	}
}
