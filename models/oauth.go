package models

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	anthropicOAuthClientID       = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
	anthropicAuthorizeURL        = "https://claude.ai/oauth/authorize"
	anthropicTokenURL            = "https://platform.claude.com/v1/oauth/token"
	anthropicScopes              = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
	codexOAuthClientID           = "app_EMoamEEZ73f0CkXaXp7hrann"
	codexAuthorizeURL            = "https://auth.openai.com/oauth/authorize"
	codexTokenURL                = "https://auth.openai.com/oauth/token"
	codexScopes                  = "openid profile email offline_access"
	codexOAuthOriginator         = "llm-go"
	maxOAuthResponseBytes  int64 = 1 << 20
)

// OAuthAuthorization is the non-interactive first half of an authorization-code
// PKCE flow. Applications present URL and retain State and Verifier until the
// redirect code is received.
type OAuthAuthorization struct {
	URL         string
	State       string
	Verifier    string
	RedirectURI string
}

// AnthropicOAuth implements Anthropic's provider-private Claude subscription
// PKCE flow without owning a browser, callback server, or user interface. Its
// default endpoint and client ID are unofficial, unstable, and replaceable via
// the corresponding fields. It retains but never closes HTTPClient and starts
// no background work. Do not mutate its fields during use; HTTPClient and Now
// must support any concurrent calls made by the application.
type AnthropicOAuth struct {
	HTTPClient   *http.Client
	AuthorizeURL string
	TokenURL     string
	ClientID     string
	Now          func() time.Time
}

// CodexOAuth implements OpenAI's provider-private ChatGPT Codex PKCE flow
// without owning interactive UI. Its default endpoint and client ID are
// unofficial, unstable, and replaceable via the corresponding fields. It
// retains but never closes HTTPClient and starts no background work. Do not
// mutate its fields during use; HTTPClient and Now must support any concurrent
// calls made by the application.
type CodexOAuth struct {
	HTTPClient   *http.Client
	AuthorizeURL string
	TokenURL     string
	ClientID     string
	// Originator identifies the private browser flow and defaults to "llm-go",
	// matching the Codex client. It can be replaced for deployments that have an
	// assigned originator.
	Originator string
	Now        func() time.Time
}

// Begin creates an Anthropic authorization URL and caller-owned PKCE state.
// Applications must retain the returned value and compare it during Exchange.
func (o AnthropicOAuth) Begin(redirectURI string) (OAuthAuthorization, error) {
	verifier, challenge, err := newPKCE()
	if err != nil {
		return OAuthAuthorization{}, err
	}
	state, err := newOAuthState()
	if err != nil {
		return OAuthAuthorization{}, err
	}
	clientID := defaultString(o.ClientID, anthropicOAuthClientID)
	values := url.Values{
		"code":                  {"true"},
		"client_id":             {clientID},
		"response_type":         {"code"},
		"redirect_uri":          {redirectURI},
		"scope":                 {anthropicScopes},
		"code_challenge":        {challenge},
		"code_challenge_method": {"S256"},
		"state":                 {state},
	}
	return authorization(defaultString(o.AuthorizeURL, anthropicAuthorizeURL), redirectURI, state, verifier, values)
}

// Exchange validates the retained authorization and callback values before
// exchanging an Anthropic authorization code. Invalid inputs perform no I/O.
func (o AnthropicOAuth) Exchange(ctx context.Context, authorization OAuthAuthorization, code, state string) (StoredCredential, error) {
	if ctx == nil {
		return StoredCredential{}, fmt.Errorf("anthropic OAuth context must not be nil")
	}
	if err := validateOAuthExchange("Anthropic", authorization, code, state); err != nil {
		return StoredCredential{}, err
	}
	body := map[string]string{
		"grant_type": "authorization_code", "client_id": defaultString(o.ClientID, anthropicOAuthClientID),
		"code": code, "state": state, "redirect_uri": authorization.RedirectURI, "code_verifier": authorization.Verifier,
	}
	return oauthJSON(ctx, o.client(), defaultString(o.TokenURL, anthropicTokenURL), body, StoredCredential{}, o.now())
}

// Refresh exchanges a current Anthropic OAuth refresh token. The returned
// credential is caller-owned.
func (o AnthropicOAuth) Refresh(ctx context.Context, current StoredCredential) (StoredCredential, error) {
	if ctx == nil {
		return StoredCredential{}, fmt.Errorf("anthropic OAuth context must not be nil")
	}
	if current.Type != CredentialOAuth || strings.TrimSpace(current.RefreshToken) == "" {
		return StoredCredential{}, fmt.Errorf("anthropic OAuth refresh token is required")
	}
	body := map[string]string{"grant_type": "refresh_token", "client_id": defaultString(o.ClientID, anthropicOAuthClientID), "refresh_token": current.RefreshToken}
	return oauthJSON(ctx, o.client(), defaultString(o.TokenURL, anthropicTokenURL), body, current, o.now())
}

// Begin creates a Codex authorization URL and caller-owned PKCE state.
// Applications must retain the returned value and compare it during Exchange.
func (o CodexOAuth) Begin(redirectURI string) (OAuthAuthorization, error) {
	verifier, challenge, err := newPKCE()
	if err != nil {
		return OAuthAuthorization{}, err
	}
	state, err := newOAuthState()
	if err != nil {
		return OAuthAuthorization{}, err
	}
	values := url.Values{
		"client_id": {defaultString(o.ClientID, codexOAuthClientID)}, "response_type": {"code"},
		"redirect_uri": {redirectURI}, "scope": {codexScopes}, "code_challenge": {challenge},
		"code_challenge_method":      {"S256"},
		"state":                      {state},
		"id_token_add_organizations": {"true"},
		"codex_cli_simplified_flow":  {"true"},
		"originator":                 {defaultString(o.Originator, codexOAuthOriginator)},
	}
	return authorization(defaultString(o.AuthorizeURL, codexAuthorizeURL), redirectURI, state, verifier, values)
}

// Exchange validates the retained authorization and callback values before
// exchanging a Codex authorization code. Invalid inputs perform no I/O.
func (o CodexOAuth) Exchange(ctx context.Context, authorization OAuthAuthorization, code, state string) (StoredCredential, error) {
	if ctx == nil {
		return StoredCredential{}, fmt.Errorf("codex OAuth context must not be nil")
	}
	if err := validateOAuthExchange("Codex", authorization, code, state); err != nil {
		return StoredCredential{}, err
	}
	values := url.Values{
		"grant_type": {"authorization_code"}, "client_id": {defaultString(o.ClientID, codexOAuthClientID)},
		"code": {code}, "code_verifier": {authorization.Verifier}, "redirect_uri": {authorization.RedirectURI},
	}
	return oauthForm(ctx, o.client(), defaultString(o.TokenURL, codexTokenURL), values, StoredCredential{}, o.now())
}

// Refresh exchanges a current Codex OAuth refresh token. The returned
// credential is caller-owned.
func (o CodexOAuth) Refresh(ctx context.Context, current StoredCredential) (StoredCredential, error) {
	if ctx == nil {
		return StoredCredential{}, fmt.Errorf("codex OAuth context must not be nil")
	}
	if current.Type != CredentialOAuth || strings.TrimSpace(current.RefreshToken) == "" {
		return StoredCredential{}, fmt.Errorf("codex OAuth refresh token is required")
	}
	values := url.Values{"grant_type": {"refresh_token"}, "refresh_token": {current.RefreshToken}, "client_id": {defaultString(o.ClientID, codexOAuthClientID)}}
	return oauthForm(ctx, o.client(), defaultString(o.TokenURL, codexTokenURL), values, current, o.now())
}

// RefreshConfig returns a CredentialManager configuration that retains a copy
// of o and invokes its Anthropic refresh flow.
func (o AnthropicOAuth) RefreshConfig(provider string, before time.Duration) CredentialRefreshConfig {
	return CredentialRefreshConfig{Provider: provider, RefreshBefore: before, Refresh: o.Refresh}
}

// RefreshConfig returns a CredentialManager configuration that retains a copy
// of o and invokes its Codex refresh flow.
func (o CodexOAuth) RefreshConfig(provider string, before time.Duration) CredentialRefreshConfig {
	return CredentialRefreshConfig{Provider: provider, RefreshBefore: before, Refresh: o.Refresh}
}

func validateOAuthExchange(provider string, authorization OAuthAuthorization, code, state string) error {
	if strings.TrimSpace(authorization.State) == "" {
		return fmt.Errorf("%s OAuth expected state is required", provider)
	}
	if strings.TrimSpace(state) == "" {
		return fmt.Errorf("%s OAuth callback state is required", provider)
	}
	if state != authorization.State {
		return fmt.Errorf("%s OAuth state mismatch", provider)
	}
	if strings.TrimSpace(code) == "" {
		return fmt.Errorf("%s OAuth authorization code is required", provider)
	}
	if strings.TrimSpace(authorization.Verifier) == "" {
		return fmt.Errorf("%s OAuth PKCE verifier is required", provider)
	}
	if strings.TrimSpace(authorization.RedirectURI) == "" {
		return fmt.Errorf("%s OAuth redirect URI is required", provider)
	}
	return nil
}

func authorization(rawURL, redirectURI, state, verifier string, values url.Values) (OAuthAuthorization, error) {
	if strings.TrimSpace(redirectURI) == "" {
		return OAuthAuthorization{}, fmt.Errorf("OAuth redirect URI must not be empty")
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme != "https" && parsed.Scheme != "http" || parsed.Host == "" {
		return OAuthAuthorization{}, fmt.Errorf("invalid OAuth authorization URL %q", rawURL)
	}
	parsed.RawQuery = values.Encode()
	return OAuthAuthorization{URL: parsed.String(), State: state, Verifier: verifier, RedirectURI: redirectURI}, nil
}

func newOAuthState() (string, error) {
	data := make([]byte, 16)
	if _, err := rand.Read(data); err != nil {
		return "", fmt.Errorf("generate OAuth state: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(data), nil
}

func newPKCE() (string, string, error) {
	data := make([]byte, 32)
	if _, err := rand.Read(data); err != nil {
		return "", "", fmt.Errorf("generate PKCE verifier: %w", err)
	}
	verifier := base64.RawURLEncoding.EncodeToString(data)
	digest := sha256.Sum256([]byte(verifier))
	return verifier, base64.RawURLEncoding.EncodeToString(digest[:]), nil
}

type oauthTokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
}

type oauthErrorResponse struct {
	Error string `json:"error"`
}

func oauthJSON(ctx context.Context, client *http.Client, endpoint string, body map[string]string, current StoredCredential, now time.Time) (StoredCredential, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return StoredCredential{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(encoded)))
	if err != nil {
		return StoredCredential{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	return executeOAuth(client, request, current, now)
}

func oauthForm(ctx context.Context, client *http.Client, endpoint string, values url.Values, current StoredCredential, now time.Time) (StoredCredential, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(values.Encode()))
	if err != nil {
		return StoredCredential{}, err
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return executeOAuth(client, request, current, now)
}

func executeOAuth(client *http.Client, request *http.Request, current StoredCredential, now time.Time) (StoredCredential, error) {
	safeClient := *client
	safeClient.CheckRedirect = func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse }
	response, err := safeClient.Do(request)
	if err != nil {
		return StoredCredential{}, err
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxOAuthResponseBytes+1))
	if err != nil {
		return StoredCredential{}, fmt.Errorf("read OAuth response: %w", err)
	}
	if len(body) > int(maxOAuthResponseBytes) {
		return StoredCredential{}, fmt.Errorf("OAuth response exceeds %d bytes", maxOAuthResponseBytes)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		var endpointError oauthErrorResponse
		_ = json.Unmarshal(body, &endpointError)
		if safe := safeOAuthErrorCode(endpointError.Error); safe != "" {
			return StoredCredential{}, fmt.Errorf("OAuth token endpoint returned HTTP %d (%s)", response.StatusCode, safe)
		}
		return StoredCredential{}, fmt.Errorf("OAuth token endpoint returned HTTP %d", response.StatusCode)
	}
	var token oauthTokenResponse
	if err := json.Unmarshal(body, &token); err != nil {
		return StoredCredential{}, fmt.Errorf("decode OAuth response: %w", err)
	}
	if strings.TrimSpace(token.AccessToken) == "" || token.ExpiresIn <= 0 {
		return StoredCredential{}, fmt.Errorf("OAuth response is missing access_token or expires_in")
	}
	const maxExpiresIn = int64((1<<63 - 1) / time.Second)
	if token.ExpiresIn > maxExpiresIn {
		return StoredCredential{}, fmt.Errorf("OAuth response expires_in exceeds supported range")
	}
	if token.RefreshToken == "" {
		token.RefreshToken = current.RefreshToken
	}
	if strings.TrimSpace(token.RefreshToken) == "" {
		return StoredCredential{}, fmt.Errorf("OAuth response is missing refresh_token")
	}
	expiresAfter := time.Duration(token.ExpiresIn) * time.Second
	credential := StoredCredential{Type: CredentialOAuth, AccessToken: token.AccessToken, RefreshToken: token.RefreshToken, ExpiresAt: now.Add(expiresAfter)}
	if err := credential.Validate(); err != nil {
		return StoredCredential{}, err
	}
	return credential, nil
}

func safeOAuthErrorCode(value string) string {
	switch strings.TrimSpace(value) {
	case "invalid_request", "invalid_client", "invalid_grant", "unauthorized_client",
		"unsupported_grant_type", "invalid_scope", "access_denied", "server_error", "temporarily_unavailable":
		return strings.TrimSpace(value)
	default:
		return ""
	}
}

func (o AnthropicOAuth) client() *http.Client {
	if o.HTTPClient != nil {
		return o.HTTPClient
	}
	return http.DefaultClient
}
func (o CodexOAuth) client() *http.Client {
	if o.HTTPClient != nil {
		return o.HTTPClient
	}
	return http.DefaultClient
}
func (o AnthropicOAuth) now() time.Time {
	if o.Now != nil {
		return o.Now()
	}
	return time.Now()
}
func (o CodexOAuth) now() time.Time {
	if o.Now != nil {
		return o.Now()
	}
	return time.Now()
}
func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
}
