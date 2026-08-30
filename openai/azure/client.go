// Package azure implements Azure OpenAI's Responses API with llm-go's neutral
// model contracts. It shares the OpenAI Responses wire codec while applying
// Azure endpoint normalization, API-version query parameters, deployment names,
// and api-key authentication.
package azure

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	openairesponses "github.com/XiaoConstantine/llm-go/openai/responses"
)

const (
	defaultProvider   = "azure-openai-responses"
	defaultAPIVersion = "v1"
)

// Config configures an Azure OpenAI Responses client. Model and APIKey are
// required. DeploymentName defaults to Model. BaseURL takes precedence over
// ResourceName; when only ResourceName is set, the endpoint is constructed as
// https://<resource>.openai.azure.com/openai/v1. APIVersion defaults to "v1".
type Config struct {
	Provider       string
	Model          string
	DeploymentName string
	Capabilities   []llm.Capability
	APIKey         string
	APIVersion     string
	ResourceName   string
	BaseURL        string
	HTTPClient     *http.Client
	Headers        http.Header
}

// Client is an immutable Azure OpenAI Responses client. It is safe for
// concurrent use when its configured HTTP transport is safe for concurrent use.
type Client struct {
	provider string
	model    string
	inner    *openairesponses.Client
}

var _ llm.Generator = (*Client)(nil)

// New constructs an Azure OpenAI Responses client.
func New(config Config) (*Client, error) {
	return newClient(config, nil)
}

// NewWithCompatibility constructs a client with Responses compatibility
// metadata. compatibility is copied during construction.
func NewWithCompatibility(config Config, compatibility *llm.OpenAIResponsesCompatibility) (*Client, error) {
	return newClient(config, compatibility)
}

func newClient(config Config, compatibility *llm.OpenAIResponsesCompatibility) (*Client, error) {
	provider := strings.TrimSpace(config.Provider)
	if provider == "" {
		provider = defaultProvider
	}
	model := strings.TrimSpace(config.Model)
	if model == "" {
		return nil, configError(provider, "model must not be empty")
	}
	deployment := strings.TrimSpace(config.DeploymentName)
	if deployment == "" {
		deployment = model
	}
	apiKey := strings.TrimSpace(config.APIKey)
	if apiKey == "" {
		return nil, configError(provider, "API key must not be empty")
	}
	apiVersion := strings.TrimSpace(config.APIVersion)
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}
	if strings.ContainsAny(apiVersion, "?#") {
		return nil, configError(provider, "API version must not contain a query or fragment")
	}
	baseURL, err := resolveBaseURL(provider, config.BaseURL, config.ResourceName)
	if err != nil {
		return nil, err
	}
	httpClient := azureHTTPClient(config.HTTPClient, apiKey, apiVersion)
	innerConfig := openairesponses.Config{Provider: provider, Model: model, RequestModel: deployment, Capabilities: config.Capabilities,
		APIKey: apiKey, BaseURL: baseURL, HTTPClient: httpClient, Headers: config.Headers}
	var inner *openairesponses.Client
	if compatibility == nil {
		inner, err = openairesponses.New(innerConfig)
	} else {
		value := *compatibility
		inner, err = openairesponses.NewWithCompatibility(innerConfig, &value)
	}
	if err != nil {
		return nil, err
	}
	return &Client{provider: provider, model: model, inner: inner}, nil
}

// Info describes the configured catalog model. The deployment name is a wire
// concern and does not replace the model identity reported to callers.
func (c *Client) Info() llm.ModelInfo {
	info := c.inner.Info()
	info.Model = c.model
	info.API = llm.APIAzureOpenAIResponses
	return info
}

// Generate performs one Azure Responses request.
func (c *Client) Generate(ctx context.Context, request llm.Request) (*llm.Response, error) {
	return c.inner.Generate(ctx, request)
}

// Stream starts one Azure Responses event stream.
func (c *Client) Stream(ctx context.Context, request llm.Request) (llm.Stream, error) {
	return c.inner.Stream(ctx, request)
}

func resolveBaseURL(provider, rawBaseURL, rawResourceName string) (string, error) {
	baseURL := strings.TrimSpace(rawBaseURL)
	resourceName := strings.TrimSpace(rawResourceName)
	if baseURL == "" {
		if resourceName == "" {
			return "", configError(provider, "Azure OpenAI base URL is required; set BaseURL or ResourceName")
		}
		if strings.ContainsAny(resourceName, "/?#") {
			return "", configError(provider, "resource name must not contain URL separators")
		}
		baseURL = "https://" + resourceName + ".openai.azure.com/openai/v1"
	}
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return "", configError(provider, "invalid base URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", configError(provider, "base URL scheme must be http or https")
	}
	if parsed.Host == "" {
		return "", configError(provider, "base URL must be absolute")
	}
	if parsed.User != nil {
		return "", configError(provider, "base URL must not contain user information")
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" || parsed.ForceQuery {
		return "", configError(provider, "base URL must not contain a query or fragment")
	}
	path := strings.TrimRight(parsed.Path, "/")
	azureHost := strings.HasSuffix(strings.ToLower(parsed.Hostname()), ".openai.azure.com") ||
		strings.HasSuffix(strings.ToLower(parsed.Hostname()), ".cognitiveservices.azure.com")
	if azureHost && (path == "" || path == "/openai") {
		path = "/openai/v1"
	}
	parsed.Path = path
	parsed.RawPath = ""
	return strings.TrimRight(parsed.String(), "/"), nil
}

func azureHTTPClient(configured *http.Client, apiKey, apiVersion string) *http.Client {
	base := configured
	if base == nil {
		base = http.DefaultClient
	}
	clone := *base
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	clone.Transport = azureTransport{base: transport, apiKey: apiKey, apiVersion: apiVersion}
	return &clone
}

type azureTransport struct {
	base       http.RoundTripper
	apiKey     string
	apiVersion string
}

func (t azureTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	clone := request.Clone(request.Context())
	clone.Header = request.Header.Clone()
	clone.Header.Del("Authorization")
	clone.Header.Set("Api-Key", t.apiKey)
	query := clone.URL.Query()
	query.Set("api-version", t.apiVersion)
	clone.URL.RawQuery = query.Encode()
	return t.base.RoundTrip(clone)
}

func configError(provider, format string, args ...any) *llm.Error {
	return &llm.Error{Kind: llm.KindInvalidRequest, Op: "configure", Provider: provider, Err: fmt.Errorf(format, args...)}
}
