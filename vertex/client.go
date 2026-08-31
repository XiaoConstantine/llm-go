// Package vertex implements Google Vertex AI GenerateContent using llm-go's
// neutral model contracts.
package vertex

import (
	"net/http"
	"net/url"
	"strings"

	llm "github.com/XiaoConstantine/llm-go"
	"github.com/XiaoConstantine/llm-go/gemini"
)

const defaultAPIVersion = "v1"

// Client is a Vertex-configured GenerateContent client.
type Client = gemini.Client

// Config configures a Vertex AI client. Model is required. APIKey selects
// Vertex API-key authentication. When APIKey is empty, Project and Location are
// required; HTTPClient must already authenticate requests, or a nil HTTPClient
// uses Application Default Credentials. Provider defaults to "google-vertex".
//
// BaseURL is optional and primarily intended for gateways and tests. APIVersion
// defaults to v1. Capabilities and Headers have the same ownership and feature
// semantics as the Gemini GenerateContent adapter.
type Config struct {
	Provider     string
	Model        string
	Capabilities []llm.Capability
	Reasoning    bool
	APIKey       string
	Project      string
	Location     string
	BaseURL      string
	APIVersion   string
	HTTPClient   *http.Client
	Headers      http.Header
}

// New constructs a Vertex AI client.
func New(config Config) (*Client, error) {
	apiVersion := config.APIVersion
	if apiVersion == "" {
		apiVersion = defaultAPIVersion
	}
	baseURL := config.BaseURL
	if strings.Contains(baseURL, "{location}") {
		baseURL = ""
	} else if normalizedBase, versionPath, ok := normalizeCustomBaseURL(baseURL, apiVersion); ok {
		baseURL, apiVersion = normalizedBase, versionPath
	}
	return gemini.NewVertex(gemini.Config{Provider: config.Provider, Model: config.Model,
		Capabilities: config.Capabilities, Reasoning: config.Reasoning, APIKey: config.APIKey, BaseURL: baseURL,
		APIVersion: apiVersion, HTTPClient: config.HTTPClient, Headers: config.Headers}, config.Project, config.Location)
}

func normalizeCustomBaseURL(value, defaultVersion string) (string, string, bool) {
	parsed, err := url.Parse(strings.TrimSpace(value))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", "", false
	}
	segments := strings.FieldsFunc(parsed.EscapedPath(), func(character rune) bool { return character == '/' })
	hasVersion := false
	for _, segment := range segments {
		if vertexAPIVersionSegment(segment) {
			hasVersion = true
			break
		}
	}
	if !hasVersion {
		segments = append(segments, defaultVersion)
	}
	parsed.Path, parsed.RawPath = "", ""
	return parsed.String(), strings.Join(segments, "/"), true
}

func vertexAPIVersionSegment(segment string) bool {
	value := strings.ToLower(segment)
	if len(value) < 2 || value[0] != 'v' {
		return false
	}
	index := 1
	for index < len(value) && value[index] >= '0' && value[index] <= '9' {
		index++
	}
	if index == 1 {
		return false
	}
	if strings.HasPrefix(value[index:], "beta") {
		index += len("beta")
		for index < len(value) && value[index] >= '0' && value[index] <= '9' {
			index++
		}
	}
	return index == len(value)
}
