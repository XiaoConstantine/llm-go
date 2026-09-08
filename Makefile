.PHONY: fmt fmt-check lint staticcheck check

# Keep this in sync with .github/workflows/ci.yml.
GOLANGCI_LINT_VERSION ?= v2.13.1
GOLANGCI_LINT ?= $(shell go env GOPATH)/bin/golangci-lint

STATICCHECK_VERSION ?= latest
STATICCHECK ?= $(shell go env GOPATH)/bin/staticcheck

fmt:
	gofmt -w .

fmt-check:
	@files="$$(gofmt -l .)"; \
	if [ -n "$$files" ]; then \
		echo "The following Go files are not formatted:"; \
		echo "$$files"; \
		exit 1; \
	fi

# Install the pinned linter with this module's Go toolchain. Distribution
# packages may have been built by a Go version too old to lint this module.
$(GOLANGCI_LINT):
	go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

lint: $(GOLANGCI_LINT)
	@current="$$($(GOLANGCI_LINT) version --short 2>/dev/null || true)"; \
	want="$(GOLANGCI_LINT_VERSION:v%=%)"; \
	if [ "$$current" != "$$want" ]; then \
		echo "Installing golangci-lint $(GOLANGCI_LINT_VERSION) with $$(go version)"; \
		go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION); \
	fi
	$(GOLANGCI_LINT) run --timeout=5m ./...

$(STATICCHECK):
	go install honnef.co/go/tools/cmd/staticcheck@$(STATICCHECK_VERSION)

staticcheck: $(STATICCHECK)
	$(STATICCHECK) ./...

check: fmt-check lint staticcheck
