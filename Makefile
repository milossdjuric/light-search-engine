.PHONY: build clean server test race integration test-all bench

BIN := bin

# The native kernels (AVX2 C on amd64, NEON C++ on arm64) live in
# internal/codec and are compiled by cgo as part of `go build`; they need a C
# compiler (plus a C++ one on arm64) and CGO_ENABLED=1, but no separate step.

build:
	mkdir -p $(BIN)
	CGO_ENABLED=1 go build -o $(BIN)/search-server  ./cmd/server/
	CGO_ENABLED=1 go build -o $(BIN)/search-client  ./cmd/client/
	CGO_ENABLED=1 go build -o $(BIN)/search-ingest  ./cmd/ingest/

server:
	CGO_ENABLED=1 go run ./cmd/server/

test:
	go test ./... -v

race:
	go test -race ./...

integration:
	go test -tags=integration ./internal/integration/... -v

# Full local verification: build, unit tests, race detector, integration tests.
test-all: build test race integration

bench:
	go test ./... -bench=. -benchmem

clean:
	rm -rf $(BIN) search-server
