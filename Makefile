.PHONY: build clean server test race integration test-all bench native

BIN := bin

# Rust delta-varbyte decoder linked by internal/retrieval/index/deltavarbyte_cgo.go
# (amd64+cgo). The target triple must match that file's #cgo LDFLAGS path.
native:
	cargo build --release --manifest-path native/deltavarbyte/Cargo.toml --target x86_64-unknown-linux-gnu

build: native
	mkdir -p $(BIN)
	CGO_ENABLED=1 go build -o $(BIN)/search-server  ./cmd/server/
	CGO_ENABLED=1 go build -o $(BIN)/search-client  ./cmd/client/
	CGO_ENABLED=1 go build -o $(BIN)/search-ingest  ./cmd/ingest/

server: native
	CGO_ENABLED=1 go run ./cmd/server/

test: native
	go test ./... -v

race: native
	go test -race ./...

integration:
	go test -tags=integration ./internal/integration/... -v

# Full local verification: build, unit tests, race detector, integration tests.
test-all: build test race integration

bench: native
	go test ./... -bench=. -benchmem

clean:
	rm -rf $(BIN) search-server native/deltavarbyte/target
