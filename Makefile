.PHONY: build clean server test race integration test-all bench

BIN := bin

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
