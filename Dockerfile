# Rust delta-varbyte decoder (native/deltavarbyte), statically linked by
# internal/retrieval/index/deltavarbyte_cgo.go. Built for musl to match the
# alpine Go builder.
FROM rust:alpine AS native
RUN apk add --no-cache musl-dev
WORKDIR /native
COPY native/deltavarbyte/ .
RUN cargo build --release --target x86_64-unknown-linux-musl

FROM golang:1.26-alpine AS builder

# CGO is required for AVX2 blockpack path (blockpack_cgo.go)
RUN apk add --no-cache gcc musl-dev

WORKDIR /src
COPY go.mod go.sum ./
COPY vendor/ vendor/

COPY . .
# deltavarbyte_cgo.go's #cgo LDFLAGS points at the gnu target dir; place the
# musl-built archive there.
COPY --from=native /native/target/x86_64-unknown-linux-musl/release/libdeltavarbyte.a \
     native/deltavarbyte/target/x86_64-unknown-linux-gnu/release/libdeltavarbyte.a
RUN CGO_ENABLED=1 GOOS=linux go build -mod=vendor -o /search-server ./cmd/server/main.go

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

WORKDIR /app
COPY --from=builder /search-server /app/search-server
COPY configs/default.yaml /app/configs/default.yaml

# HTTP API and gRPC replication
EXPOSE 8080 9090

ENTRYPOINT ["/app/search-server"]
CMD ["-config", "/app/configs/default.yaml"]
