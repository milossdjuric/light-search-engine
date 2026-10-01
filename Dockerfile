FROM golang:1.26-alpine AS builder

# CGO builds the native kernels in internal/codec: AVX2 C on amd64, NEON C++
# on arm64 (hence g++ for arm64 images).
RUN apk add --no-cache gcc g++ musl-dev

WORKDIR /src
COPY go.mod go.sum ./
COPY vendor/ vendor/

COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -mod=vendor -o /search-server ./cmd/server/

FROM alpine:3.19

# libstdc++ (and its libgcc dependency): needed by arm64 images, whose binary
# is linked with g++ for the NEON kernels; amd64 binaries only need libc.
RUN apk add --no-cache ca-certificates libstdc++

WORKDIR /app
COPY --from=builder /search-server /app/search-server
COPY configs/default.yaml /app/configs/default.yaml

# HTTP API and gRPC replication
EXPOSE 8080 9090

ENTRYPOINT ["/app/search-server"]
CMD ["-config", "/app/configs/default.yaml"]
