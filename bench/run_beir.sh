#!/usr/bin/env bash
# bench/run_beir.sh — build, start server with a clean data dir, run BEIR multi-dataset eval
#
# Usage:
#   ./bench/run_beir.sh [--datasets nfcorpus,scifact,arguana,fiqa,trec-covid]
#                       [--workers N] [--top-k N] [--output FILE] [--config FILE]
#                       [--skip-build] [--keep-data]
#
# The server is started once against a temporary clean data directory.
# run_beir.py handles per-dataset: reset → ingest → flush+merge → eval.
# The server is killed on exit (trap).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS="nfcorpus,scifact,arguana,fiqa,trec-covid"
WORKERS=4
TOP_K=100
OUTPUT="bench/beir_results.json"
CONFIG="configs/bench.yaml"
SERVER_BIN="$ROOT/search-server"
SERVER_LOG="$ROOT/logs/beir_server.log"
BEIR_DATA="$ROOT/data/beir"
READY_TIMEOUT=120
POLL_INTERVAL=2
SKIP_BUILD=0
KEEP_DATA=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets)    DATASETS="$2";    shift 2 ;;
        --workers)     WORKERS="$2";     shift 2 ;;
        --top-k)       TOP_K="$2";       shift 2 ;;
        --output)      OUTPUT="$2";      shift 2 ;;
        --config)      CONFIG="$2";      shift 2 ;;
        --skip-build)  SKIP_BUILD=1;     shift ;;
        --keep-data)   KEEP_DATA=1;      shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$ROOT"

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ "$KEEP_DATA" -eq 0 && -d "$BEIR_DATA" ]]; then
        echo "cleaning up $BEIR_DATA ..."
        rm -rf "$BEIR_DATA"
    fi
}
trap cleanup EXIT

# 1. build
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    if [[ ! -f "$SERVER_BIN" ]] || find cmd/server internal -newer "$SERVER_BIN" -name '*.go' | grep -q .; then
        echo "[1/4] building search-server..."
        CGO_ENABLED=1 go build -o "$SERVER_BIN" ./cmd/server/
    else
        echo "[1/4] binary up-to-date, skipping build"
    fi
else
    echo "[1/4] skipping build (--skip-build)"
fi

# 2. clean data dir, stop any leftover server
echo "[2/4] preparing clean data directory..."
pkill -f "search-server" 2>/dev/null || true
sleep 1

rm -rf "$BEIR_DATA"
mkdir -p "$BEIR_DATA"
mkdir -p "$(dirname "$SERVER_LOG")"

# 3. start server
echo "[3/4] starting server (config=$CONFIG, data=$BEIR_DATA)..."

# Override data_dir via env so we use an isolated beir-specific directory.
SEARCH_STORAGE_DATA_DIR="$BEIR_DATA" \
    "$SERVER_BIN" -config "$CONFIG" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "    server PID: $SERVER_PID"

echo "    waiting for ready..."
ELAPSED=0
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process died — check $SERVER_LOG" >&2
        exit 1
    fi
    RESP=$(curl -s http://localhost:8080/health 2>/dev/null || true)
    READY=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ready',''))" 2>/dev/null || true)
    if [[ "$READY" == "True" || "$READY" == "true" ]]; then
        echo "    ready after ${ELAPSED}s"
        break
    fi
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
    if [[ $ELAPSED -ge $READY_TIMEOUT ]]; then
        echo "ERROR: server not ready after ${READY_TIMEOUT}s — check $SERVER_LOG" >&2
        exit 1
    fi
    [[ $((ELAPSED % 10)) -ne 0 ]] || echo "    still starting... ${ELAPSED}s"
done

# 4. run BEIR eval
echo "[4/4] running BEIR eval (datasets=$DATASETS, workers=$WORKERS, top_k=$TOP_K)..."
source bench/.venv/bin/activate
python3 bench/run_beir.py \
    --datasets "$DATASETS" \
    --workers  "$WORKERS" \
    --top-k    "$TOP_K" \
    --server   "http://localhost:8080" \
    --output   "$OUTPUT"

echo ""
echo "Results written to $OUTPUT"
