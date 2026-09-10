#!/usr/bin/env bash
# bench/run_eval.sh — build, restart server (MADV_WILLNEED warmup), run eval
# Usage: ./bench/run_eval.sh [--workers N] [--top-k N] [--output FILE] [--config FILE]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKERS=4
TOP_K=100
OUTPUT="bench/engine_results.json"
CONFIG="configs/run.yaml"
SERVER_BIN="./search-server"
SERVER_LOG="logs/server.log"
WARM_TIMEOUT=600    # seconds to wait for warm=true (no startup merge; warmup-only is fast)
POLL_INTERVAL=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --workers) WORKERS="$2"; shift 2 ;;
        --top-k)   TOP_K="$2";   shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        --config)  CONFIG="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$ROOT"

# 1. build if binary is missing or stale
if [[ ! -f "$SERVER_BIN" ]] || find cmd/server -newer "$SERVER_BIN" -name '*.go' | grep -q .; then
    echo "[1/4] building search-server..."
    CGO_ENABLED=1 go build -o search-server ./cmd/server/
else
    echo "[1/4] binary up-to-date, skipping build"
fi

# 2. kill any running instance
echo "[2/4] stopping existing server (if any)..."
pkill -f "search-server" 2>/dev/null || true
sleep 1

# 3. start server, wait for warm=true, then dd-confirm page cache
# Why this order matters:
#   - Server startup allocates ~3 GB of Go heap (docID maps for 8.8M docs)
#   - If dd runs BEFORE startup, the OS evicts those page-cache pages to give
#     Go its heap → pages are cold again by the first query
#   - Running dd AFTER ready=true means Go has already claimed its heap;
#     the dd-loaded pages stay hot because nothing else needs that RAM
echo "[3/4] starting server..."
mkdir -p logs
"$SERVER_BIN" -config "$CONFIG" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "    server PID: $SERVER_PID"

    # Wait for warm=true — this means startup merge AND Go warmup are both done.
    # Do NOT poll for ready=true instead: ready is set before the startup merge
    # runs, so queries issued at ready=true hit unmapped/cold merged segments.
echo "    waiting for warm=true (startup merge + warmup)..."
ELAPSED=0
while true; do
    RESP=$(curl -s http://localhost:8080/health 2>/dev/null || true)
    WARM=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('warm',''))" 2>/dev/null || true)
    if [[ "$WARM" == "True" || "$WARM" == "true" ]]; then
        echo "    warm after ${ELAPSED}s"
        break
    fi
    # Show segment count while waiting so we can track merge progress.
    SEGS=$(curl -s http://localhost:8080/segments 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
    if [[ $ELAPSED -ge $WARM_TIMEOUT ]]; then
        echo "ERROR: server not warm after ${WARM_TIMEOUT}s" >&2
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    [[ $ELAPSED -le 10 ]] || echo "    merging/warming... ${ELAPSED}s  segments=${SEGS}"
done

# Belt-and-suspenders: dd the final merged segment files to ensure page cache
# is hot. Go warmup already did this, so dd hits the cache and finishes fast.
echo "    dd-confirming page cache (post-merge segments)..."
TOTAL_MB=0
DD_PIDS=()
while IFS= read -r -d '' seg; do
    MB=$(( $(stat -c%s "$seg") / 1048576 ))
    TOTAL_MB=$(( TOTAL_MB + MB ))
    dd if="$seg" of=/dev/null bs=128M 2>/dev/null &
    DD_PIDS+=($!)
done < <(find data -name "*.seg" -print0)
wait "${DD_PIDS[@]}"
echo "    confirmed ${TOTAL_MB} MB in page cache"

# 4. run eval
echo "[4/4] running eval (workers=$WORKERS, top_k=$TOP_K)..."
source bench/.venv/bin/activate
python3 bench/run_bench.py \
    --skip-ingest \
    --systems search-engine \
    --workers "$WORKERS" \
    --top-k "$TOP_K" \
    --output "$OUTPUT"

echo ""
echo "Results written to $OUTPUT"
