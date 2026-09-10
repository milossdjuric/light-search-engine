#!/usr/bin/env bash
# bench/sweep_segments.sh — sweep startup_merge_target and measure the tradeoff
#
# Usage: ./bench/sweep_segments.sh [--targets "1 2 3 5 8"] [--sample N] [--workers N]
#
# Requires: already-ingested index in data/ (no re-ingest between runs).
# Overrides startup_merge_target via SEARCH_INDEX_STARTUP_MERGE_TARGET env var.
# Results written to bench/sweep_segments_results.md (markdown table).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGETS=(1 2 3 5 8)
SAMPLE=500
WORKERS=4
TOP_K=100
CONFIG="configs/run.yaml"
SERVER_BIN="./search-server"
SERVER_LOG="logs/server_sweep.log"
OUTPUT_MD="bench/sweep_segments_results.md"
WARM_TIMEOUT=900
POLL_INTERVAL=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --targets) IFS=' ' read -ra TARGETS <<< "$2"; shift 2 ;;
        --sample)  SAMPLE="$2";  shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$ROOT"

# build
if [[ ! -f "$SERVER_BIN" ]] || find cmd/server -newer "$SERVER_BIN" -name '*.go' | grep -q .; then
    echo "Building search-server..."
    CGO_ENABLED=1 go build -o search-server ./cmd/server/
fi

mkdir -p logs bench
source bench/.venv/bin/activate

# result accumulator
RESULT_ROWS=()

# sweep
for N in "${TARGETS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════"
    echo "  startup_merge_target = ${N}"
    echo "════════════════════════════════════════════"

    # Kill any existing server
    pkill -f "search-server" 2>/dev/null || true
    sleep 1

    # Start server with target override
    SEARCH_INDEX_STARTUP_MERGE_TARGET=$N \
        "$SERVER_BIN" -config "$CONFIG" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "  server PID: $SERVER_PID"

    # Wait for warm=true, tracking startup time
    START_TS=$(date +%s)
    ELAPSED=0
    echo "  waiting for warm=true..."
    while true; do
        RESP=$(curl -s http://localhost:8080/health 2>/dev/null || true)
        WARM=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('warm',''))" 2>/dev/null || true)
        if [[ "$WARM" == "True" || "$WARM" == "true" ]]; then
            STARTUP_SECS=$(( $(date +%s) - START_TS ))
            echo "  warm after ${STARTUP_SECS}s"
            break
        fi
        sleep "$POLL_INTERVAL"
        ELAPSED=$((ELAPSED + POLL_INTERVAL))
        if [[ $ELAPSED -ge $WARM_TIMEOUT ]]; then
            echo "ERROR: server not warm after ${WARM_TIMEOUT}s" >&2
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
        SEGS=$(curl -s http://localhost:8080/segments 2>/dev/null \
            | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
        echo "  ${ELAPSED}s  segments=${SEGS}"
    done

    # Actual segment count after merge
    ACTUAL_SEGS=$(curl -s http://localhost:8080/segments 2>/dev/null \
        | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
    echo "  segments after merge: ${ACTUAL_SEGS}"

    # dd warm the merged segments
    DD_PIDS=()
    while IFS= read -r -d '' seg; do
        dd if="$seg" of=/dev/null bs=128M 2>/dev/null &
        DD_PIDS+=($!)
    done < <(find data -name "*.seg" -print0)
    wait "${DD_PIDS[@]}"

    # Run sampled eval
    echo "  running ${SAMPLE} sampled queries..."
    SWEEP_OUT="bench/sweep_n${N}_results.json"
    python3 bench/run_bench.py \
        --skip-ingest \
        --systems search-engine \
        --workers "$WORKERS" \
        --top-k "$TOP_K" \
        --sample "$SAMPLE" \
        --output "$SWEEP_OUT" 2>&1 | tail -20

    # Parse results
    if [[ -f "$SWEEP_OUT" ]]; then
        ROW=$(python3 - <<EOF
import json, sys
d = json.load(open("$SWEEP_OUT")).get("search-engine", {})
ndcg  = d.get("ndcg10",  0)
mrr   = d.get("mrr10",   0)
r100  = d.get("recall100", 0)
mean_ms = d.get("mean_ms", 0)
p95_ms  = d.get("p95_ms",  0)
qps     = d.get("qps",     0)
print(f"| {$N:2d} | {$ACTUAL_SEGS:7} | {$STARTUP_SECS:12d}s | {ndcg:.4f} | {mrr:.4f} | {r100:.4f} | {mean_ms:7.0f} ms | {p95_ms:6.0f} ms | {qps:5.1f} |")
EOF
)
        RESULT_ROWS+=("$ROW")
        echo "  → $ROW"
    fi

    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
done

# write markdown table
{
    echo "# Segment Count Sweep Results"
    echo ""
    echo "Sample: ${SAMPLE} queries  |  workers: ${WORKERS}  |  top_k: ${TOP_K}"
    echo ""
    echo "| N  | segs  | startup time | nDCG@10 | MRR@10 | R@100  | mean lat | p95 lat |  QPS  |"
    echo "|----|-------|--------------|---------|--------|--------|----------|---------|-------|"
    for row in "${RESULT_ROWS[@]}"; do
        echo "$row"
    done
} > "$OUTPUT_MD"

echo ""
echo "════════════════════════════════════════════"
echo "Results written to $OUTPUT_MD"
echo ""
cat "$OUTPUT_MD"
