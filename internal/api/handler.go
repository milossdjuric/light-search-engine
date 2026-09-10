package api

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/bytedance/sonic"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"search-eval-platform/internal/cluster"
	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/internal/retrieval/bm25f"
	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/internal/retrieval/tfidf"
	"search-eval-platform/internal/search"
	"search-eval-platform/pkg/types"
)

// Handler holds the dependencies injected into all HTTP handlers.
type Handler struct {
	shards      *search.ShardManager // non-nil in standalone/shard modes
	client      *cluster.Client      // non-nil in coordinator mode
	mode        string
	bm25K1      float64
	bm25B       float64
	apiKey      string // if non-empty, write endpoints require Authorization: Bearer <key>
	localShards []int  // local shard IDs for this node; reported in /health for K8s discovery
	ready       atomic.Bool
	warm        atomic.Bool
}

// SetReady marks the handler as ready to serve search requests.
func (h *Handler) SetReady() { h.ready.Store(true) }

// SetLocalShards records the shard IDs owned by this node, reported via /health.
func (h *Handler) SetLocalShards(shards []int) { h.localShards = shards }

// SetWarm marks the handler as having completed targeted warmup.
func (h *Handler) SetWarm() { h.warm.Store(true) }

// ResetWarm clears the warm flag (called during background merge when
// merged segments are not yet warmed).
func (h *Handler) ResetWarm() { h.warm.Store(false) }

// NewHandler creates a Handler wired to the given ShardManager (standalone/shard mode).
func NewHandler(shards *search.ShardManager, mode string, k1, b float64, apiKey string) *Handler {
	return &Handler{shards: shards, mode: mode, bm25K1: k1, bm25B: b, apiKey: apiKey}
}

// NewCoordinatorHandler creates a Handler for coordinator mode.
func NewCoordinatorHandler(client *cluster.Client, mode string, k1, b float64, apiKey string) *Handler {
	return &Handler{client: client, mode: mode, bm25K1: k1, bm25B: b, apiKey: apiKey}
}

// requireAuth wraps a handler to enforce the API key when one is configured.
// Requests must include "Authorization: Bearer <key>". Read endpoints bypass this.
func (h *Handler) requireAuth(next http.HandlerFunc) http.HandlerFunc {
	if h.apiKey == "" {
		return next
	}
	return func(w http.ResponseWriter, r *http.Request) {
		const prefix = "Bearer "
		auth := r.Header.Get("Authorization")
		if len(auth) <= len(prefix) || auth[:len(prefix)] != prefix || auth[len(prefix):] != h.apiKey {
			WriteError(w, http.StatusUnauthorized, "unauthorized")
			return
		}
		next(w, r)
	}
}

// Register mounts all routes onto mux using Go 1.22 method+path patterns.
func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("POST /index", h.requireAuth(h.handleIndex))
	mux.HandleFunc("DELETE /index/{id}", h.requireAuth(h.handleDelete))
	mux.HandleFunc("POST /index/bulk", h.requireAuth(h.handleBulk))
	mux.HandleFunc("POST /index/flush", h.requireAuth(h.handleFlush))
	mux.HandleFunc("POST /index/merge", h.requireAuth(h.handleMerge))
	mux.HandleFunc("GET /search", h.handleSearch)
	mux.HandleFunc("GET /health", h.handleHealth)
	mux.HandleFunc("GET /segments", h.handleSegments)
	mux.HandleFunc("GET /shards", h.handleShards)
	mux.HandleFunc("POST /admin/reset", h.requireAuth(h.handleReset))
	mux.HandleFunc("POST /admin/gc", h.requireAuth(h.handleGC))
}

// RegisterInternal mounts internal (shard-node-only) routes.
// These endpoints are used by coordinator nodes and are not part of the
// public API.
func (h *Handler) RegisterInternal(mux *http.ServeMux) {
	mux.HandleFunc("POST /internal/index", h.handleInternalIndex)
	mux.HandleFunc("POST /internal/search", h.handleInternalSearch)
}

// POST /index
// Accepts a single document object OR a JSON array of documents.
// {"id":"...", "text":"...", "metadata":{...}}

type indexDocRequest struct {
	ID       string            `json:"id"`
	Text     string            `json:"text"`
	Fields   map[string]string `json:"fields,omitempty"`
	Metadata map[string]string `json:"metadata"`
}

func (h *Handler) handleIndex(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 32<<20)) // 32 MB
	if err != nil {
		WriteError(w, http.StatusBadRequest, "failed to read body")
		return
	}

	// Detect array vs object.
	trimmed := bytes.TrimSpace(body)
	var docs []indexDocRequest
	if len(trimmed) > 0 && trimmed[0] == '[' {
		if err := sonic.Unmarshal(trimmed, &docs); err != nil {
			WriteError(w, http.StatusBadRequest, "invalid JSON array: "+err.Error())
			return
		}
	} else {
		var single indexDocRequest
		if err := sonic.Unmarshal(trimmed, &single); err != nil {
			WriteError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
			return
		}
		docs = []indexDocRequest{single}
	}

	var batch []types.Document
	for _, d := range docs {
		if d.ID == "" || d.Text == "" {
			WriteError(w, http.StatusBadRequest, "each document requires non-empty id and text")
			return
		}
		batch = append(batch, types.Document{ID: d.ID, Text: d.Text, Fields: d.Fields, Metadata: d.Metadata})
	}

	if h.client != nil {
		// Coordinator mode: route each doc to the owning shard node.
		for _, doc := range batch {
			if err := h.client.IndexDoc(r.Context(), doc); err != nil {
				slog.Error("handleIndex coordinator", "err", err)
				WriteError(w, http.StatusInternalServerError, "index error: "+err.Error())
				return
			}
		}
	} else {
		if err := h.shards.IndexBatch(r.Context(), batch); err != nil {
			slog.Error("handleIndex", "err", err)
			WriteError(w, http.StatusInternalServerError, "index error: "+err.Error())
			return
		}
	}
	WriteJSON(w, http.StatusOK, map[string]int{"indexed": len(batch)})
}

// DELETE /index/{id}

func (h *Handler) handleDelete(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		WriteError(w, http.StatusBadRequest, "missing document id")
		return
	}

	if h.client != nil {
		if err := h.client.DeleteDoc(r.Context(), id); err != nil {
			slog.Error("handleDelete coordinator", "id", id, "err", err)
			WriteError(w, http.StatusInternalServerError, "delete error: "+err.Error())
			return
		}
	} else {
		if err := h.shards.DeleteDoc(r.Context(), id); err != nil {
			slog.Error("handleDelete", "id", id, "err", err)
			WriteError(w, http.StatusInternalServerError, "delete error: "+err.Error())
			return
		}
	}
	WriteJSON(w, http.StatusOK, map[string]string{"deleted": id})
}

// POST /index/bulk
// NDJSON stream: one JSON document object per line.

func (h *Handler) handleBulk(w http.ResponseWriter, r *http.Request) {
	var failed int
	var batch []types.Document

	scanner := bufio.NewScanner(io.LimitReader(r.Body, 256<<20)) // 256 MB
	scanner.Buffer(make([]byte, 1<<20), 1<<20)                    // 1 MB per line

	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var req indexDocRequest
		if err := sonic.Unmarshal(line, &req); err != nil {
			slog.Warn("handleBulk unmarshal", "err", err)
			failed++
			continue
		}
		if req.ID == "" || req.Text == "" {
			failed++
			continue
		}
		batch = append(batch, types.Document{ID: req.ID, Text: req.Text, Fields: req.Fields, Metadata: req.Metadata})
	}
	if err := scanner.Err(); err != nil {
		WriteError(w, http.StatusBadRequest, "error reading body: "+err.Error())
		return
	}

	indexed := 0
	if len(batch) > 0 {
		if h.client != nil {
			// Coordinator: no batch API — fan out docs one at a time.
			for _, doc := range batch {
				if e := h.client.IndexDoc(r.Context(), doc); e != nil {
					slog.Error("handleBulk coordinator IndexDoc", "doc_id", doc.ID, "err", e)
					failed++
				} else {
					indexed++
				}
			}
		} else {
			walMode := r.URL.Query().Get("wal") // "off" skips WAL; "" or "sync"/"async" use WAL
			var indexErr error
			if walMode == "off" {
				indexErr = h.shards.IndexBatchNoWAL(r.Context(), batch)
			} else {
				indexErr = h.shards.IndexBatch(r.Context(), batch)
			}
			if indexErr != nil {
				slog.Error("handleBulk IndexBatch", "err", indexErr)
				WriteError(w, http.StatusInternalServerError, "index error: "+indexErr.Error())
				return
			}
			indexed = len(batch)
		}
	}

	WriteJSON(w, http.StatusOK, map[string]int{"indexed": indexed, "failed": failed})
}

// POST /index/flush

func (h *Handler) handleFlush(w http.ResponseWriter, r *http.Request) {
	if h.shards == nil {
		WriteJSON(w, http.StatusOK, map[string]string{"status": "flushed"})
		return
	}
	if err := h.shards.Flush(r.Context()); err != nil {
		WriteError(w, http.StatusInternalServerError, "flush error: "+err.Error())
		return
	}
	WriteJSON(w, http.StatusOK, map[string]string{"status": "flushed"})
}

// POST /index/merge
// Flushes all buffers, then merges segments.
//
// Query params:
//
//	max_segments=N  (optional) — block until each shard has ≤ N segments.
//	                             If omitted, signals the background merge goroutine
//	                             and returns immediately (async, existing behaviour).
func (h *Handler) handleMerge(w http.ResponseWriter, r *http.Request) {
	if h.shards == nil {
		WriteJSON(w, http.StatusOK, map[string]string{"status": "merge triggered"})
		return
	}
	if err := h.shards.Flush(r.Context()); err != nil {
		WriteError(w, http.StatusInternalServerError, "pre-merge flush error: "+err.Error())
		return
	}

	if ms := r.URL.Query().Get("max_segments"); ms != "" {
		n, err := strconv.Atoi(ms)
		if err != nil || n < 1 {
			WriteError(w, http.StatusBadRequest, "max_segments must be a positive integer")
			return
		}
		if err := h.shards.ForceMerge(r.Context(), n); err != nil {
			WriteError(w, http.StatusInternalServerError, "force merge error: "+err.Error())
			return
		}
		WriteJSON(w, http.StatusOK, map[string]string{"status": "force merge complete"})
		return
	}

	h.shards.TriggerMerge()
	WriteJSON(w, http.StatusOK, map[string]string{"status": "merge triggered"})
}

// GET /search
// Query params: q, top_k (default 10), retriever (bm25|tfidf, default bm25)

func (h *Handler) handleSearch(w http.ResponseWriter, r *http.Request) {
	if !h.ready.Load() {
		WriteError(w, http.StatusServiceUnavailable, "server is starting up")
		return
	}

	q := r.URL.Query().Get("q")
	if q == "" {
		WriteError(w, http.StatusBadRequest, "missing query parameter 'q'")
		return
	}

	topK := 10
	if v := r.URL.Query().Get("top_k"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			topK = n
		}
	}

	retrieverLabel := r.URL.Query().Get("retriever")
	if retrieverLabel == "" {
		retrieverLabel = "bm25"
	}

	noSnippet := r.URL.Query().Get("no_snippet") == "1"

	start := time.Now()
	var results []types.SearchResult
	var err error
	degraded := false

	if h.client != nil {
		// Coordinator mode: fan out to shard nodes.
		results, degraded, err = h.client.Search(r.Context(), q, topK, retrieverLabel, "")
	} else {
		scorer := h.scorerForRetriever(retrieverLabel)
		var shardDegraded bool
		results, shardDegraded, err = h.shards.Search(r.Context(), q, topK, scorer, noSnippet)
		if shardDegraded {
			degraded = true
		}
	}
	elapsed := time.Since(start)

	if err != nil {
		WriteError(w, http.StatusInternalServerError, "search error: "+err.Error())
		return
	}

	if degraded {
		w.Header().Set("X-Degraded-Shards", "true")
	}

	WriteJSON(w, http.StatusOK, SearchResponse{
		Results: results,
		Total:   len(results),
		TookMs:  elapsed.Milliseconds(),
	})
}

// scorerForRetriever returns the index.Scorer for the given retriever name.
// Defaults to BM25 with configured parameters.
func (h *Handler) scorerForRetriever(name string) index.Scorer {
	switch name {
	case "tfidf":
		return tfidf.NewScorerOnly()
	case "bm25f":
		return bm25f.NewScorerOnly(h.bm25K1)
	default: // "bm25" or ""
		return bm25.NewScorerOnly(h.bm25K1, h.bm25B)
	}
}

// GET /health

func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	shardCount := 0
	if h.shards != nil {
		shardCount = h.shards.NumShards()
	}
	ready := h.ready.Load()
	status := http.StatusOK
	if !ready {
		status = http.StatusServiceUnavailable
	}
	WriteJSON(w, status, HealthResponse{
		Status:      "ok",
		Mode:        h.mode,
		Shards:      shardCount,
		Ready:       ready,
		Warm:        h.warm.Load(),
		LocalShards: h.localShards,
	})
}

// GET /segments

func (h *Handler) handleSegments(w http.ResponseWriter, r *http.Request) {
	if h.shards == nil {
		WriteJSON(w, http.StatusOK, []SegmentInfo{})
		return
	}
	recs := h.shards.SegmentRecords()
	out := make([]SegmentInfo, 0, len(recs))
	for _, r := range recs {
		out = append(out, SegmentInfo{
			ShardID:   r.ShardID,
			SegmentID: r.SegmentID,
			Level:     r.Level,
			DocCount:  r.DocCount,
			SizeBytes: r.SizeBytes,
			Path:      r.Path,
			FlushSeq:  r.FlushSeq,
		})
	}
	WriteJSON(w, http.StatusOK, out)
}

// GET /shards

func (h *Handler) handleShards(w http.ResponseWriter, r *http.Request) {
	if h.shards == nil {
		WriteJSON(w, http.StatusOK, []ShardInfo{})
		return
	}

	nShards := h.shards.NumShards()
	recs := h.shards.SegmentRecords()
	docCounts := h.shards.PerShardDocCount()
	statuses := h.shards.ShardStatuses()

	type agg struct {
		sizeBytes int64
		segments  int
	}
	aggs := make([]agg, nShards)
	for _, rec := range recs {
		idx, err := strconv.Atoi(strings.TrimPrefix(rec.ShardID, "shard"))
		if err != nil || idx < 0 || idx >= nShards {
			continue
		}
		aggs[idx].sizeBytes += rec.SizeBytes
		aggs[idx].segments++
	}

	out := make([]ShardInfo, nShards)
	for i := range out {
		st := statuses[i]
		out[i] = ShardInfo{
			Shard:       fmt.Sprintf("shard%d", i),
			State:       st.State,
			Docs:        docCounts[i],
			SizeBytes:   aggs[i].sizeBytes,
			Segments:    aggs[i].segments,
			DeletedDocs: st.DeletedDocs,
			BufferDocs:  st.BufferDocs,
			WALSeq:      st.WALSeq,
			MergeDebt:   st.MergeDebt,
		}
	}
	WriteJSON(w, http.StatusOK, out)
}

// POST /admin/reset
// Wipes all indexed data: in-memory buffers, WAL, segment files, and the
// segment manifest. The server stays running; documents can be re-indexed
// immediately. Only available in shard / standalone mode.

func (h *Handler) handleReset(w http.ResponseWriter, r *http.Request) {
	if h.shards == nil {
		WriteJSON(w, http.StatusOK, map[string]string{"status": "reset"})
		return
	}
	if err := h.shards.Reset(r.Context()); err != nil {
		slog.Error("handleReset", "err", err)
		WriteError(w, http.StatusInternalServerError, "reset error: "+err.Error())
		return
	}
	WriteJSON(w, http.StatusOK, map[string]string{"status": "reset"})
}

// POST /admin/gc
// Recycles the builder temp files in the free-buffer pool across all shards.
// Call after a large ingest to release OS disk blocks accumulated by temp files
// during flush cycles. The server stays running; indexing continues normally.

func (h *Handler) handleGC(w http.ResponseWriter, r *http.Request) {
	if h.shards == nil {
		WriteJSON(w, http.StatusOK, map[string]string{"status": "gc"})
		return
	}
	if err := h.shards.GC(); err != nil {
		slog.Error("handleGC", "err", err)
		WriteError(w, http.StatusInternalServerError, "gc error: "+err.Error())
		return
	}
	WriteJSON(w, http.StatusOK, map[string]string{"status": "gc"})
}

// Internal endpoints (shard nodes only)

// handleInternalIndex accepts a document from the coordinator and indexes it
// directly (without routing through ShardManager's FNV partitioning — the
// coordinator already determined the owning shard).
func (h *Handler) handleInternalIndex(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 32<<20))
	if err != nil {
		WriteError(w, http.StatusBadRequest, "failed to read body")
		return
	}
	var req indexDocRequest
	if err := sonic.Unmarshal(bytes.TrimSpace(body), &req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.ID == "" || req.Text == "" {
		WriteError(w, http.StatusBadRequest, "id and text required")
		return
	}
	doc := types.Document{ID: req.ID, Text: req.Text, Fields: req.Fields, Metadata: req.Metadata}
	if err := h.shards.IndexDoc(r.Context(), doc); err != nil {
		WriteError(w, http.StatusInternalServerError, "index error: "+err.Error())
		return
	}
	WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// internalSearchRequest carries search parameters plus an optional precomputed
type internalSearchRequest struct {
	Q         string `json:"q"`
	TopK      int    `json:"top_k"`
	Retriever string `json:"retriever"`
}

// handleInternalSearch performs a local search on this shard node, optionally
// using a precomputed global IDF map supplied by the coordinator.
func (h *Handler) handleInternalSearch(w http.ResponseWriter, r *http.Request) {
	var req internalSearchRequest
	if err := sonic.ConfigDefault.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if req.Q == "" {
		WriteError(w, http.StatusBadRequest, "q required")
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}

	scorer := h.scorerForRetriever(req.Retriever)
	start := time.Now()
	results, _, err := h.shards.Search(r.Context(), req.Q, req.TopK, scorer)
	elapsed := time.Since(start)

	if err != nil {
		WriteError(w, http.StatusInternalServerError, "search error: "+err.Error())
		return
	}
	WriteJSON(w, http.StatusOK, SearchResponse{
		Results: results,
		Total:   len(results),
		TookMs:  elapsed.Milliseconds(),
	})
}

