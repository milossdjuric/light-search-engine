package api

import (
	"net/http"

	"github.com/bytedance/sonic"

	"search-eval-platform/pkg/types"
)

// SearchResponse is the JSON body returned by GET /search.
type SearchResponse struct {
	Results []types.SearchResult `json:"results"`
	Total   int                  `json:"total"`
	TookMs  int64                `json:"took_ms"`
}

// SegmentInfo is one element of the GET /segments response.
type SegmentInfo struct {
	ShardID    string `json:"shard_id"`
	SegmentID  string `json:"segment_id"`
	Level      int    `json:"level"`
	DocCount   int    `json:"doc_count"`
	SizeBytes  int64  `json:"size_bytes"`
	Path       string `json:"path"`
	FlushSeq   int64  `json:"flush_seq"`
}

// ShardInfo is one element of the GET /shards response.
type ShardInfo struct {
	Shard       string `json:"shard"`
	State       string `json:"state"`
	Docs        int    `json:"docs"`
	SizeBytes   int64  `json:"size_bytes"`
	Segments    int    `json:"segments"`
	DeletedDocs int    `json:"deleted_docs"`
	BufferDocs  int    `json:"buffer_docs"`
	WALSeq      int64  `json:"wal_seq"`
	MergeDebt   int    `json:"merge_debt"`
}

// HealthResponse is the JSON body returned by GET /health.
type HealthResponse struct {
	Status      string `json:"status"`
	Mode        string `json:"mode"`
	Shards      int    `json:"shards"`
	Ready       bool   `json:"ready"`
	Warm        bool   `json:"warm"`
	LocalShards []int  `json:"local_shards,omitempty"` // present on shard nodes; used by K8s membership watcher
}

// ErrorResponse is the JSON body for error responses.
type ErrorResponse struct {
	Error string `json:"error"`
}

// WriteJSON serialises v as JSON and writes it with the given HTTP status code.
func WriteJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := sonic.ConfigDefault.NewEncoder(w).Encode(v); err != nil {
		// Headers already sent; best-effort log via stderr is fine here.
		_ = err
	}
}

// WriteError writes a JSON error response with the given HTTP status code.
func WriteError(w http.ResponseWriter, status int, msg string) {
	WriteJSON(w, status, ErrorResponse{Error: msg})
}
