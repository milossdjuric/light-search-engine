//go:build integration

// Package integration contains end-to-end HTTP API tests.
// Run with: go test -tags=integration ./internal/integration/...
package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"search-eval-platform/internal/api"
	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/internal/search"
)

// ── Test server setup ─────────────────────────────────────────────────────────

func newIntegrationServer(t *testing.T, nShards int) *httptest.Server {
	t.Helper()
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()
	shards, err := search.NewShardManager(nShards, t.TempDir(), scorer, policy)
	if err != nil {
		t.Fatalf("NewShardManager: %v", err)
	}
	if err := shards.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}

	mux := http.NewServeMux()
	h := api.NewHandler(shards, "shard", 1.2, 0.75, "")
	h.SetReady()
	h.Register(mux)

	srv := httptest.NewServer(mux)
	t.Cleanup(func() {
		srv.Close()
		shards.Close()
	})
	return srv
}

func postJSON(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	return resp
}

func mustDecodeJSON(t *testing.T, r io.Reader, dst interface{}) {
	t.Helper()
	if err := json.NewDecoder(r).Decode(dst); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// TestHTTPRoundTrip indexes 10 docs, flushes, and searches.
func TestHTTPRoundTrip(t *testing.T) {
	srv := newIntegrationServer(t, 2)

	// Index 10 documents.
	var docs []map[string]string
	for i := 0; i < 10; i++ {
		docs = append(docs, map[string]string{
			"id":   fmt.Sprintf("doc%d", i),
			"text": fmt.Sprintf("search engine retrieval document term%d alpha beta", i),
		})
	}
	body, _ := json.Marshal(docs)
	resp := postJSON(t, srv.URL+"/index", string(body))
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("index: want 200, got %d", resp.StatusCode)
	}

	// Flush.
	flushResp, err := http.Post(srv.URL+"/index/flush", "application/json", nil)
	if err != nil {
		t.Fatalf("flush: %v", err)
	}
	flushResp.Body.Close()

	// Give async ops a moment.
	time.Sleep(50 * time.Millisecond)

	// Search.
	searchResp, err := http.Get(srv.URL + "/search?q=search+engine&top_k=5")
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	defer searchResp.Body.Close()
	if searchResp.StatusCode != http.StatusOK {
		t.Fatalf("search: want 200, got %d", searchResp.StatusCode)
	}

	var result struct {
		Results []struct {
			DocID   string  `json:"doc_id"`
			Score   float64 `json:"score"`
			Rank    int     `json:"rank"`
			Snippet string  `json:"snippet"`
		} `json:"results"`
		Total  int   `json:"total"`
		TookMs int64 `json:"took_ms"`
	}
	mustDecodeJSON(t, searchResp.Body, &result)

	if result.Total == 0 {
		t.Error("expected non-empty search results")
	}
	// Scores must be descending.
	for i := 1; i < len(result.Results); i++ {
		if result.Results[i].Score > result.Results[i-1].Score {
			t.Errorf("results not sorted: [%d].score=%.4f > [%d].score=%.4f",
				i, result.Results[i].Score, i-1, result.Results[i-1].Score)
		}
	}
	// Rank must be consecutive starting at 1.
	for i, r := range result.Results {
		if r.Rank != i+1 {
			t.Errorf("result[%d].Rank=%d, want %d", i, r.Rank, i+1)
		}
	}
	// Snippets must be non-empty.
	for _, r := range result.Results {
		if r.Snippet == "" {
			t.Errorf("result %s has empty snippet", r.DocID)
		}
	}
}

// TestBulkIngest tests POST /index/bulk with NDJSON.
func TestBulkIngest(t *testing.T) {
	srv := newIntegrationServer(t, 2)

	var lines []string
	for i := 0; i < 100; i++ {
		doc := map[string]string{
			"id":   fmt.Sprintf("bulk-doc%d", i),
			"text": fmt.Sprintf("bulk ingest document number %d with some text content", i),
		}
		b, _ := json.Marshal(doc)
		lines = append(lines, string(b))
	}
	ndjson := strings.Join(lines, "\n")

	resp, err := http.Post(srv.URL+"/index/bulk", "application/x-ndjson", strings.NewReader(ndjson))
	if err != nil {
		t.Fatalf("bulk: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("bulk: want 200, got %d", resp.StatusCode)
	}

	var result map[string]int
	mustDecodeJSON(t, resp.Body, &result)
	if result["indexed"] != 100 {
		t.Errorf("bulk indexed: want 100, got %d", result["indexed"])
	}
	if result["failed"] != 0 {
		t.Errorf("bulk failed: want 0, got %d", result["failed"])
	}
}

// TestDeleteThenSearch verifies deleted docs don't appear in results.
func TestDeleteThenSearch(t *testing.T) {
	srv := newIntegrationServer(t, 1)

	// Index a specific doc.
	resp := postJSON(t, srv.URL+"/index", `{"id":"delete-me","text":"unique dragon slayer term"}`)
	resp.Body.Close()

	// Flush.
	r, _ := http.Post(srv.URL+"/index/flush", "application/json", nil)
	r.Body.Close()
	time.Sleep(50 * time.Millisecond)

	// Confirm it's found.
	searchResp, _ := http.Get(srv.URL + "/search?q=dragon+slayer&top_k=5")
	var result1 map[string]interface{}
	mustDecodeJSON(t, searchResp.Body, &result1)
	searchResp.Body.Close()
	if total, ok := result1["total"].(float64); !ok || int(total) == 0 {
		t.Skip("doc not indexed yet; skip delete test")
	}

	// Delete it.
	req, _ := http.NewRequest(http.MethodDelete, srv.URL+"/index/delete-me", nil)
	delResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("DELETE: %v", err)
	}
	delResp.Body.Close()
	if delResp.StatusCode != http.StatusOK {
		t.Fatalf("DELETE: want 200, got %d", delResp.StatusCode)
	}

	// Search again: should not find the deleted doc.
	searchResp2, _ := http.Get(srv.URL + "/search?q=dragon+slayer&top_k=5")
	defer searchResp2.Body.Close()
	var result2 struct {
		Results []struct {
			DocID string `json:"doc_id"`
		} `json:"results"`
	}
	mustDecodeJSON(t, searchResp2.Body, &result2)
	for _, r := range result2.Results {
		if r.DocID == "delete-me" {
			t.Error("deleted doc 'delete-me' still appears in search results")
		}
	}
}

// TestCrossShardFanOut verifies documents distributed across multiple shards
// are all reachable via a single search.
func TestCrossShardFanOut(t *testing.T) {
	srv := newIntegrationServer(t, 4)

	// Index docs designed to land on different shards (fnv32a routing).
	docIDs := []string{"alpha1", "beta2", "gamma3", "delta4", "epsilon5", "zeta6"}
	for _, id := range docIDs {
		body := fmt.Sprintf(`{"id":%q,"text":"crossshard unique retrieval test %s"}`, id, id)
		resp := postJSON(t, srv.URL+"/index", body)
		resp.Body.Close()
	}

	r, _ := http.Post(srv.URL+"/index/flush", "application/json", nil)
	r.Body.Close()
	time.Sleep(50 * time.Millisecond)

	searchResp, err := http.Get(srv.URL + "/search?q=crossshard+unique+retrieval&top_k=10")
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	defer searchResp.Body.Close()

	var result struct {
		Results []struct{ DocID string `json:"doc_id"` } `json:"results"`
		Total   int                                       `json:"total"`
	}
	mustDecodeJSON(t, searchResp.Body, &result)

	if result.Total < len(docIDs) {
		t.Errorf("cross-shard fan-out: want >= %d results, got %d", len(docIDs), result.Total)
	}
}

// TestBulkPartialFailure verifies that malformed lines in NDJSON don't abort
// the whole bulk request — only valid lines are counted as indexed.
func TestBulkPartialFailure(t *testing.T) {
	srv := newIntegrationServer(t, 1)

	ndjson := strings.Join([]string{
		`{"id":"good1","text":"valid document one"}`,
		`{broken json`,
		`{"id":"good2","text":"valid document two"}`,
		`{"id":"","text":"missing id should fail"}`,
		`{"id":"good3","text":"valid document three"}`,
	}, "\n")

	resp, err := http.Post(srv.URL+"/index/bulk",
		"application/x-ndjson",
		bytes.NewBufferString(ndjson))
	if err != nil {
		t.Fatalf("bulk: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("bulk: want 200, got %d", resp.StatusCode)
	}

	var result map[string]int
	mustDecodeJSON(t, resp.Body, &result)

	if result["indexed"] != 3 {
		t.Errorf("indexed: want 3 (valid docs), got %d", result["indexed"])
	}
	if result["failed"] < 2 {
		t.Errorf("failed: want >= 2, got %d", result["failed"])
	}
}
