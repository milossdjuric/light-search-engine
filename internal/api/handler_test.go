package api_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"search-eval-platform/internal/api"
	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/internal/search"
)

func newTestServer(t *testing.T) *httptest.Server {
	t.Helper()
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()
	shards, err := search.NewShardManager(2, t.TempDir(), scorer, policy)
	if err != nil {
		t.Fatalf("NewShardManager: %v", err)
	}
	if err := shards.Start(); err != nil {
		t.Fatalf("shards.Start: %v", err)
	}

	mux := http.NewServeMux()
	h := api.NewHandler(shards, "shard", 1.2, 0.75, "")
	h.Register(mux)
	h.RegisterInternal(mux)
	h.SetReady()

	srv := httptest.NewServer(mux)
	t.Cleanup(func() {
		srv.Close()
		shards.Close()
	})
	return srv
}

func TestHandlerHealth(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("health: want 200, got %d", resp.StatusCode)
	}
	var body map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&body)
	if body["status"] != "ok" {
		t.Errorf("health status: want 'ok', got %v", body["status"])
	}
}

func TestHandlerIndexSingleDoc(t *testing.T) {
	srv := newTestServer(t)
	body := `{"id":"d1","text":"hello world search engine"}`
	resp, err := http.Post(srv.URL+"/index", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("index: want 200, got %d", resp.StatusCode)
	}
	var result map[string]int
	json.NewDecoder(resp.Body).Decode(&result)
	if result["indexed"] != 1 {
		t.Errorf("indexed: want 1, got %d", result["indexed"])
	}
}

func TestHandlerIndexArray(t *testing.T) {
	srv := newTestServer(t)
	body := `[{"id":"a1","text":"first document"},{"id":"a2","text":"second document"}]`
	resp, err := http.Post(srv.URL+"/index", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("want 200, got %d", resp.StatusCode)
	}
	var result map[string]int
	json.NewDecoder(resp.Body).Decode(&result)
	if result["indexed"] != 2 {
		t.Errorf("indexed: want 2, got %d", result["indexed"])
	}
}

func TestHandlerIndexMissingFields(t *testing.T) {
	srv := newTestServer(t)
	body := `{"id":"","text":"no id provided"}`
	resp, err := http.Post(srv.URL+"/index", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("missing id: want 400, got %d", resp.StatusCode)
	}
}

func TestHandlerSearch(t *testing.T) {
	srv := newTestServer(t)
	// Index a doc first
	http.Post(srv.URL+"/index", "application/json",
		strings.NewReader(`{"id":"s1","text":"distributed search engine query ranking"}`))

	resp, err := http.Get(srv.URL + "/search?q=search+engine&top_k=5")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("search: want 200, got %d", resp.StatusCode)
	}
	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)
	results := result["results"].([]interface{})
	if len(results) == 0 {
		t.Error("expected at least one search result")
	}
}

func TestHandlerSearchMissingQuery(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Get(srv.URL + "/search")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("missing q: want 400, got %d", resp.StatusCode)
	}
}

func TestHandlerDelete(t *testing.T) {
	srv := newTestServer(t)
	// Index
	http.Post(srv.URL+"/index", "application/json",
		strings.NewReader(`{"id":"del1","text":"please delete me from index"}`))

	// Verify it appears
	resp, _ := http.Get(srv.URL + "/search?q=delete+me")
	var before map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&before)
	resp.Body.Close()
	if len(before["results"].([]interface{})) == 0 {
		t.Skip("document not found before delete (may be in buffer)")
	}

	// Delete
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodDelete,
		srv.URL+"/index/del1", nil)
	delResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	delResp.Body.Close()
	if delResp.StatusCode != http.StatusOK {
		t.Errorf("delete: want 200, got %d", delResp.StatusCode)
	}

	// Verify it no longer appears
	resp2, _ := http.Get(srv.URL + "/search?q=delete+me")
	var after map[string]interface{}
	json.NewDecoder(resp2.Body).Decode(&after)
	resp2.Body.Close()
	if results, ok := after["results"].([]interface{}); ok {
		for _, r := range results {
			doc, ok := r.(map[string]interface{})
			if ok && doc["doc_id"] == "del1" {
				t.Error("deleted doc still appears in search results")
			}
		}
	}
}

func TestHandlerBulkNDJSON(t *testing.T) {
	srv := newTestServer(t)
	ndjson := `{"id":"b1","text":"bulk document one alpha"}
{"id":"b2","text":"bulk document two beta"}
{"id":"b3","text":"bulk document three gamma"}
`
	resp, err := http.Post(srv.URL+"/index/bulk", "application/x-ndjson",
		bytes.NewBufferString(ndjson))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("bulk: want 200, got %d", resp.StatusCode)
	}
	var result map[string]int
	json.NewDecoder(resp.Body).Decode(&result)
	if result["indexed"] != 3 {
		t.Errorf("bulk indexed: want 3, got %d", result["indexed"])
	}
	if result["failed"] != 0 {
		t.Errorf("bulk failed: want 0, got %d", result["failed"])
	}
}

func TestHandlerSegments(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Get(srv.URL + "/segments")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("segments: want 200, got %d", resp.StatusCode)
	}
}

func TestHandlerShards(t *testing.T) {
	srv := newTestServer(t)

	// Index a doc so buffer_docs > 0.
	http.Post(srv.URL+"/index", "application/json",
		strings.NewReader(`{"id":"sh1","text":"shard status test"}`))

	resp, err := http.Get(srv.URL + "/shards")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("shards: want 200, got %d", resp.StatusCode)
	}

	var shards []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&shards); err != nil {
		t.Fatalf("shards decode: %v", err)
	}
	if len(shards) == 0 {
		t.Fatal("expected at least one shard")
	}
	for _, s := range shards {
		if _, ok := s["shard"]; !ok {
			t.Error("missing 'shard' field")
		}
		state, _ := s["state"].(string)
		if state == "" {
			t.Error("missing or empty 'state' field")
		}
		if _, ok := s["deleted_docs"]; !ok {
			t.Error("missing 'deleted_docs' field")
		}
		if _, ok := s["wal_seq"]; !ok {
			t.Error("missing 'wal_seq' field")
		}
		if _, ok := s["merge_debt"]; !ok {
			t.Error("missing 'merge_debt' field")
		}
	}
	// buffer_docs should be > 0 across at least one shard since we indexed a doc.
	totalBuffer := 0
	for _, s := range shards {
		if v, ok := s["buffer_docs"].(float64); ok {
			totalBuffer += int(v)
		}
	}
	if totalBuffer == 0 {
		t.Error("expected buffer_docs > 0 after indexing a doc")
	}
}

func TestHandlerFlush(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Post(srv.URL+"/index/flush", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("flush: want 200, got %d", resp.StatusCode)
	}
}

func TestHandlerMerge(t *testing.T) {
	srv := newTestServer(t)
	http.Post(srv.URL+"/index", "application/json", strings.NewReader(`{"id":"m1","text":"merge trigger doc"}`))

	resp, err := http.Post(srv.URL+"/index/merge", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("merge: want 200, got %d", resp.StatusCode)
	}
	var result map[string]string
	json.NewDecoder(resp.Body).Decode(&result)
	if result["status"] != "merge triggered" {
		t.Errorf("merge status: want 'merge triggered', got %v", result["status"])
	}
}

func TestHandlerMergeMaxSegments(t *testing.T) {
	srv := newTestServer(t)
	http.Post(srv.URL+"/index", "application/json", strings.NewReader(`{"id":"m2","text":"force merge doc"}`))

	resp, err := http.Post(srv.URL+"/index/merge?max_segments=1", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("force merge: want 200, got %d", resp.StatusCode)
	}
	var result map[string]string
	json.NewDecoder(resp.Body).Decode(&result)
	if result["status"] != "force merge complete" {
		t.Errorf("force merge status: want 'force merge complete', got %v", result["status"])
	}
}

func TestHandlerMergeInvalidMaxSegments(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Post(srv.URL+"/index/merge?max_segments=notanumber", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("invalid max_segments: want 400, got %d", resp.StatusCode)
	}
}

func TestHandlerAdminReset(t *testing.T) {
	srv := newTestServer(t)
	http.Post(srv.URL+"/index", "application/json", strings.NewReader(`{"id":"r1","text":"reset target doc"}`))
	http.Post(srv.URL+"/index/flush", "application/json", nil)

	resp, err := http.Post(srv.URL+"/admin/reset", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("reset: want 200, got %d", resp.StatusCode)
	}

	segResp, err := http.Get(srv.URL + "/segments")
	if err != nil {
		t.Fatal(err)
	}
	defer segResp.Body.Close()
	var segs []map[string]any
	json.NewDecoder(segResp.Body).Decode(&segs)
	if len(segs) != 0 {
		t.Errorf("expected no segments after reset, got %d", len(segs))
	}
}

func TestHandlerAdminGC(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Post(srv.URL+"/admin/gc", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("gc: want 200, got %d", resp.StatusCode)
	}
}

func TestHandlerInternalIndexAndSearch(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Post(srv.URL+"/internal/index", "application/json",
		strings.NewReader(`{"id":"int1","text":"internal shard index doc"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("internal index: want 200, got %d", resp.StatusCode)
	}

	searchResp, err := http.Post(srv.URL+"/internal/search", "application/json",
		strings.NewReader(`{"q":"internal shard","top_k":5}`))
	if err != nil {
		t.Fatal(err)
	}
	defer searchResp.Body.Close()
	if searchResp.StatusCode != http.StatusOK {
		t.Errorf("internal search: want 200, got %d", searchResp.StatusCode)
	}
	var result map[string]interface{}
	json.NewDecoder(searchResp.Body).Decode(&result)
	results, _ := result["results"].([]interface{})
	if len(results) == 0 {
		t.Error("expected internal search to find the internally indexed doc")
	}
}

func TestHandlerInternalSearchMissingQuery(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Post(srv.URL+"/internal/search", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("internal search missing q: want 400, got %d", resp.StatusCode)
	}
}

func TestHandlerInternalIndexMissingFields(t *testing.T) {
	srv := newTestServer(t)
	resp, err := http.Post(srv.URL+"/internal/index", "application/json", strings.NewReader(`{"id":"","text":""}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("internal index missing fields: want 400, got %d", resp.StatusCode)
	}
}

func newTestServerWithAPIKey(t *testing.T, apiKey string) *httptest.Server {
	t.Helper()
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()
	shards, err := search.NewShardManager(1, t.TempDir(), scorer, policy)
	if err != nil {
		t.Fatalf("NewShardManager: %v", err)
	}
	if err := shards.Start(); err != nil {
		t.Fatalf("shards.Start: %v", err)
	}

	mux := http.NewServeMux()
	h := api.NewHandler(shards, "shard", 1.2, 0.75, apiKey)
	h.Register(mux)
	h.RegisterInternal(mux)
	h.SetReady()

	srv := httptest.NewServer(mux)
	t.Cleanup(func() {
		srv.Close()
		shards.Close()
	})
	return srv
}

func TestHandlerRequireAuth(t *testing.T) {
	srv := newTestServerWithAPIKey(t, "secret123")
	body := `{"id":"auth1","text":"auth protected doc"}`

	// No Authorization header -> 401.
	resp, err := http.Post(srv.URL+"/index", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("no auth: want 401, got %d", resp.StatusCode)
	}

	// Wrong key -> 401.
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/index", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer wrongkey")
	resp2, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusUnauthorized {
		t.Errorf("wrong key: want 401, got %d", resp2.StatusCode)
	}

	// Correct key -> 200.
	req2, _ := http.NewRequest(http.MethodPost, srv.URL+"/index", strings.NewReader(body))
	req2.Header.Set("Authorization", "Bearer secret123")
	resp3, err := http.DefaultClient.Do(req2)
	if err != nil {
		t.Fatal(err)
	}
	resp3.Body.Close()
	if resp3.StatusCode != http.StatusOK {
		t.Errorf("correct key: want 200, got %d", resp3.StatusCode)
	}

	// Read endpoints bypass auth even without a key.
	healthResp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	healthResp.Body.Close()
	if healthResp.StatusCode != http.StatusOK {
		t.Errorf("health without auth: want 200, got %d", healthResp.StatusCode)
	}
}
