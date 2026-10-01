package api_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"search-eval-platform/internal/api"
	"search-eval-platform/internal/cluster"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/shard"
)

// newShardNode starts an in-process shard node that owns the given global
// shard IDs (reported via /health local_shards, like cluster.local_shards).
func newShardNode(t *testing.T, owns []int) (addr string, sm *shard.ShardManager) {
	addr, sm, _ = newCountingShardNode(t, owns)
	return addr, sm
}

// newCountingShardNode is newShardNode that also counts requests per path.
func newCountingShardNode(t *testing.T, owns []int) (addr string, sm *shard.ShardManager, hits *pathCounter) {
	t.Helper()
	sm, err := shard.NewShardManager(len(owns), t.TempDir(), scoring.NewBM25(1.2, 0.75), shard.DefaultTieredMergePolicy())
	if err != nil {
		t.Fatal(err)
	}
	if err := sm.Start(); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	h := api.NewHandler(sm, "shard", 1.2, 0.75, "")
	h.Register(mux)
	h.RegisterInternal(mux)
	h.SetLocalShards(owns)
	h.SetReady()
	hits = &pathCounter{n: map[string]int{}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.add(r.URL.Path)
		mux.ServeHTTP(w, r)
	}))
	t.Cleanup(func() { srv.Close(); sm.Close() })
	return strings.TrimPrefix(srv.URL, "http://"), sm, hits
}

type pathCounter struct {
	mu sync.Mutex
	n  map[string]int
}

func (c *pathCounter) add(path string) { c.mu.Lock(); c.n[path]++; c.mu.Unlock() }
func (c *pathCounter) get(path string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n[path]
}

// A coordinator configured only with static node addresses must learn shard
// ownership from the nodes and route writes and searches to them. Before the
// fix every write failed with "ring: no primary for shard N".
func TestCoordinatorWithStaticNodesIndexesAndSearches(t *testing.T) {
	addrA, smA := newShardNode(t, []int{0, 1})
	addrB, smB := newShardNode(t, []int{2, 3})

	ring := cluster.NewRing(4)
	if n := cluster.NewStaticMemberWatcher(ring, []string{addrA, addrB}).Refresh(context.Background()); n != 2 {
		t.Fatalf("discovered %d nodes, want 2", n)
	}
	client := cluster.NewClient(ring, "score", 150)
	mux := http.NewServeMux()
	coord := api.NewCoordinatorHandler(client, "coordinator", 1.2, 0.75, "")
	coord.Register(mux)
	coord.SetReady()
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	const n = 40
	for i := 0; i < n; i++ {
		body := fmt.Sprintf(`{"id":"doc-%d","text":"coordinated platypus number %d"}`, i, i)
		resp, err := http.Post(srv.URL+"/index", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("POST /index doc-%d via coordinator: status %d", i, resp.StatusCode)
		}
	}

	resp, err := http.Get(srv.URL + "/search?top_k=100&q=" + url.QueryEscape("coordinated platypus"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /search via coordinator: status %d", resp.StatusCode)
	}
	var out struct {
		Results []struct {
			DocID   string `json:"doc_id"`
			Rank    int    `json:"rank"`
			Snippet string `json:"snippet"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if len(out.Results) != n {
		t.Fatalf("coordinator search returned %d docs, want %d", len(out.Results), n)
	}
	for i, r := range out.Results {
		if r.Rank != i+1 {
			t.Fatalf("result %d (%s) has rank %d, want %d", i, r.DocID, r.Rank, i+1)
		}
	}
	if out.Results[0].Snippet == "" {
		t.Error("coordinator search dropped snippets")
	}

	// no_snippet=1 must be honoured through the coordinator too.
	resp2, err := http.Get(srv.URL + "/search?no_snippet=1&q=" + url.QueryEscape("coordinated platypus"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	var out2 struct {
		Results []struct {
			Snippet string `json:"snippet"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp2.Body).Decode(&out2); err != nil {
		t.Fatal(err)
	}
	for _, r := range out2.Results {
		if r.Snippet != "" {
			t.Fatal("coordinator returned snippets despite no_snippet=1")
		}
	}
	// Both nodes must have received part of the corpus.
	for name, sm := range map[string]*shard.ShardManager{"A": smA, "B": smB} {
		total := 0
		for _, c := range sm.PerShardDocCount() {
			total += c
		}
		if total == 0 || total == n {
			t.Errorf("node %s holds %d of %d docs; want a share of them", name, total, n)
		}
	}
}

// Bulk ingest through the coordinator must forward one batch per shard node,
// not one HTTP request per document.
func TestCoordinatorBulkForwardsOneBatchPerNode(t *testing.T) {
	addrA, _, hitsA := newCountingShardNode(t, []int{0, 1})
	addrB, _, hitsB := newCountingShardNode(t, []int{2, 3})
	ring := cluster.NewRing(4)
	cluster.NewStaticMemberWatcher(ring, []string{addrA, addrB}).Refresh(context.Background())
	mux := http.NewServeMux()
	coord := api.NewCoordinatorHandler(cluster.NewClient(ring, "score", 150), "coordinator", 1.2, 0.75, "")
	coord.Register(mux)
	coord.SetReady()
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	const n = 200
	var body strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&body, `{"id":"b-%d","text":"bulk wombat %d"}`+"\n", i, i)
	}
	resp, err := http.Post(srv.URL+"/index/bulk", "application/x-ndjson", strings.NewReader(body.String()))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct{ Indexed, Failed int }
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Indexed != n || out.Failed != 0 {
		t.Fatalf("bulk via coordinator: indexed=%d failed=%d, want %d/0", out.Indexed, out.Failed, n)
	}
	for name, h := range map[string]*pathCounter{"A": hitsA, "B": hitsB} {
		if got := h.get("/index/bulk"); got != 1 {
			t.Errorf("node %s got %d /index/bulk requests, want 1", name, got)
		}
		if got := h.get("/index"); got != 0 {
			t.Errorf("node %s got %d single-doc /index requests, want 0", name, got)
		}
	}

	sr, err := http.Get(srv.URL + "/search?top_k=500&q=wombat")
	if err != nil {
		t.Fatal(err)
	}
	defer sr.Body.Close()
	var res struct {
		Results []json.RawMessage `json:"results"`
	}
	if err := json.NewDecoder(sr.Body).Decode(&res); err != nil {
		t.Fatal(err)
	}
	if len(res.Results) != n {
		t.Fatalf("search after bulk found %d docs, want %d", len(res.Results), n)
	}
}
