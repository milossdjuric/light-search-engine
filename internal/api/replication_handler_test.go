package api_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"search-eval-platform/internal/api"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/shard"
)

func newReplTestServer(t *testing.T, configure func(*api.Handler)) (*httptest.Server, *shard.ShardManager) {
	t.Helper()
	shards, err := shard.NewShardManager(1, t.TempDir(), scoring.NewBM25(1.2, 0.75), shard.DefaultTieredMergePolicy())
	if err != nil {
		t.Fatal(err)
	}
	if err := shards.Start(); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	h := api.NewHandler(shards, "shard", 1.2, 0.75, "")
	configure(h)
	h.Register(mux)
	h.RegisterInternal(mux)
	h.SetReady()
	srv := httptest.NewServer(mux)
	t.Cleanup(func() { srv.Close(); shards.Close() })
	return srv, shards
}

// TestReadOnlyReplicaRejectsWrites verifies a replica refuses client writes
// (they'd diverge from its primary and collide with its seq space) while
// still serving reads and local maintenance.
func TestReadOnlyReplicaRejectsWrites(t *testing.T) {
	srv, _ := newReplTestServer(t, func(h *api.Handler) { h.SetReadOnly("read-only replica of primary:9090") })

	writes := []struct{ method, path, body string }{
		{http.MethodPost, "/index", `{"id":"a","text":"x"}`},
		{http.MethodPost, "/index/bulk", `{"id":"a","text":"x"}` + "\n"},
		{http.MethodDelete, "/index/a", ""},
		{http.MethodPost, "/admin/reset", ""},
		{http.MethodPost, "/internal/index", `{"id":"a","text":"x"}`},
	}
	for _, w := range writes {
		req, _ := http.NewRequest(w.method, srv.URL+w.path, strings.NewReader(w.body))
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusServiceUnavailable {
			t.Errorf("%s %s on replica: status %d, want 503", w.method, w.path, resp.StatusCode)
		}
	}
	for _, path := range []string{"/search?q=x", "/health"} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("GET %s on replica: status %d, want 200", path, resp.StatusCode)
		}
	}
	resp, err := http.Post(srv.URL+"/index/flush", "application/json", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("POST /index/flush on replica: status %d, want 200 (local maintenance stays allowed)", resp.StatusCode)
	}
}

// TestHealthReportsReplication verifies /health includes the replication
// status provider's output.
func TestHealthReportsReplication(t *testing.T) {
	srv, _ := newReplTestServer(t, func(h *api.Handler) {
		h.SetReplicationStatus(func() any { return map[string]any{"role": "replica", "dead_letters": 3} })
	})
	resp, err := http.Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body struct {
		Replication map[string]any `json:"replication"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body.Replication["role"] != "replica" || body.Replication["dead_letters"] != float64(3) {
		t.Errorf("health replication = %v, want role=replica dead_letters=3", body.Replication)
	}
}

// TestPrimaryIgnoresWALOff verifies a replication primary logs ?wal=off bulk
// writes to the WAL anyway — unlogged writes would never reach replicas.
func TestPrimaryIgnoresWALOff(t *testing.T) {
	srv, shards := newReplTestServer(t, func(h *api.Handler) { h.SetRequireWAL(true) })
	var hooked int
	shards.Shard(0).SetWALHook(func(shard.WALEntry) { hooked++ })

	resp, err := http.Post(srv.URL+"/index/bulk?wal=off", "application/x-ndjson",
		strings.NewReader(`{"id":"a","text":"x"}`+"\n"+`{"id":"b","text":"y"}`+"\n"))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("bulk status %d", resp.StatusCode)
	}
	if hooked != 2 {
		t.Errorf("WAL hook saw %d entries, want 2 — ?wal=off must be ignored on a primary", hooked)
	}
}
