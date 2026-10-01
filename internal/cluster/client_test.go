package cluster

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"search-eval-platform/pkg/types"
)

// TestNodeForRetriesPrimaryAfterResetTimeout verifies that once a tripped
// primary's circuit breaker has been open longer than its reset timeout,
// nodeFor actually gives it another chance via Allow() — the only method
// that can transition Open->HalfOpen — instead of permanently routing
// everything to a replica because a stale IsOpen() read never advances the
// breaker's state on its own.
func TestNodeForRetriesPrimaryAfterResetTimeout(t *testing.T) {
	ring := NewRing(1)
	primary := &NodeMeta{NodeID: "primary", HTTPAddr: "primary:8080", Shards: []int{0}}
	replica := &NodeMeta{NodeID: "replica", HTTPAddr: "replica:8080", Shards: []int{0}}
	ring.Rebuild([]*NodeMeta{primary, replica})

	c := NewClient(ring, "score", 150)

	// Trip the primary's breaker with a short reset timeout so the test
	// doesn't need to wait out the real 30s default.
	cb := NewCircuitBreaker(3, 10*time.Millisecond)
	for i := 0; i < 3; i++ {
		allowed, done := cb.Allow()
		if !allowed {
			t.Fatalf("setup: Allow() unexpectedly denied on failure %d", i)
		}
		done(false)
	}
	if cb.State() != "open" {
		t.Fatal("setup: breaker did not trip open after 3 failures")
	}
	c.cbMu.Lock()
	c.breakers[primary.NodeID] = cb
	c.cbMu.Unlock()

	time.Sleep(20 * time.Millisecond) // past the 10ms reset timeout

	node, done, degraded, err := c.nodeFor(0)
	if err != nil {
		t.Fatalf("nodeFor returned error: %v", err)
	}
	if node.NodeID != primary.NodeID {
		t.Errorf("nodeFor returned %q, want primary %q — a recovered primary should get a HalfOpen probe via Allow(), not be permanently skipped",
			node.NodeID, primary.NodeID)
	}
	if degraded {
		t.Error("nodeFor reported degraded=true, want false — this is the primary's own probe, not a replica fallback")
	}
	if done == nil {
		t.Fatal("nodeFor returned a nil done callback for an allowed request")
	}
	done(true)
}

// TestClientForwardsAPIKeyOnWrites verifies the coordinator presents the
// cluster API key when it forwards index/delete to a shard's public write
// endpoints — otherwise every coordinator write 401s once shards set
// server.api_key.
func TestClientForwardsAPIKeyOnWrites(t *testing.T) {
	got := map[string]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got[r.Method] = r.Header.Get("Authorization")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	ring := NewRing(1)
	ring.Rebuild([]*NodeMeta{{NodeID: "n1", HTTPAddr: strings.TrimPrefix(srv.URL, "http://"), Shards: []int{0}}})
	c := NewClient(ring, "score", 150)
	c.SetAPIKey("secret")

	if err := c.IndexDoc(context.Background(), types.Document{ID: "d1", Text: "hello"}); err != nil {
		t.Fatalf("IndexDoc: %v", err)
	}
	if err := c.DeleteDoc(context.Background(), "d1"); err != nil {
		t.Fatalf("DeleteDoc: %v", err)
	}
	for _, m := range []string{http.MethodPost, http.MethodDelete} {
		if got[m] != "Bearer secret" {
			t.Errorf("%s: Authorization = %q, want %q", m, got[m], "Bearer secret")
		}
	}
}
