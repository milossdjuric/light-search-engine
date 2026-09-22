package cluster

import (
	"testing"
	"time"
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
	if !cb.IsOpen() {
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
