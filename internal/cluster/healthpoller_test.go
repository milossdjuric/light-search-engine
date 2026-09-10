package cluster_test

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"search-eval-platform/internal/cluster"
)

func TestHealthPollerTripsCircuitBreaker(t *testing.T) {
	// Healthy server that always returns 200.
	healthy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer healthy.Close()

	// Dead server that always returns 503.
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer dead.Close()

	nodes := []*cluster.NodeMeta{
		{NodeID: "healthy-node", HTTPAddr: healthy.Listener.Addr().String()},
		{NodeID: "dead-node",    HTTPAddr: dead.Listener.Addr().String()},
	}

	breakers := map[string]*cluster.CircuitBreaker{
		"healthy-node": cluster.NewCircuitBreaker(3, 30*time.Second),
		"dead-node":    cluster.NewCircuitBreaker(3, 30*time.Second),
	}

	var pollCount atomic.Int32
	getBreaker := func(nodeID string) *cluster.CircuitBreaker {
		pollCount.Add(1)
		return breakers[nodeID]
	}

	poller := cluster.NewHealthPoller(nodes, getBreaker, 20*time.Millisecond)
	poller.Start()
	defer poller.Stop()

	// Wait for enough polls to trip the dead-node breaker (needs 3 failures).
	time.Sleep(200 * time.Millisecond)

	// healthy-node breaker must be closed (not open).
	if breakers["healthy-node"].IsOpen() {
		t.Error("healthy-node circuit breaker is open; want closed")
	}
	// dead-node breaker must be open after 3+ failures.
	if !breakers["dead-node"].IsOpen() {
		t.Error("dead-node circuit breaker is not open after repeated 503s; want open")
	}
}
