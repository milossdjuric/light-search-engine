package cluster

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fakeK8sServer returns an httptest.Server that serves:
//   - GET /health → healthProbeResponse with the given localShards
//   - Any other path → streaming watch events from watchEvents (one JSON object per string)
func fakeK8sServer(t *testing.T, localShards []int, watchEvents []string) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(healthProbeResponse{LocalShards: localShards})
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("ResponseWriter does not implement http.Flusher")
			return
		}
		for _, ev := range watchEvents {
			w.Write([]byte(ev + "\n"))
			flusher.Flush()
		}
		// Block until client disconnects so the watcher reads all events.
		<-r.Context().Done()
	})

	return httptest.NewServer(mux)
}

// watchEventJSON builds a minimal ADDED Endpoints watch event whose one
// address points back to the fake server at the given IP:port.
func watchEventJSON(ip, port string) string {
	return `{"type":"ADDED","object":{"subsets":[{"addresses":[{"ip":"` + ip + `","targetRef":{"name":"pod-0"}}],"ports":[{"name":"http","port":` + port + `}]}]}}`
}

func TestK8sMemberWatcherRebuildRing(t *testing.T) {
	ring := NewRing(4)

	srv := fakeK8sServer(t, []int{0, 1, 2, 3}, nil) // health server; no watch events yet
	defer srv.Close()

	// Parse host and port from srv.URL (http://127.0.0.1:PORT)
	addr := strings.TrimPrefix(srv.URL, "http://")
	parts := strings.SplitN(addr, ":", 2)
	ip, port := parts[0], parts[1]

	// Build a watch event that points the watcher at the fake server's /health.
	event := watchEventJSON(ip, port)

	watchSrv := fakeK8sServer(t, []int{0, 1, 2, 3}, []string{event})
	defer watchSrv.Close()

	watchAddr := strings.TrimPrefix(watchSrv.URL, "http://")
	watchParts := strings.SplitN(watchAddr, ":", 2)
	watchPort := watchParts[1]
	_ = watchPort

	// Build a watcher that talks to our fake watch server.
	w := &K8sMemberWatcher{
		ring:   ring,
		cfg:    K8sMemberConfig{HTTPPort: 8080, GRPCPort: 9090},
		client: &http.Client{Timeout: 5 * time.Second},
	}

	// Override rebuild to use our fake health server's address directly.
	// We inject a synthetic endpointOrError instead of going through Run().
	ep := &endpointOrError{
		Subsets: []endpointSubset{
			{
				Addresses: []endpointAddress{
					{IP: ip, TargetRef: &objectRef{Name: "pod-0"}},
				},
				Ports: []endpointPort{
					{Name: "http", Port: mustParsePort(t, port)},
				},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	w.rebuild(ctx, ep)

	nodes := ring.Nodes()
	if len(nodes) != 1 {
		t.Fatalf("expected 1 node after rebuild, got %d", len(nodes))
	}
	node := nodes[0]
	if node.NodeID != "pod-0" {
		t.Errorf("NodeID = %q, want pod-0", node.NodeID)
	}
	if len(node.Shards) != 4 {
		t.Errorf("Shards = %v, want [0 1 2 3]", node.Shards)
	}
	for i, s := range node.Shards {
		if s != i {
			t.Errorf("Shards[%d] = %d, want %d", i, s, i)
		}
	}
	prim, err := ring.Primary(0)
	if err != nil {
		t.Fatalf("ring.Primary(0): %v", err)
	}
	if prim.NodeID != "pod-0" {
		t.Errorf("Primary(0).NodeID = %q, want pod-0", prim.NodeID)
	}
}

func TestK8sMemberWatcherSkipsUnhealthyNode(t *testing.T) {
	ring := NewRing(4)

	// Server that returns 500 on /health.
	badSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer badSrv.Close()

	addr := strings.TrimPrefix(badSrv.URL, "http://")
	parts := strings.SplitN(addr, ":", 2)
	ip, port := parts[0], parts[1]

	w := &K8sMemberWatcher{
		ring:   ring,
		cfg:    K8sMemberConfig{HTTPPort: mustParsePort(t, port), GRPCPort: 9090},
		client: &http.Client{Timeout: 2 * time.Second},
	}

	ep := &endpointOrError{
		Subsets: []endpointSubset{
			{
				Addresses: []endpointAddress{
					{IP: ip, TargetRef: &objectRef{Name: "bad-pod"}},
				},
				Ports: []endpointPort{
					{Name: "http", Port: mustParsePort(t, port)},
				},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	w.rebuild(ctx, ep)

	// Ring should NOT be updated because probe failed.
	if len(ring.Nodes()) != 0 {
		t.Errorf("expected ring empty after failed probe, got %d nodes", len(ring.Nodes()))
	}
}

func TestK8sMemberWatcherNoLocalShards(t *testing.T) {
	ring := NewRing(4)

	// Server that returns empty local_shards (not in shard mode).
	badSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(healthProbeResponse{LocalShards: nil})
	}))
	defer badSrv.Close()

	addr := strings.TrimPrefix(badSrv.URL, "http://")
	parts := strings.SplitN(addr, ":", 2)
	ip, port := parts[0], parts[1]

	w := &K8sMemberWatcher{
		ring:   ring,
		cfg:    K8sMemberConfig{HTTPPort: mustParsePort(t, port), GRPCPort: 9090},
		client: &http.Client{Timeout: 2 * time.Second},
	}

	ep := &endpointOrError{
		Subsets: []endpointSubset{
			{
				Addresses: []endpointAddress{
					{IP: ip, TargetRef: &objectRef{Name: "coord-pod"}},
				},
				Ports: []endpointPort{
					{Name: "http", Port: mustParsePort(t, port)},
				},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	w.rebuild(ctx, ep)

	if len(ring.Nodes()) != 0 {
		t.Errorf("expected ring empty when no local_shards, got %d nodes", len(ring.Nodes()))
	}
}

func TestK8sMemberWatcherMultipleNodes(t *testing.T) {
	ring := NewRing(8)

	// Two fake shard servers.
	srv0 := fakeK8sServer(t, []int{0, 1, 2, 3}, nil)
	defer srv0.Close()
	srv1 := fakeK8sServer(t, []int{4, 5, 6, 7}, nil)
	defer srv1.Close()

	addr0 := strings.TrimPrefix(srv0.URL, "http://")
	p0 := strings.SplitN(addr0, ":", 2)
	addr1 := strings.TrimPrefix(srv1.URL, "http://")
	p1 := strings.SplitN(addr1, ":", 2)

	w := &K8sMemberWatcher{
		ring:   ring,
		cfg:    K8sMemberConfig{HTTPPort: 8080, GRPCPort: 9090},
		client: &http.Client{Timeout: 5 * time.Second},
	}

	ep := &endpointOrError{
		Subsets: []endpointSubset{
			{
				Addresses: []endpointAddress{
					{IP: p0[0], TargetRef: &objectRef{Name: "shard-0"}},
					{IP: p1[0], TargetRef: &objectRef{Name: "shard-1"}},
				},
				Ports: []endpointPort{
					{Name: "http", Port: mustParsePort(t, p0[1])},
				},
			},
		},
	}

	// Each address probes on the same port (from the subset), but we need
	// individual ports. Override by using two subsets instead.
	ep = &endpointOrError{
		Subsets: []endpointSubset{
			{
				Addresses: []endpointAddress{
					{IP: p0[0], TargetRef: &objectRef{Name: "shard-0"}},
				},
				Ports: []endpointPort{{Name: "http", Port: mustParsePort(t, p0[1])}},
			},
			{
				Addresses: []endpointAddress{
					{IP: p1[0], TargetRef: &objectRef{Name: "shard-1"}},
				},
				Ports: []endpointPort{{Name: "http", Port: mustParsePort(t, p1[1])}},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	w.rebuild(ctx, ep)

	nodes := ring.Nodes()
	if len(nodes) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(nodes))
	}

	// Both shards 0–3 and 4–7 should have primaries.
	for _, shardID := range []int{0, 1, 2, 3, 4, 5, 6, 7} {
		prim, err := ring.Primary(shardID)
		if err != nil {
			t.Errorf("ring.Primary(%d): %v", shardID, err)
			continue
		}
		_ = prim
	}
}

func mustParsePort(t *testing.T, s string) int {
	t.Helper()
	n, err := parseInt(s)
	if err != nil {
		t.Fatalf("parse port %q: %v", s, err)
	}
	return n
}

func parseInt(s string) (int, error) {
	var n int
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, &portError{s}
		}
		n = n*10 + int(c-'0')
	}
	return n, nil
}

type portError struct{ s string }

func (e *portError) Error() string { return "invalid port: " + e.s }
