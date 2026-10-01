package cluster

import (
	"context"
	"log/slog"
	"net/http"
	"slices"
	"sync"
	"time"
)

// StaticMemberWatcher keeps the Ring in sync with a fixed list of shard-node
// addresses (cluster.bootstrap_addrs): it probes each node's /health for the
// shards it owns and rebuilds the ring when that mapping changes. A node that
// fails a probe keeps its last known shards — liveness is the circuit
// breakers' job — and a node that was down at startup is added once it answers.
type StaticMemberWatcher struct {
	ring   *Ring
	addrs  []string
	client *http.Client

	mu    sync.Mutex
	known map[string][]int // addr → last shards it reported
}

// NewStaticMemberWatcher creates a watcher for the given node addresses.
func NewStaticMemberWatcher(ring *Ring, addrs []string) *StaticMemberWatcher {
	return &StaticMemberWatcher{
		ring:   ring,
		addrs:  addrs,
		client: &http.Client{Timeout: 5 * time.Second},
		known:  make(map[string][]int, len(addrs)),
	}
}

// Refresh probes every node once, rebuilds the ring if any node's shards
// changed, and returns how many nodes have known shards.
func (w *StaticMemberWatcher) Refresh(ctx context.Context) int {
	w.mu.Lock()
	defer w.mu.Unlock()

	changed := false
	for _, addr := range w.addrs {
		shards, err := ProbeLocalShards(ctx, w.client, addr)
		if err != nil {
			slog.Debug("staticmember: probe failed", "addr", addr, "err", err)
			continue
		}
		if !slices.Equal(w.known[addr], shards) {
			w.known[addr] = shards
			changed = true
		}
	}
	if changed {
		var nodes []*NodeMeta
		for _, addr := range w.addrs { // config order decides primaries on overlap
			if shards, ok := w.known[addr]; ok {
				nodes = append(nodes, &NodeMeta{NodeID: addr, HTTPAddr: addr, Shards: shards, Role: "shard"})
			}
		}
		w.ring.Rebuild(nodes)
		slog.Info("staticmember: ring rebuilt", "nodes", len(nodes), "configured", len(w.addrs))
	}
	return len(w.known)
}

// Run refreshes every interval until ctx is cancelled.
func (w *StaticMemberWatcher) Run(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			w.Refresh(ctx)
		}
	}
}
