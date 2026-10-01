package cluster_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"search-eval-platform/internal/cluster"
)

// fakeShardNode serves /health with the given local_shards; down=true makes it
// answer 503 without a body, like a node that is restarting.
func fakeShardNode(t *testing.T, shards []int, down *atomic.Bool) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if down != nil && down.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		parts := make([]string, len(shards))
		for i, s := range shards {
			parts[i] = fmt.Sprint(s)
		}
		fmt.Fprintf(w, `{"status":"ok","mode":"shard","local_shards":[%s]}`, strings.Join(parts, ","))
	}))
	t.Cleanup(srv.Close)
	return strings.TrimPrefix(srv.URL, "http://")
}

func primaryAddr(t *testing.T, ring *cluster.Ring, shard int) string {
	t.Helper()
	n, err := ring.Primary(shard)
	if err != nil {
		t.Fatalf("Primary(%d): %v", shard, err)
	}
	return n.HTTPAddr
}

// Static bootstrap nodes must be probed for the shards they own; before this,
// the coordinator built them with no shards and every write failed with
// "ring: no primary for shard N".
func TestStaticMemberWatcherLearnsShardsFromHealth(t *testing.T) {
	a := fakeShardNode(t, []int{0, 1}, nil)
	b := fakeShardNode(t, []int{2, 3}, nil)
	ring := cluster.NewRing(4)
	w := cluster.NewStaticMemberWatcher(ring, []string{a, b})

	if n := w.Refresh(context.Background()); n != 2 {
		t.Fatalf("Refresh discovered %d nodes, want 2", n)
	}
	for s, want := range map[int]string{0: a, 1: a, 2: b, 3: b} {
		if got := primaryAddr(t, ring, s); got != want {
			t.Errorf("shard %d primary = %s, want %s", s, got, want)
		}
	}
}

// A node that fails one probe keeps its last known shards in the ring (its
// circuit breaker handles liveness) instead of vanishing from routing.
func TestStaticMemberWatcherKeepsLastKnownShardsOnProbeFailure(t *testing.T) {
	var bDown atomic.Bool
	a := fakeShardNode(t, []int{0}, nil)
	b := fakeShardNode(t, []int{1}, &bDown)
	ring := cluster.NewRing(2)
	w := cluster.NewStaticMemberWatcher(ring, []string{a, b})
	w.Refresh(context.Background())

	bDown.Store(true)
	w.Refresh(context.Background())
	if got := primaryAddr(t, ring, 1); got != b {
		t.Fatalf("shard 1 primary after failed probe = %s, want %s", got, b)
	}
}

// A node that was down at startup is picked up by a later refresh.
func TestStaticMemberWatcherPicksUpLateNode(t *testing.T) {
	var bDown atomic.Bool
	bDown.Store(true)
	a := fakeShardNode(t, []int{0}, nil)
	b := fakeShardNode(t, []int{1}, &bDown)
	ring := cluster.NewRing(2)
	w := cluster.NewStaticMemberWatcher(ring, []string{a, b})

	if n := w.Refresh(context.Background()); n != 1 {
		t.Fatalf("first Refresh discovered %d nodes, want 1", n)
	}
	if _, err := ring.Primary(1); err == nil {
		t.Fatal("shard 1 has a primary before its node ever answered")
	}
	bDown.Store(false)
	w.Refresh(context.Background())
	if got := primaryAddr(t, ring, 1); got != b {
		t.Fatalf("shard 1 primary = %s, want %s", got, b)
	}
}
