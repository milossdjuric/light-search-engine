package main

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/shard"
	"search-eval-platform/pkg/types"
)

func openShards(t *testing.T, dir string) *shard.ShardManager {
	t.Helper()
	sm, err := shard.NewShardManager(2, dir, scoring.NewBM25(1.2, 0.75), shard.DefaultTieredMergePolicy(),
		shard.ShardManagerOptions{GlobalFusion: "score"})
	if err != nil {
		t.Fatalf("NewShardManager: %v", err)
	}
	if err := sm.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	return sm
}

// waitCaughtUp blocks until every replica shard has applied everything its
// primary shard has written.
func waitCaughtUp(t *testing.T, primary, replica *replicationNode) {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		p := primary.Status().(replicationStatus)
		r := replica.Status().(replicationStatus)
		caught := true
		for i := range p.Primary {
			if r.Replicas[i].AppliedSeq != p.Primary[i].HeadSeq {
				caught = false
			}
		}
		if caught {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("replica never caught up: primary %+v, replica %+v", primary.Status(), replica.Status())
}

func searchIDs(t *testing.T, shards *shard.ShardManager, q string) map[string]bool {
	t.Helper()
	if err := shards.Flush(context.Background()); err != nil { // buffer snapshot is only refreshed on flush
		t.Fatal(err)
	}
	res, _, err := shards.Search(context.Background(), q, 100, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	ids := map[string]bool{}
	for _, r := range res {
		ids[r.DocID] = true
	}
	return ids
}

// TestReplicationEndToEnd runs a primary and a replica shard node in-process
// over real gRPC and checks catch-up, live streaming, deletes, reset and a
// replica restart resuming from its saved position.
func TestReplicationEndToEnd(t *testing.T) {
	ctx := context.Background()
	primaryShards := openShards(t, t.TempDir())
	defer primaryShards.Close()
	pn, err := startReplication(replicationConfig{Role: "primary", ListenAddr: "127.0.0.1:0"}, primaryShards)
	if err != nil {
		t.Fatal(err)
	}
	defer pn.stop()

	// Written before the replica exists: served by WAL catch-up.
	for i := 0; i < 20; i++ {
		if err := primaryShards.IndexDoc(ctx, types.Document{ID: fmt.Sprintf("early%d", i), Text: "alpha early"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := primaryShards.DeleteDoc(ctx, "early0"); err != nil {
		t.Fatal(err)
	}

	replicaDir := t.TempDir()
	stateDir := filepath.Join(replicaDir, "replication")
	replicaShards := openShards(t, replicaDir)
	rn, err := startReplication(replicationConfig{Role: "replica", PrimaryAddr: pn.Addr(), StateDir: stateDir}, replicaShards)
	if err != nil {
		t.Fatal(err)
	}
	waitCaughtUp(t, pn, rn)

	// Written while connected: live stream.
	for i := 0; i < 10; i++ {
		if err := primaryShards.IndexDoc(ctx, types.Document{ID: fmt.Sprintf("live%d", i), Text: "alpha live"}); err != nil {
			t.Fatal(err)
		}
	}
	waitCaughtUp(t, pn, rn)

	got := searchIDs(t, replicaShards, "alpha")
	if len(got) != 29 || got["early0"] || !got["early1"] || !got["live9"] {
		t.Fatalf("replica has %d docs (early0=%v early1=%v live9=%v), want 29 without the deleted early0",
			len(got), got["early0"], got["early1"], got["live9"])
	}

	// Replica restart: stop, close, write more on the primary, reopen.
	rn.stop()
	if err := replicaShards.Close(); err != nil {
		t.Fatal(err)
	}
	rn.afterClose()
	for i := 0; i < 5; i++ {
		if err := primaryShards.IndexDoc(ctx, types.Document{ID: fmt.Sprintf("offline%d", i), Text: "alpha offline"}); err != nil {
			t.Fatal(err)
		}
	}
	replicaShards = openShards(t, replicaDir)
	defer replicaShards.Close()
	rn, err = startReplication(replicationConfig{Role: "replica", PrimaryAddr: pn.Addr(), StateDir: stateDir}, replicaShards)
	if err != nil {
		t.Fatal(err)
	}
	defer rn.stop()
	waitCaughtUp(t, pn, rn)
	if got := searchIDs(t, replicaShards, "alpha"); len(got) != 34 {
		t.Fatalf("after replica restart: %d docs, want 34", len(got))
	}

	// Reset on the primary wipes the replica too.
	if err := primaryShards.Reset(ctx); err != nil {
		t.Fatal(err)
	}
	if err := primaryShards.IndexDoc(ctx, types.Document{ID: "post-reset", Text: "alpha fresh"}); err != nil {
		t.Fatal(err)
	}
	waitCaughtUp(t, pn, rn)
	if got := searchIDs(t, replicaShards, "alpha"); len(got) != 1 || !got["post-reset"] {
		t.Fatalf("after primary reset replica has %v, want only post-reset", got)
	}

	for _, r := range rn.Status().(replicationStatus).Replicas {
		if r.DeadLetters != 0 || r.NeedsResync {
			t.Errorf("replica shard %s unhealthy: %+v", r.ShardID, r)
		}
	}
}

// TestReplicaNeedsResyncWhenPrimaryAlreadyFlushed verifies that a fresh,
// empty replica joining a primary whose early writes are already flushed
// into segments (WAL gone) is reported as needing a re-seed instead of
// silently replicating only the tail.
func TestReplicaNeedsResyncWhenPrimaryAlreadyFlushed(t *testing.T) {
	ctx := context.Background()
	primaryDir := t.TempDir()
	first := openShards(t, primaryDir)
	for i := 0; i < 20; i++ {
		if err := first.IndexDoc(ctx, types.Document{ID: fmt.Sprintf("d%d", i), Text: "x"}); err != nil {
			t.Fatal(err)
		}
	}
	// Close flushes into segments and drops the WAL; the restarted primary's
	// ring buffer is empty, so seq 1.. is retained nowhere.
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	primaryShards := openShards(t, primaryDir)
	defer primaryShards.Close()
	pn, err := startReplication(replicationConfig{Role: "primary", ListenAddr: "127.0.0.1:0"}, primaryShards)
	if err != nil {
		t.Fatal(err)
	}
	defer pn.stop()
	if err := primaryShards.IndexDoc(ctx, types.Document{ID: "after-restart", Text: "x"}); err != nil {
		t.Fatal(err)
	}

	replicaShards := openShards(t, t.TempDir())
	defer replicaShards.Close()
	rn, err := startReplication(replicationConfig{Role: "replica", PrimaryAddr: pn.Addr(), StateDir: t.TempDir()}, replicaShards)
	if err != nil {
		t.Fatal(err)
	}
	defer rn.stop()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		all := true
		for _, r := range rn.Status().(replicationStatus).Replicas {
			all = all && r.NeedsResync
		}
		if all {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("replicas never reported needs_resync: %+v", rn.Status())
}
