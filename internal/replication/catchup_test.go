package replication

import (
	"context"
	"sync"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// walCatchUp returns a CatchUpFunc serving the given seqs as if they were the
// entries still retained in the primary's WAL files.
func walCatchUp(seqs ...uint64) CatchUpFunc {
	return func(afterSeq uint64, fn func(*WALEntry) error) error {
		for _, s := range seqs {
			if s <= afterSeq {
				continue
			}
			if err := fn(&WALEntry{Seq: s, Op: "index", DocId: "d"}); err != nil {
				return err
			}
		}
		return nil
	}
}

// liveStream is a server stream that stays open until stopAt has been sent.
type liveStream struct {
	fakeServerStream
	cancel context.CancelFunc
	stopAt uint64
	mu     sync.Mutex
}

func newLiveStream(stopAt uint64) *liveStream {
	ctx, cancel := context.WithCancel(context.Background())
	return &liveStream{fakeServerStream: fakeServerStream{ctx: ctx}, cancel: cancel, stopAt: stopAt}
}

func (l *liveStream) SendMsg(m any) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if e, ok := m.(*WALEntry); ok && e.Op != "heartbeat" {
		l.sent = append(l.sent, e)
		if e.Seq == l.stopAt {
			l.cancel()
		}
	}
	return nil
}

func sentSeqs(es []*WALEntry) []uint64 {
	var out []uint64
	for _, e := range es {
		out = append(out, e.Seq)
	}
	return out
}

// TestStreamToReplicaWALThenRingThenLiveNoDuplicates verifies catch-up reads
// the WAL, then the ring, then the live channel, delivering every seq exactly
// once and in order even though the three sources overlap.
func TestStreamToReplicaWALThenRingThenLiveNoDuplicates(t *testing.T) {
	p := NewPrimaryReplicator("shard0", walCatchUp(1, 2, 3, 4, 5))
	for s := uint64(4); s <= 7; s++ { // ring overlaps the WAL tail
		p.Append(&WALEntry{Seq: s, Op: "index", DocId: "d"})
	}
	ch := make(chan *WALEntry, 4)
	ch <- &WALEntry{Seq: 7, Op: "index"} // overlaps the ring
	ch <- &WALEntry{Seq: 8, Op: "index"}

	ls := newLiveStream(8)
	err := p.streamToReplica(0, ch, &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: ls})
	if err != nil {
		t.Fatalf("streamToReplica: %v", err)
	}
	got := sentSeqs(ls.sent)
	want := []uint64{1, 2, 3, 4, 5, 6, 7, 8}
	if len(got) != len(want) {
		t.Fatalf("sent %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sent %v, want %v", got, want)
		}
	}
}

// TestStreamToReplicaReportsUnrecoverableGap verifies that when the entries a
// replica needs are no longer retained (already flushed into segments and
// their WAL files deleted), the primary says so explicitly with
// FailedPrecondition instead of streaming a gap the replica can never fill.
func TestStreamToReplicaReportsUnrecoverableGap(t *testing.T) {
	p := NewPrimaryReplicator("shard0", walCatchUp(5, 6))
	ls := newLiveStream(0)
	err := p.streamToReplica(0, make(chan *WALEntry), &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: ls})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("err = %v, want FailedPrecondition", err)
	}
	if len(ls.sent) != 0 {
		t.Errorf("sent %v before reporting the gap, want nothing", sentSeqs(ls.sent))
	}
}

// TestStreamToReplicaLiveGapForcesReconnect verifies a live entry dropped by
// a full replica channel ends the stream (so the replica reconnects and
// catches up) instead of silently skipping it.
func TestStreamToReplicaLiveGapForcesReconnect(t *testing.T) {
	p := NewPrimaryReplicator("shard0", nil)
	ch := make(chan *WALEntry, 2)
	ch <- &WALEntry{Seq: 1, Op: "index"}
	ch <- &WALEntry{Seq: 3, Op: "index"} // seq 2 was dropped
	ls := newLiveStream(99)
	err := p.streamToReplica(0, ch, &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: ls})
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("err = %v, want Unavailable (reconnect to catch up)", err)
	}
	if got := sentSeqs(ls.sent); len(got) != 1 || got[0] != 1 {
		t.Errorf("sent %v, want [1]", got)
	}
}

// TestServerRoutesByShardAndRejectsUnknown verifies the multi-shard gRPC
// server dispatches StreamWAL to the right shard's primary.
func TestServerRoutesByShardAndRejectsUnknown(t *testing.T) {
	srv := NewServer()
	p := NewPrimaryReplicator("shard1", walCatchUp(1))
	srv.Register(p)
	defer p.Close()

	ls := newLiveStream(1)
	if err := srv.StreamWAL(&StreamRequest{ShardId: "shard1"}, &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: ls}); err != nil {
		t.Fatalf("StreamWAL(shard1): %v", err)
	}
	if got := sentSeqs(ls.sent); len(got) != 1 || got[0] != 1 {
		t.Errorf("sent %v, want [1]", got)
	}

	err := srv.StreamWAL(&StreamRequest{ShardId: "nope"}, &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: newLiveStream(0)})
	if status.Code(err) != codes.NotFound {
		t.Errorf("StreamWAL(unknown) err = %v, want NotFound", err)
	}
}

// TestConcurrentReplicasOfSameShardDoNotCollide is a regression test for
// every replica of a shard being registered under the same key
// (shardID+"-replica"), so a second replica silently replaced the first.
func TestConcurrentReplicasOfSameShardDoNotCollide(t *testing.T) {
	p := NewPrimaryReplicator("shard0", nil)
	defer p.Close()
	s1, s2 := newLiveStream(0), newLiveStream(0)
	p.AddReplica(0, &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: s1})
	p.AddReplica(0, &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: s2})
	if n := p.ReplicaCount(); n != 2 {
		t.Fatalf("ReplicaCount = %d, want 2", n)
	}
	s1.cancel()
	s2.cancel()
}

// TestStreamToReplicaReportsGapWhenNothingRetained verifies a primary that
// retains no entries at all (everything flushed, ring empty after restart)
// still reports a replica behind its last seq as needing a re-seed, rather
// than streaming nothing and looking healthy.
func TestStreamToReplicaReportsGapWhenNothingRetained(t *testing.T) {
	p := NewPrimaryReplicator("shard0", walCatchUp())
	p.SetLastSeqFunc(func() uint64 { return 14 })

	err := p.streamToReplica(0, make(chan *WALEntry), &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: newFakeStream()})
	if status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("replica at 0, primary at 14, nothing retained: err = %v, want FailedPrecondition", err)
	}

	err = p.streamToReplica(14, make(chan *WALEntry), &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: newFakeStream()})
	if err != nil {
		t.Fatalf("replica already at 14: err = %v, want nil", err)
	}
	if got := p.HeadSeq(); got != 14 {
		t.Errorf("HeadSeq = %d, want 14 from the last-seq func", got)
	}
}
