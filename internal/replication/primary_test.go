package replication

import (
	"context"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// fakeServerStream is a minimal grpc.ServerStream that records every message
// passed to SendMsg, for use as the transport behind a
// WALReplication_StreamWALServer in tests.
type fakeServerStream struct {
	ctx  context.Context
	sent []*WALEntry
}

func (f *fakeServerStream) SetHeader(metadata.MD) error  { return nil }
func (f *fakeServerStream) SendHeader(metadata.MD) error { return nil }
func (f *fakeServerStream) SetTrailer(metadata.MD)       {}
func (f *fakeServerStream) Context() context.Context     { return f.ctx }
func (f *fakeServerStream) RecvMsg(any) error            { return nil }
func (f *fakeServerStream) SendMsg(m any) error {
	if e, ok := m.(*WALEntry); ok {
		f.sent = append(f.sent, e)
	}
	return nil
}

// newFakeStream builds a WALReplication_StreamWALServer whose context is
// already cancelled, so streamToReplica runs its catch-up phases and then
// returns immediately once it reaches the live-stream select loop.
func newFakeStream() *fakeServerStream {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return &fakeServerStream{ctx: ctx}
}

// TestStreamToReplicaCatchUpSendsAllRingEntries verifies that a full
// catch-up request (fromSeq=0) delivers every entry currently held in the
// ring buffer, including the very first one appended.
func TestStreamToReplicaCatchUpSendsAllRingEntries(t *testing.T) {
	p := NewPrimaryReplicator("shard0", "/nonexistent/wal")

	for seq := uint64(1); seq <= 3; seq++ {
		p.Append(&WALEntry{Seq: seq, Op: "index", DocId: "d"})
	}

	fake := newFakeStream()
	stream := &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: fake}

	p.streamToReplica("node1", 0, make(chan *WALEntry), stream)

	var gotSeqs []uint64
	for _, e := range fake.sent {
		gotSeqs = append(gotSeqs, e.Seq)
	}
	want := []uint64{1, 2, 3}
	if len(gotSeqs) != len(want) {
		t.Fatalf("got %d entries %v, want %d entries %v", len(gotSeqs), gotSeqs, len(want), want)
	}
	for i, s := range want {
		if gotSeqs[i] != s {
			t.Errorf("entry %d: got seq %d, want %d (full sequence: %v)", i, gotSeqs[i], s, gotSeqs)
		}
	}
}

// TestRingBufferReadDuringConcurrentAppendIsRaceFree fills the ring buffer
// completely, then concurrently runs a full-ring catch-up read
// (streamToReplica) against a goroutine that keeps calling Append — which
// writes new entries into the same indices the catch-up loop is reading.
// streamToReplica's ring read (p.ring[seq%ringBufferSize]) must take p.mu
// like Append's write does; run with -race to catch a torn read otherwise.
func TestRingBufferReadDuringConcurrentAppendIsRaceFree(t *testing.T) {
	p := NewPrimaryReplicator("shard-race", "/nonexistent/wal")

	for seq := uint64(1); seq <= ringBufferSize; seq++ {
		p.Append(&WALEntry{Seq: seq, Op: "index", DocId: "d"})
	}

	fake := newFakeStream()
	stream := &grpc.GenericServerStream[struct{}, WALEntry]{ServerStream: fake}

	done := make(chan struct{})
	go func() {
		defer close(done)
		for seq := uint64(ringBufferSize + 1); seq <= 2*ringBufferSize; seq++ {
			p.Append(&WALEntry{Seq: seq, Op: "index", DocId: "d"})
		}
	}()

	p.streamToReplica("racer", 0, make(chan *WALEntry), stream)
	<-done
}
