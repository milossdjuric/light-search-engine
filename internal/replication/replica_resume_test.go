package replication

import (
	"errors"
	"io"
	"path/filepath"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// errReceiver delivers entries, then returns err.
type errReceiver struct {
	fakeEntryReceiver
	err error
}

func (e *errReceiver) Recv() (*WALEntry, error) {
	if ent, err := e.fakeEntryReceiver.Recv(); err == nil {
		return ent, nil
	}
	return nil, e.err
}

// TestApplyStreamSkipsDuplicates verifies entries at or below appliedSeq (the
// primary's overlapping catch-up sources, or a resend after reconnect) are
// skipped rather than treated as a sequence gap.
func TestApplyStreamSkipsDuplicates(t *testing.T) {
	var applied []uint64
	r := NewReplicaApplier("shard0", "unused", func(e *WALEntry) error {
		applied = append(applied, e.Seq)
		return nil
	})
	r.appliedSeq.Store(100)
	err := r.applyStream(&fakeEntryReceiver{entries: []*WALEntry{
		{Seq: 99, Op: "index"}, {Seq: 100, Op: "index"}, {Seq: 101, Op: "index"},
	}})
	if !errors.Is(err, io.EOF) {
		t.Fatalf("err = %v, want io.EOF", err)
	}
	if len(applied) != 1 || applied[0] != 101 {
		t.Errorf("applied %v, want [101]", applied)
	}
}

// TestNeedsResyncTrackedFromPrimary verifies a FailedPrecondition from the
// primary (entries no longer retained) is surfaced in Status().NeedsResync,
// and cleared once entries apply again.
func TestNeedsResyncTrackedFromPrimary(t *testing.T) {
	r := NewReplicaApplier("shard0", "primary:9090", func(e *WALEntry) error { return nil })
	gone := status.Error(codes.FailedPrecondition, "seqs 1..4 no longer retained")
	r.applyStream(&errReceiver{err: gone})

	st := r.Status()
	if !st.NeedsResync {
		t.Fatalf("Status().NeedsResync = false after FailedPrecondition, want true (status %+v)", st)
	}
	if st.LastError == "" {
		t.Error("Status().LastError empty, want the primary's message")
	}

	r.applyStream(&fakeEntryReceiver{entries: []*WALEntry{{Seq: 1, Op: "index"}}})
	if r.Status().NeedsResync {
		t.Error("NeedsResync still true after entries applied again")
	}
}

// TestAppliedSeqPersistence verifies the replica resumes from its saved
// position after a restart, and falls back to the initial seq when nothing
// has been saved yet.
func TestAppliedSeqPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "shard0.replica.seq")

	r1 := NewReplicaApplier("shard0", "unused", func(e *WALEntry) error { return nil })
	r1.SetSeqPath(path)
	if err := r1.loadAppliedSeq(42); err != nil {
		t.Fatal(err)
	}
	if got := r1.AppliedSeq(); got != 42 {
		t.Fatalf("with no saved seq, AppliedSeq = %d, want initial 42", got)
	}
	r1.appliedSeq.Store(77)
	if err := r1.SaveAppliedSeq(); err != nil {
		t.Fatal(err)
	}

	r2 := NewReplicaApplier("shard0", "unused", func(e *WALEntry) error { return nil })
	r2.SetSeqPath(path)
	if err := r2.loadAppliedSeq(42); err != nil {
		t.Fatal(err)
	}
	if got := r2.AppliedSeq(); got != 77 {
		t.Fatalf("after restart AppliedSeq = %d, want saved 77", got)
	}
}
