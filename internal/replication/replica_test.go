package replication

import (
	"errors"
	"io"
	"testing"
)

// fakeEntryReceiver replays a fixed sequence of WAL entries, then returns
// io.EOF, mimicking a stream that hangs up after delivering its backlog.
type fakeEntryReceiver struct {
	entries []*WALEntry
	i       int
}

func (f *fakeEntryReceiver) Recv() (*WALEntry, error) {
	if f.i >= len(f.entries) {
		return nil, io.EOF
	}
	e := f.entries[f.i]
	f.i++
	return e, nil
}

// TestApplyStreamStopsOnApplyErrorWithoutAdvancingAppliedSeq verifies that
// when apply() fails for an entry, applyStream does not silently move on to
// later entries on the same stream: appliedSeq must stay behind the failed
// entry (so a reconnect re-requests it) and later entries in the same
// backlog must not be applied ahead of it.
func TestApplyStreamStopsOnApplyErrorWithoutAdvancingAppliedSeq(t *testing.T) {
	var appliedDocIDs []string
	r := NewReplicaApplier("shard0", "unused", func(op, docID, text string, metadata map[string]string) error {
		if docID == "fail" {
			return errors.New("boom")
		}
		appliedDocIDs = append(appliedDocIDs, docID)
		return nil
	})
	r.appliedSeq.Store(99)

	recv := &fakeEntryReceiver{entries: []*WALEntry{
		{Seq: 100, Op: "index", DocId: "fail"},
		{Seq: 101, Op: "index", DocId: "ok"},
	}}

	if err := r.applyStream(recv); err == nil {
		t.Fatal("expected applyStream to return an error when apply fails, got nil")
	}

	if got := r.AppliedSeq(); got != 99 {
		t.Errorf("appliedSeq = %d, want 99 (must not advance past the failed seq=100 entry)", got)
	}
	if len(appliedDocIDs) != 0 {
		t.Errorf("apply() ran for %v, want none (seq=101 must not be applied ahead of the failed seq=100)", appliedDocIDs)
	}
}

// TestApplyStreamDetectsSequenceGapAndForcesResync verifies that a gap in
// the incoming Seq sequence (e.g. an entry dropped by the primary's full
// fan-out channel — see PrimaryReplicator.Append's select/default) is
// treated as a stream error rather than silently applied. Applying seq=3
// straight after seq=1 would permanently and silently lose seq=2; instead
// applyStream must return an error so run()'s reconnect-with-backoff loop
// re-requests FromSeq starting at the last good seq, letting the primary's
// catch-up phase resend the missing entry.
func TestApplyStreamDetectsSequenceGapAndForcesResync(t *testing.T) {
	var appliedDocIDs []string
	r := NewReplicaApplier("shard0", "unused", func(op, docID, text string, metadata map[string]string) error {
		appliedDocIDs = append(appliedDocIDs, docID)
		return nil
	})
	r.appliedSeq.Store(1)

	// seq=2 is missing; the stream jumps straight to seq=3.
	recv := &fakeEntryReceiver{entries: []*WALEntry{
		{Seq: 3, Op: "index", DocId: "d3"},
	}}

	if err := r.applyStream(recv); err == nil {
		t.Fatal("expected applyStream to return an error on a sequence gap, got nil")
	}

	if got := r.AppliedSeq(); got != 1 {
		t.Errorf("appliedSeq = %d, want 1 (must not advance past the gap by applying seq=3 directly)", got)
	}
	if len(appliedDocIDs) != 0 {
		t.Errorf("apply() ran for %v, want none (seq=3 must not be applied until the gap at seq=2 is resolved)", appliedDocIDs)
	}
}
