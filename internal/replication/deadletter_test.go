package replication

import (
	"bufio"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// poisonedBacklog is the backlog a reconnect re-sends: the poisoned entry
// (seq 100) followed by a healthy one (seq 101).
func poisonedBacklog() *fakeEntryReceiver {
	return &fakeEntryReceiver{entries: []*WALEntry{
		{Seq: 100, Op: "index", DocId: "poison", Text: "bad", Metadata: map[string]string{"k": "v"}},
		{Seq: 101, Op: "index", DocId: "ok", Text: "good"},
	}}
}

// TestApplyStreamDeadLettersEntryAfterMaxAttempts verifies a permanently
// failing entry is retried (one attempt per reconnect) up to
// maxApplyAttempts, then set aside so replication continues past it instead
// of livelocking forever.
func TestApplyStreamDeadLettersEntryAfterMaxAttempts(t *testing.T) {
	var applied []string
	r := NewReplicaApplier("shard0", "unused", func(e *WALEntry) error {
		docID := e.DocId
		if docID == "poison" {
			return errors.New("boom")
		}
		applied = append(applied, docID)
		return nil
	})
	r.appliedSeq.Store(99)

	for attempt := 1; attempt < maxApplyAttempts; attempt++ {
		if err := r.applyStream(poisonedBacklog()); err == nil || errors.Is(err, io.EOF) {
			t.Fatalf("attempt %d: err = %v, want the apply error (retry via reconnect)", attempt, err)
		}
		if got := r.AppliedSeq(); got != 99 {
			t.Fatalf("attempt %d: appliedSeq = %d, want 99 (not yet given up)", attempt, got)
		}
		if n := len(r.DeadLetters()); n != 0 {
			t.Fatalf("attempt %d: %d dead letters, want 0 before the limit", attempt, n)
		}
	}

	// Final attempt: entry is dead-lettered and the stream moves on.
	if err := r.applyStream(poisonedBacklog()); !errors.Is(err, io.EOF) {
		t.Fatalf("final attempt: err = %v, want io.EOF (poisoned entry skipped, backlog drained)", err)
	}
	if got := r.AppliedSeq(); got != 101 {
		t.Errorf("appliedSeq = %d, want 101", got)
	}
	if len(applied) != 1 || applied[0] != "ok" {
		t.Errorf("applied = %v, want [ok]", applied)
	}
	dls := r.DeadLetters()
	if len(dls) != 1 {
		t.Fatalf("dead letters = %d, want 1", len(dls))
	}
	if dl := dls[0]; dl.Seq != 100 || dl.DocID != "poison" || dl.Attempts != maxApplyAttempts || dl.Err != "boom" {
		t.Errorf("dead letter = %+v, want seq 100 / poison / %d attempts / boom", dl, maxApplyAttempts)
	}
	if got := r.DeadLetterCount(); got != 1 {
		t.Errorf("DeadLetterCount = %d, want 1", got)
	}
}

// TestApplyStreamTransientFailureDoesNotDeadLetter verifies an entry that
// fails a few times and then applies is never set aside, and that its
// failure count doesn't leak into a later, different failing entry.
func TestApplyStreamTransientFailureDoesNotDeadLetter(t *testing.T) {
	fails := maxApplyAttempts - 1
	r := NewReplicaApplier("shard0", "unused", func(e *WALEntry) error {
		docID := e.DocId
		if docID == "poison" && fails > 0 {
			fails--
			return errors.New("transient")
		}
		return nil
	})
	r.appliedSeq.Store(99)

	for i := 0; i < maxApplyAttempts; i++ {
		r.applyStream(poisonedBacklog())
	}
	if got := r.AppliedSeq(); got != 101 {
		t.Fatalf("appliedSeq = %d, want 101 (entry recovered on its last try)", got)
	}
	if n := r.DeadLetterCount(); n != 0 {
		t.Fatalf("DeadLetterCount = %d, want 0 for a transient failure", n)
	}

	// A new failing entry starts its own count from 1.
	r2fail := &fakeEntryReceiver{entries: []*WALEntry{{Seq: 102, Op: "index", DocId: "poison"}}}
	fails = 1
	if err := r.applyStream(r2fail); err == nil || errors.Is(err, io.EOF) {
		t.Fatalf("err = %v, want the apply error on a fresh entry's first failure", err)
	}
	if n := r.DeadLetterCount(); n != 0 {
		t.Errorf("DeadLetterCount = %d, want 0 — a different entry must not inherit the old failure count", n)
	}
}

// TestDeadLetterPersistedToFile verifies the full entry (including text and
// metadata, so it can be replayed by hand) is appended to the dead-letter
// file as one JSON line.
func TestDeadLetterPersistedToFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "shard0.deadletter.ndjson")
	r := NewReplicaApplier("shard0", "unused", func(e *WALEntry) error {
		docID := e.DocId
		if docID == "poison" {
			return errors.New("boom")
		}
		return nil
	})
	r.SetDeadLetterPath(path)
	r.appliedSeq.Store(99)
	for i := 0; i < maxApplyAttempts; i++ {
		r.applyStream(poisonedBacklog())
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("dead-letter file not written: %v", err)
	}
	defer f.Close()
	var lines []DeadLetter
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var dl DeadLetter
		if err := json.Unmarshal(sc.Bytes(), &dl); err != nil {
			t.Fatalf("bad JSON line %q: %v", sc.Text(), err)
		}
		lines = append(lines, dl)
	}
	if len(lines) != 1 {
		t.Fatalf("lines = %d, want 1", len(lines))
	}
	dl := lines[0]
	if dl.Seq != 100 || dl.Op != "index" || dl.DocID != "poison" || dl.Text != "bad" || dl.Metadata["k"] != "v" || dl.ShardID != "shard0" {
		t.Errorf("persisted dead letter = %+v, want the full original entry", dl)
	}
}
