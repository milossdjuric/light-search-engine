package shard

import (
	"context"
	"testing"

	"search-eval-platform/internal/scoring"
	"search-eval-platform/pkg/types"
)

func openTestSM(t *testing.T, dir string) *SegmentManager {
	t.Helper()
	sm, err := NewSegmentManager("shard0", dir, scoring.NewBM25(1.2, 0.75), DefaultTieredMergePolicy())
	if err != nil {
		t.Fatalf("NewSegmentManager: %v", err)
	}
	sm.Start()
	if err := sm.LoadFromMeta(); err != nil {
		t.Fatalf("LoadFromMeta: %v", err)
	}
	return sm
}

// crashTestSM simulates a process crash: buffered WAL bytes reach the file,
// background goroutines stop, but the in-memory buffer is never flushed to a
// segment (Close would flush it).
func crashTestSM(sm *SegmentManager) {
	sm.walMu.Lock()
	sm.walBuf.Flush() //nolint:errcheck
	sm.walMu.Unlock()
	close(sm.stopCh)
	sm.wg.Wait()
	sm.flushWg.Wait()
}

// TestWALSeqContinuesAfterRestartWithFlushedWAL is a regression test for
// nextSeq restarting at 1 when every WAL entry had already been flushed into
// a segment before a clean shutdown. Replay skips seq <= the manifest's
// MaxFlushSeq, so documents written after that restart (seq 1, 2, ...) were
// silently dropped by the next crash recovery.
func TestWALSeqContinuesAfterRestartWithFlushedWAL(t *testing.T) {
	dir := t.TempDir()

	sm1 := openTestSM(t, dir)
	for _, id := range []string{"a", "b", "c"} {
		if err := sm1.IndexDocument(types.Document{ID: id, Text: "alpha " + id}); err != nil {
			t.Fatal(err)
		}
	}
	if err := sm1.Close(); err != nil { // flushes → segment flush_seq 3, WAL rotated away
		t.Fatal(err)
	}

	sm2 := openTestSM(t, dir)
	if err := sm2.IndexDocument(types.Document{ID: "after-restart", Text: "bravo"}); err != nil {
		t.Fatal(err)
	}
	crashTestSM(sm2)

	sm3 := openTestSM(t, dir)
	defer sm3.Close()
	res, err := sm3.Search("bravo", 10, scoring.NewBM25(1.2, 0.75))
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].DocID != "after-restart" {
		t.Fatalf("search after crash recovery = %v, want [after-restart] — doc written after a restart was lost on replay", res)
	}
}

// TestResetKeepsWALSeqMonotonic verifies /admin/reset never rewinds sequence
// numbers — not in memory, and not after a restart (the floor is persisted in
// the manifest) — so replicas tracking the primary's seq never mistake new
// writes for already-applied ones. It also emits a "reset" entry to the WAL
// hook so replicas can wipe their copy too.
func TestResetKeepsWALSeqMonotonic(t *testing.T) {
	dir := t.TempDir()
	sm := openTestSM(t, dir)

	var hooked []WALEntry
	sm.SetWALHook(func(e WALEntry) { hooked = append(hooked, e) })

	for _, id := range []string{"a", "b"} {
		if err := sm.IndexDocument(types.Document{ID: id, Text: "x"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := sm.Reset(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := sm.IndexDocument(types.Document{ID: "c", Text: "x"}); err != nil {
		t.Fatal(err)
	}

	if len(hooked) != 4 {
		t.Fatalf("hook saw %d entries, want 4 (index a, index b, reset, index c): %+v", len(hooked), hooked)
	}
	if hooked[2].Op != opReset {
		t.Errorf("entry 3 op = %q, want %q", hooked[2].Op, opReset)
	}
	for i := 1; i < len(hooked); i++ {
		if hooked[i].Seq != hooked[i-1].Seq+1 {
			t.Fatalf("seqs not contiguous and increasing across reset: %d then %d", hooked[i-1].Seq, hooked[i].Seq)
		}
	}
	lastSeq := hooked[3].Seq
	if err := sm.Close(); err != nil {
		t.Fatal(err)
	}

	sm2 := openTestSM(t, dir)
	defer sm2.Close()
	var after []WALEntry
	sm2.SetWALHook(func(e WALEntry) { after = append(after, e) })
	if err := sm2.IndexDocument(types.Document{ID: "d", Text: "x"}); err != nil {
		t.Fatal(err)
	}
	if len(after) != 1 || after[0].Seq <= lastSeq {
		t.Fatalf("seq after restart = %+v, want > %d", after, lastSeq)
	}
}

// TestScanWALReturnsEntriesAfterSeq verifies ScanWAL yields every WAL entry
// above afterSeq in order, with fields and metadata intact, for both the
// plain and the LZ4-compressed binary WAL formats (the replication primary's
// catch-up reads through it).
func TestScanWALReturnsEntriesAfterSeq(t *testing.T) {
	for _, compression := range []string{"", "lz4"} {
		t.Run("compression="+compression, func(t *testing.T) {
			sm, err := NewSegmentManager("shard0", t.TempDir(), scoring.NewBM25(1.2, 0.75), DefaultTieredMergePolicy(),
				SegmentManagerOptions{WALCompression: compression})
			if err != nil {
				t.Fatal(err)
			}
			sm.Start()
			defer sm.Close()
			if err := sm.LoadFromMeta(); err != nil {
				t.Fatal(err)
			}

			docs := []types.Document{
				{ID: "a", Text: "one"},
				{ID: "b", Text: "two", Fields: map[string]string{"title": "T"}, Metadata: map[string]string{"k": "v"}},
				{ID: "c", Text: "three"},
			}
			for _, d := range docs {
				if err := sm.IndexDocument(d); err != nil {
					t.Fatal(err)
				}
			}
			if err := sm.DeleteDocument("a"); err != nil {
				t.Fatal(err)
			}

			var got []WALEntry
			if err := sm.ScanWAL(1, func(e WALEntry) error { got = append(got, e); return nil }); err != nil {
				t.Fatal(err)
			}
			if len(got) != 3 {
				t.Fatalf("got %d entries, want 3 (seq 2..4): %+v", len(got), got)
			}
			for i, want := range []struct {
				seq int64
				op  string
				id  string
			}{{2, opIndex, "b"}, {3, opIndex, "c"}, {4, opDelete, "a"}} {
				if got[i].Seq != want.seq || got[i].Op != want.op || got[i].DocID != want.id {
					t.Errorf("entry %d = {%d %s %s}, want {%d %s %s}", i, got[i].Seq, got[i].Op, got[i].DocID, want.seq, want.op, want.id)
				}
			}
			if got[0].Text != "two" || got[0].Fields["title"] != "T" || got[0].Metadata["k"] != "v" {
				t.Errorf("entry b lost text/fields/metadata: %+v", got[0])
			}
		})
	}
}
