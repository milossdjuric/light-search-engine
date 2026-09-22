package search_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/internal/search"
	"search-eval-platform/pkg/types"
)

func newTestSegMgr(t *testing.T, dataDir string) *search.SegmentManager {
	t.Helper()
	return newTestSegMgrWithShard(t, dataDir, "shard0")
}

func newTestSegMgrWithShard(t *testing.T, dataDir, shardID string) *search.SegmentManager {
	t.Helper()
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()
	sm, err := search.NewSegmentManager(shardID, dataDir, scorer, policy)
	if err != nil {
		t.Fatalf("NewSegmentManager: %v", err)
	}
	sm.Start()
	t.Cleanup(func() { sm.Close() })
	return sm
}

// TestSegmentManagerFlushTrigger verifies that after indexing documents and
// explicitly calling Flush, search still returns the expected results from the
// newly-created segment.
func TestSegmentManagerFlushTrigger(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	docs := []types.Document{
		{ID: "d1", Text: "alpha beta gamma"},
		{ID: "d2", Text: "delta epsilon zeta"},
		{ID: "d3", Text: "eta theta iota"},
	}
	for _, d := range docs {
		if err := sm.IndexDocument(d); err != nil {
			t.Fatalf("IndexDocument %s: %v", d.ID, err)
		}
	}

	// Explicitly flush so documents move from buffer into a segment file.
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	results, err := sm.Search("alpha", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Error("expected at least one result for 'alpha' after flush")
	}
	found := false
	for _, r := range results {
		if r.DocID == "d1" {
			found = true
		}
	}
	if !found {
		t.Error("expected d1 in results for 'alpha'")
	}
}

// TestSegmentManagerTombstoneHidesDoc verifies that a document is findable
// before deletion and absent from results after DeleteDocument is called.
func TestSegmentManagerTombstoneHidesDoc(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	doc := types.Document{ID: "victim", Text: "find me before delete"}
	if err := sm.IndexDocument(doc); err != nil {
		t.Fatal(err)
	}

	// Flush so the doc lives in a real segment, not just the buffer.
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// Confirm the document is visible before deletion.
	before, err := sm.Search("find", 10, nil)
	if err != nil {
		t.Fatalf("Search (before delete): %v", err)
	}
	found := false
	for _, r := range before {
		if r.DocID == "victim" {
			found = true
		}
	}
	if !found {
		t.Error("document should be findable before deletion")
	}

	// Delete and confirm the document is now hidden.
	if err := sm.DeleteDocument("victim"); err != nil {
		t.Fatalf("DeleteDocument: %v", err)
	}

	after, err := sm.Search("find", 10, nil)
	if err != nil {
		t.Fatalf("Search (after delete): %v", err)
	}
	for _, r := range after {
		if r.DocID == "victim" {
			t.Error("deleted document still appears in search results")
		}
	}
}

// TestSegmentManagerReindexAfterDeleteNoDuplicate verifies that a document
// which was flushed to a segment, deleted, then re-indexed (still sitting in
// the buffer, pre-merge) appears exactly once in search results — not once
// from the stale on-disk copy and once from the fresh buffer copy.
func TestSegmentManagerReindexAfterDeleteNoDuplicate(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	doc := types.Document{ID: "docX", Text: "the quick brown fox"}
	if err := sm.IndexDocument(doc); err != nil {
		t.Fatal(err)
	}
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := sm.DeleteDocument("docX"); err != nil {
		t.Fatalf("DeleteDocument: %v", err)
	}
	if err := sm.IndexDocument(doc); err != nil {
		t.Fatalf("re-index: %v", err)
	}

	results, err := sm.Search("quick fox", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	count := 0
	for _, r := range results {
		if r.DocID == "docX" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected docX to appear exactly once after delete+reindex, got %d", count)
	}
}

// TestSegmentManagerReindexSupersedesOldSegmentContent verifies that when a
// document is re-indexed with different text (no delete needed), the stale
// on-disk copy is no longer findable via its old content — the old segment
// copy is superseded by the buffer's live copy, not merely out-ranked.
func TestSegmentManagerReindexSupersedesOldSegmentContent(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	if err := sm.IndexDocument(types.Document{ID: "docY", Text: "original stale content alpha"}); err != nil {
		t.Fatal(err)
	}
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := sm.IndexDocument(types.Document{ID: "docY", Text: "updated fresh content beta"}); err != nil {
		t.Fatalf("re-index: %v", err)
	}

	stale, err := sm.Search("stale alpha", 10, nil)
	if err != nil {
		t.Fatalf("Search (old terms): %v", err)
	}
	for _, r := range stale {
		if r.DocID == "docY" {
			t.Error("docY matched on superseded content — old segment copy should no longer be findable")
		}
	}

	fresh, err := sm.Search("fresh beta", 10, nil)
	if err != nil {
		t.Fatalf("Search (new terms): %v", err)
	}
	found := false
	for _, r := range fresh {
		if r.DocID == "docY" {
			found = true
		}
	}
	if !found {
		t.Error("docY should be findable via its current (buffer) content")
	}
}

// TestSegmentManagerDocCount verifies that DocCount reflects the number of
// documents currently held in the buffer and any flushed segments.
func TestSegmentManagerDocCount(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	docs := []types.Document{
		{ID: "x1", Text: "hello world"},
		{ID: "x2", Text: "world peace"},
	}
	for _, d := range docs {
		if err := sm.IndexDocument(d); err != nil {
			t.Fatalf("IndexDocument %s: %v", d.ID, err)
		}
	}

	if n := sm.DocCount(); n != 2 {
		t.Errorf("DocCount before flush: want 2, got %d", n)
	}

	// DocCount should remain 2 after flushing (buffer → segment).
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if n := sm.DocCount(); n != 2 {
		t.Errorf("DocCount after flush: want 2, got %d", n)
	}
}

// TestSegmentManagerSearchBufferAndSegment verifies that documents in both a
// flushed segment and the in-memory buffer are returned by Search.
//
// Searches are performed sequentially (one after the other) to avoid
// triggering a pre-existing data race in MaxScoreSearcher.Search where
// unique := tokens[:0] re-uses the caller's slice backing array, causing
// concurrent goroutines to race on the shared memory when two searches run
// at the same time over different segments.
func TestSegmentManagerSearchBufferAndSegment(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	// Index and flush one doc so it lives in a segment.
	if err := sm.IndexDocument(types.Document{ID: "seg-doc", Text: "retrieval search engine"}); err != nil {
		t.Fatal(err)
	}
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	// Index a second doc that stays in the buffer.
	if err := sm.IndexDocument(types.Document{ID: "buf-doc", Text: "retrieval ranking systems"}); err != nil {
		t.Fatal(err)
	}

	// Search for a term present in both docs. The SegmentManager fans out
	// to the flushed segment and the in-memory buffer snapshot.
	results, err := sm.Search("retrieval", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	ids := make(map[string]bool)
	for _, r := range results {
		ids[r.DocID] = true
	}
	if !ids["seg-doc"] {
		t.Error("seg-doc (in segment) not found in search results")
	}
	if !ids["buf-doc"] {
		t.Error("buf-doc (in buffer) not found in search results")
	}
}

// TestSearchAllFetchKTight verifies that for a single-segment shard with an
// empty buffer, the search returns exactly topK results (regression guard for
// tight fetchK = topK * actualSources instead of topK * (allSegs+1)).
func TestSearchAllFetchKTight(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	for i := 0; i < 50; i++ {
		_ = sm.IndexDocument(types.Document{
			ID:   fmt.Sprintf("fetch%d", i),
			Text: fmt.Sprintf("fetchK tight optimisation keyword term%d", i),
		})
	}
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	results, err := sm.Search("fetchK", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 10 {
		t.Errorf("got %d results, want exactly 10", len(results))
	}
}

// TestInterSegmentMaxScorePruning indexes docs into two segments and verifies
// that search returns correct, deduplicated results (correctness regression
// test for seed-based inter-segment MaxScore 3rd-tier pruning).
func TestInterSegmentMaxScorePruning(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	for i := 0; i < 30; i++ {
		_ = sm.IndexDocument(types.Document{
			ID:   fmt.Sprintf("s1d%d", i),
			Text: fmt.Sprintf("pruning test unique keyword alpha term%d", i),
		})
	}
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush 1: %v", err)
	}

	for i := 0; i < 30; i++ {
		_ = sm.IndexDocument(types.Document{
			ID:   fmt.Sprintf("s2d%d", i),
			Text: fmt.Sprintf("pruning test unique keyword beta term%d", i),
		})
	}
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush 2: %v", err)
	}

	results, err := sm.Search("pruning keyword", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 10 {
		t.Errorf("want 10 results, got %d", len(results))
	}
	seen := make(map[string]struct{})
	for _, r := range results {
		if _, dup := seen[r.DocID]; dup {
			t.Errorf("duplicate docID in results: %s", r.DocID)
		}
		seen[r.DocID] = struct{}{}
	}
}

// TestSegmentManagerForceMergeDeletesPurgesTombstonedDoc verifies that a
// merge triggered via ForceMergeDeletes physically drops a tombstoned doc
// from the output segment, and that the resulting segment record's
// DeletedDocs count reflects that the doc is gone (0), not stale.
func TestSegmentManagerForceMergeDeletesPurgesTombstonedDoc(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	docs := []types.Document{
		{ID: "keep", Text: "alpha beta gamma"},
		{ID: "victim", Text: "delta epsilon zeta"},
	}
	for _, d := range docs {
		if err := sm.IndexDocument(d); err != nil {
			t.Fatalf("IndexDocument %s: %v", d.ID, err)
		}
	}
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := sm.DeleteDocument("victim"); err != nil {
		t.Fatalf("DeleteDocument: %v", err)
	}
	// Force the segment to load: DeletedDocs is only tracked for loaded
	// segments (incrementDeletedDocsLocked skips nil placeholders), and a
	// fresh Flush() leaves the segment nil until first accessed.
	if _, err := sm.Search("alpha", 10, nil); err != nil {
		t.Fatalf("Search (force load): %v", err)
	}

	if err := sm.ForceMergeDeletes(context.Background(), 0); err != nil {
		t.Fatalf("ForceMergeDeletes: %v", err)
	}

	recs := sm.SegmentRecords()
	if len(recs) != 1 {
		t.Fatalf("want 1 segment record after merge, got %d", len(recs))
	}
	if recs[0].DocCount != 1 {
		t.Errorf("merged segment DocCount: want 1 (victim purged), got %d", recs[0].DocCount)
	}
	if recs[0].DeletedDocs != 0 {
		t.Errorf("merged segment DeletedDocs: want 0 (victim physically gone), got %d", recs[0].DeletedDocs)
	}
}

// TestSegmentManagerForceMergeDeletesPrunesTombstone verifies that once a
// merge physically purges a tombstoned doc from every segment, its tombstone
// entry is also removed — leaving it behind would grow sm.tombstones
// unboundedly over a shard's lifetime and add redundant scan work to every
// future merge for a doc that's already gone.
func TestSegmentManagerForceMergeDeletesPrunesTombstone(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	docs := []types.Document{
		{ID: "keep", Text: "alpha beta gamma"},
		{ID: "victim", Text: "delta epsilon zeta"},
	}
	for _, d := range docs {
		if err := sm.IndexDocument(d); err != nil {
			t.Fatalf("IndexDocument %s: %v", d.ID, err)
		}
	}
	if err := sm.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	if err := sm.DeleteDocument("victim"); err != nil {
		t.Fatalf("DeleteDocument: %v", err)
	}
	if _, err := sm.Search("alpha", 10, nil); err != nil {
		t.Fatalf("Search (force load): %v", err)
	}

	if got := sm.Status().DeletedDocs; got != 1 {
		t.Fatalf("DeletedDocs before merge = %d, want 1", got)
	}

	if err := sm.ForceMergeDeletes(context.Background(), 0); err != nil {
		t.Fatalf("ForceMergeDeletes: %v", err)
	}

	if got := sm.Status().DeletedDocs; got != 0 {
		t.Errorf("DeletedDocs after merge = %d, want 0 (tombstone for physically-purged doc should be pruned)", got)
	}
}

// TestSegmentManagerResetDoesNotDeletePrefixCollidingShard verifies that
// Reset on shard "shard1" does not delete segment files belonging to
// "shard10", which shares a naive string prefix ("shard1") but is a
// different shard.
func TestSegmentManagerResetDoesNotDeletePrefixCollidingShard(t *testing.T) {
	dataDir := t.TempDir()

	sm1 := newTestSegMgrWithShard(t, dataDir, "shard1")
	sm10 := newTestSegMgrWithShard(t, dataDir, "shard10")

	if err := sm10.IndexDocument(types.Document{ID: "victim", Text: "keep me safe"}); err != nil {
		t.Fatalf("IndexDocument: %v", err)
	}
	if err := sm10.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	segDir := filepath.Join(dataDir, "segments")
	before, err := os.ReadDir(segDir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(before) == 0 {
		t.Fatal("expected shard10 to have written at least one segment file before reset")
	}

	if err := sm1.Reset(context.Background()); err != nil {
		t.Fatalf("shard1 Reset: %v", err)
	}

	after, err := os.ReadDir(segDir)
	if err != nil {
		t.Fatalf("ReadDir after reset: %v", err)
	}
	if len(after) != len(before) {
		t.Errorf("shard1.Reset() deleted shard10's segment files: before=%d files, after=%d files", len(before), len(after))
	}
}
