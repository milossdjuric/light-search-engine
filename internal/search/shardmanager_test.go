package search_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/internal/retrieval/bm25f"
	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/internal/search"
	"search-eval-platform/pkg/types"
)

func newTestShardManager(t *testing.T, nShards int) *search.ShardManager {
	t.Helper()
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()
	sm, err := search.NewShardManager(nShards, t.TempDir(), scorer, policy)
	if err != nil {
		t.Fatalf("NewShardManager: %v", err)
	}
	if err := sm.Start(); err != nil {
		t.Fatalf("ShardManager.Start: %v", err)
	}
	t.Cleanup(func() {
		sm.Close()
	})
	return sm
}

// TestShardManagerFNVRoutingDeterminism indexes 20 documents across 4 shards
// and confirms that a cross-shard fan-out search returns all 20.
func TestShardManagerFNVRoutingDeterminism(t *testing.T) {
	sm := newTestShardManager(t, 4)
	ctx := context.Background()

	for i := 0; i < 20; i++ {
		doc := types.Document{
			ID:   fmt.Sprintf("doc%03d", i),
			Text: fmt.Sprintf("document number %d with unique content", i),
		}
		if err := sm.IndexDoc(ctx, doc); err != nil {
			t.Fatalf("IndexDoc %s: %v", doc.ID, err)
		}
	}

	// The word "document" appears in every doc; all 20 should be returned.
	results, _, err := sm.Search(ctx, "document", 20, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 20 {
		t.Errorf("cross-shard fan-out: want 20 results, got %d", len(results))
	}
}

// TestShardManagerDeleteDoc verifies that a document deleted via DeleteDoc no
// longer appears in search results across any shard.
func TestShardManagerDeleteDoc(t *testing.T) {
	sm := newTestShardManager(t, 2)
	ctx := context.Background()

	doc := types.Document{ID: "del-me", Text: "delete this document please"}
	if err := sm.IndexDoc(ctx, doc); err != nil {
		t.Fatalf("IndexDoc: %v", err)
	}

	if err := sm.DeleteDoc(ctx, "del-me"); err != nil {
		t.Fatalf("DeleteDoc: %v", err)
	}

	results, _, err := sm.Search(ctx, "delete", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range results {
		if r.DocID == "del-me" {
			t.Error("deleted doc still appears in search results")
		}
	}
}


// TestShardManagerNumShards confirms NumShards returns the value passed at
// construction time.
func TestShardManagerNumShards(t *testing.T) {
	sm := newTestShardManager(t, 3)
	if sm.NumShards() != 3 {
		t.Errorf("NumShards: want 3, got %d", sm.NumShards())
	}
}

// TestShardManagerTotalDocs verifies TotalDocs sums doc counts across all
// shards.
func TestShardManagerTotalDocs(t *testing.T) {
	sm := newTestShardManager(t, 2)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		doc := types.Document{
			ID:   fmt.Sprintf("tdoc%d", i),
			Text: fmt.Sprintf("total docs test content %d", i),
		}
		if err := sm.IndexDoc(ctx, doc); err != nil {
			t.Fatalf("IndexDoc: %v", err)
		}
	}

	if n := sm.TotalDocs(); n != 5 {
		t.Errorf("TotalDocs: want 5, got %d", n)
	}
}

// TestShardManagerBM25FSearch verifies end-to-end BM25F scoring through
// ShardManager: documents indexed with named fields should rank by pseudo-TF
// (title hits outrank body-only hits) when searched with the BM25F scorer.
func TestShardManagerBM25FSearch(t *testing.T) {
	scorer := bm25f.NewScorerOnly(1.2)
	policy := search.DefaultTieredMergePolicy()
	fields := []index.FieldConfig{
		{Name: "title", Weight: 2.5, B: 0.45},
		{Name: "body", Weight: 1.0, B: 0.75},
	}
	sm, err := search.NewShardManager(2, t.TempDir(), scorer, policy,
		search.ShardManagerOptions{BM25FFields: fields})
	if err != nil {
		t.Fatalf("NewShardManager: %v", err)
	}
	if err := sm.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { sm.Close() })

	ctx := context.Background()

	// docA: "indexing" in title (high weight).
	if err := sm.IndexDoc(ctx, types.Document{
		ID:   "docA",
		Text: "indexing inverted index structures",
		Fields: map[string]string{
			"title": "indexing inverted index",
			"body":  "some generic content about storage",
		},
	}); err != nil {
		t.Fatalf("IndexDoc docA: %v", err)
	}
	// docB: "indexing" in body only (low weight).
	if err := sm.IndexDoc(ctx, types.Document{
		ID:   "docB",
		Text: "storage engine with indexing support",
		Fields: map[string]string{
			"title": "storage engine design",
			"body":  "this document covers indexing in depth",
		},
	}); err != nil {
		t.Fatalf("IndexDoc docB: %v", err)
	}

	results, _, err := sm.Search(ctx, "indexing", 10, scorer)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) < 2 {
		t.Fatalf("BM25F search: want at least 2 results, got %d", len(results))
	}
	if results[0].DocID != "docA" {
		t.Errorf("BM25F ranking: want docA first (title hit), got %s first", results[0].DocID)
	}
}

// TestShardManagerFlushPersistsResults verifies that documents remain
// searchable after an explicit Flush call moves them from buffer to segment.
func TestShardManagerFlushPersistsResults(t *testing.T) {
	sm := newTestShardManager(t, 2)
	ctx := context.Background()

	docs := []types.Document{
		{ID: "fp1", Text: "persistent retrieval result"},
		{ID: "fp2", Text: "persistent ranking system"},
	}
	for _, d := range docs {
		if err := sm.IndexDoc(ctx, d); err != nil {
			t.Fatalf("IndexDoc %s: %v", d.ID, err)
		}
	}

	if err := sm.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	results, _, err := sm.Search(ctx, "persistent", 10, nil)
	if err != nil {
		t.Fatalf("Search after flush: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("after flush: want 2 results, got %d", len(results))
	}
}

// TestShardManagerPartialFailure validates the new 3-value signature and
// confirms degraded=false when all shards are healthy.
func TestShardManagerPartialFailure(t *testing.T) {
	sm := newTestShardManager(t, 2)
	ctx := context.Background()

	docs := []types.Document{
		{ID: "a1", Text: "resilience partial failure test alpha"},
		{ID: "b1", Text: "resilience partial failure test beta"},
		{ID: "c1", Text: "resilience partial failure test gamma"},
		{ID: "d1", Text: "resilience partial failure test delta"},
	}
	for _, d := range docs {
		if err := sm.IndexDoc(ctx, d); err != nil {
			t.Fatalf("IndexDoc %s: %v", d.ID, err)
		}
	}
	if err := sm.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	results, degraded, err := sm.Search(ctx, "resilience", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if degraded {
		t.Error("expected degraded=false when all shards healthy")
	}
	if len(results) == 0 {
		t.Error("expected results from healthy cluster")
	}
}

// TestShardManagerSearchReturnsDegradedBool is a minimal contract test that
// verifies the Search signature returns a bool degraded value.
func TestShardManagerSearchReturnsDegradedBool(t *testing.T) {
	sm := newTestShardManager(t, 1)
	ctx := context.Background()
	if err := sm.IndexDoc(ctx, types.Document{ID: "z1", Text: "query term present"}); err != nil {
		t.Fatalf("IndexDoc: %v", err)
	}
	results, degraded, err := sm.Search(ctx, "query", 5, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if degraded {
		t.Error("single healthy shard: expected degraded=false")
	}
	if len(results) == 0 {
		t.Error("expected at least one result")
	}
}

// TestShardManagerDegradedOnCorruptSegment verifies that when one shard's
// segment files are corrupted the search returns without panicking.
// If shard0 had matching docs, degraded=true is expected; if all docs routed
// to shard1 the test is skipped (no corruption was injected).
func TestShardManagerDegradedOnCorruptSegment(t *testing.T) {
	dir := t.TempDir()
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()

	sm, err := search.NewShardManager(2, dir, scorer, policy)
	if err != nil {
		t.Fatalf("NewShardManager: %v", err)
	}
	if err := sm.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	ctx := context.Background()

	// Index docs spread across both shards (use IDs that hash to different shards).
	for i := 0; i < 20; i++ {
		d := types.Document{
			ID:   fmt.Sprintf("doc%d", i),
			Text: fmt.Sprintf("corruption resilience test term%d", i),
		}
		if err := sm.IndexDoc(ctx, d); err != nil {
			t.Fatalf("IndexDoc: %v", err)
		}
	}
	if err := sm.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	sm.Close()

	// Corrupt .seg files belonging to shard0.
	// Segments are stored at <dir>/segments/ by the LocalStore (key = "segments/<name>").
	segDir := filepath.Join(dir, "segments")
	entries, err := os.ReadDir(segDir)
	if err != nil {
		t.Fatalf("ReadDir segments: %v", err)
	}
	corruptedAny := false
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "shard0") && strings.HasSuffix(e.Name(), ".seg") {
			path := filepath.Join(segDir, e.Name())
			if werr := os.WriteFile(path, []byte("CORRUPT_GARBAGE_DATA"), 0644); werr != nil {
				t.Fatalf("corrupt segment: %v", werr)
			}
			corruptedAny = true
		}
	}
	if !corruptedAny {
		t.Skip("no shard0 segment files found — docs may have all routed to shard1")
	}

	// Reopen — shard0 will fail to load its corrupt segment.
	sm2, err := search.NewShardManager(2, dir, scorer, policy)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if err := sm2.Start(); err != nil {
		t.Fatalf("reopen Start: %v", err)
	}
	defer sm2.Close()

	results, degraded, err := sm2.Search(ctx, "resilience", 20, nil)
	if err != nil {
		t.Fatalf("Search returned error; expected partial degraded results from healthy shard1, got: %v", err)
	}
	if !degraded {
		t.Error("expected degraded=true: shard0 segment is corrupt, search should report partial results")
	}
	if len(results) == 0 {
		t.Error("expected results from healthy shard1")
	}
	t.Logf("results=%d, degraded=%v", len(results), degraded)
}

// TestShardManagerAllShardsFailed verifies that when all shard segment files
// are corrupted the search returns without panicking.
func TestShardManagerAllShardsFailed(t *testing.T) {
	dir := t.TempDir()
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()

	sm, err := search.NewShardManager(2, dir, scorer, policy)
	if err != nil {
		t.Fatalf("NewShardManager: %v", err)
	}
	if err := sm.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	ctx := context.Background()

	// Use a range wide enough that both shards receive docs under the default consistent ring.
	// af0-af19 → shard0, af20-af39 → shard1 with 2 shards and 150 vnodes.
	for i := 0; i < 40; i++ {
		d := types.Document{ID: fmt.Sprintf("af%d", i), Text: "all shards fail test keyword"}
		if err := sm.IndexDoc(ctx, d); err != nil {
			t.Fatalf("IndexDoc: %v", err)
		}
	}
	if err := sm.Flush(ctx); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	// Guard: both shards must have received docs for the "all fail" test to be meaningful.
	for i, c := range sm.PerShardDocCount() {
		if c == 0 {
			t.Skipf("shard %d received no docs — both shards must be populated", i)
		}
	}
	sm.Close()

	// Corrupt ALL .seg files (stored at <dir>/segments/ by LocalStore).
	segDir := filepath.Join(dir, "segments")
	entries, err := os.ReadDir(segDir)
	if err != nil {
		t.Fatalf("ReadDir segments: %v", err)
	}
	if len(entries) == 0 {
		t.Skip("no segment files found — nothing to corrupt")
	}
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".seg") {
			if werr := os.WriteFile(filepath.Join(segDir, e.Name()), []byte("BAD"), 0644); werr != nil {
				t.Fatalf("corrupt segment %s: %v", e.Name(), werr)
			}
		}
	}

	sm2, err := search.NewShardManager(2, dir, scorer, policy)
	if err != nil {
		t.Fatalf("reopen NewShardManager: %v", err)
	}
	if err := sm2.Start(); err != nil {
		t.Fatalf("reopen Start: %v", err)
	}
	defer sm2.Close()

	_, _, searchErr := sm2.Search(ctx, "fail", 10, nil)
	if searchErr == nil {
		t.Error("expected error when all shards have corrupt segments, got nil")
	}
	t.Logf("all-corrupt search result: err=%v", searchErr)
}

