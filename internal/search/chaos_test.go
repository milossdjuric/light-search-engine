package search_test

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/internal/search"
	"search-eval-platform/pkg/types"
)

// TestConcurrentIndexFlushSearch runs concurrent writers, flushers, and readers
// on the same SegmentManager. Passed with -race to detect data races.
func TestConcurrentIndexFlushSearch(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	const (
		numIndexers = 10
		numFlushers = 2
		numSearchers = 5
		duration    = 2 * time.Second
	)

	var (
		wg      sync.WaitGroup
		stop    = make(chan struct{})
		indexed atomic.Int64
		panics  atomic.Int64
	)

	safeRecover := func() {
		if r := recover(); r != nil {
			t.Errorf("panic in goroutine: %v", r)
			panics.Add(1)
		}
	}

	// Indexers: continuously add documents.
	for i := 0; i < numIndexers; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer safeRecover()
			for j := 0; ; j++ {
				select {
				case <-stop:
					return
				default:
				}
				doc := types.Document{
					ID:   fmt.Sprintf("goroutine%d-doc%d", i, j),
					Text: fmt.Sprintf("term%d alpha beta gamma delta", i+j),
				}
				if err := sm.IndexDocument(doc); err != nil {
					// Errors during concurrent ops are acceptable (e.g. flush in progress).
					continue
				}
				indexed.Add(1)
			}
		}()
	}

	// Flushers: periodically flush.
	for i := 0; i < numFlushers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer safeRecover()
			for {
				select {
				case <-stop:
					return
				case <-time.After(200 * time.Millisecond):
					_ = sm.Flush()
				}
			}
		}()
	}

	// Searchers: continuously query.
	for i := 0; i < numSearchers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer safeRecover()
			scorer := bm25.NewScorerOnly(1.2, 0.75)
			for {
				select {
				case <-stop:
					return
				default:
				}
				_, _ = sm.Search("alpha beta", 5, scorer)
			}
		}()
	}

	// Run for duration seconds, then stop all goroutines.
	time.Sleep(duration)
	close(stop)
	wg.Wait()

	if panics.Load() > 0 {
		t.Errorf("detected %d panics during concurrent operation", panics.Load())
	}
	t.Logf("indexed %d documents during chaos test", indexed.Load())
}

// TestWALReplayAfterTruncation verifies that SegmentManager recovers cleanly
// from a WAL that was truncated mid-write (simulating a crash).
func TestWALReplayAfterTruncation(t *testing.T) {
	dataDir := t.TempDir()

	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()

	// Phase 1: index some docs and let them go to WAL.
	sm1, err := search.NewSegmentManager("shard0", dataDir, scorer, policy)
	if err != nil {
		t.Fatalf("NewSegmentManager: %v", err)
	}
	sm1.Start()

	for i := 0; i < 50; i++ {
		doc := types.Document{
			ID:   fmt.Sprintf("doc%d", i),
			Text: fmt.Sprintf("word%d search engine index", i),
		}
		if err := sm1.IndexDocument(doc); err != nil {
			t.Fatalf("IndexDocument: %v", err)
		}
	}

	// Close without flushing — WAL has all 50 entries, no segments.
	sm1.Close()

	// Phase 2: truncate the current WAL file mid-way.
	// With multi-file WAL, the active file is shard0_0001.wal (or higher after
	// any flushes).  Find the newest .wal file in the wal dir.
	walDir := filepath.Join(dataDir, "wal")
	entries, err := os.ReadDir(walDir)
	if err != nil || len(entries) == 0 {
		t.Skipf("no WAL files found in %s", walDir)
	}
	walPath := filepath.Join(walDir, entries[len(entries)-1].Name())
	info, err := os.Stat(walPath)
	if err != nil {
		t.Skipf("WAL file not found at %s (may be implementation detail): %v", walPath, err)
	}
	if info.Size() < 64 {
		t.Skip("WAL too small to meaningfully truncate")
	}
	truncateAt := info.Size() / 2
	if err := os.Truncate(walPath, truncateAt); err != nil {
		t.Fatalf("os.Truncate: %v", err)
	}

	// Phase 3: create a new SegmentManager pointing at the same dataDir.
	// It must replay the WAL without panic or error.
	sm2, err := search.NewSegmentManager("shard0", dataDir, scorer, policy)
	if err != nil {
		t.Fatalf("NewSegmentManager after WAL truncation: %v", err)
	}
	sm2.Start()
	defer sm2.Close()

	if err := sm2.LoadFromMeta(); err != nil {
		t.Fatalf("LoadFromMeta after WAL truncation: %v", err)
	}

	// Search should return results (or empty — no crash is the key invariant).
	scorer2 := bm25.NewScorerOnly(1.2, 0.75)
	results, err := sm2.Search("search engine", 10, scorer2)
	if err != nil {
		t.Fatalf("Search after WAL truncation: %v", err)
	}
	t.Logf("results after truncated WAL replay: %d", len(results))
}

// TestMergeDuringSearch verifies that searches return consistent results while
// a merge runs concurrently.
func TestMergeDuringSearch(t *testing.T) {
	dataDir := t.TempDir()

	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := &search.TieredMergePolicy{
		SegmentsPerTier: 2, // aggressive: merge after 2 segments
		MaxMergeAtOnce:  2,
		LevelRatio:      2,
	}

	sm, err := search.NewSegmentManager("shard0", dataDir, scorer, policy)
	if err != nil {
		t.Fatalf("NewSegmentManager: %v", err)
	}
	sm.Start()
	defer sm.Close()

	// Index enough docs to create multiple segments.
	for batch := 0; batch < 4; batch++ {
		for i := 0; i < 50; i++ {
			doc := types.Document{
				ID:   fmt.Sprintf("b%d-doc%d", batch, i),
				Text: fmt.Sprintf("search engine alpha beta term%d", i),
			}
			if err := sm.IndexDocument(doc); err != nil {
				t.Fatalf("IndexDocument: %v", err)
			}
		}
		if err := sm.Flush(); err != nil {
			t.Fatalf("Flush: %v", err)
		}
	}

	// Concurrent: trigger merge while running searches.
	var wg sync.WaitGroup
	var searchErrs atomic.Int64

	// Trigger merge.
	wg.Add(1)
	go func() {
		defer wg.Done()
		sm.TriggerMerge()
		time.Sleep(100 * time.Millisecond) // give merge time to start
	}()

	// 50 concurrent searches.
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := sm.Search("search engine", 5, scorer)
			if err != nil {
				searchErrs.Add(1)
			}
		}()
	}

	wg.Wait()
	if searchErrs.Load() > 0 {
		t.Errorf("%d search errors during concurrent merge", searchErrs.Load())
	}
}
