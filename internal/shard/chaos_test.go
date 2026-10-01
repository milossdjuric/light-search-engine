package shard_test

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/shard"
	"search-eval-platform/pkg/types"
)

// TestConcurrentIndexFlushSearch runs concurrent writers, flushers, and readers
// on the same SegmentManager. Passed with -race to detect data races.
func TestConcurrentIndexFlushSearch(t *testing.T) {
	sm := newTestSegMgr(t, t.TempDir())

	const (
		numIndexers  = 10
		numFlushers  = 2
		numSearchers = 5
		duration     = 2 * time.Second
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
			scorer := scoring.NewBM25(1.2, 0.75)
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

// TestWALReplayAfterTruncation simulates a crash that cut the WAL mid-record:
// docs are written with sync durability, the data dir is copied while the
// shard is still running (the on-disk state a crash would leave), the WAL in
// the copy is truncated halfway, and a new SegmentManager replays it. Replay
// must not fail, and must recover a non-empty prefix of the writes — every
// record before the cut, none after it.
func TestWALReplayAfterTruncation(t *testing.T) {
	scorer := scoring.NewBM25(1.2, 0.75)
	policy := shard.DefaultTieredMergePolicy()
	live := t.TempDir()

	sm1, err := shard.NewSegmentManager("shard0", live, scorer, policy, shard.SegmentManagerOptions{WALDurability: "sync"})
	if err != nil {
		t.Fatalf("NewSegmentManager: %v", err)
	}
	sm1.Start()
	const n = 50
	for i := 0; i < n; i++ {
		doc := types.Document{ID: fmt.Sprintf("doc%02d", i), Text: fmt.Sprintf("word%d search engine index", i)}
		if err := sm1.IndexDocument(doc); err != nil {
			t.Fatalf("IndexDocument: %v", err)
		}
	}

	// Crash snapshot: copy the data dir before Close (which would flush).
	crashed := t.TempDir()
	if err := os.CopyFS(crashed, os.DirFS(live)); err != nil {
		t.Fatalf("copy data dir: %v", err)
	}
	sm1.Close()

	walFiles, _ := filepath.Glob(filepath.Join(crashed, "wal", "*.wal"))
	var walPath string
	var size int64
	for _, f := range walFiles {
		if fi, err := os.Stat(f); err == nil && fi.Size() > size {
			walPath, size = f, fi.Size()
		}
	}
	if walPath == "" {
		t.Fatalf("no non-empty WAL file in crash snapshot (files: %v)", walFiles)
	}
	if err := os.Truncate(walPath, size/2); err != nil {
		t.Fatalf("os.Truncate: %v", err)
	}

	sm2, err := shard.NewSegmentManager("shard0", crashed, scorer, policy)
	if err != nil {
		t.Fatalf("NewSegmentManager after WAL truncation: %v", err)
	}
	sm2.Start()
	defer sm2.Close()
	if err := sm2.LoadFromMeta(); err != nil {
		t.Fatalf("LoadFromMeta after WAL truncation: %v", err)
	}

	results, err := sm2.Search("search engine", n, scorer)
	if err != nil {
		t.Fatalf("Search after WAL truncation: %v", err)
	}
	got := map[string]bool{}
	for _, r := range results {
		got[r.DocID] = true
	}
	if len(got) == 0 || len(got) >= n {
		t.Fatalf("recovered %d of %d docs after truncating the WAL halfway; want a non-empty proper prefix", len(got), n)
	}
	for i := 0; i < len(got); i++ {
		if id := fmt.Sprintf("doc%02d", i); !got[id] {
			t.Fatalf("recovered docs are not a prefix of the writes: %s missing but %d docs recovered", id, len(got))
		}
	}
	t.Logf("recovered %d of %d docs from the truncated WAL", len(got), n)
}

// TestMergeDuringSearch verifies that searches return consistent results while
// a merge runs concurrently.
func TestMergeDuringSearch(t *testing.T) {
	dataDir := t.TempDir()

	scorer := scoring.NewBM25(1.2, 0.75)
	policy := &shard.TieredMergePolicy{
		SegmentsPerTier: 2, // aggressive: merge after 2 segments
		MaxMergeAtOnce:  2,
		LevelRatio:      2,
	}

	sm, err := shard.NewSegmentManager("shard0", dataDir, scorer, policy)
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
