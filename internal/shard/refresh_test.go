package shard_test

import (
	"testing"
	"time"

	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/shard"
	"search-eval-platform/pkg/types"
)

func newRefreshSegMgr(t *testing.T, refresh time.Duration) *shard.SegmentManager {
	t.Helper()
	sm, err := shard.NewSegmentManager("shard0", t.TempDir(), scoring.NewBM25(1.2, 0.75),
		shard.DefaultTieredMergePolicy(), shard.SegmentManagerOptions{RefreshInterval: refresh})
	if err != nil {
		t.Fatalf("NewSegmentManager: %v", err)
	}
	sm.Start()
	t.Cleanup(func() { sm.Close() })
	return sm
}

func searchHas(t *testing.T, sm *shard.SegmentManager, q, docID string) bool {
	t.Helper()
	res, err := sm.Search(q, 10, scoring.NewBM25(1.2, 0.75))
	if err != nil {
		t.Fatalf("Search(%q): %v", q, err)
	}
	for _, r := range res {
		if r.DocID == docID {
			return true
		}
	}
	return false
}

// indexAndSnapshot indexes a first doc and searches once, so the shard has a
// built buffer snapshot before the doc under test is written.
func indexAndSnapshot(t *testing.T, sm *shard.SegmentManager) {
	t.Helper()
	if err := sm.IndexDocument(types.Document{ID: "first", Text: "alpha beta"}); err != nil {
		t.Fatal(err)
	}
	if !searchHas(t, sm, "alpha", "first") {
		t.Fatal("first doc not visible")
	}
}

// A doc written after the buffer snapshot was built must become searchable
// once the refresh interval has passed — without waiting for a flush.
func TestBufferWriteVisibleAfterRefreshInterval(t *testing.T) {
	sm := newRefreshSegMgr(t, 20*time.Millisecond)
	indexAndSnapshot(t, sm)

	if err := sm.IndexDocument(types.Document{ID: "late", Text: "zebra crossing"}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(30 * time.Millisecond)
	if !searchHas(t, sm, "zebra", "late") {
		t.Fatal("doc written after the snapshot is not searchable after the refresh interval")
	}
}

// RefreshInterval 0 means the next search after a write sees it.
func TestBufferWriteVisibleImmediatelyWithZeroRefresh(t *testing.T) {
	sm := newRefreshSegMgr(t, 0)
	indexAndSnapshot(t, sm)

	if err := sm.IndexDocument(types.Document{ID: "late", Text: "zebra crossing"}); err != nil {
		t.Fatal(err)
	}
	if !searchHas(t, sm, "zebra", "late") {
		t.Fatal("doc not searchable immediately with RefreshInterval 0")
	}
}

// Within the interval the snapshot is deliberately reused (that is what keeps
// concurrent writers from forcing a rebuild on every search); a flush still
// makes the doc visible.
func TestBufferSnapshotReusedWithinRefreshInterval(t *testing.T) {
	sm := newRefreshSegMgr(t, time.Hour)
	indexAndSnapshot(t, sm)

	if err := sm.IndexDocument(types.Document{ID: "late", Text: "zebra crossing"}); err != nil {
		t.Fatal(err)
	}
	if searchHas(t, sm, "zebra", "late") {
		t.Fatal("snapshot was rebuilt inside the refresh interval")
	}
	if err := sm.Flush(); err != nil {
		t.Fatal(err)
	}
	if !searchHas(t, sm, "zebra", "late") {
		t.Fatal("doc not visible after flush")
	}
}
