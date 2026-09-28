package search

import (
	"fmt"
	"reflect"
	"testing"

	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/pkg/types"
)

// TestSearch_CachesBufferDocIDs_AcrossRepeatedQueries is a white-box
// regression test for Search() rebuilding a map[string]struct{} of every
// buffered doc ID from scratch on every call, even when the buffer hasn't
// changed since the last query. bufferDocIDs plays the same "point-in-time
// buffer snapshot" role as bufferIdx, which is already cached and only
// rebuilt after a flush (with documented-acceptable staleness in between —
// see the comment above the bufferIdx rebuild in Search()); bufferDocIDs
// should follow the same caching lifecycle instead of paying O(buffer size)
// map-build cost on every single query.
func TestSearch_CachesBufferDocIDs_AcrossRepeatedQueries(t *testing.T) {
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := DefaultTieredMergePolicy()
	sm, err := NewSegmentManager("shard0", t.TempDir(), scorer, policy)
	if err != nil {
		t.Fatalf("NewSegmentManager: %v", err)
	}
	sm.Start()
	t.Cleanup(func() { sm.Close() })

	if err := sm.IndexDocument(types.Document{ID: "d1", Text: "alpha beta gamma"}); err != nil {
		t.Fatalf("IndexDocument: %v", err)
	}

	if _, err := sm.Search("alpha", 10, nil); err != nil {
		t.Fatalf("Search (1st): %v", err)
	}
	sm.mu.RLock()
	first := sm.bufferDocIDs
	sm.mu.RUnlock()
	if first == nil {
		t.Fatal("expected sm.bufferDocIDs to be populated after the first Search")
	}

	if _, err := sm.Search("alpha", 10, nil); err != nil {
		t.Fatalf("Search (2nd): %v", err)
	}
	sm.mu.RLock()
	second := sm.bufferDocIDs
	sm.mu.RUnlock()

	if reflect.ValueOf(first).Pointer() != reflect.ValueOf(second).Pointer() {
		t.Error("Search rebuilt bufferDocIDs on a repeated query with no intervening writes; expected the cached map to be reused (same underlying map as bufferIdx's caching)")
	}
}

func makeLiveBufferBenchDocs(n int) []types.Document {
	vocab := []string{"search", "engine", "index", "query", "retrieval", "bm25", "ranking", "segment", "merge", "document"}
	docs := make([]types.Document, n)
	for i := 0; i < n; i++ {
		docs[i] = types.Document{
			ID:   fmt.Sprintf("doc-%d", i),
			Text: fmt.Sprintf("%s %s %s %s the quick brown fox %d", vocab[i%len(vocab)], vocab[(i+1)%len(vocab)], vocab[(i+3)%len(vocab)], vocab[(i+7)%len(vocab)], i),
		}
	}
	return docs
}

// BenchmarkLiveBufferSearch measures SegmentManager.Search against the
// in-memory buffer (no Flush), the code path fixed by the UseFOR32 forwarding
// bug in shard_search.go. Run with -cpuprofile and compare against
// BenchmarkMaxScoreSearch's profile: the UseFOR32=true sub-benchmark should
// no longer show intcomp.UncompressDeltaVarByteUint64 as a hot function.
func BenchmarkLiveBufferSearch(b *testing.B) {
	for _, useFOR32 := range []bool{false, true} {
		name := "UseFOR32=false"
		if useFOR32 {
			name = "UseFOR32=true"
		}
		b.Run(name, func(b *testing.B) {
			scorer := bm25.NewScorerOnly(1.2, 0.75)
			policy := DefaultTieredMergePolicy()
			sm, err := NewSegmentManager("shard0", b.TempDir(), scorer, policy, SegmentManagerOptions{
				UseFOR32: useFOR32,
			})
			if err != nil {
				b.Fatalf("NewSegmentManager: %v", err)
			}
			sm.Start()
			b.Cleanup(func() { sm.Close() })

			for _, doc := range makeLiveBufferBenchDocs(10_000) {
				if err := sm.IndexDocument(doc); err != nil {
					b.Fatalf("IndexDocument: %v", err)
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := sm.Search("search engine", 10, nil); err != nil {
					b.Fatalf("Search: %v", err)
				}
			}
		})
	}
}

// TestSearch_BufferSnapshot_HonorsUseFOR32 is a white-box regression test for
// a bug where SegmentManagerOptions.UseFOR32 was correctly threaded into the
// on-disk segment writer (flush.go) but silently dropped when Search rebuilds
// the in-memory buffer snapshot (shard_search.go), so a shard configured with
// UseFOR32: true still decoded live (unflushed) query traffic through the
// slower intcomp-based codec instead of the SIMD-accelerated FOR32 one.
func TestSearch_BufferSnapshot_HonorsUseFOR32(t *testing.T) {
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := DefaultTieredMergePolicy()
	sm, err := NewSegmentManager("shard0", t.TempDir(), scorer, policy, SegmentManagerOptions{
		UseFOR32: true,
	})
	if err != nil {
		t.Fatalf("NewSegmentManager: %v", err)
	}
	sm.Start()
	t.Cleanup(func() { sm.Close() })

	if err := sm.IndexDocument(types.Document{ID: "d1", Text: "alpha beta gamma"}); err != nil {
		t.Fatalf("IndexDocument: %v", err)
	}

	// Search (without an intervening Flush) forces shard_search.go to build
	// sm.bufferIdx from the still-in-memory buffer.
	if _, err := sm.Search("alpha", 10, nil); err != nil {
		t.Fatalf("Search: %v", err)
	}

	sm.mu.RLock()
	bufIdx := sm.bufferIdx
	sm.mu.RUnlock()

	if bufIdx == nil {
		t.Fatal("expected sm.bufferIdx to be populated after Search")
	}
	if !bufIdx.UsesFOR32() {
		t.Error("SegmentManager configured with UseFOR32: true, but the buffer snapshot used for live queries was built without it")
	}
}
