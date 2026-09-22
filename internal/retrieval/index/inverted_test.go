package index_test

import (
	"fmt"
	"testing"

	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/pkg/types"
)

var testDocs = []types.Document{
	{ID: "a", Text: "the quick brown fox jumps over the lazy dog"},
	{ID: "b", Text: "quick brown foxes run fast over hills"},
	{ID: "c", Text: "lazy dogs sleep all day in the sun"},
	{ID: "d", Text: "the fox and the hound are good friends"},
}

func TestNewInvertedIndexBuildAndSearch(t *testing.T) {
	idx := index.NewInvertedIndex(testDocs)
	if idx.DocCount() != len(testDocs) {
		t.Fatalf("DocCount: want %d, got %d", len(testDocs), idx.DocCount())
	}
	if idx.AvgDocLen() <= 0 {
		t.Error("AvgDocLen should be positive")
	}
}

func TestPostingIterNext(t *testing.T) {
	idx := index.NewInvertedIndex(testDocs)
	it := idx.Iterator("quick")
	if it == nil {
		t.Fatal("Iterator returned nil for 'quick'")
	}
	count := 0
	for it.Next() {
		count++
		if it.TF() <= 0 {
			t.Error("TF should be > 0")
		}
	}
	if count != 2 {
		t.Errorf("'quick' should appear in 2 docs, got %d", count)
	}
}

func TestPostingIterMissingTerm(t *testing.T) {
	idx := index.NewInvertedIndex(testDocs)
	it := idx.Iterator("nonexistent_term_xyz")
	if it != nil {
		t.Error("Iterator for unknown term should return nil")
	}
}

func TestPostingIterSkipTo(t *testing.T) {
	docs := make([]types.Document, 300)
	for i := range docs {
		docs[i] = types.Document{
			ID:   fmt.Sprintf("doc%04d", i),
			Text: "common term appears here",
		}
	}
	idx := index.NewInvertedIndex(docs)
	it := idx.Iterator("common")
	if it == nil {
		t.Fatal("Iterator for 'common' should not be nil")
	}
	if !it.Next() {
		t.Fatal("expected at least one posting")
	}
	firstID := it.DocID()
	target := firstID + 50
	found := it.SkipTo(target)
	if found && it.DocID() < target {
		t.Errorf("after SkipTo(%d): docID=%d is less than target", target, it.DocID())
	}
	// SkipTo beyond all documents should return false.
	if it.SkipTo(^uint64(0)) {
		t.Error("SkipTo(MaxUint64) should return false")
	}
}

// TestIndexBuilderReAddBeforeBuildDoesNotDuplicatePostings verifies that
// calling Add() twice for the same docID before Build() (an upsert within one
// flush window) results in the second call's content fully replacing the
// first's — not both being merged into duplicate postings for the same doc.
func TestIndexBuilderReAddBeforeBuildDoesNotDuplicatePostings(t *testing.T) {
	b := index.NewIndexBuilder()
	defer b.Close()

	b.Add("d1", []string{"hello", "world"})
	b.Add("d1", []string{"hello", "there"}) // re-added before Build(): last write should win

	idx := b.Build()

	if idx.DocCount() != 1 {
		t.Fatalf("DocCount: want 1, got %d", idx.DocCount())
	}

	it := idx.Iterator("hello")
	if it == nil {
		t.Fatal("Iterator returned nil for 'hello'")
	}
	count := 0
	for it.Next() {
		count++
	}
	if count != 1 {
		t.Errorf("'hello' should appear in exactly 1 doc after re-Add, got %d occurrences (duplicate postings for re-indexed doc)", count)
	}

	// The re-add replaced the doc's content: "world" (only in the first,
	// superseded Add) must be gone; "there" (from the winning second Add)
	// must be present.
	if it2 := idx.Iterator("world"); it2 != nil && it2.Next() {
		t.Error("term 'world' from the superseded first Add() should not appear in the index")
	}
	it3 := idx.Iterator("there")
	if it3 == nil || !it3.Next() {
		t.Error("term 'there' from the second (winning) Add() should appear in the index")
	}
}

// TestIndexBuilderSnapshotAfterReAddIsRepeatable verifies that Snapshot()
// (documented as callable "without clearing the builder") can be called a
// second time after a re-Add() happened before the first call, without
// panicking or corrupting doc lengths — the superseded-doc compaction must
// not mutate the builder's own state.
func TestIndexBuilderSnapshotAfterReAddIsRepeatable(t *testing.T) {
	b := index.NewIndexBuilder()
	defer b.Close()

	b.Add("d1", []string{"hello", "world"})
	b.Add("d1", []string{"hello", "there"}) // re-added before Build(): triggers supersede compaction

	first := b.Snapshot()
	if first.DocCount() != 1 {
		t.Fatalf("first Snapshot DocCount: want 1, got %d", first.DocCount())
	}

	second := b.Snapshot() // must not panic
	if second.DocCount() != 1 {
		t.Fatalf("second Snapshot DocCount: want 1, got %d", second.DocCount())
	}
	if got, want := second.DocLen("d1"), first.DocLen("d1"); got != want {
		t.Errorf("second Snapshot DocLen(d1) = %d, want %d (same as first)", got, want)
	}
}

func TestMemoryEstimateGrows(t *testing.T) {
	b := index.NewIndexBuilder()
	before := b.MemoryEstimate()
	b.Add("doc1", index.Tokenize("hello world this is a test"))
	after := b.MemoryEstimate()
	if after <= before {
		t.Errorf("MemoryEstimate should grow: before=%d after=%d", before, after)
	}
}

func TestIndexBuilderBuildEmpty(t *testing.T) {
	b := index.NewIndexBuilder()
	idx := b.Build()
	if idx.DocCount() != 0 {
		t.Errorf("empty build: DocCount=%d", idx.DocCount())
	}
}

func TestTermUB(t *testing.T) {
	idx := index.NewInvertedIndex(testDocs)
	ub := idx.TermUB("quick")
	if ub <= 0 {
		t.Errorf("TermUB for 'quick' should be > 0, got %f", ub)
	}
	if idx.TermUB("nonexistent_xyz") != 0 {
		t.Error("TermUB for unknown term should be 0")
	}
}

func TestDocLen(t *testing.T) {
	idx := index.NewInvertedIndex(testDocs)
	dl := idx.DocLen("a")
	if dl <= 0 {
		t.Errorf("DocLen for 'a' should be > 0, got %d", dl)
	}
	if idx.DocLen("nonexistent") != 0 {
		t.Error("DocLen for unknown doc should be 0")
	}
}
