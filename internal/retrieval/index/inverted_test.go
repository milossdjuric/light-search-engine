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
