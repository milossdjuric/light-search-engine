package search_test

import (
	"fmt"
	"path/filepath"
	"testing"

	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/internal/search"
	"search-eval-platform/pkg/types"
)

// ── Helper ────────────────────────────────────────────────────────────────────

func makeBenchSegmentDocs(n int) []types.Document {
	words := []string{
		"search", "engine", "index", "query", "retrieval", "document",
		"term", "frequency", "ranking", "segment", "merge", "flush",
	}
	docs := make([]types.Document, n)
	for i := range docs {
		text := ""
		for j := 0; j < 15; j++ {
			if text != "" {
				text += " "
			}
			text += words[(i+j)%len(words)]
		}
		docs[i] = types.Document{
			ID:   fmt.Sprintf("doc%d", i),
			Text: text,
		}
	}
	return docs
}

// ── Segment write/load benchmark ──────────────────────────────────────────────

// BenchmarkSegmentWriteLoad measures the round-trip cost of writing then
// loading a segment with 1000 documents.
func BenchmarkSegmentWriteLoad(b *testing.B) {
	docs := makeBenchSegmentDocs(1000)
	dir := b.TempDir()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		path := filepath.Join(dir, fmt.Sprintf("seg%d.seg", i))
		builder := index.NewIndexBuilder()
		for _, doc := range docs {
			builder.Add(doc.ID, index.Tokenize(doc.Text))
		}
		if err := search.WriteSegment(path, builder); err != nil {
			b.Fatalf("WriteSegment: %v", err)
		}
		if _, err := search.LoadSegment(path); err != nil {
			b.Fatalf("LoadSegment: %v", err)
		}
	}
}

// BenchmarkSegmentSearch measures search latency on a single loaded segment.
func BenchmarkSegmentSearch(b *testing.B) {
	docs := makeBenchSegmentDocs(2000)
	dir := b.TempDir()
	path := filepath.Join(dir, "bench.seg")

	builder := index.NewIndexBuilder()
	for _, doc := range docs {
		builder.Add(doc.ID, index.Tokenize(doc.Text))
	}
	if err := search.WriteSegment(path, builder); err != nil {
		b.Fatalf("WriteSegment: %v", err)
	}
	seg, err := search.LoadSegment(path)
	if err != nil {
		b.Fatalf("LoadSegment: %v", err)
	}

	scorer := bm25.NewScorerOnly(1.2, 0.75)
	tokens := []string{"search", "engine", "index"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seg.Search(tokens, 10, scorer)
	}
}

// ── ShardManager search benchmark ────────────────────────────────────────────

// BenchmarkShardManagerSearch measures end-to-end search through ShardManager.
func BenchmarkShardManagerSearch(b *testing.B) {
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	policy := search.DefaultTieredMergePolicy()

	shards, err := search.NewShardManager(2, b.TempDir(), scorer, policy)
	if err != nil {
		b.Fatalf("NewShardManager: %v", err)
	}
	if err := shards.Start(); err != nil {
		b.Fatalf("Start: %v", err)
	}
	b.Cleanup(func() { shards.Close() })

	docs := makeBenchSegmentDocs(500)
	for _, doc := range docs {
		if err := shards.IndexDoc(b.Context(), doc); err != nil {
			b.Fatalf("IndexDoc: %v", err)
		}
	}
	if err := shards.Flush(b.Context()); err != nil {
		b.Fatalf("Flush: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = shards.Search(b.Context(), "search engine", 10, scorer)
	}
}

