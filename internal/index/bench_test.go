package index_test

import (
	"fmt"
	"testing"

	"search-eval-platform/internal/analysis"
	"search-eval-platform/internal/index"
	"search-eval-platform/pkg/types"
)

// ── IndexBuilder benchmarks ───────────────────────────────────────────────────

var benchDocs []types.Document

func init() {
	benchDocs = makeBenchDocs(1000)
}

func makeBenchDocs(n int) []types.Document {
	words := []string{
		"search", "engine", "index", "query", "retrieval", "document",
		"term", "frequency", "relevance", "ranking", "bm25", "tfidf",
		"segment", "posting", "inverted", "compress", "delta", "block",
		"shard", "merge", "flush", "wal", "replica", "cluster",
	}
	docs := make([]types.Document, n)
	for i := range docs {
		// Each doc has ~20 words, cycling through vocabulary.
		text := ""
		for j := 0; j < 20; j++ {
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

// BenchmarkIndexBuild measures IndexBuilder.Add + Build() for 1000 docs.
func BenchmarkIndexBuild(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		builder := index.NewIndexBuilder()
		for _, doc := range benchDocs {
			builder.Add(doc.ID, analysis.Tokenize(doc.Text))
		}
		builder.Build()
	}
}

// BenchmarkIndexBuildNDocs parametrises the doc count.
func BenchmarkIndexBuildNDocs(b *testing.B) {
	for _, n := range []int{100, 500, 1000, 5000} {
		docs := makeBenchDocs(n)
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				builder := index.NewIndexBuilder()
				for _, doc := range docs {
					builder.Add(doc.ID, analysis.Tokenize(doc.Text))
				}
				builder.Build()
			}
		})
	}
}

// ── MaxScore search benchmarks ────────────────────────────────────────────────

// benchIndex is built once for search benchmarks (large enough to be meaningful).
var benchIndex *index.InvertedIndex

func init() {
	docs := makeBenchDocs(10_000)
	builder := index.NewIndexBuilder()
	for _, doc := range docs {
		builder.Add(doc.ID, analysis.Tokenize(doc.Text))
	}
	benchIndex = builder.Build()
}

// BenchmarkMaxScoreSearch measures top-10 search over a 10k-doc in-memory index.
func BenchmarkMaxScoreSearch(b *testing.B) {
	scorer := benchBM25Scorer{}
	searcher := index.NewMaxScoreSearcher(benchIndex, scorer)
	queries := [][]string{
		{"search", "engine"},
		{"index", "query", "retrieval"},
		{"bm25", "ranking"},
		{"segment", "merge"},
		{"document", "frequency", "term"},
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		searcher.Search(q, 10)
	}
}

// benchBM25Scorer is a minimal BM25 scorer for benchmarks.
type benchBM25Scorer struct{}

func (benchBM25Scorer) ScoreTerm(tf, df, dl int, avgDL float64, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	const k1, bParam = 1.2, 0.75
	idf := 1.0 // simplified
	tfNorm := float64(tf) * (k1 + 1) / (float64(tf) + k1*(1-bParam+bParam*float64(dl)/avgDL))
	return idf * tfNorm
}

func (benchBM25Scorer) UpperBound(df, minDL int, avgDL float64, N int) float64 {
	return (1.2 + 1) / 1.0 // simplified constant upper bound
}

func (benchBM25Scorer) IDF(df, N int) float64 {
	return 1.0 // simplified
}

func (benchBM25Scorer) ScoreWithIDF(tf int, idf float64, dl int, avgDL float64) float64 {
	const k1, bParam = 1.2, 0.75
	tfNorm := float64(tf) * (k1 + 1) / (float64(tf) + k1*(1-bParam+bParam*float64(dl)/avgDL))
	return idf * tfNorm
}
