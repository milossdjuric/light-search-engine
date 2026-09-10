package index_test

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"search-eval-platform/internal/retrieval/bm25"
	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/pkg/types"
)

// naiveBM25 scores all documents for the given tokens and returns top-K.
// This is the reference implementation to validate MaxScoreSearcher against.
func naiveBM25(idx *index.InvertedIndex, scorer index.Scorer, tokens []string, topK int) []types.ScoredDocument {
	scores := make(map[string]float64)
	N := idx.DocCount()
	avgDL := idx.AvgDocLen()

	for _, tok := range tokens {
		it := idx.Iterator(tok)
		if it == nil {
			continue
		}
		df := idx.DF(tok)
		for it.Next() {
			docID := idx.DocStringID(it.DocID())
			dl := idx.DocLen(docID)
			scores[docID] += scorer.ScoreTerm(it.TF(), df, dl, avgDL, N)
		}
	}

	type entry struct {
		docID string
		score float64
	}
	var all []entry
	for id, sc := range scores {
		all = append(all, entry{id, sc})
	}
	sort.Slice(all, func(i, j int) bool {
		if math.Abs(all[i].score-all[j].score) > 1e-12 {
			return all[i].score > all[j].score
		}
		return all[i].docID < all[j].docID
	})
	if len(all) > topK {
		all = all[:topK]
	}
	out := make([]types.ScoredDocument, len(all))
	for i, e := range all {
		out[i] = types.ScoredDocument{DocID: e.docID, Score: e.score, Rank: i + 1}
	}
	return out
}

func TestMaxScoreMatchesNaiveBM25(t *testing.T) {
	// Build 1000 random documents from a small vocabulary.
	vocab := []string{
		"search", "engine", "index", "query", "document", "term", "score",
		"rank", "retrieval", "text", "data", "fast", "efficient", "build",
		"merge", "shard", "segment", "flush", "buffer", "write",
	}
	rng := rand.New(rand.NewSource(42))

	docs := make([]types.Document, 1000)
	for i := range docs {
		wordCount := 10 + rng.Intn(40)
		words := make([]string, wordCount)
		for j := range words {
			words[j] = vocab[rng.Intn(len(vocab))]
		}
		text := ""
		for j, w := range words {
			if j > 0 {
				text += " "
			}
			text += w
		}
		docs[i] = types.Document{ID: fmt.Sprintf("doc%04d", i), Text: text}
	}

	idx := index.NewInvertedIndex(docs)
	scorer := bm25.NewScorerOnly(1.2, 0.75)

	queries := []string{
		"search engine",
		"fast retrieval",
		"document index term",
		"score rank",
		"shard segment merge",
		"buffer flush write",
		"query text data",
		"efficient build",
		"rank score retrieval",
		"index shard query",
	}

	topK := 10
	for _, q := range queries {
		tokens := index.Tokenize(q)

		maxScorer := index.NewMaxScoreSearcher(idx, scorer)
		got := maxScorer.Search(tokens, topK)

		want := naiveBM25(idx, scorer, tokens, topK)

		if len(got) != len(want) {
			t.Errorf("query %q: MaxScore len=%d, naive len=%d", q, len(got), len(want))
			continue
		}
		for i := range got {
			if got[i].DocID != want[i].DocID {
				t.Errorf("query %q rank %d: MaxScore=%s (%.6f) naive=%s (%.6f)",
					q, i+1, got[i].DocID, got[i].Score, want[i].DocID, want[i].Score)
			}
		}
	}
}

func TestMaxScoreEmptyIndex(t *testing.T) {
	idx := index.NewInvertedIndex(nil)
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	ms := index.NewMaxScoreSearcher(idx, scorer)
	results := ms.Search([]string{"anything"}, 10)
	if len(results) != 0 {
		t.Errorf("empty index: expected 0 results, got %d", len(results))
	}
}

func TestMaxScoreTopKCap(t *testing.T) {
	idx := index.NewInvertedIndex(testDocs)
	scorer := bm25.NewScorerOnly(1.2, 0.75)
	ms := index.NewMaxScoreSearcher(idx, scorer)
	results := ms.Search(index.Tokenize("the fox"), 2)
	if len(results) > 2 {
		t.Errorf("expected at most 2 results, got %d", len(results))
	}
	// Verify descending score order.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score+1e-12 {
			t.Errorf("results not sorted: [%d]=%.6f > [%d]=%.6f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}
