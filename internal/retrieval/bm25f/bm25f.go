package bm25f

import (
	"math"

	"search-eval-platform/internal/retrieval/index"
)

const defaultK1 = 1.2

// BM25F implements index.Scorer for multi-field BM25F scoring.
//
// The TF value passed to ScoreTerm is the pre-computed pseudo-TF scaled by
// index.BM25FScaleFactor and stored as an integer in the posting list.
// The dl and avgDocLen parameters are intentionally unused — per-field length
// normalization was applied during indexing by IndexBuilder.BuildBM25F.
type BM25F struct {
	k1 float64
}

// NewScorerOnly returns a BM25F scorer with the given k1 parameter.
func NewScorerOnly(k1 float64) *BM25F {
	return &BM25F{k1: k1}
}

// NewDefaultScorerOnly returns a BM25F scorer with the default k1 (1.2).
func NewDefaultScorerOnly() *BM25F {
	return NewScorerOnly(defaultK1)
}

// ScoreTerm implements index.Scorer.
// tf is the stored pseudo-TF: round(actual_pseudo_tf * BM25FScaleFactor).
func (b *BM25F) ScoreTerm(tf, df, dl int, avgDocLen float64, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	pseudoTF := float64(tf) / index.BM25FScaleFactor
	idf := math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
	return idf * (b.k1 + 1) * pseudoTF / (b.k1 + pseudoTF)
}

// UpperBound implements index.Scorer.
// Returns a conservative upper bound: IDF * (k1+1), which is the limit of the
// saturation term as pseudoTF → ∞. The actual bound is stored in TermUB
// (computed from actual scores during BuildBM25F).
func (b *BM25F) UpperBound(df, minDocLen int, avgDocLen float64, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	idf := math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
	return idf * (b.k1 + 1)
}

// IDF implements index.Scorer.
func (b *BM25F) IDF(df, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	return math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
}

// ScoreWithIDF implements index.Scorer using a precomputed IDF.
// dl and avgDocLen are unused — per-field length normalization was applied at index time.
func (b *BM25F) ScoreWithIDF(tf int, idf float64, dl int, avgDocLen float64) float64 {
	pseudoTF := float64(tf) / index.BM25FScaleFactor
	return idf * (b.k1 + 1) * pseudoTF / (b.k1 + pseudoTF)
}

var _ index.Scorer = (*BM25F)(nil)
