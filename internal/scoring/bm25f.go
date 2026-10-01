package scoring

import (
	"math"
)

// BM25F implements Scorer for multi-field BM25F scoring.
//
// The TF value passed to ScoreTerm is the pre-computed pseudo-TF scaled by
// BM25FScaleFactor and stored as an integer in the posting list.
// The dl and avgDocLen parameters are intentionally unused — per-field length
// normalization was applied during indexing by IndexBuilder.BuildBM25F.
type BM25F struct {
	k1 float64
}

// NewBM25F returns a BM25F scorer with the given k1 parameter.
func NewBM25F(k1 float64) *BM25F {
	return &BM25F{k1: k1}
}

// NewDefaultBM25F returns a BM25F scorer with the default k1 (1.2).
func NewDefaultBM25F() *BM25F {
	return NewBM25F(DefaultK1)
}

// ScoreTerm implements Scorer.
// tf is the stored pseudo-TF: round(actual_pseudo_tf * BM25FScaleFactor).
func (b *BM25F) ScoreTerm(tf, df, dl int, avgDocLen float64, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	pseudoTF := float64(tf) / BM25FScaleFactor
	idf := math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
	return idf * (b.k1 + 1) * pseudoTF / (b.k1 + pseudoTF)
}

// UpperBound implements Scorer.
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

// IDF implements Scorer.
func (b *BM25F) IDF(df, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	return math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
}

// ScoreWithIDF implements Scorer using a precomputed IDF.
// dl and avgDocLen are unused — per-field length normalization was applied at index time.
func (b *BM25F) ScoreWithIDF(tf int, idf float64, dl int, avgDocLen float64) float64 {
	pseudoTF := float64(tf) / BM25FScaleFactor
	return idf * (b.k1 + 1) * pseudoTF / (b.k1 + pseudoTF)
}

var _ Scorer = (*BM25F)(nil)

// FieldConfig defines BM25F scoring parameters for one document field.
type FieldConfig struct {
	Name   string  // field name, e.g. "title", "body", "url"
	Weight float64 // w_f: contribution multiplier (higher = more important)
	B      float64 // b_f: length normalization strength (0=none, 1=full)
}

// BM25FScaleFactor converts a float pseudo-TF to an integer for storage.
// Stored TF = round(pseudoTF * BM25FScaleFactor).
const BM25FScaleFactor = 1000.0
