package scoring

import (
	"math"
)

// DefaultK1 and DefaultB are the standard BM25 parameters.
const (
	DefaultK1 = 1.2
	DefaultB  = 0.75
)

// BM25 implements Scorer using the BM25 ranking function.
type BM25 struct {
	k1 float64
	b  float64
}

// NewBM25 returns a BM25 scorer configured with custom parameters.
// Use this when SegmentManager or ShardManager need a stateless scorer.
func NewBM25(k1, b float64) *BM25 {
	return &BM25{k1: k1, b: b}
}

// NewDefaultBM25 returns a BM25 scorer with default parameters (k1=1.2, b=0.75).
func NewDefaultBM25() *BM25 {
	return NewBM25(DefaultK1, DefaultB)
}

// ScoreTerm implements Scorer.
// idf = ln((N - df + 0.5) / (df + 0.5) + 1)
// tfNorm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgDocLen))
func (bm *BM25) ScoreTerm(tf, df, dl int, avgDocLen float64, N int) float64 {
	idf := math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
	tfNorm := float64(tf) * (bm.k1 + 1) / (float64(tf) + bm.k1*(1-bm.b+bm.b*float64(dl)/avgDocLen))
	return idf * tfNorm
}

// UpperBound implements Scorer.
// Maximum BM25 score for this term occurs at tf=1 and the smallest document length.
// We use dl=1 as a conservative lower bound on document length.
func (bm *BM25) UpperBound(df, minDocLen int, avgDocLen float64, N int) float64 {
	dl := minDocLen
	if dl < 1 {
		dl = 1
	}
	return bm.ScoreTerm(1, df, dl, avgDocLen, N)
}

// IDF implements Scorer.
func (bm *BM25) IDF(df, N int) float64 {
	return math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
}

// ScoreWithIDF implements Scorer using a precomputed IDF.
func (bm *BM25) ScoreWithIDF(tf int, idf float64, dl int, avgDocLen float64) float64 {
	tfNorm := float64(tf) * (bm.k1 + 1) / (float64(tf) + bm.k1*(1-bm.b+bm.b*float64(dl)/avgDocLen))
	return idf * tfNorm
}

// Ensure BM25 satisfies Scorer at compile time.
var _ Scorer = (*BM25)(nil)
