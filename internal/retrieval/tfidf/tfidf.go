package tfidf

import (
	"math"

	"search-eval-platform/internal/retrieval/index"
)

// TFIDF implements index.Scorer using TF-IDF ranking.
type TFIDF struct{}

// NewScorerOnly returns a TFIDF scorer.
func NewScorerOnly() *TFIDF {
	return &TFIDF{}
}

// ScoreTerm implements index.Scorer.
// score = log(N / df) * log(1 + tf)
func (t *TFIDF) ScoreTerm(tf, df, dl int, avgDocLen float64, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	idf := math.Log(float64(N) / float64(df))
	tfLog := math.Log(1 + float64(tf))
	return idf * tfLog
}

// UpperBound implements index.Scorer.
// Maximum TF-IDF score at tf=1: log(N/df) * log(2)
func (t *TFIDF) UpperBound(df, minDocLen int, avgDocLen float64, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	idf := math.Log(float64(N) / float64(df))
	return idf * math.Log(2)
}

// IDF implements index.Scorer.
func (t *TFIDF) IDF(df, N int) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	return math.Log(float64(N) / float64(df))
}

// ScoreWithIDF implements index.Scorer using a precomputed IDF.
func (t *TFIDF) ScoreWithIDF(tf int, idf float64, dl int, avgDocLen float64) float64 {
	return idf * math.Log(1+float64(tf))
}

// Ensure TFIDF satisfies index.Scorer at compile time.
var _ index.Scorer = (*TFIDF)(nil)
