// Package scoring holds the ranking functions (BM25, TF-IDF, BM25F) behind the
// Scorer interface used by the MaxScore query engine.
package scoring

// Scorer computes BM25 (or TF-IDF) scores and upper bounds.
type Scorer interface {
	// ScoreTerm returns the score contribution of one term occurrence.
	ScoreTerm(tf, df, dl int, avgDocLen float64, N int) float64

	// UpperBound returns the maximum possible ScoreTerm value for this term.
	// Used to compute per-term UBt for MaxScore pruning.
	UpperBound(df, minDocLen int, avgDocLen float64, N int) float64

	// IDF returns the inverse document frequency component for a term.
	// Depends only on df and N, so it can be precomputed once per query term.
	IDF(df, N int) float64

	// ScoreWithIDF scores a single term hit using a precomputed IDF value.
	// Avoids redundant math.Log calls in the MaxScore inner loop.
	ScoreWithIDF(tf int, idf float64, dl int, avgDocLen float64) float64
}
