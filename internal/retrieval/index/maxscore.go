package index

import (
	"container/heap"
	"fmt"
	"sort"

	"search-eval-platform/pkg/types"
)

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

// IDFCacher is an optional interface that an IndexReader may implement to
// provide cross-query IDF caching. MaxScoreSearcher type-asserts the reader
// to this interface at construction time; if present, IDF values are reused
// across queries for the same (term, scorer) pair instead of recomputing
// math.Log each time.
type IDFCacher interface {
	IDFCached(term, scorerType string, df, N int, scorer Scorer) float64
}

// heapEntry is one element in the top-K min-heap.
type heapEntry struct {
	docID string
	score float64
}

// scoreHeap is a min-heap over heapEntry, ordered by score ascending.
// The root is always the lowest-scoring document in the current top-K.
type scoreHeap []heapEntry

func (h scoreHeap) Len() int            { return len(h) }
func (h scoreHeap) Less(i, j int) bool  { return h[i].score < h[j].score }
func (h scoreHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *scoreHeap) Push(x interface{}) { *h = append(*h, x.(heapEntry)) }
func (h *scoreHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// Min returns the current minimum score in the heap (the threshold θ).
// Returns 0 if the heap is empty.
func (h scoreHeap) Min() float64 {
	if len(h) == 0 {
		return 0
	}
	return h[0].score
}

// ReplaceMin replaces the root entry with a new one and fixes the heap.
func (h *scoreHeap) ReplaceMin(docID string, score float64) {
	(*h)[0] = heapEntry{docID, score}
	heap.Fix(h, 0)
}

// Sorted returns the heap contents as a descending-score []ScoredDocument with ranks assigned.
func (h scoreHeap) Sorted() []types.ScoredDocument {
	tmp := make(scoreHeap, len(h))
	copy(tmp, h)

	results := make([]types.ScoredDocument, 0, len(tmp))
	for tmp.Len() > 0 {
		e := heap.Pop(&tmp).(heapEntry)
		results = append(results, types.ScoredDocument{DocID: e.docID, Score: e.score})
	}
	for i, j := 0, len(results)-1; i < j; i, j = i+1, j-1 {
		results[i], results[j] = results[j], results[i]
	}
	for i := range results {
		results[i].Rank = i + 1
	}
	return results
}

// termEntry holds per-term state for the MaxScore algorithm.
type termEntry struct {
	iter *PostingIter
	ub   float64
	df   int
	idf  float64 // precomputed IDF — avoids math.Log per document hit
	term string
}

// MaxScoreSearcher implements Block-Max MaxScore query evaluation.
// It accepts any IndexReader, so it works with both InvertedIndex and
// the Segment adapter introduced in Phase 2.
type MaxScoreSearcher struct {
	index     IndexReader
	scorer    Scorer
	idfCacher IDFCacher // non-nil when index implements IDFCacher
}

// NewMaxScoreSearcher creates a MaxScoreSearcher backed by idx and scorer.
func NewMaxScoreSearcher(idx IndexReader, scorer Scorer) *MaxScoreSearcher {
	ms := &MaxScoreSearcher{index: idx, scorer: scorer}
	if c, ok := idx.(IDFCacher); ok {
		ms.idfCacher = c
	}
	return ms
}

// findPivot returns the LARGEST i where suffixUB[i] >= θ (n if none does).
// suffixUB[i] is the sum of ub over entries[i:], and entries are
// ub-ascending, so suffixUB is non-increasing — once the condition fails
// for some i it fails for every larger i too, so a forward scan can stop at
// the first failure and keep the last i that still held. Terms before the
// pivot (the low-UB head) are optional: even without them, entries[p:]
// alone could still reach θ. θ=0 until the heap fills, in which case every
// term is mandatory (p=0) since nothing can yet be safely skipped.
func findPivot(suffixUB []float64, θ float64) int {
	n := len(suffixUB)
	if θ == 0 {
		return 0
	}
	p := n
	for i := 0; i < n; i++ {
		if suffixUB[i] < θ {
			break
		}
		p = i
	}
	return p
}

// Search returns the top-K documents for the given tokens using Block-Max MaxScore.
func (s *MaxScoreSearcher) Search(tokens []string, topK int) []types.ScoredDocument {
	// Cache index-level constants — each call is a map/slice read; hoisting them
	// out avoids repeated interface dispatch in the hot loop.
	N := s.index.DocCount()
	if topK <= 0 || N == 0 {
		return nil
	}
	avgDocLen := s.index.AvgDocLen()

	// Deduplicate tokens.
	seen := make(map[string]bool)
	unique := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if !seen[t] {
			seen[t] = true
			unique = append(unique, t)
		}
	}
	tokens = unique

	// Build termEntries for terms that exist in the index.
	// Precompute IDF once per term — eliminates math.Log per document hit.
	// When the reader implements IDFCacher, IDF values are also shared across
	// queries for the same (term, scorer) pair.
	scorerType := fmt.Sprintf("%T", s.scorer)
	entries := make([]termEntry, 0, len(tokens))
	for _, t := range tokens {
		iter := s.index.Iterator(t)
		if iter == nil {
			continue
		}
		df := s.index.DF(t)
		var idf float64
		if s.idfCacher != nil {
			idf = s.idfCacher.IDFCached(t, scorerType, df, N, s.scorer)
		} else {
			idf = s.scorer.IDF(df, N)
		}
		entries = append(entries, termEntry{
			iter: iter,
			ub:   s.index.TermUB(t),
			df:   df,
			idf:  idf,
			term: t,
		})
	}
	if len(entries) == 0 {
		return nil
	}

	// Sort by ub ascending (optional terms first, highest impact last).
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].ub < entries[j].ub
	})

	// Pre-advance all iterators.
	alive := make([]bool, len(entries))
	for i := range entries {
		alive[i] = entries[i].iter.Next()
	}

	// Compute suffix upper bounds: suffixUB[i] = sum(ub for entries[i..n-1]).
	n := len(entries)
	suffixUB := make([]float64, n)
	suffixUB[n-1] = entries[n-1].ub
	for i := n - 2; i >= 0; i-- {
		suffixUB[i] = suffixUB[i+1] + entries[i].ub
	}

	h := make(scoreHeap, 0, topK)
	heap.Init(&h)
	θ := 0.0

	for {
		p := findPivot(suffixUB, θ)
		if p == n {
			break
		}

		// Candidate = minimum docID among alive mandatory terms (i >= p).
		var candidateDoc uint64
		candidateFound := false
		for i := p; i < n; i++ {
			if !alive[i] {
				continue
			}
			if !candidateFound || entries[i].iter.DocID() < candidateDoc {
				candidateDoc = entries[i].iter.DocID()
				candidateFound = true
			}
		}
		if !candidateFound {
			break
		}

		// Block-max check: if the combined block-level max impact ≤ θ,
		// skip all alive terms past their current block.
		if θ > 0 {
			blockMaxSum := 0.0
			for i := 0; i < n; i++ {
				if !alive[i] {
					continue
				}
				bmi := float64(entries[i].iter.BlockMaxImpact())
				if bmi == 0 {
					bmi = entries[i].ub
				}
				blockMaxSum += bmi
			}
			if blockMaxSum <= θ {
				allDone := true
				for i := 0; i < n; i++ {
					if !alive[i] {
						continue
					}
					blkIdx := entries[i].iter.BlockIdx()
					var skipTarget uint64
					if last, ok := entries[i].iter.SkipLastDocInBlock(blkIdx); ok {
						skipTarget = last + 1
					} else {
						skipTarget = entries[i].iter.DocID() + 1
					}
					alive[i] = entries[i].iter.SkipTo(skipTarget)
					if alive[i] {
						allDone = false
					}
				}
				if allDone {
					break
				}
				continue
			}
		}

		// Score candidateDoc.
		docStrID := s.index.DocStringID(candidateDoc)
		dl := s.index.DocLen(docStrID)
		score := 0.0

		for i := 0; i < n; i++ {
			if !alive[i] {
				continue
			}
			iter := entries[i].iter
			if alive[i] = iter.SkipTo(candidateDoc); !alive[i] {
				continue
			}
			if iter.DocID() == candidateDoc {
				score += s.scorer.ScoreWithIDF(iter.TF(), entries[i].idf, dl, avgDocLen)
				alive[i] = iter.Next()
			} else if i >= p {
				alive[i] = iter.SkipTo(candidateDoc + 1)
			}
		}

		if h.Len() < topK {
			heap.Push(&h, heapEntry{docStrID, score})
			if h.Len() == topK {
				θ = h.Min()
			}
		} else if score > θ {
			h.ReplaceMin(docStrID, score)
			θ = h.Min()
		}

		// Advance optional terms still at ≤ candidateDoc.
		for i := 0; i < p; i++ {
			if alive[i] && entries[i].iter.DocID() <= candidateDoc {
				alive[i] = entries[i].iter.SkipTo(candidateDoc + 1)
			}
		}
	}

	return h.Sorted()
}
