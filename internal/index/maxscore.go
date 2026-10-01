package index

import (
	"container/heap"
	"fmt"
	"sort"

	"search-eval-platform/internal/scoring"
	"search-eval-platform/pkg/types"
)

// IDFCacher is an optional interface that an IndexReader may implement to
// provide cross-query IDF caching. MaxScoreSearcher type-asserts the reader
// to this interface at construction time; if present, IDF values are reused
// across queries for the same (term, scorer) pair instead of recomputing
// math.Log each time.
type IDFCacher interface {
	IDFCached(term, scorerType string, df, N int, scorer scoring.Scorer) float64
}

// heapEntry is one element in the top-K min-heap.
type heapEntry struct {
	docNum uint64 // numeric doc ID; resolved to the string ID only for the final top-K
	score  float64
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
func (h *scoreHeap) ReplaceMin(docNum uint64, score float64) {
	(*h)[0] = heapEntry{docNum, score}
	heap.Fix(h, 0)
}

// Sorted returns the heap contents as a descending-score []ScoredDocument with
// ranks assigned, resolving numeric doc IDs through docID.
func (h scoreHeap) Sorted(docID func(uint64) string) []types.ScoredDocument {
	tmp := make(scoreHeap, len(h))
	copy(tmp, h)

	results := make([]types.ScoredDocument, 0, len(tmp))
	for tmp.Len() > 0 {
		e := heap.Pop(&tmp).(heapEntry)
		results = append(results, types.ScoredDocument{DocID: docID(e.docNum), Score: e.score})
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
	qtf  float64 // query-term frequency: weight applied to this term's score and bounds
	term string
}

// MaxScoreSearcher implements Block-Max MaxScore query evaluation.
// It accepts any IndexReader, so it works with both InvertedIndex and
// the Segment adapter introduced in Phase 2.
type MaxScoreSearcher struct {
	index     IndexReader
	scorer    scoring.Scorer
	idfCacher IDFCacher         // non-nil when index implements IDFCacher
	docLenNum DocLenByNumReader // non-nil when index implements DocLenByNumReader
}

// DocLenByNumReader is an optional IndexReader extension returning a
// document's length by numeric ID. MaxScore uses it to skip the string-ID
// round trip (DocStringID + map lookup) for every scored candidate.
type DocLenByNumReader interface {
	DocLenByNum(numID uint64) int
}

// NewMaxScoreSearcher creates a MaxScoreSearcher backed by idx and scorer.
func NewMaxScoreSearcher(idx IndexReader, scorer scoring.Scorer) *MaxScoreSearcher {
	ms := &MaxScoreSearcher{index: idx, scorer: scorer}
	if c, ok := idx.(IDFCacher); ok {
		ms.idfCacher = c
	}
	if d, ok := idx.(DocLenByNumReader); ok {
		ms.docLenNum = d
	}
	return ms
}

// findPivot returns the MaxScore pivot p: entries[:p] are non-essential
// (never used to generate candidates), entries[p:] are essential.
//
// Entries are ub-ascending and suffixUB[i] = sum(ub over entries[i:]), so
// the combined upper bound of the head entries[:i] is suffixUB[0]-suffixUB[i].
// Skipping the head is only rank-safe while that bound is below θ: a doc
// matching nothing but head terms then cannot enter the top-K. p is the
// largest such i; p == n means no document can beat θ any more. θ == 0 (heap
// not yet full) makes every term essential.
//
// The UBs are float32-rounded at build time, so the head's bound is inflated
// by a tiny relative margin to stay on the safe side of θ.
func findPivot(suffixUB []float64, θ float64) int {
	n := len(suffixUB)
	if θ == 0 || n == 0 {
		return 0
	}
	total := suffixUB[0]
	p := 0
	for i := 1; i <= n; i++ {
		head := total
		if i < n {
			head = total - suffixUB[i]
		}
		if head*(1+1e-6) >= θ {
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

	// Collapse repeated tokens into one entry weighted by its query-term
	// frequency: a term given k times contributes k× its score (and k× its
	// upper bounds), matching Lucene's duplicate-clause semantics.
	qtf := make(map[string]int, len(tokens))
	unique := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if qtf[t] == 0 {
			unique = append(unique, t)
		}
		qtf[t]++
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
		w := float64(qtf[t])
		entries = append(entries, termEntry{
			iter: iter,
			ub:   w * s.index.TermUB(t),
			df:   df,
			idf:  idf,
			qtf:  w,
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
	// termBound[i] bounds term i's contribution to the current candidate
	// (its block max around candidateDoc, else its UB); set by the
	// block-max check, used for the scoring early exit.
	termBound := make([]float64, n)

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

		// Block-max check. Every doc in [candidateDoc, minEnd] is bounded by
		// the sum, over alive terms, of the max impact of that term's block
		// covering the range: the current block for a term positioned at/after
		// candidateDoc (it has no postings in between), or — for a lagging
		// non-essential term — the block containing candidateDoc, found
		// through the skip list alone (shallow, no decoding). minEnd is the
		// earliest of those blocks' last docs. A term without block data
		// contributes its global UB, which holds everywhere. If the bound can't
		// beat θ, the range is skipped by advancing the essential
		// (candidate-generating) terms past minEnd; non-essential terms are
		// advanced lazily when a later candidate is scored.
		if θ > 0 {
			bound := 0.0
			minEnd := ^uint64(0)
			for i := 0; i < n; i++ {
				termBound[i] = 0
				if !alive[i] {
					continue
				}
				it := entries[i].iter
				var impact float32
				var end uint64
				ok := false
				if it.DocID() >= candidateDoc {
					impact = it.BlockMaxImpact()
					end, ok = it.SkipLastDocInBlock(it.BlockIdx())
				} else {
					impact, end, ok = it.ShallowBlock(candidateDoc)
				}
				if !ok || impact == 0 {
					termBound[i] = entries[i].ub
					bound += termBound[i]
					continue
				}
				termBound[i] = entries[i].qtf * float64(impact)
				bound += termBound[i]
				if end < minEnd {
					minEnd = end
				}
			}
			if bound*(1+1e-6) <= θ && minEnd != ^uint64(0) {
				anyAlive := false
				for i := p; i < n; i++ {
					if alive[i] && entries[i].iter.DocID() <= minEnd {
						alive[i] = entries[i].iter.SkipTo(minEnd + 1)
					}
					anyAlive = anyAlive || alive[i]
				}
				if !anyAlive {
					break
				}
				continue
			}
		}

		// Score candidateDoc: essential terms first (they all advance past
		// it, driving candidate generation), then non-essential terms from the
		// highest UB down, stopping as soon as the remaining non-essential
		// UBs can no longer lift the score above θ — the doc can't enter the
		// top-K, so the (typically long, common-word) lists are left alone.
		var dl int
		if s.docLenNum != nil {
			dl = s.docLenNum.DocLenByNum(candidateDoc)
		} else {
			dl = s.index.DocLen(s.index.DocStringID(candidateDoc))
		}
		score := 0.0

		for i := p; i < n; i++ {
			if !alive[i] {
				continue
			}
			iter := entries[i].iter
			if alive[i] = iter.SkipTo(candidateDoc); !alive[i] {
				continue
			}
			if iter.DocID() == candidateDoc {
				score += entries[i].qtf * s.scorer.ScoreWithIDF(iter.TF(), entries[i].idf, dl, avgDocLen)
				alive[i] = iter.Next()
			}
		}
		full := h.Len() == topK
		// rest bounds what the not-yet-scored non-essential terms can still
		// add to this candidate. termBound is only filled by the block-max
		// check, which runs when θ > 0, so the early exit requires that too.
		earlyExit := full && θ > 0
		rest := 0.0
		if earlyExit {
			for i := 0; i < p; i++ {
				rest += termBound[i]
			}
		}
		for i := p - 1; i >= 0; i-- {
			if !alive[i] {
				continue
			}
			if earlyExit && score+rest*(1+1e-6) <= θ {
				break
			}
			rest -= termBound[i]
			iter := entries[i].iter
			if alive[i] = iter.SkipTo(candidateDoc); !alive[i] {
				continue
			}
			if iter.DocID() == candidateDoc {
				score += entries[i].qtf * s.scorer.ScoreWithIDF(iter.TF(), entries[i].idf, dl, avgDocLen)
			}
		}

		if !full {
			heap.Push(&h, heapEntry{candidateDoc, score})
			if h.Len() == topK {
				θ = h.Min()
			}
		} else if score > θ {
			h.ReplaceMin(candidateDoc, score)
			θ = h.Min()
		}
	}

	return h.Sorted(s.index.DocStringID)
}
