package segment

import (
	"encoding/binary"
	"fmt"
	"os"
	"runtime"
	"strings"

	bloom "github.com/bits-and-blooms/bloom/v3"

	"search-eval-platform/internal/index"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/pkg/types"
)

// LoadSegment reads a .seg file from path into a Segment.
// It also loads the bloom filter sidecar (<path>.bloom) if present.
func LoadSegment(path string) (*Segment, error) {
	// Map the segment file into the OS page cache instead of copying it to the
	// Go heap. On Unix this uses mmap(MAP_SHARED|PROT_READ); on other platforms
	// it falls back to os.ReadFile. The mapping is released by a GC finalizer
	// when the *Segment becomes unreachable — this is safe because in-flight
	// searches hold a *Segment reference that keeps the mapping alive.
	data, err := mmapFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadSegment %s: %w", path, err)
	}
	seg, err := parseSegment(data)
	if err != nil {
		_ = munmapFile(data)
		return nil, err
	}
	if len(data) > 0 {
		seg.mmapData = data
		// parseSegment already did a sequential read-through (header, FST, term
		// metadata). Now switch to random-access mode so posting list traversals
		// (which jump around the file) don't trigger useless kernel read-ahead.
		setMadviseRandom(data)
		hintHugepage(data) // 2 MB THP pages — reduces TLB pressure on posting list traversal
		hintDontDump(data) // exclude from core dumps; data is on disk and reloadable
		runtime.SetFinalizer(seg, func(s *Segment) {
			_ = munmapFile(s.mmapData)
		})
	}
	// Load bloom sidecar if it exists (non-fatal if missing).
	if f, ferr := os.Open(path + ".bloom"); ferr == nil {
		var bf bloom.BloomFilter
		if _, rerr := bf.ReadFrom(f); rerr == nil {
			seg.bloom = &bf
		}
		f.Close()
	}
	// Attempt to load stored fields sidecar (optional; absent for old segments).
	_ = LoadStoredFields(seg, strings.TrimSuffix(path, ".seg")+".seg.fld")
	seg.blockCache = index.NewBlockCache(0) // 0 = default size (4096 entries ≈ 8 MB)
	return seg, nil
}

// DocCount returns the number of documents in this segment.
func (s *Segment) DocCount() int { return int(s.docCount) }

// ordinalFor returns the ordinal for term via FST lookup.
func (s *Segment) ordinalFor(term string) (int, bool) {
	ord, exists, err := s.fst.Get([]byte(term))
	if err != nil || !exists {
		return 0, false
	}
	return int(ord), true
}

// TermUB returns the precomputed per-term upper bound, or 0 if absent.
func (s *Segment) TermUB(term string) float64 {
	ord, ok := s.ordinalFor(term)
	if !ok {
		return 0
	}
	return float64(s.infos[ord].UB)
}

// DF returns the document frequency for term, or 0 if absent.
func (s *Segment) DF(term string) int {
	ord, ok := s.ordinalFor(term)
	if !ok {
		return 0
	}
	return int(s.infos[ord].DF)
}

// MergeSegments  — k-way merge via min-heap over sorted term iterators

// SegmentIndexAdapter wraps a *Segment so it can be passed to
// index.NewMaxScoreSearcher, giving full MaxScore search on-disk segments.
type SegmentIndexAdapter struct{ seg *Segment }

// IDFCached returns the cached IDF for term+scorer, computing and storing it on
// the first call. Safe for concurrent use; segments are immutable so the cached
// value is always valid.
func (a *SegmentIndexAdapter) IDFCached(term, scorerType string, df, N int, scorer scoring.Scorer) float64 {
	key := term + "\x00" + scorerType
	if v, ok := a.seg.idfCache.Load(key); ok {
		return v.(float64)
	}
	val := scorer.IDF(df, N)
	a.seg.idfCache.Store(key, val)
	return val
}

// AsIndexReader wraps seg in a SegmentIndexAdapter.
func (s *Segment) AsIndexReader() *SegmentIndexAdapter {
	return &SegmentIndexAdapter{seg: s}
}

// Iterator returns a PostingIter for term with its skip list wired up, or nil
// if not present. Attaching the skip list enables block-max MaxScore skipping
// on on-disk segments.
func (a *SegmentIndexAdapter) Iterator(term string) *index.PostingIter {
	ord, ok := a.seg.ordinalFor(term)
	if !ok {
		return nil
	}
	info := a.seg.infos[ord]
	postSlice := a.seg.postData[info.PostingsOffset : info.PostingsOffset+info.PostingsLen]
	if len(postSlice) < 8 {
		return nil
	}
	skipLen := binary.LittleEndian.Uint64(postSlice[:8])
	skipData := postSlice[8 : 8+skipLen]
	rawPost := postSlice[8+skipLen:]
	sl := index.LoadSkipFromBytes(skipData)
	it := index.NewPostingIterWithSkip(rawPost, sl, int(info.DF))
	if a.seg.blockCache != nil {
		it.WithBlockCache(a.seg.blockCache, ord)
	}
	return it
}

// TermUB returns the per-term upper bound stored in the segment.
func (a *SegmentIndexAdapter) TermUB(term string) float64 {
	return a.seg.TermUB(term)
}

// DF returns the document frequency for term, or 0 if absent.
func (a *SegmentIndexAdapter) DF(term string) int {
	ord, ok := a.seg.ordinalFor(term)
	if !ok {
		return 0
	}
	return int(a.seg.infos[ord].DF)
}

// DocCount returns the number of documents in the segment.
func (a *SegmentIndexAdapter) DocCount() int { return int(a.seg.docCount) }

// AvgDocLen returns the average document length across all docs in the segment.
func (a *SegmentIndexAdapter) AvgDocLen() float64 {
	if a.seg.docCount == 0 {
		return 0
	}
	return float64(a.seg.totalTokens) / float64(a.seg.docCount)
}

// DocLen returns the length (token count) of the document with the given string ID.
func (a *SegmentIndexAdapter) DocLen(docID string) int {
	num, ok := a.seg.docIDToNum[docID]
	if !ok {
		return 0
	}
	return int(a.seg.docLens[num])
}

// DocLenByNum returns the token count of the doc with numeric ID numID
// (index.DocLenByNumReader).
func (a *SegmentIndexAdapter) DocLenByNum(numID uint64) int {
	if numID >= uint64(len(a.seg.docLens)) {
		return 0
	}
	return int(a.seg.docLens[numID])
}

// DocStringID converts a numeric docID to its string form.
func (a *SegmentIndexAdapter) DocStringID(numID uint64) string {
	if numID >= a.seg.docCount {
		return ""
	}
	return a.seg.docIDs[numID]
}

// HasTerm reports whether the document with the given string docID contains term.
// Uses SkipTo on the on-disk posting list for an O(log N) membership check.
func (s *Segment) HasTerm(term, docID string) bool {
	numID, ok := s.docIDToNum[docID]
	if !ok {
		return false
	}
	it := s.AsIndexReader().Iterator(term)
	if it == nil {
		return false
	}
	it.SkipTo(numID)
	return it.DocID() == numID
}

// Search returns the top-K results for tokens using MaxScoreSearcher.
func (s *Segment) Search(tokens []string, topK int, scorer scoring.Scorer) []types.ScoredDocument {
	searcher := index.NewMaxScoreSearcher(s.AsIndexReader(), scorer)
	return searcher.Search(tokens, topK)
}

// HasDoc reports whether docID is stored in this segment (tombstoned or not).
func (s *Segment) HasDoc(docID string) bool {
	_, ok := s.docIDToNum[docID]
	return ok
}

// MayContainAny reports whether the segment's bloom sidecar admits at least one
// of tokens. It returns true when there is no bloom filter or no tokens, since
// the segment can't be ruled out then.
func (s *Segment) MayContainAny(tokens []string) bool {
	if s.bloom == nil || len(tokens) == 0 {
		return true
	}
	for _, tok := range tokens {
		if s.bloom.TestString(tok) {
			return true
		}
	}
	return false
}
