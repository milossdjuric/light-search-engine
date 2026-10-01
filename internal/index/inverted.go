// Package index builds and queries inverted indexes: IndexBuilder accumulates
// postings and builds an immutable InvertedIndex (BM25 or BM25F impacts), and
// MaxScoreSearcher runs rank-safe block-max MaxScore top-K over any IndexReader
// (in-memory InvertedIndex or an on-disk segment).
package index

import (
	"encoding/binary"
	"math"
)

// IndexReader is the read-only interface that MaxScoreSearcher and other
// consumers use to query an index. Both InvertedIndex and the Segment adapter
// implement it, enabling search across both in-memory and on-disk indexes.
type IndexReader interface {
	Iterator(term string) *PostingIter
	TermUB(term string) float64
	DF(term string) int
	DocCount() int
	AvgDocLen() float64
	DocLen(docID string) int
	DocStringID(numID uint64) string
}

// InvertedIndex is an immutable, lock-free inverted index.
type InvertedIndex struct {
	termIndex  map[string]int // term → ordinal
	terms      []string
	data       []byte
	offsets    []int64 // offsets[i]:offsets[i+1] = posting data for ordinal i
	dfs        []int   // document frequency per term ordinal
	skipLists  []*SkipList
	ub         []float64 // precomputed max BM25 UBt per ordinal
	docIDs     []string  // numeric uint64 → string docID
	docIDIndex map[string]uint64
	docLengths map[string]int
	avgDocLen  float64
	N          int
}

// termOrdinalFor returns the ordinal for term, or false if not found.
func (idx *InvertedIndex) termOrdinalFor(term string) (int, bool) {
	ord, ok := idx.termIndex[term]
	return ord, ok
}

// TermUB returns the precomputed upper-bound BM25 score for term.
// Returns 0 if the term is not in the index.
func (idx *InvertedIndex) TermUB(term string) float64 {
	ord, ok := idx.termOrdinalFor(term)
	if !ok {
		return 0
	}
	return idx.ub[ord]
}

// DocCount returns the number of documents in the index.
func (idx *InvertedIndex) DocCount() int {
	return idx.N
}

// AvgDocLen returns the average document length.
func (idx *InvertedIndex) AvgDocLen() float64 {
	return idx.avgDocLen
}

// DocLen returns the length of the document with the given string ID.
func (idx *InvertedIndex) DocLen(docID string) int {
	return idx.docLengths[docID]
}

// DocStringID converts a numeric docID to its string form.
func (idx *InvertedIndex) DocStringID(numID uint64) string {
	if int(numID) >= len(idx.docIDs) {
		return ""
	}
	return idx.docIDs[numID]
}

// Iterator returns a PostingIter for the given term, or nil if the term is absent.
func (idx *InvertedIndex) Iterator(term string) *PostingIter {
	ord, ok := idx.termOrdinalFor(term)
	if !ok {
		return nil
	}
	return idx.iteratorByOrdinal(ord)
}

// iteratorByOrdinal returns a PostingIter for the term at the given ordinal.
func (idx *InvertedIndex) iteratorByOrdinal(ord int) *PostingIter {
	start := idx.offsets[ord]
	end := idx.offsets[ord+1]
	it := &PostingIter{
		data:     idx.data[start:end],
		skip:     idx.skipLists[ord],
		df:       idx.dfs[ord],
		bufPos:   0,
		bufLen:   0,
		blockIdx: -1,
	}
	return it
}

// Terms returns the sorted term list of the index.
func (idx *InvertedIndex) Terms() []string {
	return idx.terms
}

// DF returns the document frequency for the given term, or 0 if absent.
func (idx *InvertedIndex) DF(term string) int {
	ord, ok := idx.termOrdinalFor(term)
	if !ok {
		return 0
	}
	return idx.dfs[ord]
}

// RawPostingBytes returns the raw posting bytes for the term at the given ordinal.
// The bytes are in the Phase 1 in-memory format (FOR-delta blocks + LEB128 tail).
func (idx *InvertedIndex) RawPostingBytes(ord int) []byte {
	if ord < 0 || ord >= len(idx.terms) {
		return nil
	}
	start := idx.offsets[ord]
	end := idx.offsets[ord+1]
	return idx.data[start:end]
}

// HasTerm reports whether the document with the given string docID contains term.
// Uses SkipTo on the posting list for an O(log N) membership check.
func (idx *InvertedIndex) HasTerm(term, docID string) bool {
	numID, ok := idx.docIDIndex[docID]
	if !ok {
		return false
	}
	it := idx.Iterator(term)
	if it == nil {
		return false
	}
	it.SkipTo(numID)
	return it.DocID() == numID
}

// RawSkipBytes serialises the skip list for the given ordinal into a compact
// binary representation for embedding in .seg files.
//
// Format per L0 entry (28 bytes):
//
//	[8] lastDocID
//	[8] byteOffset
//	[8] deltaBase
//	[4] blockMaxImpact (float32)
//
// Followed by L1 entries (16 bytes each):
//
//	[8] lastDocID
//	[8] l0Index
//
// Layout: [4 nL0][4 nL1][nL0 * 28 bytes][nL1 * 16 bytes]
func (idx *InvertedIndex) RawSkipBytes(ord int) []byte {
	if ord < 0 || ord >= len(idx.skipLists) || idx.skipLists[ord] == nil {
		return nil
	}
	sl := idx.skipLists[ord]
	nL0 := len(sl.l0DocIDs)
	nL1 := len(sl.l1DocIDs)

	buf := make([]byte, 8+nL0*28+nL1*16)
	off := 0
	putU32 := func(v uint32) { binary.LittleEndian.PutUint32(buf[off:], v); off += 4 }
	putU64 := func(v uint64) { binary.LittleEndian.PutUint64(buf[off:], v); off += 8 }
	putF32 := func(v float32) {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(v))
		off += 4
	}

	putU32(uint32(nL0))
	putU32(uint32(nL1))
	for i := 0; i < nL0; i++ {
		putU64(sl.l0DocIDs[i])
		putU64(uint64(sl.l0Offsets[i]))
		putU64(sl.l0Bases[i])
		putF32(sl.l0Impact[i])
	}
	for i := 0; i < nL1; i++ {
		putU64(sl.l1DocIDs[i])
		putU64(uint64(sl.l1L0Idx[i]))
	}
	return buf
}
