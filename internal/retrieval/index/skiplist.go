package index

const l1Stride = 32 // L1 covers one entry per 32 L0 entries (4096 docs)

// SkipList is a two-level skip list over posting list blocks.
//
// L0: one entry per 128-doc block (matching FOR-delta block boundaries).
// L1: one entry per 32 L0 entries (4096-doc granularity).
//
// Critical invariant: l0Bases[i] is the last docID of block i-1 (or 0 for i=0).
// Restoring this before decompressing block i ensures
//
//	base + blockDocBuf[0] == first absolute docID of block i.
type SkipList struct {
	l0DocIDs  []uint64  // last docID in block i
	l0Offsets []int     // byte offset of block i's start in posting data (relative to term start)
	l0Bases   []uint64  // delta base BEFORE block i (= l0DocIDs[i-1] or 0 for i=0)
	l0Impact  []float32 // max BM25 score contribution for any doc in block i

	l1DocIDs []uint64 // last docID covered by L1 entry j
	l1L0Idx  []int    // first L0 index covered by L1 entry j
}

// BuildSkipList constructs a SkipList from pre-computed L0 data.
// l0DocIDs, l0Offsets, l0Bases, and l0Impact must all have the same length
// (one entry per block).
func BuildSkipList(l0DocIDs []uint64, l0Offsets []int, l0Bases []uint64, l0Impact []float32) *SkipList {
	sl := &SkipList{
		l0DocIDs:  l0DocIDs,
		l0Offsets: l0Offsets,
		l0Bases:   l0Bases,
		l0Impact:  l0Impact,
	}

	// Build L1: one entry per l1Stride L0 blocks.
	for i := 0; i < len(l0DocIDs); i += l1Stride {
		end := i + l1Stride - 1
		if end >= len(l0DocIDs) {
			end = len(l0DocIDs) - 1
		}
		sl.l1DocIDs = append(sl.l1DocIDs, l0DocIDs[end])
		sl.l1L0Idx = append(sl.l1L0Idx, i)
	}

	return sl
}

// SkipTo finds the first block whose last docID >= targetDocID.
// Returns (offset, base, blockIdx) for that block.
// Returns (0, 0, 0) if no skip is useful (target is before or at the first block).
//
// The caller must set pos=offset, base=base, blockIdx=blockIdx, and force
// a reload before calling Next().
func (sl *SkipList) SkipTo(targetDocID uint64) (offset int, base uint64, blockIdx int) {
	if len(sl.l0DocIDs) == 0 {
		return 0, 0, 0
	}

	// Lower-bound search: find the first L1 entry whose docID >= targetDocID.
	// Uses the standard half-open [lo, hi) pattern so lo == hi == result index
	// when the loop ends, and lo == len means "no L1 entry covers target".
	lo, hi := 0, len(sl.l1DocIDs)
	for lo < hi {
		mid := (lo + hi) / 2
		if sl.l1DocIDs[mid] < targetDocID {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	if lo >= len(sl.l1DocIDs) {
		// All L1 entries have docID < target; target is past the end of all blocks.
		return 0, 0, 0
	}

	// Linear scan L0 within the L1 bucket.
	l0Start := sl.l1L0Idx[lo]
	l0End := l0Start + l1Stride
	if l0End > len(sl.l0DocIDs) {
		l0End = len(sl.l0DocIDs)
	}

	for i := l0Start; i < l0End; i++ {
		if sl.l0DocIDs[i] >= targetDocID {
			return sl.l0Offsets[i], sl.l0Bases[i], i
		}
	}

	return 0, 0, 0
}

// BlockMaxImpact returns the precomputed maximum BM25 impact for blockIdx.
// Returns 0 if blockIdx is out of range.
func (sl *SkipList) BlockMaxImpact(blockIdx int) float32 {
	if sl == nil || blockIdx < 0 || blockIdx >= len(sl.l0Impact) {
		return 0
	}
	return sl.l0Impact[blockIdx]
}

// Len returns the number of L0 blocks in the skip list.
func (sl *SkipList) Len() int {
	if sl == nil {
		return 0
	}
	return len(sl.l0DocIDs)
}

// LastDocIDInBlock returns the last docID stored in block blockIdx and true,
// or (0, false) if blockIdx is out of range. Used by MaxScore to compute the
// skip target (lastDocID + 1) when pruning an entire block.
func (sl *SkipList) LastDocIDInBlock(blockIdx int) (uint64, bool) {
	if sl == nil || blockIdx < 0 || blockIdx >= len(sl.l0DocIDs) {
		return 0, false
	}
	return sl.l0DocIDs[blockIdx], true
}
