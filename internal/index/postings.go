package index

import (
	"search-eval-platform/internal/codec"
)

// NewPostingIterWithSkip creates a PostingIter with an attached SkipList,
// enabling block-max MaxScore skipping on on-disk segments.
func NewPostingIterWithSkip(data []byte, skip *SkipList, df int) *PostingIter {
	return &PostingIter{
		data:     data,
		skip:     skip,
		df:       df,
		blockIdx: -1,
	}
}

// WithBlockCache attaches a shared block cache to the iterator, enabling
// decoded-block reuse across repeated accesses to the same posting list block.
// termOrd is the term's ordinal within the segment and forms half of the cache key.
func (it *PostingIter) WithBlockCache(cache *BlockCache, termOrd int) *PostingIter {
	it.blockCache = cache
	it.termOrd = termOrd
	return it
}

// PostingIter iterates over the posting list for one term.
type PostingIter struct {
	data    []byte
	skip    *SkipList
	pos     int    // current read position in data
	base    uint64 // delta base for current block
	currDoc uint64 // current docID (numeric)
	currTF  int    // current TF
	df      int    // total entries in this posting list
	read    int    // number of entries decoded so far

	blockIdx int // current block index (-1 = not started)
	docBuf   [codec.BlockSize]uint64
	tfBuf    [codec.BlockSize]uint64

	shallowIdx int // ShallowBlock's forward-only cursor into the skip list

	bufPos int // position within current decoded block
	bufLen int // number of valid entries in current decoded block

	// blockCache, when non-nil, caches decoded full blocks to avoid redundant
	// FOR-delta decoding on repeated accesses to the same block.
	blockCache *BlockCache
	termOrd    int // term ordinal within the segment (block cache key component)
}

// loadBlock decodes the next block into docBuf/tfBuf: a full block via
// codec.UnpackFOR32Into (or the block cache), a partial tail as LEB128 varints.
func (it *PostingIter) loadBlock() {
	remaining := it.df - it.read
	if remaining <= 0 {
		it.bufPos = 0
		it.bufLen = 0
		return
	}

	if remaining >= codec.BlockSize {
		// Full block: check cache first, then fall back to format-specific decoder.
		if it.blockCache != nil {
			if entry, ok := it.blockCache.get(it.termOrd, it.blockIdx); ok {
				it.docBuf = entry.docBuf
				it.tfBuf = entry.tfBuf
				it.pos += entry.docBytes + entry.tfBytes
				it.bufLen = codec.BlockSize
			} else {
				docN := codec.UnpackFOR32Into(it.data[it.pos:], it.docBuf[:])
				it.pos += docN
				tfN := codec.UnpackFOR32Into(it.data[it.pos:], it.tfBuf[:])
				it.pos += tfN
				it.bufLen = codec.BlockSize
				it.blockCache.set(it.termOrd, it.blockIdx, &cachedBlock{
					docBuf:   it.docBuf,
					tfBuf:    it.tfBuf,
					docBytes: docN,
					tfBytes:  tfN,
				})
			}
		} else {
			docN := codec.UnpackFOR32Into(it.data[it.pos:], it.docBuf[:])
			it.pos += docN
			it.pos += codec.UnpackFOR32Into(it.data[it.pos:], it.tfBuf[:])
			it.bufLen = codec.BlockSize
		}
	} else {
		// Partial tail: interleaved LEB128 (delta, tf) pairs — same for all versions.
		for i := 0; i < remaining; i++ {
			delta, n := codec.ReadVarint(it.data, it.pos)
			it.pos += n
			tf, n := codec.ReadVarint(it.data, it.pos)
			it.pos += n
			it.docBuf[i] = delta
			it.tfBuf[i] = tf
		}
		it.bufLen = remaining
	}
	it.bufPos = 0
}

// Next advances the iterator to the next posting.
// Returns false when exhausted.
func (it *PostingIter) Next() bool {
	if it.bufPos >= it.bufLen {
		if it.read >= it.df {
			return false
		}
		it.blockIdx++
		it.loadBlock()
		if it.bufLen == 0 {
			return false
		}
	}

	delta := it.docBuf[it.bufPos]
	it.currDoc = it.base + delta
	it.base = it.currDoc
	it.currTF = int(it.tfBuf[it.bufPos])
	it.bufPos++
	it.read++
	return true
}

// DocID returns the current numeric docID.
func (it *PostingIter) DocID() uint64 {
	return it.currDoc
}

// TF returns the current term frequency.
func (it *PostingIter) TF() int {
	return it.currTF
}

// BlockIdx returns the current block index (0-based).
func (it *PostingIter) BlockIdx() int {
	return it.blockIdx
}

// BlockMaxImpact returns the precomputed max BM25 impact for the current block.
// Returns 0 if no skip list is attached or the block is out of range.
func (it *PostingIter) BlockMaxImpact() float32 {
	return it.skip.BlockMaxImpact(it.blockIdx)
}

// ShallowBlock returns the max impact and last docID of the block that
// would contain target, using only the skip list (no decoding, iterator
// position unchanged). ok is false when that isn't known (no skip list, or
// target past the last full block).
//
// Targets must be non-decreasing across calls (MaxScore candidates are): the
// lookup walks a forward-only cursor, amortised O(1) per call, falling back
// to a binary search only for a long jump.
func (it *PostingIter) ShallowBlock(target uint64) (impact float32, lastDoc uint64, ok bool) {
	sl := it.skip
	if sl == nil {
		return 0, 0, false
	}
	n := len(sl.l0DocIDs)
	idx := it.shallowIdx
	if it.blockIdx > idx {
		idx = it.blockIdx
	}
	for steps := 0; idx < n && sl.l0DocIDs[idx] < target; steps++ {
		if steps == 8 { // far away: binary search instead of walking
			b, found := sl.BlockFor(target)
			if !found {
				idx = n
			} else {
				idx = b
			}
			break
		}
		idx++
	}
	it.shallowIdx = idx
	if idx >= n {
		return 0, 0, false
	}
	return sl.BlockMaxImpact(idx), sl.l0DocIDs[idx], true
}

// SkipLastDocInBlock returns the last docID in blockIdx and true, delegating
// to the attached SkipList. Returns (0, false) if unavailable.
func (it *PostingIter) SkipLastDocInBlock(blockIdx int) (uint64, bool) {
	return it.skip.LastDocIDInBlock(blockIdx)
}

// SkipTo advances the iterator to the first entry with docID >= target.
// Returns false if no such entry exists.
func (it *PostingIter) SkipTo(target uint64) bool {
	if it.currDoc >= target && it.read > 0 {
		return true
	}

	// Try skip list for a coarse jump.
	if it.skip != nil && it.read < it.df {
		offset, base, blkIdx := it.skip.SkipTo(target)
		if offset > it.pos || (offset == 0 && blkIdx == 0 && it.read == 0) {
			// Only jump forward.
			if offset > it.pos {
				it.pos = offset
				it.base = base
				it.blockIdx = blkIdx - 1 // will be incremented by Next() → loadBlock()
				it.read = blkIdx * codec.BlockSize
				it.bufPos = 0
				it.bufLen = 0
			}
		}
	}

	// Linear scan forward.
	for {
		if it.read >= it.df {
			return false
		}
		if !it.Next() {
			return false
		}
		if it.currDoc >= target {
			return true
		}
	}
}
