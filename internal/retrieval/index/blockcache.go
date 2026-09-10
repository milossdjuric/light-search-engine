package index

import lru "github.com/hashicorp/golang-lru/v2"

// blockCacheDefaultSize is the maximum number of decoded blocks held per segment.
// Each entry is 2×BlockSize×8 = 2048 bytes, so 4096 entries ≈ 8 MB.
const blockCacheDefaultSize = 4096

// cachedBlock holds the decoded docBuf and tfBuf arrays for one full posting
// list block. Values are delta-encoded (same format as PostingIter.docBuf /
// tfBuf), so the caller must apply the running delta base when reading docIDs.
// docBytes and tfBytes record how many bytes each encoded block occupies so
// that the iterator can advance its read cursor on a cache hit without
// re-invoking the decoder.
type cachedBlock struct {
	docBuf   [BlockSize]uint64
	tfBuf    [BlockSize]uint64
	docBytes int
	tfBytes  int
}

// BlockCache is a goroutine-safe LRU cache of decoded posting list blocks,
// shared across all PostingIter instances for the same segment.
// Key encodes (termOrdinal uint32, blockIndex uint32) into one uint64.
type BlockCache struct {
	lru *lru.Cache[uint64, *cachedBlock]
}

// NewBlockCache creates a BlockCache with the given capacity.
// size <= 0 uses blockCacheDefaultSize.
func NewBlockCache(size int) *BlockCache {
	if size <= 0 {
		size = blockCacheDefaultSize
	}
	c, _ := lru.New[uint64, *cachedBlock](size)
	return &BlockCache{lru: c}
}

func (bc *BlockCache) key(termOrd, blockIdx int) uint64 {
	return uint64(uint32(termOrd))<<32 | uint64(uint32(blockIdx))
}

func (bc *BlockCache) get(termOrd, blockIdx int) (*cachedBlock, bool) {
	return bc.lru.Get(bc.key(termOrd, blockIdx))
}

func (bc *BlockCache) set(termOrd, blockIdx int, e *cachedBlock) {
	bc.lru.Add(bc.key(termOrd, blockIdx), e)
}
