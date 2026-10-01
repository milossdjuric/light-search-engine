package segment

import (
	"log/slog"
	"sort"
)

// Prefetch advises the kernel to load all segment pages into the page cache.
// The call is non-blocking; actual prefetch happens asynchronously. Call after
// LoadSegment during startup warmup to avoid cold-load latency on first search.
func (s *Segment) Prefetch() {
	prefetchSegmentData(s.mmapData)
}

// WarmTopKTerms touches the posting data pages for the k highest-DF terms,
// ensuring they are resident in the OS page cache before queries arrive.
// This is the targeted Lucene-style warmup: high-DF terms appear in most
// queries and cover the hot posting lists with minimal I/O.
func (s *Segment) WarmTopKTerms(k int) {
	if k <= 0 || len(s.infos) == 0 || len(s.postData) == 0 {
		return
	}
	// Build a slice of (df, ordinal) pairs and partial-sort to find top-k.
	type dfOrd struct {
		df  uint32
		ord int
	}
	order := make([]dfOrd, len(s.infos))
	for i, info := range s.infos {
		order[i] = dfOrd{df: info.DF, ord: i}
	}
	// Sort descending by DF; take only the top-k.
	sort.Slice(order, func(i, j int) bool { return order[i].df > order[j].df })
	if len(order) > k {
		order = order[:k]
	}
	// Touch every 4 KB page of each top-k term's posting list.
	var acc byte
	for _, di := range order {
		info := s.infos[di.ord]
		end := info.PostingsOffset + info.PostingsLen
		if end > uint64(len(s.postData)) {
			continue
		}
		data := s.postData[info.PostingsOffset:end]
		for i := 0; i < len(data); i += 4096 {
			acc ^= data[i]
		}
	}
	_ = acc
}

// WarmFull warms the entire segment into the OS page cache.
//
// It covers the full mmapData region — FST, term metadata, skip lists, doc
// lengths, and posting data — not just postData. This matters because queries
// access all of these on every search, not just the posting lists.
//
// Strategy: call MADV_WILLNEED first so the kernel issues async prefetch at
// full SSD sequential bandwidth (~500 MB/s), then stride-scan to
// synchronously wait for all pages to land before returning. The stride scan
// after WILLNEED is ~10× faster than a cold stride scan because the kernel
// has already queued the reads.
func (s *Segment) WarmFull() {
	if len(s.mmapData) == 0 {
		return
	}
	// Switch to sequential mode so the kernel issues aggressive read-ahead
	// during the stride scan — fewer individual I/O ops than random mode.
	setMadviseSequential(s.mmapData)
	prefetchSegmentData(s.mmapData) // MADV_WILLNEED: kick off async prefetch now
	var acc byte
	for i := 0; i < len(s.mmapData); i += 4096 {
		acc ^= s.mmapData[i]
	}
	_ = acc
	// All pages are now in RAM. Restore random-access mode for query serving,
	// then synchronously collapse to 2 MB huge pages (Linux 6.1+, silent no-op
	// on older kernels). Pages are physically contiguous after sequential load,
	// so collapse succeeds quickly and all subsequent TLB misses hit 2 MB entries.
	setMadviseRandom(s.mmapData)
	collapseHugepages(s.mmapData)
}

// SizeBytes returns the size of the mmap'd backing data in bytes.
// Returns 0 for in-memory-only (non-mmap'd) segments.
func (s *Segment) SizeBytes() int64 {
	return int64(len(s.mmapData))
}

// TryMlock attempts to lock the segment's mmap pages in RAM using mlock(2).
// Prevents OS eviction under memory pressure. Requires CAP_IPC_LOCK or
// adequate RLIMIT_MEMLOCK; logs a debug message and returns nil on EPERM.
func (s *Segment) TryMlock() error {
	if err := tryMlockData(s.mmapData); err != nil {
		slog.Debug("mlock unavailable, pages may be evicted under memory pressure",
			"err", err)
		return nil // non-fatal
	}
	return nil
}

// WriteSegment

// AdviseSequential switches the segment's mapping to sequential read-ahead
// (used while a merge streams through it).
func (s *Segment) AdviseSequential() { setMadviseSequential(s.mmapData) }

// EvictPageCache asks the kernel to drop the segment's pages from the page
// cache (after it has been merged away). No-op on non-Linux platforms.
func (s *Segment) EvictPageCache() { evictPageCache(s.mmapData) }
