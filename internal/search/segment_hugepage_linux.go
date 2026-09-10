//go:build linux

package search

import "syscall"

// Linux madvise constants not yet in Go's syscall package for linux/amd64.
// Values are architecture-independent on x86 and match the kernel headers.
const (
	madvDontDump = 16 // MADV_DONTDUMP: exclude from core dumps (Linux 3.4+)
	madvCold     = 20 // MADV_COLD: deprioritise in LRU (Linux 5.4+)  -- kept for reference
	madvPageout  = 21 // MADV_PAGEOUT: drop clean pages immediately (Linux 5.4+)
	madvCollapse = 25 // MADV_COLLAPSE: synchronous THP collapse (Linux 6.1+)
)

// hintHugepage asks the kernel to back data with 2 MB transparent huge pages
// when THP is in madvise or always mode. Reduces TLB pressure on random-access
// posting list traversals: a 500 MB segment needs ~131K TLB entries at 4 KB
// pages but only ~250 at 2 MB pages. The kernel promotes pages asynchronously
// via khugepaged; use collapseHugepages for a synchronous guarantee.
func hintHugepage(data []byte) {
	if len(data) > 0 {
		_ = syscall.Madvise(data, syscall.MADV_HUGEPAGE)
	}
}

// hintDontDump excludes the mmap'd region from core dumps. Segment data is
// read-only and can always be reloaded from disk — including it in a core dump
// wastes disk space and makes crash analysis harder (1-2 GB of posting data
// drowns the Go heap state that actually matters for debugging).
func hintDontDump(data []byte) {
	if len(data) > 0 {
		_ = syscall.Madvise(data, madvDontDump)
	}
}

// evictPageCache drops the segment's pages from the OS page cache immediately.
// Because the segment files are clean (we never write to mmap'd data), the
// kernel simply releases the pages — no I/O. Call on merge input segments
// right after MergeSegmentsWithOptions returns so the OS reclaims that RAM
// before the new merged segment is warmed. Without this, LRU eviction is
// implicit and the old pages may linger, competing with the new segment.
func evictPageCache(data []byte) {
	if len(data) > 0 {
		_ = syscall.Madvise(data, madvPageout)
	}
}

// collapseHugepages synchronously promotes all pages in data to 2 MB huge
// pages. Blocks the caller until collapse is complete or fails. Call at the
// end of WarmFull() after all pages are loaded into RAM by the linear scan —
// at that point pages are likely physically contiguous, so collapse succeeds
// quickly. After this returns, every subsequent TLB miss on this region hits a
// 2 MB entry instead of a 4 KB one. Fails silently on kernels before 6.1 or
// when memory is too fragmented (EINVAL/ENOMEM returned, both ignored).
func collapseHugepages(data []byte) {
	if len(data) > 0 {
		_ = syscall.Madvise(data, madvCollapse)
	}
}
