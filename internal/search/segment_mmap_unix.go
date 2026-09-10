//go:build unix

package search

import (
	"fmt"
	"os"
	"syscall"
)

// mmapFile maps path into the OS page cache and returns the byte slice.
// The caller must ensure munmapFile is eventually called (via GC finalizer).
// Closing the file descriptor after mmap is safe on all Unix systems —
// the mapping persists until explicitly unmapped.
func mmapFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("mmap open %s: %w", path, err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("mmap stat %s: %w", path, err)
	}
	size := int(fi.Size())
	if size == 0 {
		return []byte{}, nil
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s (%d bytes): %w", path, size, err)
	}
	// No madvise here — the caller (LoadSegment) does a sequential parse pass
	// first, then switches to MADV_RANDOM for search access.
	return data, nil
}

// setMadviseRandom switches the mmap'd region to random-access mode, disabling
// kernel read-ahead. Call after the initial sequential parse pass is complete.
func setMadviseRandom(data []byte) {
	if len(data) > 0 {
		_ = syscall.Madvise(data, syscall.MADV_RANDOM)
	}
}

// setMadviseSequential enables aggressive kernel read-ahead for sequential
// access patterns (warmup scans, merge reads). Call setMadviseRandom when
// done to restore random-access mode for query serving.
func setMadviseSequential(data []byte) {
	if len(data) > 0 {
		_ = syscall.Madvise(data, syscall.MADV_SEQUENTIAL)
	}
}

// munmapFile releases the mapping returned by mmapFile.
func munmapFile(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return syscall.Munmap(data)
}

// prefetchSegmentData advises the kernel to prefetch all pages of data into
// the page cache asynchronously. Used for background hints; for guaranteed
// residency use WarmTopKTerms which reads pages synchronously.
func prefetchSegmentData(data []byte) {
	if len(data) > 0 {
		_ = syscall.Madvise(data, syscall.MADV_WILLNEED)
	}
}

// tryMlockData attempts to lock data pages in RAM so the OS cannot evict them
// under memory pressure. Requires CAP_IPC_LOCK or adequate RLIMIT_MEMLOCK.
// Returns the error without panicking so callers can log and continue.
func tryMlockData(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return syscall.Mlock(data)
}
