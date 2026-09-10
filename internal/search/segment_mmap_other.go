//go:build !unix

package search

import "os"

// mmapFile falls back to os.ReadFile on non-Unix platforms.
func mmapFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

// munmapFile is a no-op on non-Unix platforms (GC handles heap memory).
func munmapFile(_ []byte) error { return nil }

// prefetchSegmentData is a no-op on non-Unix platforms.
func prefetchSegmentData(_ []byte) {}

// setMadviseRandom is a no-op on non-Unix platforms.
func setMadviseRandom(_ []byte) {}

// setMadviseSequential is a no-op on non-Unix platforms.
func setMadviseSequential(_ []byte) {}

// tryMlockData is a no-op on non-Unix platforms.
func tryMlockData(_ []byte) error { return nil }
