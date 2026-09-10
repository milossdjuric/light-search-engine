//go:build !linux

package search

// hintHugepage is a no-op on non-Linux platforms (MADV_HUGEPAGE is Linux-only).
func hintHugepage(_ []byte) {}

// hintDontDump is a no-op on non-Linux platforms.
func hintDontDump(_ []byte) {}

// evictPageCache is a no-op on non-Linux platforms.
func evictPageCache(_ []byte) {}

// collapseHugepages is a no-op on non-Linux platforms.
func collapseHugepages(_ []byte) {}
