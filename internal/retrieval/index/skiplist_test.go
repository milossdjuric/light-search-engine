package index_test

import (
	"testing"

	"search-eval-platform/internal/retrieval/index"
)

// buildTestSkipList creates a skip list with 3 blocks:
//
//	block 0: docIDs 0..127,   offset=0,   base=0,     impact=10.0
//	block 1: docIDs 128..255, offset=100, base=127,   impact=8.0
//	block 2: docIDs 256..383, offset=200, base=255,   impact=5.0
func buildTestSkipList() *index.SkipList {
	l0DocIDs := []uint64{127, 255, 383}
	l0Offsets := []int{0, 100, 200}
	l0Bases := []uint64{0, 127, 255}
	l0Impact := []float32{10.0, 8.0, 5.0}
	return index.BuildSkipList(l0DocIDs, l0Offsets, l0Bases, l0Impact)
}

func TestSkipToExactHit(t *testing.T) {
	sl := buildTestSkipList()
	// Target = first docID of block 1 (128): skip to block 1
	off, base, blk := sl.SkipTo(128)
	if off != 100 {
		t.Errorf("offset: want 100, got %d", off)
	}
	if base != 127 {
		t.Errorf("base: want 127, got %d", base)
	}
	if blk != 1 {
		t.Errorf("blockIdx: want 1, got %d", blk)
	}
}

func TestSkipToBetweenCheckpoints(t *testing.T) {
	sl := buildTestSkipList()
	// Target = 200 (inside block 1: 128..255)
	off, base, blk := sl.SkipTo(200)
	if off != 100 {
		t.Errorf("offset: want 100, got %d", off)
	}
	if base != 127 {
		t.Errorf("base: want 127, got %d", base)
	}
	if blk != 1 {
		t.Errorf("blockIdx: want 1, got %d", blk)
	}
}

func TestSkipToBeforeFirst(t *testing.T) {
	sl := buildTestSkipList()
	// Target = 0: no useful skip (we're already at the start)
	off, base, blk := sl.SkipTo(0)
	// Should return (0, 0, 0) — no skip useful
	if off != 0 || base != 0 || blk != 0 {
		t.Errorf("before first: want (0,0,0), got (%d,%d,%d)", off, base, blk)
	}
}

func TestSkipToAfterLast(t *testing.T) {
	sl := buildTestSkipList()
	// Target = 999 is past all block last-docIDs (127, 255, 383).
	// No L1 entry covers 999 → (0, 0, 0): no useful skip available.
	off, base, blk := sl.SkipTo(999)
	if off != 0 || base != 0 || blk != 0 {
		t.Errorf("past last: want (0,0,0), got (%d,%d,%d)", off, base, blk)
	}
}

func TestBlockMaxImpact(t *testing.T) {
	sl := buildTestSkipList()
	if got := sl.BlockMaxImpact(0); got != 10.0 {
		t.Errorf("block 0: want 10.0, got %f", got)
	}
	if got := sl.BlockMaxImpact(1); got != 8.0 {
		t.Errorf("block 1: want 8.0, got %f", got)
	}
	if got := sl.BlockMaxImpact(2); got != 5.0 {
		t.Errorf("block 2: want 5.0, got %f", got)
	}
	// Out of range → 0
	if got := sl.BlockMaxImpact(99); got != 0 {
		t.Errorf("out of range: want 0, got %f", got)
	}
}

func TestSkipListLen(t *testing.T) {
	sl := buildTestSkipList()
	if sl.Len() != 3 {
		t.Errorf("want 3 L0 entries, got %d", sl.Len())
	}
}

func TestLastDocIDInBlock(t *testing.T) {
	sl := buildTestSkipList()
	id, ok := sl.LastDocIDInBlock(1)
	if !ok {
		t.Fatal("expected ok=true for block 1")
	}
	if id != 255 {
		t.Errorf("want 255, got %d", id)
	}
	_, ok = sl.LastDocIDInBlock(99)
	if ok {
		t.Error("expected ok=false for out-of-range block")
	}
}
