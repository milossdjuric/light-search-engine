package shard

import (
	"testing"

	"search-eval-platform/internal/segment"
	"search-eval-platform/internal/storage"
)

// TestOrderSegmentsByFlushSeqSortsAscending verifies that segments are
// reordered to match ascending FlushSeq of their corresponding manifest
// record, regardless of the order candidates arrived in. SelectMerge sorts
// candidates by deletion score and forceMerge sorts them by size — neither
// is chronological — so without this, segment.MergeSegmentsWithOptions's
// "highest index wins" duplicate-docID resolution doesn't actually reflect
// which segment holds the most recent content for a doc.
func TestOrderSegmentsByFlushSeqSortsAscending(t *testing.T) {
	segOld := &segment.Segment{}
	segMid := &segment.Segment{}
	segNew := &segment.Segment{}

	// Deliberately out of chronological order, as a deletion-score or
	// size sort would produce.
	segsToMerge := []*segment.Segment{segNew, segOld, segMid}
	candidates := []storage.SegmentRecord{
		{SegmentID: "new", FlushSeq: 30},
		{SegmentID: "old", FlushSeq: 10},
		{SegmentID: "mid", FlushSeq: 20},
	}

	orderSegmentsByFlushSeq(segsToMerge, candidates)

	want := []*segment.Segment{segOld, segMid, segNew}
	for i, s := range want {
		if segsToMerge[i] != s {
			t.Errorf("index %d: got %p, want %p (full order: %v)", i, segsToMerge[i], s, segsToMerge)
		}
	}
}
