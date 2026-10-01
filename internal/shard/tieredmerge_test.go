package shard_test

import (
	"fmt"
	"testing"

	"search-eval-platform/internal/shard"
	"search-eval-platform/internal/storage"
)

func makeSegRecs(n, level int) []storage.SegmentRecord {
	recs := make([]storage.SegmentRecord, n)
	for i := range recs {
		recs[i] = storage.SegmentRecord{
			SegmentID: fmt.Sprintf("seg%02d", i),
			ShardID:   "shard0",
			Level:     level,
			DocCount:  100,
			FlushSeq:  int64(i),
		}
	}
	return recs
}

func TestSelectMerge15L0Segments(t *testing.T) {
	policy := &shard.TieredMergePolicy{
		SegmentsPerTier: 10,
		MaxMergeAtOnce:  10,
		LevelRatio:      10,
	}
	segs := makeSegRecs(15, 0)
	selected := policy.SelectMerge(segs)
	if len(selected) != 10 {
		t.Errorf("want 10 segments selected, got %d", len(selected))
	}
	for _, s := range selected {
		if s.Level != 0 {
			t.Errorf("expected level 0, got %d", s.Level)
		}
	}
}

func TestSelectMergeFewerThanTier(t *testing.T) {
	policy := shard.DefaultTieredMergePolicy()
	segs := makeSegRecs(5, 0)
	selected := policy.SelectMerge(segs)
	if len(selected) != 0 {
		t.Errorf("fewer than SegmentsPerTier: expected 0 selected, got %d", len(selected))
	}
}

func TestSelectMergeEmptyInput(t *testing.T) {
	policy := shard.DefaultTieredMergePolicy()
	selected := policy.SelectMerge(nil)
	if len(selected) != 0 {
		t.Error("nil input: expected empty result")
	}
}

func TestNextLevel(t *testing.T) {
	if shard.NextLevel(0) != 1 {
		t.Error("NextLevel(0) should be 1")
	}
	if shard.NextLevel(3) != 4 {
		t.Error("NextLevel(3) should be 4")
	}
}

func TestSelectMergeMultipleLevels(t *testing.T) {
	policy := &shard.TieredMergePolicy{
		SegmentsPerTier: 10,
		MaxMergeAtOnce:  10,
		LevelRatio:      10,
	}
	segs := append(makeSegRecs(12, 0), makeSegRecs(12, 1)...)
	selected := policy.SelectMerge(segs)
	if len(selected) == 0 {
		t.Error("expected some segments to be selected")
	}
	if len(selected) > 10 {
		t.Errorf("selected too many: %d > MaxMergeAtOnce=10", len(selected))
	}
}
