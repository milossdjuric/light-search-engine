package search

import (
	"sort"

	"search-eval-platform/internal/storage/manifest"
)

// TieredMergePolicy selects which segments to merge next using a tiered
// strategy that gives O(log N) write amplification.
//
// Segments are grouped by their Level field (L0 = freshly flushed).
// When a level has more than SegmentsPerTier segments, SelectMerge returns
// the MaxMergeAtOnce highest-scored ones. Score = effectiveSize × deletionScore,
// so segments with many tombstones are prioritised even if they are small.
//
// Default values match Lucene's TieredMergePolicy defaults:
//   - SegmentsPerTier     = 10
//   - MaxMergeAtOnce      = 10
//   - LevelRatio          = 10
//   - DeletionWeight      = 2.0  (α in deletion_score = 1 + α*(deleted/total))
//   - FloorSegmentMB      = 2    (treat tiny segments as at least this size)
//   - ExpungeDeletesThreshold = 0.25 (opportunistic expunge above 25% deleted)
type TieredMergePolicy struct {
	SegmentsPerTier   int
	MaxMergeAtOnce    int
	LevelRatio        int
	MaxMergeSizeBytes int64 // cap on total input bytes per merge; 0 = unlimited

	// DeletionWeight (α) scales the deletion penalty.
	// A segment with f deleted docs scores as (1 + α*f/total) × its effective size.
	// 0 disables deletion-aware scoring (pure size-based, like the old behaviour).
	DeletionWeight float64

	// FloorSegmentMB: tiny segments are treated as at least this size in the
	// tiering math, preventing 1-at-a-time merges of micro-flushes.
	FloorSegmentMB int64

	// ExpungeDeletesThreshold: segments with deletion ratio ≥ this value are
	// eligible for opportunistic expunge even when no tier is over-capacity.
	// 0 disables opportunistic expunge.
	ExpungeDeletesThreshold float64
}

// DefaultTieredMergePolicy returns a TieredMergePolicy with default values.
func DefaultTieredMergePolicy() *TieredMergePolicy {
	return &TieredMergePolicy{
		SegmentsPerTier:         10,
		MaxMergeAtOnce:          10,
		LevelRatio:              10,
		DeletionWeight:          2.0,
		FloorSegmentMB:          2,
		ExpungeDeletesThreshold: 0.25,
	}
}

// SelectMerge returns the segments that should be merged in the next merge
// operation, or nil if no merge is needed.
//
// It scans levels from lowest (0) upward and returns the MaxMergeAtOnce
// highest-scored segments from the first level that exceeds SegmentsPerTier.
// Score = effectiveSize × deletionScore, where:
//
//	effectiveSize    = max(SizeBytes, FloorSegmentMB * 1 MB)
//	deletionScore    = 1 + DeletionWeight * (DeletedDocs / DocCount)
//
// Sorting by score descending ensures the most wasteful segments are merged
// first, reducing search-time tombstone filtering and reclaiming disk space
// faster than a pure-age ordering would.
func (p *TieredMergePolicy) SelectMerge(segs []manifest.SegmentRecord) []manifest.SegmentRecord {
	if len(segs) == 0 {
		return nil
	}

	floorBytes := p.FloorSegmentMB * 1024 * 1024

	// Group active segments by level.
	byLevel := make(map[int][]manifest.SegmentRecord)
	for _, s := range segs {
		byLevel[s.Level] = append(byLevel[s.Level], s)
	}

	// Collect levels in ascending order.
	levels := make([]int, 0, len(byLevel))
	for l := range byLevel {
		levels = append(levels, l)
	}
	sort.Ints(levels)

	for _, level := range levels {
		group := byLevel[level]
		if len(group) <= p.SegmentsPerTier {
			continue
		}

		// Score each segment.
		type scored struct {
			rec   manifest.SegmentRecord
			score float64
		}
		sg := make([]scored, len(group))
		for i, s := range group {
			effSize := s.SizeBytes
			if effSize < floorBytes {
				effSize = floorBytes
			}
			if effSize == 0 {
				effSize = 1 // avoid zero score for legacy records with unknown size
			}
			delScore := 1.0
			if p.DeletionWeight > 0 && s.DocCount > 0 && s.DeletedDocs > 0 {
				delScore = 1.0 + p.DeletionWeight*float64(s.DeletedDocs)/float64(s.DocCount)
			}
			sg[i] = scored{rec: s, score: float64(effSize) * delScore}
		}

		// Sort by score descending: most wasteful segments merged first.
		sort.Slice(sg, func(i, j int) bool { return sg[i].score > sg[j].score })

		n := p.MaxMergeAtOnce
		if n > len(sg) {
			n = len(sg)
		}
		candidates := make([]manifest.SegmentRecord, n)
		for i := range candidates {
			candidates[i] = sg[i].rec
		}

		// Cap total input size to bound worst-case merge I/O duration.
		if p.MaxMergeSizeBytes > 0 {
			var total int64
			for i, c := range candidates {
				if total+c.SizeBytes > p.MaxMergeSizeBytes && i >= 2 {
					candidates = candidates[:i]
					break
				}
				total += c.SizeBytes
			}
		}

		return candidates
	}
	return nil
}

// SelectExpungeDeletes returns segments whose deletion ratio meets or exceeds
// ExpungeDeletesThreshold, regardless of tier-level capacity. This enables
// opportunistic reclaim of tombstoned space even when no tier is over-full.
// Returns nil when ExpungeDeletesThreshold is 0 or no segment qualifies.
func (p *TieredMergePolicy) SelectExpungeDeletes(segs []manifest.SegmentRecord) []manifest.SegmentRecord {
	if p.ExpungeDeletesThreshold <= 0 || p.ExpungeDeletesThreshold >= 1 {
		return nil
	}
	var result []manifest.SegmentRecord
	for _, s := range segs {
		if s.DocCount > 0 && float64(s.DeletedDocs)/float64(s.DocCount) >= p.ExpungeDeletesThreshold {
			result = append(result, s)
		}
	}
	return result
}

// NextLevel returns the output level for a merge of segments at the given
// input level. All input segments must be at the same level.
func NextLevel(inputLevel int) int { return inputLevel + 1 }
