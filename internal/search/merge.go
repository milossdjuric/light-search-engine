package search

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/internal/storage/manifest"
)

// runMerge is the background goroutine that checks for merge candidates whenever
// a flush occurs, or periodically.
func (sm *SegmentManager) runMerge() {
	defer sm.wg.Done()

	// mergeCtx is cancelled as soon as stopCh is closed so that any in-progress
	// executeMerge can bail out promptly instead of running to completion.
	mergeCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		select {
		case <-sm.stopCh:
			cancel()
		case <-mergeCtx.Done():
		}
	}()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-sm.stopCh:
			return
		case <-sm.mergeCh:
			sm.tryMerge(mergeCtx)
		case <-ticker.C:
			sm.tryMerge(mergeCtx)
		}
	}
}

// TriggerMerge sends a non-blocking signal to the background merge goroutine.
func (sm *SegmentManager) TriggerMerge() {
	select {
	case sm.mergeCh <- struct{}{}:
	default:
	}
}

// tryMerge checks the merge policy and executes one merge if needed.
// ctx should be derived from stopCh so that a merge in progress during shutdown
// can be aborted promptly.
func (sm *SegmentManager) tryMerge(ctx context.Context) {
	// Back-pressure: if fewer than half the pool slots are free, background
	// flushes are queued up — don't add merge I/O on top and starve them.
	// The next flush completion will re-signal mergeCh, so no work is lost.
	if len(sm.freeBuffers) < cap(sm.freeBuffers)/2 {
		slog.Debug("tryMerge: back-pressure, skipping", "shard", sm.shardID, "free", len(sm.freeBuffers), "cap", cap(sm.freeBuffers))
		return
	}

	sm.mu.RLock()
	recs := make([]manifest.SegmentRecord, len(sm.segRecords))
	copy(recs, sm.segRecords)
	sm.mu.RUnlock()

	candidates := sm.mergePolicy.SelectMerge(recs)
	if len(candidates) == 0 {
		// No tier is over-capacity; check opportunistic expunge-deletes.
		candidates = sm.mergePolicy.SelectExpungeDeletes(recs)
		if len(candidates) > 0 {
			slog.Info("tryMerge: expunge-deletes", "shard", sm.shardID, "n_segs", len(recs), "n_candidates", len(candidates))
		}
	} else {
		slog.Info("tryMerge: candidates selected", "shard", sm.shardID, "n_segs", len(recs), "n_candidates", len(candidates))
	}
	if len(candidates) == 0 {
		return
	}
	if err := sm.executeMerge(ctx, candidates, false); err != nil && !errors.Is(err, context.Canceled) {
		slog.Warn("background merge failed", "shard", sm.shardID, "err", err)
	}
}

// mergeWait consumes n bytes from the merge I/O rate limiter before the caller
// performs disk I/O. Burst is configured to be ≥ one segment's size, so this
// issues a single WaitN per segment — one pause per segment load, matching the
// actual disk I/O boundary and giving flush goroutines a clean window between
// segment reads.
func (sm *SegmentManager) mergeWait(ctx context.Context, n int64) error {
	if n <= 0 {
		return nil
	}
	// Clamp to burst in case a segment is larger than the configured burst.
	if n > int64(sm.mergeLimiter.Burst()) {
		n = int64(sm.mergeLimiter.Burst())
	}
	return sm.mergeLimiter.WaitN(ctx, int(n))
}

// executeMerge loads candidate segments, merges them, registers the result,
// and marks the inputs as merged in the manifest.
func (sm *SegmentManager) executeMerge(ctx context.Context, candidates []manifest.SegmentRecord, skipRateLimit bool) error {
	if len(candidates) == 0 {
		return nil
	}
	// Serialize merge operations within this shard. tryMerge (background goroutine)
	// and forceMerge (startup goroutine) can run concurrently; both generate outID
	// from maxFlushSeq and would collide on the same tmpPath and output key.
	sm.mergeMu.Lock()
	defer sm.mergeMu.Unlock()
	sm.merging.Store(true)
	defer sm.merging.Store(false)

	// TOCTOU guard: candidates were selected before this lock was acquired.
	// Another merge may have consumed one of them while we were waiting. Verify
	// all candidates are still in sm.segRecords; if not, bail silently so the
	// caller can re-select from the current state.
	sm.mu.RLock()
	activeIDs := make(map[string]struct{}, len(sm.segRecords))
	for _, r := range sm.segRecords {
		activeIDs[r.SegmentID] = struct{}{}
	}
	tombstonesSnapshot := make(map[string]struct{}, len(sm.tombstones))
	for docID := range sm.tombstones {
		tombstonesSnapshot[docID] = struct{}{}
	}
	sm.mu.RUnlock()
	for _, c := range candidates {
		if _, ok := activeIDs[c.SegmentID]; !ok {
			slog.Debug("executeMerge: candidate consumed by concurrent merge, skipping",
				"shard", sm.shardID, "segment", c.SegmentID)
			return nil
		}
	}

	outLevel := NextLevel(candidates[0].Level)

	// Resolve local paths for input segments (via store; may download from S3).
	// The merge rate limiter throttles reads so flush I/O isn't starved.
	segsToMerge := make([]*Segment, 0, len(candidates))
	for _, rec := range candidates {
		localPath, err := sm.store.LocalPath(ctx, rec.Path)
		if err != nil {
			return fmt.Errorf("executeMerge LocalPath %s: %w", rec.Path, err)
		}
		// Throttle read: consume tokens proportional to file size before loading.
		// skipRateLimit is true during startup force-merge (no concurrent ingest).
		if sm.mergeLimiter != nil && !skipRateLimit && rec.SizeBytes > 0 {
			if err := sm.mergeWait(ctx, rec.SizeBytes); err != nil {
				return err
			}
		}
		seg, err := LoadSegment(localPath)
		if err != nil {
			return fmt.Errorf("executeMerge LoadSegment %s: %w", rec.Path, err)
		}
		segsToMerge = append(segsToMerge, seg)
	}
	// segsToMerge is built in candidates order, which is not chronological
	// (SelectMerge sorts by deletion score, forceMerge by size) — reorder it
	// oldest-to-newest by FlushSeq so MergeSegmentsWithOptions's "highest
	// index wins" duplicate-docID resolution actually reflects which input
	// segment holds a doc's most recent content, not merge-candidate order.
	orderSegmentsByFlushSeq(segsToMerge, candidates)
	// Merge reads terms in sorted FST order — file positions are mostly
	// ascending. Switch to sequential mode so the kernel issues aggressive
	// read-ahead instead of treating each read as an isolated random access.
	// Deferred eviction drops these pages as soon as the merge is done,
	// freeing RAM for the new merged segment before it is warmed.
	for _, seg := range segsToMerge {
		setMadviseSequential(seg.mmapData)
	}
	defer func() {
		for _, seg := range segsToMerge {
			evictPageCache(seg.mmapData)
		}
	}()

	// Compute a unique output sequence number: max FlushSeq of inputs.
	var maxSeq int64
	for _, rec := range candidates {
		if rec.FlushSeq > maxSeq {
			maxSeq = rec.FlushSeq
		}
	}

	outID := fmt.Sprintf("%s_%d_L%d", sm.shardID, maxSeq, outLevel)
	outKey := fmt.Sprintf("segments/%s.seg", outID)
	tmpPath := filepath.Join(sm.tmpSegDir, fmt.Sprintf("%s.seg", outID))

	mergeOpts := SegmentWriteOptions{
		Compression: sm.compressionMode,
		BloomFPRate: sm.bloomFPRate,
		UseFOR32:    sm.useFOR32,
		SkipStoredFields: sm.skipStoredFields,
		BuildOpts: index.BuildOptions{
			BM25FScaled: len(sm.bm25fFields) > 0,
			BM25FK1:     1.2,
		},
	}
	cleanupTmp := func() {
		os.Remove(tmpPath)
		os.Remove(tmpPath + ".bloom")
		os.Remove(strings.TrimSuffix(tmpPath, ".seg") + ".seg.fld")
	}
	if err := MergeSegmentsWithOptions(tmpPath, segsToMerge, mergeOpts, tombstonesSnapshot); err != nil {
		cleanupTmp()
		return fmt.Errorf("executeMerge MergeSegments: %w", err)
	}

	// Check for shutdown after the expensive FST build — bail before any
	// manifest writes so we don't leave a half-registered segment.
	if ctx.Err() != nil {
		cleanupTmp()
		return ctx.Err()
	}

	// Upload bloom sidecar before the segment.
	if sm.bloomFPRate > 0 {
		bloomTmp := tmpPath + ".bloom"
		if _, statErr := os.Stat(bloomTmp); statErr == nil {
			if err := sm.store.PutFile(ctx, outKey+".bloom", bloomTmp); err != nil {
				os.Remove(bloomTmp)
			}
		}
	}

	// Upload stored fields sidecar if written by MergeSegmentsWithOptions.
	fldTmpPath := strings.TrimSuffix(tmpPath, ".seg") + ".seg.fld"
	if _, statErr := os.Stat(fldTmpPath); statErr == nil {
		if err := sm.store.PutFile(ctx, outKey+".fld", fldTmpPath); err != nil {
			os.Remove(fldTmpPath) // non-fatal; search still works without it
		}
	}

	// Upload merged segment (tmpPath consumed by PutFile).
	if err := sm.store.PutFile(ctx, outKey, tmpPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("executeMerge PutFile: %w", err)
	}

	// Resolve local path for loading.
	localOutPath, err := sm.store.LocalPath(ctx, outKey)
	if err != nil {
		return fmt.Errorf("executeMerge LocalPath output: %w", err)
	}
	outSeg, err := LoadSegment(localOutPath)
	if err != nil {
		return fmt.Errorf("executeMerge LoadSegment output: %w", err)
	}

	var outSizeBytes int64
	if fi, statErr := os.Stat(localOutPath); statErr == nil {
		outSizeBytes = fi.Size()
	}

	// Recompute DeletedDocs from the current (not the pre-merge-snapshot)
	// tombstone set: a doc can be deleted while the merge is running, after
	// tombstonesSnapshot was taken but before it's excluded from a future
	// merge. Scanning the merged output against live tombstones now (same
	// pattern as loadNilSegments) keeps this segment's count accurate instead
	// of staying stuck at a stale value until the next merge cycle.
	//
	// While scanning, also collect tombstones that this merge physically
	// purged (present in an input segment, absent from the output): a
	// purged doc can only ever have lived in one segment at a time (it
	// would have cleared its own tombstone on re-index — see the WAL
	// replay/re-index fix elsewhere in this package), so once it's gone
	// from the merge output it's gone shard-wide and the tombstone entry
	// can be dropped. Leaving it behind would grow sm.tombstones
	// unboundedly over the shard's lifetime and add redundant scan work
	// here on every future merge for a doc that's already gone.
	sm.mu.RLock()
	var outDeletedDocs int64
	var purgedTombstones []string
	for docID := range sm.tombstones {
		if _, ok := outSeg.docIDToNum[docID]; ok {
			outDeletedDocs++
			continue
		}
		for _, seg := range segsToMerge {
			if _, ok := seg.docIDToNum[docID]; ok {
				purgedTombstones = append(purgedTombstones, docID)
				break
			}
		}
	}
	sm.mu.RUnlock()

	if len(purgedTombstones) > 0 {
		sm.mu.Lock()
		for _, docID := range purgedTombstones {
			delete(sm.tombstones, docID)
		}
		sm.mu.Unlock()
		for _, docID := range purgedTombstones {
			// Best-effort: a leftover manifest tombstone record for an
			// already-purged doc is harmless (nothing left for it to hide).
			_ = sm.meta.RemoveTombstone(sm.shardID, docID)
		}
	}

	outRec := manifest.SegmentRecord{
		SegmentID:   outID,
		ShardID:     sm.shardID,
		Level:       outLevel,
		DocCount:    outSeg.DocCount(),
		DeletedDocs: outDeletedDocs,
		Path:        outKey, // object-store key
		FlushSeq:    maxSeq,
		SizeBytes:   outSizeBytes,
	}

	// Register output and remove inputs atomically in the manifest.
	// Safety invariant: output is registered before inputs are removed,
	// so a crash leaves both old and new segments visible (safe to recover).
	removeIDs := make([]string, len(candidates))
	for i, rec := range candidates {
		removeIDs[i] = rec.SegmentID
	}
	if err := sm.meta.AddThenRemoveSegments(outRec, removeIDs); err != nil {
		return fmt.Errorf("executeMerge AddThenRemoveSegments: %w", err)
	}

	// Update in-memory state atomically.
	sm.mu.Lock()
	candidateIDs := make(map[string]bool, len(candidates))
	for _, rec := range candidates {
		candidateIDs[rec.SegmentID] = true
	}
	newSegs := make([]*Segment, 0, len(sm.segments))
	newRecs := make([]manifest.SegmentRecord, 0, len(sm.segRecords))
	for i, rec := range sm.segRecords {
		if !candidateIDs[rec.SegmentID] {
			newSegs = append(newSegs, sm.segments[i])
			newRecs = append(newRecs, rec)
		}
	}
	newSegs = append(newSegs, outSeg)
	newRecs = append(newRecs, outRec)
	sm.segments = newSegs
	sm.segRecords = newRecs
	sm.mu.Unlock()

	// Delete input files from the store now that they are no longer referenced.
	for _, rec := range candidates {
		_ = sm.store.Delete(ctx, rec.Path)
		_ = sm.store.Delete(ctx, rec.Path+".bloom") // non-fatal; may not exist
		_ = sm.store.Delete(ctx, rec.Path+".fld")   // non-fatal; may not exist
	}
	return nil
}

// startupMerge runs merge passes until the policy finds no more candidates.
// Unlike the background merge goroutine it skips the flush back-pressure check
// because there is no active ingest at startup.
func (sm *SegmentManager) startupMerge(ctx context.Context) error {
	sm.mu.RLock()
	initial := len(sm.segRecords)
	sm.mu.RUnlock()
	if initial == 0 {
		return nil
	}
	slog.Info("startup merge started", "shard", sm.shardID, "segments", initial)
	t0 := time.Now()
	passes := 0
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		sm.mu.RLock()
		recs := make([]manifest.SegmentRecord, len(sm.segRecords))
		copy(recs, sm.segRecords)
		sm.mu.RUnlock()

		candidates := sm.mergePolicy.SelectMerge(recs)
		if len(candidates) == 0 {
			break
		}
		if err := sm.executeMerge(ctx, candidates, true); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			slog.Error("startup merge pass failed", "shard", sm.shardID, "err", err)
			break
		}
		passes++
	}
	sm.mu.RLock()
	remaining := len(sm.segRecords)
	sm.mu.RUnlock()
	slog.Info("startup merge complete", "shard", sm.shardID,
		"passes", passes, "segments", remaining, "elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}

// orderSegmentsByFlushSeq reorders segsToMerge in place to ascending
// FlushSeq of the corresponding entry in candidates. segsToMerge and
// candidates must be the same length and index-aligned (as executeMerge
// builds them, one LoadSegment per candidate in order).
func orderSegmentsByFlushSeq(segsToMerge []*Segment, candidates []manifest.SegmentRecord) {
	type pair struct {
		seg      *Segment
		flushSeq int64
	}
	pairs := make([]pair, len(segsToMerge))
	for i, seg := range segsToMerge {
		pairs[i] = pair{seg: seg, flushSeq: candidates[i].FlushSeq}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].flushSeq < pairs[j].flushSeq })
	for i, p := range pairs {
		segsToMerge[i] = p.seg
	}
}

// forceMerge merges until the shard has at most maxSegments segments,
// bypassing the tiered policy's candidate selection. Unlike startupMerge it
// accepts an explicit target segment count and batches candidates directly
// (no policy filtering). Used by the HTTP forceMerge endpoint.
func (sm *SegmentManager) forceMerge(ctx context.Context, maxSegments int) error {
	if maxSegments < 1 {
		maxSegments = 1
	}
	batch := sm.mergePolicy.MaxMergeAtOnce
	if batch < 2 {
		batch = 10
	}
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		sm.mu.RLock()
		count := len(sm.segRecords)
		recs := make([]manifest.SegmentRecord, len(sm.segRecords))
		copy(recs, sm.segRecords)
		sm.mu.RUnlock()

		if count <= maxSegments {
			break
		}
		// Sort smallest segments first so the first merge pass is cheap and
		// quickly reduces segment count; larger segments are merged later.
		sort.Slice(recs, func(i, j int) bool {
			si, sj := recs[i].SizeBytes, recs[j].SizeBytes
			if si == 0 {
				si = 1 << 62
			} // unknown size → treat as large
			if sj == 0 {
				sj = 1 << 62
			}
			return si < sj
		})
		// Pick up to `batch` candidates — the N smallest records.
		n := batch
		if n > len(recs) {
			n = len(recs)
		}
		candidates := recs[:n]
		if err := sm.executeMerge(ctx, candidates, true); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			return fmt.Errorf("forceMerge: %w", err)
		}
	}
	sm.mu.RLock()
	remaining := len(sm.segRecords)
	sm.mu.RUnlock()
	slog.Info("force merge complete", "shard", sm.shardID, "segments", remaining)
	return nil
}

// ForceMergeDeletes merges all segments whose deletion ratio is at or above
// minDeletedPct. It repeats until no qualifying segment remains or ctx is
// cancelled. minDeletedPct == 0 uses the policy's ExpungeDeletesThreshold.
func (sm *SegmentManager) ForceMergeDeletes(ctx context.Context, minDeletedPct float64) error {
	threshold := minDeletedPct
	if threshold <= 0 {
		threshold = sm.mergePolicy.ExpungeDeletesThreshold
	}
	if threshold <= 0 {
		threshold = 0.25
	}
	batch := sm.mergePolicy.MaxMergeAtOnce
	if batch < 2 {
		batch = 10
	}
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		sm.mu.RLock()
		recs := make([]manifest.SegmentRecord, len(sm.segRecords))
		copy(recs, sm.segRecords)
		sm.mu.RUnlock()

		var candidates []manifest.SegmentRecord
		for _, rec := range recs {
			if rec.DocCount > 0 && float64(rec.DeletedDocs)/float64(rec.DocCount) >= threshold {
				candidates = append(candidates, rec)
				if len(candidates) >= batch {
					break
				}
			}
		}
		if len(candidates) == 0 {
			break
		}
		if err := sm.executeMerge(ctx, candidates, true); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			return fmt.Errorf("ForceMergeDeletes: %w", err)
		}
	}
	sm.mu.RLock()
	remaining := len(sm.segRecords)
	sm.mu.RUnlock()
	slog.Info("ForceMergeDeletes complete", "shard", sm.shardID, "segments", remaining, "threshold", threshold)
	return nil
}
