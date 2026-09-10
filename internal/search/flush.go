package search

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/internal/storage/manifest"
)

// flushOrSwap is called with sm.mu held and always releases it before returning.
//
// Fast path (pool has a spare buffer): swap the full buffer for the empty one
// while mu is still held (~microseconds), release mu, and flush the old buffer
// in a background goroutine.  New writes proceed immediately on the fresh buffer.
//
// Slow path (all N pool buffers are in-flight): release mu, block until any
// background flush returns its buffer to the pool, re-acquire mu, re-check
// whether a flush is still needed, then do the swap and launch another
// background flush.  No synchronous I/O on either path — mu is never held
// while writing a segment file.
func (sm *SegmentManager) flushOrSwap() error {
	var newBuf *index.IndexBuilder
	select {
	case newBuf = <-sm.freeBuffers:
		// Fast path: a spare buffer was available immediately.
	default:
		// Slow path: all N buffers are busy. Release mu so background
		// goroutines can call back into sm.mu when registering their segments,
		// then block until one of them returns a buffer to the pool.
		sm.mu.Unlock()
		newBuf = <-sm.freeBuffers // blocks until any background flush completes
		sm.mu.Lock()
		// Re-check: another writer may have flushed while we were waiting.
		if sm.buffer.MemoryEstimate() < sm.memThreshold && sm.bufferDocs < sm.maxBufferDocs {
			sm.freeBuffers <- newBuf // return the slot unused
			sm.mu.Unlock()
			return nil
		}
	}

	sm.walMu.Lock()
	flushSeq := sm.nextSeq - 1
	sm.walMu.Unlock()

	oldBuf := sm.buffer
	oldDocs := sm.bufferDocs
	oldTexts := sm.bufferTexts
	sm.buffer = newBuf
	sm.bufferDocs = 0
	sm.bufferDirty = false
	sm.bufferIdx = nil
	sm.bufferTexts = make(map[string]string, sm.maxBufferDocs)
	sm.mu.Unlock() // release mu before any I/O

	sm.flushWg.Add(1)
	go sm.runBackgroundFlush(oldBuf, oldTexts, flushSeq, oldDocs)
	return nil
}

// runBackgroundFlush writes oldBuf to a segment file, registers it in the
// manifest, updates sm.segments, resets the builder, and returns it to the pool.
// It is the only place that appends to sm.segments outside of flushLocked.
// WAL rotation is intentionally skipped here; it only happens in flushLocked
// (explicit flush / Close) to keep manifest flush_seq ordering monotonic.
func (sm *SegmentManager) runBackgroundFlush(oldBuf *index.IndexBuilder, texts map[string]string, flushSeq int64, docs int) {
	defer sm.flushWg.Done()

	slog.Info("background flush started", "shard", sm.shardID, "docs", docs, "flush_seq", flushSeq)

	segName := fmt.Sprintf("%s_%d_L0.seg", sm.shardID, flushSeq)
	segKey := "segments/" + segName
	tmpPath := filepath.Join(sm.tmpSegDir, segName)

	writeOpts := SegmentWriteOptions{
		Compression: sm.compressionMode,
		BloomFPRate: sm.bloomFPRate,
		UseFOR32:    sm.useFOR32,
		BuildOpts:   index.BuildOptions{Fields: sm.bm25fFields},
	}
	if err := WriteSegmentWithOptions(tmpPath, oldBuf, writeOpts); err != nil {
		slog.Error("background flush WriteSegment", "shard", sm.shardID, "err", err)
		oldBuf.Reset()
		sm.freeBuffers <- oldBuf
		return
	}

	ctx := context.Background()

	var segSizeBytes int64
	if fi, statErr := os.Stat(tmpPath); statErr == nil {
		segSizeBytes = fi.Size()
	}

	// Fire-and-forget goroutine for stored fields sidecar — independent of segment upload.
	if !sm.skipStoredFields && len(texts) > 0 {
		fldTmpPath := tmpPath + ".fld"
		go func() {
			if werr := WriteStoredFields(fldTmpPath, texts); werr != nil {
				slog.Warn("background flush WriteStoredFields", "shard", sm.shardID, "err", werr)
			} else if perr := sm.store.PutFile(ctx, segKey+".fld", fldTmpPath); perr != nil {
				os.Remove(fldTmpPath)
				slog.Warn("background flush PutFile fld", "shard", sm.shardID, "err", perr)
			}
		}()
	}

	if err := sm.store.PutFile(ctx, segKey, tmpPath); err != nil {
		os.Remove(tmpPath)
		slog.Error("background flush PutFile", "shard", sm.shardID, "err", err)
		oldBuf.Reset()
		sm.freeBuffers <- oldBuf
		return
	}

	// Use the pre-swap doc count directly — no LoadSegment needed.
	rec := manifest.SegmentRecord{
		SegmentID: fmt.Sprintf("%s_%d_L0", sm.shardID, flushSeq),
		ShardID:   sm.shardID,
		Level:     0,
		DocCount:  docs,
		Path:      segKey,
		FlushSeq:  flushSeq,
		SizeBytes: segSizeBytes,
	}
	if regErr := sm.meta.AddSegment(rec); regErr != nil {
		slog.Error("background flush AddSegment", "shard", sm.shardID, "err", regErr)
		oldBuf.Reset()
		sm.freeBuffers <- oldBuf
		return
	}

	// Update the segment list under mu (brief critical section — no I/O).
	// Append nil instead of the loaded segment: keeps segment byte slices out of
	// memory during ingest. loadNilSegments loads them on demand at search time.
	sm.mu.Lock()
	sm.segments = append(sm.segments, nil)
	sm.segRecords = append(sm.segRecords, rec)
	sm.mu.Unlock()

	slog.Info("background flush complete", "shard", sm.shardID, "docs", docs, "flush_seq", flushSeq)

	// Return the now-empty builder to the pool — unblocks the next flushOrSwap.
	oldBuf.Reset()
	sm.freeBuffers <- oldBuf

	// Lazy goroutine: upload bloom sidecar after the buffer is returned to pool.
	// Bloom is an optimisation; segments searched before upload simply do a full scan.
	if sm.bloomFPRate > 0 {
		bloomTmp := tmpPath + ".bloom"
		if _, statErr := os.Stat(bloomTmp); statErr == nil {
			go func() {
				if err := sm.store.PutFile(ctx, segKey+".bloom", bloomTmp); err != nil {
					os.Remove(bloomTmp)
				}
			}()
		}
	}

	// Signal the merge goroutine that a new L0 segment is available.
	select {
	case sm.mergeCh <- struct{}{}:
	default:
	}
}

// flushLocked writes the current buffer to disk. Must be called with mu held.
// nextSeq is read under walMu (not mu) because appendWAL protects it with walMu.
func (sm *SegmentManager) flushLocked() error {
	sm.walMu.Lock()
	flushSeq := sm.nextSeq - 1
	sm.walMu.Unlock()

	segName := fmt.Sprintf("%s_%d_L0.seg", sm.shardID, flushSeq)
	segKey := "segments/" + segName               // key stored in manifest / object store
	tmpPath := filepath.Join(sm.tmpSegDir, segName) // local temp file

	writeOpts := SegmentWriteOptions{
		Compression: sm.compressionMode,
		BloomFPRate: sm.bloomFPRate,
		UseFOR32:    sm.useFOR32,
		BuildOpts:   index.BuildOptions{Fields: sm.bm25fFields},
	}
	if err := WriteSegmentWithOptions(tmpPath, sm.buffer, writeOpts); err != nil {
		return fmt.Errorf("flush WriteSegment: %w", err)
	}

	ctx := context.Background()

	// Upload bloom sidecar before the segment so LoadSegment can find it.
	if sm.bloomFPRate > 0 {
		bloomTmp := tmpPath + ".bloom"
		if _, statErr := os.Stat(bloomTmp); statErr == nil {
			if err := sm.store.PutFile(ctx, segKey+".bloom", bloomTmp); err != nil {
				os.Remove(bloomTmp) // non-fatal — bloom is an optimisation
			}
		}
	}

	// Write stored fields sidecar.
	if !sm.skipStoredFields && len(sm.bufferTexts) > 0 {
		fldTmpPath := tmpPath + ".fld"
		if werr := WriteStoredFields(fldTmpPath, sm.bufferTexts); werr != nil {
			slog.Warn("flushLocked WriteStoredFields", "shard", sm.shardID, "err", werr)
		} else if perr := sm.store.PutFile(ctx, segKey+".fld", fldTmpPath); perr != nil {
			os.Remove(fldTmpPath)
		}
		sm.bufferTexts = make(map[string]string)
	}

	var segSizeBytes int64
	if fi, statErr := os.Stat(tmpPath); statErr == nil {
		segSizeBytes = fi.Size()
	}

	// Upload segment (tmpPath is consumed by PutFile).
	if err := sm.store.PutFile(ctx, segKey, tmpPath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("flush PutFile: %w", err)
	}

	// Use current bufferDocs count — no LoadSegment needed.
	rec := manifest.SegmentRecord{
		SegmentID: fmt.Sprintf("%s_%d_L0", sm.shardID, flushSeq),
		ShardID:   sm.shardID,
		Level:     0,
		DocCount:  sm.bufferDocs,
		Path:      segKey,
		FlushSeq:  flushSeq,
		SizeBytes: segSizeBytes,
	}
	if err := sm.meta.AddSegment(rec); err != nil {
		return fmt.Errorf("flush AddSegment: %w", err)
	}

	sm.segments = append(sm.segments, nil) // loaded on demand at search time
	sm.segRecords = append(sm.segRecords, rec)

	// Reset buffer, reusing the temp file to avoid repeated OS-level file creation.
	sm.buffer.Reset()
	sm.bufferDocs = 0
	sm.bufferDirty = false
	sm.bufferIdx = nil

	// Rotate WAL: seal current file, open next numbered file, delete sealed file.
	if err := sm.rotateWALLocked(flushSeq); err != nil {
		return fmt.Errorf("flush rotateWAL: %w", err)
	}

	// Signal merge goroutine.
	select {
	case sm.mergeCh <- struct{}{}:
	default:
	}
	return nil
}
