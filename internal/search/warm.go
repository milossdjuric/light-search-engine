package search

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// warmConcKey is the context key for the shared warm-I/O semaphore.
// ShardManager injects it so total concurrent segment-warm goroutines across
// ALL shards stay bounded. Without this, N shards × M segments goroutines all
// do concurrent sequential page scans, turning sequential I/O into N×M
// competing streams and causing apparent freezes on large indexes.
type warmConcKey struct{}

// warmMaxParallelSegments is the default cap on concurrent mmap-touching
// goroutines when no semaphore is injected via context. Sequential page scans
// are 10–20× faster per byte than random reads, but only when the SSD queue
// is not saturated by many concurrent streams. 2 preserves near-sequential
// throughput while allowing mild pipeline overlap between segments.
const warmMaxParallelSegments = 2

// warmSem extracts the shared semaphore from ctx, or creates a local one.
func warmSem(ctx context.Context) chan struct{} {
	if sem, ok := ctx.Value(warmConcKey{}).(chan struct{}); ok && sem != nil {
		return sem
	}
	return make(chan struct{}, warmMaxParallelSegments)
}

// WarmupSegments loads all lazy-nil segments into memory (structural parse:
// FST, term metadata, doc lengths, doc IDs). Does NOT touch posting data —
// use WarmTopTerms for that. Returns quickly so the caller can mark ready.
func (sm *SegmentManager) WarmupSegments(ctx context.Context) error {
	sm.mu.RLock()
	total := len(sm.segments)
	sm.mu.RUnlock()
	if total == 0 {
		return nil
	}
	slog.Info("loading segment structure", "shard", sm.shardID, "count", total)
	t0 := time.Now()
	if err := sm.loadNilSegments(ctx); err != nil {
		return fmt.Errorf("WarmupSegments %s: %w", sm.shardID, err)
	}
	slog.Info("segment structure loaded", "shard", sm.shardID, "elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}

// WarmTopTerms touches the posting pages for the top-k highest-DF terms in
// every segment, loading them into the OS page cache synchronously.
// If mlock is true, also attempts to pin the segments' pages in RAM.
// Concurrent segment goroutines are bounded by the semaphore in ctx (injected
// by ShardManager) to prevent I/O saturation across shards.
func (sm *SegmentManager) WarmTopTerms(ctx context.Context, k int, mlock bool) error {
	sm.warming.Store(true)
	defer sm.warming.Store(false)

	sm.mu.RLock()
	segs := make([]*Segment, len(sm.segments))
	copy(segs, sm.segments)
	sm.mu.RUnlock()

	slog.Info("warming top terms", "shard", sm.shardID, "top_k", k, "segments", len(segs))
	t0 := time.Now()

	sem := warmSem(ctx)
	var wg sync.WaitGroup
	var done atomic.Int64
	// Pre-count non-nil segments before launching goroutines; goroutines read
	// total via closure and a plain int64 would race with loop modifications.
	var total int64
	for _, seg := range segs {
		if seg != nil {
			total++
		}
	}
	for _, seg := range segs {
		if seg == nil {
			continue
		}
		seg := seg
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			seg.WarmTopKTerms(k)
			if mlock {
				_ = seg.TryMlock()
			}
			n := done.Add(1)
			slog.Debug("warm segment done", "shard", sm.shardID, "progress", fmt.Sprintf("%d/%d", n, total))
		}()
	}
	wg.Wait()

	slog.Info("top terms warm", "shard", sm.shardID, "elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}

// WarmTiered warms each segment according to its size relative to threshold:
//   - segments whose mmap size is <= thresholdBytes get a full sequential
//     page scan (WarmFull) — cheap because they are small, and ensures ALL
//     their posting data is hot (LinkedIn/Twitter hot-tier approach).
//   - larger segments get targeted top-K DF warmup (WarmTopKTerms) only,
//     since a full scan of a multi-GB segment would take too long.
//
// If mlock is true, TryMlock is called on every segment after warmup so the
// OS cannot evict its pages under memory pressure.
// Concurrent segment goroutines are bounded by the semaphore in ctx (injected
// by ShardManager) to prevent I/O saturation across shards.
func (sm *SegmentManager) WarmTiered(ctx context.Context, thresholdBytes int64, topK int, mlock bool) error {
	sm.warming.Store(true)
	defer sm.warming.Store(false)

	sm.mu.RLock()
	segs := make([]*Segment, len(sm.segments))
	copy(segs, sm.segments)
	sm.mu.RUnlock()

	slog.Info("warming tiered", "shard", sm.shardID,
		"segments", len(segs),
		"threshold_mb", thresholdBytes/(1024*1024),
		"top_k", topK)
	t0 := time.Now()

	// Fire MADV_WILLNEED on all segments that will be fully scanned, before
	// launching the semaphore-bounded goroutines. Submitting all prefetch
	// requests at once lets the kernel batch them into a single saturated I/O
	// queue (~500 MB/s) rather than issuing them one-at-a-time inside the
	// semaphore (which would serialize them at ~14 MB/s per page-fault stream).
	// Large segments that get top-K warmup are excluded to avoid wasting I/O on
	// pages that will never be accessed by the targeted scan.
	for _, seg := range segs {
		if seg != nil && seg.SizeBytes() <= thresholdBytes {
			seg.Prefetch()
		}
	}

	sem := warmSem(ctx)
	var fullCount, topkCount, done atomic.Int64
	// Pre-count non-nil segments before launching goroutines so goroutines
	// can read total safely (plain int64 would race with loop modifications).
	var total int64
	for _, seg := range segs {
		if seg != nil {
			total++
		}
	}
	var wg sync.WaitGroup
	for _, seg := range segs {
		if seg == nil {
			continue
		}
		seg := seg
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			sizeMB := seg.SizeBytes() / (1024 * 1024)
			if seg.SizeBytes() <= thresholdBytes {
				seg.WarmFull()
				fullCount.Add(1)
			} else {
				seg.WarmTopKTerms(topK)
				topkCount.Add(1)
			}
			if mlock {
				_ = seg.TryMlock()
			}
			n := done.Add(1)
			slog.Debug("warm segment done", "shard", sm.shardID,
				"progress", fmt.Sprintf("%d/%d", n, total),
				"size_mb", sizeMB)
		}()
	}
	wg.Wait()

	slog.Info("tiered warm done", "shard", sm.shardID,
		"full", fullCount.Load(), "top_k", topkCount.Load(),
		"elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}
