package shard

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"sort"
	"sync"
	"time"

	bloom "github.com/bits-and-blooms/bloom/v3"

	"search-eval-platform/internal/index"
	"search-eval-platform/internal/query"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/segment"
	"search-eval-platform/internal/storage"
	"search-eval-platform/pkg/types"
)

// loadNilSegments loads any nil entries in sm.segments from disk in parallel.
// Segments are stored as nil after a flush to avoid keeping their full byte
// slices in RAM during ingest-only workloads. The first search call loads them
// on demand; subsequent callers see them already loaded.
func (sm *SegmentManager) loadNilSegments(ctx context.Context) error {
	// Fast path: no nil entries — done.
	sm.mu.RLock()
	type pending struct {
		idx  int
		path string
	}
	var work []pending
	for i, seg := range sm.segments {
		if seg == nil {
			work = append(work, pending{i, sm.segRecords[i].Path})
		}
	}
	sm.mu.RUnlock()
	if len(work) == 0 {
		return nil
	}

	// Load all nil segments in parallel — I/O dominates, so goroutines help
	// especially when segments are spread across device queues or S3 shards.
	loaded := make([]*segment.Segment, len(work))
	errs := make([]error, len(work))
	var wg sync.WaitGroup
	for j, w := range work {
		j, w := j, w
		wg.Add(1)
		go func() {
			defer wg.Done()
			localPath, err := sm.store.LocalPath(ctx, w.path)
			if err != nil {
				// A concurrent merge may have deleted the file; treat as transient.
				if errors.Is(err, storage.ErrNotFound) || errors.Is(err, os.ErrNotExist) {
					return
				}
				errs[j] = fmt.Errorf("loadNilSegments LocalPath %s: %w", w.path, err)
				return
			}
			seg, err := segment.LoadSegment(localPath)
			if err != nil {
				// File was deleted between LocalPath and segment.LoadSegment (merge race).
				if errors.Is(err, os.ErrNotExist) {
					return
				}
				errs[j] = fmt.Errorf("loadNilSegments segment.LoadSegment %s: %w", w.path, err)
				return
			}
			loaded[j] = seg
		}()
	}
	wg.Wait()

	// Surface the first non-transient error, if any.
	for _, err := range errs {
		if err != nil {
			return err
		}
	}

	// Write results back under write lock. Only fill slots that are still nil
	// (a concurrent flush could have populated one in the meantime). Also
	// guard against a concurrent merge that may have shrunk sm.segments.
	sm.mu.Lock()
	for j, w := range work {
		if w.idx < len(sm.segments) && sm.segments[w.idx] == nil {
			sm.segments[w.idx] = loaded[j]
			// Compute DeletedDocs from the current tombstone set so the merge
			// policy has accurate deletion ratios from the first merge attempt.
			if loaded[j] != nil && w.idx < len(sm.segRecords) {
				var count int64
				for docID := range sm.tombstones {
					if loaded[j].HasDoc(docID) {
						count++
					}
				}
				sm.segRecords[w.idx].DeletedDocs = count
			}
		}
	}
	sm.mu.Unlock()
	return nil
}

// Search queries all segments plus the in-memory buffer in parallel and
// RRF-merges the results. Tombstoned docs are filtered from the final list.
// If scorer is nil the SegmentManager's default scorer is used.
func (sm *SegmentManager) Search(q string, topK int, scorer scoring.Scorer) ([]types.ScoredDocument, error) {
	if scorer == nil {
		scorer = sm.scorer
	}

	pq := query.ParseQuery(query.PreprocessQuery(q), sm.tokenizer, sm.synonyms)
	// Repeated query words are kept so each scores once per occurrence (and
	// the per-segment UB sums below account for it).
	tokens := pq.ScoringTokens()
	if len(tokens) == 0 {
		return nil, nil
	}
	if err := sm.loadNilSegments(context.Background()); err != nil {
		return nil, fmt.Errorf("Search: %w", err)
	}

	sm.mu.RLock()
	segs := make([]*segment.Segment, len(sm.segments))
	copy(segs, sm.segments)
	segRecords := make([]storage.SegmentRecord, len(sm.segRecords))
	copy(segRecords, sm.segRecords)
	tombstones := sm.tombstones
	tombBloom := sm.tombstoneBloom
	bufIdx := sm.bufferIdx
	bufferDocIDs := sm.bufferDocIDs
	rebuild := sm.needsSnapshotLocked()
	sm.mu.RUnlock()

	// The buffer snapshot (the search index plus the doc-ID membership set
	// used to detect segment hits superseded by a newer buffer write — see
	// searchAll) is rebuilt when missing (right after a flush) or, after
	// writes, at most once per refresh interval. Between rebuilds searches
	// reuse the snapshot, so concurrent writers can't force a write-lock and
	// an O(buffer) rebuild on every query.
	if rebuild {
		sm.mu.Lock()
		if sm.needsSnapshotLocked() {
			sm.bufferIdx = sm.buffer.BuildWithOptions(index.BuildOptions{Fields: sm.bm25fFields})
			docIDs := make(map[string]struct{}, len(sm.bufferTexts))
			for id := range sm.bufferTexts {
				docIDs[id] = struct{}{}
			}
			sm.bufferDocIDs = docIDs
			sm.bufferDirty = false
			sm.snapshotAt = time.Now()
		}
		bufIdx = sm.bufferIdx
		bufferDocIDs = sm.bufferDocIDs
		sm.mu.Unlock()
	}

	// Oversample when filters are present so post-filtering has enough
	// candidates to fill topK after exclusions.
	fetchK := topK
	if pq.HasFilters() {
		fetchK = topK * 3
		if fetchK < 30 {
			fetchK = 30
		}
	}
	results := sm.searchAll(tokens, segs, segRecords, bufIdx, bufferDocIDs, tombstones, tombBloom, fetchK, scorer)
	if pq.HasFilters() {
		results = sm.applyQueryFilters(results, pq, segs, segRecords, bufIdx, bufferDocIDs, topK)
	}
	return results, nil
}

// DefaultPhraseBoost is the score multiplier (minus 1) applied per matched
// "quoted phrase" when SegmentManagerOptions.PhraseBoost is 0.
const DefaultPhraseBoost = 0.25

func resolvePhraseBoost(v float64) float64 {
	switch {
	case v == 0:
		return DefaultPhraseBoost
	case v < 0:
		return 0
	}
	return v
}

// applyQueryFilters enforces Must/Not constraints from pq on the scored
// (oversampled) results, and boosts each result by (1 + phraseBoost) per
// quoted phrase it contains, returning at most topK results.
//
// Phrases boost rather than filter: quotes in natural-language queries
// (quoted speech, titles) are rarely meant as a hard requirement, and a
// strict filter dropped most relevant documents for such queries.
func (sm *SegmentManager) applyQueryFilters(
	results []types.ScoredDocument,
	pq *query.ParsedQuery,
	segs []*segment.Segment,
	segRecords []storage.SegmentRecord,
	bufIdx *index.InvertedIndex,
	bufferDocIDs map[string]struct{},
	topK int,
) []types.ScoredDocument {
	boostPhrases := len(pq.Phrases) > 0 && sm.phraseBoost > 0
	out := make([]types.ScoredDocument, 0, len(results))
	for _, doc := range results {
		// Without boosting, order can't change: stop once topK survive.
		if !boostPhrases && len(out) >= topK {
			break
		}
		keep := true

		for _, term := range pq.Must {
			if !sm.docHasTermAnywhere(term, doc.DocID, segs, segRecords, bufIdx, bufferDocIDs) {
				keep = false
				break
			}
		}
		if !keep {
			continue
		}

		for _, term := range pq.Not {
			if sm.docHasTermAnywhere(term, doc.DocID, segs, segRecords, bufIdx, bufferDocIDs) {
				keep = false
				break
			}
		}
		if !keep {
			continue
		}

		if boostPhrases {
			if text := sm.loadDocText(doc.DocID, segs); text != "" {
				docTokens := sm.tokenizer.Tokenize(text)
				for _, phrase := range pq.Phrases {
					if containsTokenSeq(docTokens, phrase) {
						doc.Score *= 1 + sm.phraseBoost
					}
				}
			}
		}
		out = append(out, doc)
	}

	if boostPhrases {
		sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	}
	if len(out) > topK {
		out = out[:topK]
	}
	for i := range out {
		out[i].Rank = i + 1
	}
	return out
}

// containsTokenSeq reports whether seq occurs contiguously in tokens.
func containsTokenSeq(tokens, seq []string) bool {
	if len(seq) == 0 || len(seq) > len(tokens) {
		return false
	}
outer:
	for i := 0; i+len(seq) <= len(tokens); i++ {
		for j, t := range seq {
			if tokens[i+j] != t {
				continue outer
			}
		}
		return true
	}
	return false
}

// docHasTermAnywhere checks the (term, docID) pair against only the source
// that currently holds docID's authoritative content: the buffer if docID
// lives there (bufferDocIDs), otherwise the single most-recently-flushed
// segment that contains it. A doc updated since its last flush can
// transiently exist in more than one on-disk segment until the next merge
// consolidates them (see searchAll's recency handling); scanning every
// segment here — including stale pre-update copies — would let a Must/Not
// filter match against content the doc no longer has.
func (sm *SegmentManager) docHasTermAnywhere(term, docID string, segs []*segment.Segment, segRecords []storage.SegmentRecord, bufIdx *index.InvertedIndex, bufferDocIDs map[string]struct{}) bool {
	if bufIdx != nil && bufIdx.HasTerm(term, docID) {
		return true
	}
	if _, inBuffer := bufferDocIDs[docID]; inBuffer {
		// The buffer holds the authoritative copy for this doc — the check
		// above already covered it, so any segment copy is stale.
		return false
	}
	if seg := mostRecentSegmentFor(docID, segs, segRecords); seg != nil {
		return seg.HasTerm(term, docID)
	}
	return false
}

// mostRecentSegmentFor returns the segment (from segs, index-aligned with
// segRecords) that holds docID with the highest FlushSeq — the same
// "most recent copy wins" rule searchAll applies when merging query results.
// Returns nil if no segment in segs contains docID.
func mostRecentSegmentFor(docID string, segs []*segment.Segment, segRecords []storage.SegmentRecord) *segment.Segment {
	var best *segment.Segment
	var bestSeq int64
	for i, seg := range segs {
		if seg == nil {
			continue
		}
		if !seg.HasDoc(docID) {
			continue
		}
		var seq int64
		if i < len(segRecords) {
			seq = segRecords[i].FlushSeq
		}
		if best == nil || seq > bestSeq {
			best = seg
			bestSeq = seq
		}
	}
	return best
}

// loadDocText returns the raw stored text for docID: the in-memory buffer's
// copy if docID hasn't been flushed yet (the buffer always holds the most
// recent write), otherwise the first segment that has it. Returns "" if
// stored fields are disabled or the document is not found anywhere.
func (sm *SegmentManager) loadDocText(docID string, segs []*segment.Segment) string {
	sm.mu.RLock()
	bufText, hasBufText := sm.bufferTexts[docID]
	sm.mu.RUnlock()
	if hasBufText {
		return bufText
	}
	for _, seg := range segs {
		if text, ok := seg.GetText(docID); ok && text != "" {
			return text
		}
	}
	return ""
}

// TermDFs returns total doc count and per-token DF across all loaded segments and
// the in-memory buffer. Used for IDF-weighted snippet scoring.
func (sm *SegmentManager) TermDFs(tokens []string) (n int, df map[string]int) {
	sm.mu.RLock()
	segs := make([]*segment.Segment, len(sm.segments))
	copy(segs, sm.segments)
	recs := make([]storage.SegmentRecord, len(sm.segRecords))
	copy(recs, sm.segRecords)
	bufIdx := sm.bufferIdx
	n = sm.bufferDocs
	sm.mu.RUnlock()

	df = make(map[string]int, len(tokens))
	for i, seg := range segs {
		if seg == nil {
			n += recs[i].DocCount
			continue
		}
		n += seg.DocCount()
		for _, t := range tokens {
			if d := seg.DF(t); d > 0 {
				df[t] += d
			}
		}
	}
	if bufIdx != nil {
		for _, t := range tokens {
			if d := bufIdx.DF(t); d > 0 {
				df[t] += d
			}
		}
	}
	return n, df
}

// searchAll fans out to segments + buffer in parallel, then RRF-merges.
//
// Because segments are immutable and a re-indexed docID is always written to
// the buffer first (only reaching a new segment on the next flush), the same
// docID can transiently exist in more than one queryable source: an older,
// superseded on-disk copy plus the live buffer (or, pre-merge, an older and a
// newer flushed segment). Left unhandled this produces duplicate entries for
// the same docID, and — since snippet lookup (GetDocText) always resolves to
// the freshest copy regardless of which source actually matched — a snippet
// that doesn't correspond to the matched content. bufferDocIDs and each
// segment's FlushSeq (segRecords) let this function keep only the most
// recent copy of each docID and drop stale ones outright.
func (sm *SegmentManager) searchAll(
	tokens []string,
	segs []*segment.Segment,
	segRecords []storage.SegmentRecord,
	bufIdx *index.InvertedIndex,
	bufferDocIDs map[string]struct{},
	tombstones map[string]struct{},
	tombBloom *bloom.BloomFilter,
	topK int,
	scorer scoring.Scorer,
) []types.ScoredDocument {
	const bufferRecency = math.MaxInt64 // buffer is always the most recent source

	type segSearchResult struct {
		docs    []types.ScoredDocument
		recency int64 // FlushSeq for a segment; bufferRecency for the buffer
	}

	// Pass 1: evaluate bloom + UB-sum for each segment, collect those that pass.
	type activeSource struct {
		seg     *segment.Segment // nil = buffer
		bufIdx  *index.InvertedIndex
		ubSum   float64
		recency int64
	}
	var active []activeSource

	for i, seg := range segs {
		if seg == nil {
			// Segment was flushed after loadNilSegments ran; it contains no docs
			// yet visible to search (first search will load it on the next call).
			continue
		}
		// Bloom pre-screen.
		if !seg.MayContainAny(tokens) {
			continue
		}
		// UB-sum exact check.
		var ubSum float64
		for _, tok := range tokens {
			ubSum += seg.TermUB(tok)
		}
		if len(tokens) > 0 && ubSum == 0 {
			continue
		}
		var recency int64
		if i < len(segRecords) {
			recency = segRecords[i].FlushSeq
		}
		active = append(active, activeSource{seg: seg, ubSum: ubSum, recency: recency})
	}

	// Buffer: always included if non-nil (no bloom/UB pre-check for in-memory index).
	if bufIdx != nil {
		active = append(active, activeSource{bufIdx: bufIdx, recency: bufferRecency})
	}

	if len(active) == 0 {
		return nil
	}

	// fetchK: tight bound based on actual active sources.
	fetchK := topK * len(active)
	if fetchK < topK {
		fetchK = topK
	}

	// Sort active sources by ubSum descending so the highest-potential segment
	// runs first as the seed to establish an initial θ for 3rd-tier pruning.
	// Buffer (ubSum=0) stays last.
	sort.Slice(active, func(i, j int) bool {
		return active[i].ubSum > active[j].ubSum
	})

	// Seed pass: run the first (highest-ubSum) source synchronously.
	// θ = score of the topK-th result (or 0 if fewer than topK results).
	var θ float64
	var allResults []segSearchResult

	seed := active[0]
	var seedDocs []types.ScoredDocument
	if seed.seg != nil {
		seedDocs = seed.seg.Search(tokens, fetchK, scorer)
	} else {
		searcher := index.NewMaxScoreSearcher(seed.bufIdx, scorer)
		seedDocs = searcher.Search(tokens, fetchK)
	}
	if len(seedDocs) >= topK {
		θ = seedDocs[topK-1].Score
	}
	if len(seedDocs) > 0 {
		allResults = append(allResults, segSearchResult{docs: seedDocs, recency: seed.recency})
	}

	// Remaining sources: skip if ubSum ≤ θ (cannot improve top-K).
	remaining := active[1:]
	ch := make(chan segSearchResult, len(remaining))
	var wg sync.WaitGroup
	pruned := 0
	for _, src := range remaining {
		if θ > 0 && src.ubSum <= θ {
			pruned++
			continue
		}
		src := src
		wg.Add(1)
		go func() {
			defer wg.Done()
			if src.seg != nil {
				ch <- segSearchResult{docs: src.seg.Search(tokens, fetchK, scorer), recency: src.recency}
			} else {
				searcher := index.NewMaxScoreSearcher(src.bufIdx, scorer)
				ch <- segSearchResult{docs: searcher.Search(tokens, fetchK), recency: src.recency}
			}
		}()
	}
	wg.Wait()
	close(ch)

	if pruned > 0 {
		slog.Debug("3rd-tier inter-segment pruning", "pruned", pruned, "θ", θ)
	}

	for r := range ch {
		if len(r.docs) > 0 {
			allResults = append(allResults, r)
		}
	}

	if len(allResults) == 0 {
		return nil
	}

	// Dedup by docID across sources, keeping only the most recent copy.
	// A segment-sourced hit is dropped outright (not merely out-scored) when
	// its docID currently lives in the buffer, even if the buffer's own
	// search didn't match this query — that copy has been superseded and its
	// stale content should not be findable at all.
	best := make(map[string]types.ScoredDocument, topK*len(allResults))
	bestRecency := make(map[string]int64, len(best))
	for _, r := range allResults {
		for _, d := range r.docs {
			if r.recency < bufferRecency {
				if _, superseded := bufferDocIDs[d.DocID]; superseded {
					continue
				}
			}
			if cur, ok := bestRecency[d.DocID]; !ok || r.recency > cur {
				best[d.DocID] = d
				bestRecency[d.DocID] = r.recency
			}
		}
	}
	deduped := make([]types.ScoredDocument, 0, len(best))
	for _, d := range best {
		deduped = append(deduped, d)
	}

	// Merge by score: with FNV-uniform routing, df ∝ N in every segment so
	// per-segment BM25 IDF ≈ global IDF — scores are directly comparable.
	// Plain score merge preserves ranking fidelity; no scaling needed.
	merged := mergeByScore([][]types.ScoredDocument{deduped}, topK)

	// Filter tombstones with bloom pre-screen.
	out := merged[:0]
	for _, r := range merged {
		if tombBloom != nil && !tombBloom.TestString(r.DocID) {
			out = append(out, r)
			continue
		}
		if _, deleted := tombstones[r.DocID]; !deleted {
			out = append(out, r)
		}
	}
	for i := range out {
		out[i].Rank = i + 1
	}
	return out
}

// rrfMergeWeighted performs weighted RRF: score(d) = Σ w_i / (k + rank_i(d)).
// weights[i] is the relative weight for list i (e.g. normalised doc count of
// that source). If weights is nil or shorter than lists, missing entries default
// to 1.0 (equal weight). k controls rank sensitivity: lower k amplifies the gap
// between rank-1 and rank-2; higher k flattens it.
func rrfMergeWeighted(lists [][]types.ScoredDocument, weights []float64, topK int, k float64) []types.ScoredDocument {
	totalCandidates := 0
	for _, l := range lists {
		totalCandidates += len(l)
	}
	scores := make(map[string]float64, totalCandidates)
	for i, list := range lists {
		w := 1.0
		if i < len(weights) && weights[i] > 0 {
			w = weights[i]
		}
		for _, doc := range list {
			scores[doc.DocID] += w / (k + float64(doc.Rank))
		}
	}

	type entry struct {
		docID string
		score float64
	}
	flat := make([]entry, 0, len(scores))
	for id, sc := range scores {
		flat = append(flat, entry{id, sc})
	}
	sort.Slice(flat, func(i, j int) bool {
		if flat[i].score != flat[j].score {
			return flat[i].score > flat[j].score
		}
		return flat[i].docID < flat[j].docID
	})

	n := len(flat)
	if n > topK {
		n = topK
	}
	out := make([]types.ScoredDocument, n)
	for i := 0; i < n; i++ {
		out[i] = types.ScoredDocument{DocID: flat[i].docID, Score: flat[i].score, Rank: i + 1}
	}
	return out
}

// mergeByScore combines result lists from multiple sources and returns the
// top-K documents by score. No scaling is applied: with FNV-uniform routing
// df ∝ N in every segment, so per-segment BM25 IDF ≈ global IDF and raw
// scores are directly comparable across segments and shards.
func mergeByScore(lists [][]types.ScoredDocument, topK int) []types.ScoredDocument {
	total := 0
	for _, l := range lists {
		total += len(l)
	}
	flat := make([]types.ScoredDocument, 0, total)
	for _, list := range lists {
		flat = append(flat, list...)
	}
	sort.Slice(flat, func(i, j int) bool {
		if flat[i].Score != flat[j].Score {
			return flat[i].Score > flat[j].Score
		}
		return flat[i].DocID < flat[j].DocID
	})
	n := len(flat)
	if n > topK {
		n = topK
	}
	out := make([]types.ScoredDocument, n)
	for i := 0; i < n; i++ {
		out[i] = flat[i]
		out[i].Rank = i + 1
	}
	return out
}

// needsSnapshotLocked reports whether the buffer search snapshot must be
// (re)built: it is missing while the buffer holds docs, or writes have landed
// since it was built and the refresh interval has passed. Caller holds sm.mu
// (read or write).
func (sm *SegmentManager) needsSnapshotLocked() bool {
	if sm.bufferDocs == 0 {
		return false
	}
	if sm.bufferIdx == nil || sm.bufferDocIDs == nil {
		return true
	}
	return sm.bufferDirty && sm.refreshInterval >= 0 && time.Since(sm.snapshotAt) >= sm.refreshInterval
}
