// Package shard runs the per-shard write and read paths on top of on-disk
// segments: SegmentManager owns one shard (buffer, WAL, flush, tiered merge,
// warmup, search, snippets); ShardManager fans writes and queries out across
// shards via the consistent-hash ring and merges the results.
package shard

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"

	"search-eval-platform/internal/analysis"
	"search-eval-platform/internal/cluster"
	"search-eval-platform/internal/query"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/segment"
	"search-eval-platform/internal/storage"
	"search-eval-platform/pkg/types"
)

// ShardManager is the multi-shard orchestrator for shard-node mode.
// It routes index writes via a virtual-node consistent hash ring and fans out
// search requests to all shards in parallel.
type ShardManager struct {
	shards       []*SegmentManager
	nShards      int
	ring         *cluster.ConsistentRing
	dataDir      string
	store        *storage.LocalStore
	scorer       scoring.Scorer // default scorer (BM25 by default)
	tokenizer    *analysis.Tokenizer
	synonyms     *analysis.SynonymMap // optional query-time synonym expansion
	cache        *QueryCache
	docCache     *lru.Cache[string, string] // LRU cache for hot document texts
	globalFusion string                     // "score" | "rrf"
}

// ShardManagerOptions holds optional configuration for NewShardManager.
type ShardManagerOptions struct {
	Tokenizer            *analysis.Tokenizer
	BloomFPRate          float64
	QueryCacheSize       int
	DocCacheSize         int                   // LRU doc text cache size (entries); 0 = use default 50000
	Store                *storage.LocalStore   // nil = LocalStore(dataDir)
	BM25FFields          []scoring.FieldConfig // non-empty → BM25F scoring mode
	MaxBufferDocs        int                   // 0 = use default
	MemThresholdMB       int                   // 0 = use default
	WALDurability        string                // "async" | "sync" | "off"
	WALSyncIntervalMs    int                   // 0 = use default
	WALCompression       string                // "" | "lz4" — compress WAL payloads
	IdleFlushSecs        int                   // flush buffer if idle for N seconds; 0 = disabled
	RefreshInterval      time.Duration         // see SegmentManagerOptions.RefreshInterval
	MaxMergeSizeMB       int64                 // cap total merge input size; 0 = disabled
	MergeRateLimitMBps   float64               // MB/s token-bucket limit on merge I/O; 0 = unlimited
	MergeIOBurstMB       int                   // token-bucket burst in MB; 0 = auto (rate×0.5s)
	MergeDeletionWeight  float64               // α in deletion_score=1+α*(deleted/total); 0 = disable
	FloorSegmentMB       int64                 // treat segments smaller than this as floor size; 0 = default (2 MB)
	ExpungeDeletesPct    float64               // opportunistic expunge threshold; 0 = use policy default
	GlobalFusion         string                // "score" | "rrf" — cross-shard merge strategy (default: "score")
	BufferPoolSize       int                   // background-flush swap buffer pool size per shard; 0 = default (4)
	SkipStoredFields     bool                  // omit .seg.fld sidecars; disables snippet generation
	PhraseBoost          float64               // per-matched-phrase score multiplier minus 1; 0 = DefaultPhraseBoost, <0 = off
	VirtualNodesPerShard int                   // virtual nodes per shard for consistent-hash ring; 0 = default (150)
	Synonyms             *analysis.SynonymMap  // optional query-time synonym expansion; nil = disabled
}

// NewShardManager creates and starts N SegmentManagers (one per shard).
// scorer is the default ranking function; pass scoring.NewBM25(k1,b) for BM25.
func NewShardManager(
	nShards int,
	dataDir string,
	scorer scoring.Scorer,
	policy *TieredMergePolicy,
	opts ...ShardManagerOptions,
) (*ShardManager, error) {
	if nShards <= 0 {
		nShards = 1
	}

	var smOpts ShardManagerOptions
	if len(opts) > 0 {
		smOpts = opts[0]
	}
	if smOpts.Tokenizer == nil {
		smOpts.Tokenizer = analysis.NewTokenizer(analysis.TokenizerConfig{})
	}

	store := smOpts.Store
	if store == nil {
		var storeErr error
		store, storeErr = storage.NewLocalStore(dataDir)
		if storeErr != nil {
			return nil, fmt.Errorf("NewShardManager: create local store: %w", storeErr)
		}
	}

	segOpts := SegmentManagerOptions{
		Tokenizer:           smOpts.Tokenizer,
		Synonyms:            smOpts.Synonyms,
		BloomFPRate:         smOpts.BloomFPRate,
		Store:               store,
		BM25FFields:         smOpts.BM25FFields,
		MaxBufferDocs:       smOpts.MaxBufferDocs,
		MemThresholdMB:      smOpts.MemThresholdMB,
		WALDurability:       smOpts.WALDurability,
		WALSyncIntervalMs:   smOpts.WALSyncIntervalMs,
		WALCompression:      smOpts.WALCompression,
		IdleFlushSecs:       smOpts.IdleFlushSecs,
		RefreshInterval:     smOpts.RefreshInterval,
		MaxMergeSizeMB:      smOpts.MaxMergeSizeMB,
		MergeRateLimitMBps:  smOpts.MergeRateLimitMBps,
		MergeIOBurstMB:      smOpts.MergeIOBurstMB,
		BufferPoolSize:      smOpts.BufferPoolSize,
		SkipStoredFields:    smOpts.SkipStoredFields,
		PhraseBoost:         smOpts.PhraseBoost,
		MergeDeletionWeight: smOpts.MergeDeletionWeight,
		FloorSegmentMB:      smOpts.FloorSegmentMB,
		ExpungeDeletesPct:   smOpts.ExpungeDeletesPct,
	}

	shards := make([]*SegmentManager, nShards)
	for i := 0; i < nShards; i++ {
		shardID := fmt.Sprintf("shard%d", i)
		sm, err := NewSegmentManager(shardID, dataDir, scorer, policy, segOpts)
		if err != nil {
			return nil, fmt.Errorf("NewShardManager shard %d: %w", i, err)
		}
		shards[i] = sm
	}

	docCacheSize := smOpts.DocCacheSize
	if docCacheSize <= 0 {
		docCacheSize = 50_000
	}
	docCache, _ := lru.New[string, string](docCacheSize)

	fusion := smOpts.GlobalFusion
	if fusion == "" {
		fusion = "rrf"
	}

	ring := cluster.BuildConsistentRing(nShards, smOpts.VirtualNodesPerShard)

	return &ShardManager{
		shards:       shards,
		nShards:      nShards,
		ring:         ring,
		dataDir:      dataDir,
		store:        store,
		scorer:       scorer,
		tokenizer:    smOpts.Tokenizer,
		synonyms:     smOpts.Synonyms,
		cache:        NewQueryCache(smOpts.QueryCacheSize),
		docCache:     docCache,
		globalFusion: fusion,
	}, nil
}

// Start launches background merge goroutines for all shards and restores state
// from the JSON manifest.
func (m *ShardManager) Start() error {
	t0 := time.Now()
	for _, sm := range m.shards {
		if err := sm.LoadFromMeta(); err != nil {
			return err
		}
		sm.Start()
	}
	slog.Info("shard manager ready", "shards", m.nShards, "segments", len(m.SegmentRecords()), "elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}

// Close gracefully shuts down all shards.
func (m *ShardManager) Close() error {
	var first error
	for _, sm := range m.shards {
		if err := sm.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// shardFor maps a docID to a shard index via the consistent hash ring.
func (m *ShardManager) shardFor(docID string) int {
	return m.ring.ShardFor(docID)
}

// IndexDoc routes the document to the correct shard.
func (m *ShardManager) IndexDoc(_ context.Context, doc types.Document) error {
	m.cache.Invalidate()
	return m.shards[m.shardFor(doc.ID)].IndexDocument(doc)
}

// IndexBatch indexes a slice of documents, grouping writes by shard.
// Per-shard errors are logged and the first is returned after all groups complete.
func (m *ShardManager) IndexBatch(_ context.Context, docs []types.Document) error {
	return m.indexGrouped(docs, true)
}

// IndexBatchNoWAL is IndexBatch without WAL entries (/index/bulk?wal=off).
func (m *ShardManager) IndexBatchNoWAL(_ context.Context, docs []types.Document) error {
	return m.indexGrouped(docs, false)
}

func (m *ShardManager) indexGrouped(docs []types.Document, withWAL bool) error {
	m.cache.Invalidate()

	groups := make(map[int][]types.Document, m.nShards)
	for _, doc := range docs {
		i := m.shardFor(doc.ID)
		groups[i] = append(groups[i], doc)
	}

	var mu sync.Mutex
	var firstErr error
	var wg sync.WaitGroup
	for shardIdx, shardDocs := range groups {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := m.shards[shardIdx].indexDocs(shardDocs, withWAL); err != nil {
				slog.Error("IndexBatch", "shard", shardIdx, "wal", withWAL, "err", err)
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	return firstErr
}

// DeleteDoc marks a document as deleted in the owning shard.
func (m *ShardManager) DeleteDoc(_ context.Context, docID string) error {
	m.cache.Invalidate()
	m.docCache.Remove(docID)
	return m.shards[m.shardFor(docID)].DeleteDocument(docID)
}

// Shard returns shard i's SegmentManager (0 <= i < NumShards).
func (m *ShardManager) Shard(i int) *SegmentManager { return m.shards[i] }

// IndexOnShard indexes doc into shard i directly. Replicas use it (and
// DeleteOnShard / ResetShard) to apply a primary's writes to the same shard
// the primary wrote them to, while still invalidating the caches.
func (m *ShardManager) IndexOnShard(i int, doc types.Document) error {
	m.cache.Invalidate()
	m.docCache.Remove(doc.ID)
	return m.shards[i].IndexDocument(doc)
}

// DeleteOnShard deletes docID from shard i directly. See IndexOnShard.
func (m *ShardManager) DeleteOnShard(i int, docID string) error {
	m.cache.Invalidate()
	m.docCache.Remove(docID)
	return m.shards[i].DeleteDocument(docID)
}

// ResetShard wipes shard i. See IndexOnShard.
func (m *ShardManager) ResetShard(ctx context.Context, i int) error {
	m.cache.Invalidate()
	m.docCache.Purge()
	return m.shards[i].Reset(ctx)
}

// Flush forces a flush on all shards in parallel.
func (m *ShardManager) Flush(_ context.Context) error {
	return m.eachShard(func(sm *SegmentManager) error { return sm.Flush() })
}

// scorerID returns a stable string identifier for a scorer, used as cache key.
func scorerID(scorer scoring.Scorer) string {
	return fmt.Sprintf("%T", scorer)
}

// Search fans out BM25/TF-IDF retrieval to all shards in parallel, cross-shard
// RRF-merges, then attaches snippets.
// scorer overrides the default if non-nil.
// noSnippet skips stored-field lookups entirely; use for benchmarking or when
// the caller only needs doc IDs and scores (e.g. eval tools). When noSnippet
// is true the query-result cache is also bypassed (eval queries are unique).
// The second return value is degraded: true when one or more shards failed but
// partial results from healthy shards are returned. Degraded results are not
// cached. Only returns an error when ALL shards fail.
func (m *ShardManager) Search(
	ctx context.Context,
	q string,
	topK int,
	scorer scoring.Scorer,
	noSnippet ...bool,
) ([]types.SearchResult, bool, error) {
	if scorer == nil {
		scorer = m.scorer
	}

	skipSnippet := len(noSnippet) > 0 && noSnippet[0]

	// Preprocess then parse the query so "don't" expands to "do not",
	// trailing ? is stripped, +/-/AND/NOT/OR operators are honored, and
	// quoted phrases are extracted before tokenization.
	// The normalized token join forms a stable cache key.
	pq := query.ParseQuery(query.PreprocessQuery(q), m.tokenizer, m.synonyms)
	tokens := pq.AllTokens()
	normalizedKey := strings.Join(tokens, " ")

	sid := scorerID(scorer)
	if !skipSnippet {
		if cached, ok := m.cache.Get(normalizedKey, topK, sid); ok {
			return cached, false, nil
		}
	}

	merged, degraded, err := m.fanOutSearch(q, topK, scorer)
	if err != nil {
		return nil, false, err
	}

	// Build IDF weights for snippet scoring (Tier 2).
	var termIDFs map[string]float64
	if !skipSnippet && len(tokens) > 0 {
		globalN := 0
		globalDF := make(map[string]int, len(tokens))
		for _, sm := range m.shards {
			n, df := sm.TermDFs(tokens)
			globalN += n
			for t, d := range df {
				globalDF[t] += d
			}
		}
		termIDFs = buildTermIDFs(globalN, globalDF)
	}

	results := m.attachSnippets(merged, tokens, termIDFs, skipSnippet)
	if !skipSnippet && !degraded {
		m.cache.Set(normalizedKey, topK, sid, results)
	}
	return results, degraded, nil
}

// fanOutSearch issues Search to all shards in parallel and merges results.
//   - "score" — single-phase local IDF, merge by raw score (default; same as ES query_then_fetch)
//   - "rrf"   — single-phase local IDF, Reciprocal Rank Fusion merge
//
// If some (but not all) shards fail, partial results are returned with
// degraded=true. Only returns an error when ALL shards fail.
func (m *ShardManager) fanOutSearch(query string, topK int, scorer scoring.Scorer) ([]types.ScoredDocument, bool, error) {
	type shardResult struct {
		docs     []types.ScoredDocument
		docCount int
		err      error
	}
	rrf := m.globalFusion == "rrf"

	fetchK := topK * m.nShards
	resCh := make(chan shardResult, m.nShards)
	for _, sm := range m.shards {
		go func() {
			docs, err := sm.Search(query, fetchK, scorer)
			r := shardResult{docs: docs, err: err}
			if rrf {
				r.docCount = sm.DocCount()
			}
			resCh <- r
		}()
	}

	var allLists [][]types.ScoredDocument
	var docCounts []int
	var failedShards int
	for i := 0; i < m.nShards; i++ {
		r := <-resCh
		if r.err != nil {
			slog.Warn("shard search failed", "fusion", m.globalFusion, "err", r.err)
			failedShards++
			continue
		}
		if len(r.docs) > 0 {
			allLists = append(allLists, r.docs)
			docCounts = append(docCounts, r.docCount)
		}
	}

	if failedShards == m.nShards {
		return nil, false, fmt.Errorf("all %d shards failed", m.nShards)
	}
	degraded := failedShards > 0
	if len(allLists) == 0 {
		return nil, degraded, nil
	}
	if !rrf {
		// "score": each shard scores with its local IDF; results merge by raw
		// score. The consistent-hash ring keeps per-shard df close to uniform,
		// so local IDF ≈ global IDF (Elasticsearch's query_then_fetch default).
		return mergeByScore(allLists, topK), degraded, nil
	}

	// "rrf": weighted Reciprocal Rank Fusion, each shard weighted by its share
	// of the documents.
	totalDocs := 0
	for _, c := range docCounts {
		totalDocs += c
	}
	weights := make([]float64, len(docCounts))
	for i, c := range docCounts {
		if totalDocs > 0 {
			weights[i] = float64(c) / float64(totalDocs)
		} else {
			weights[i] = 1.0
		}
	}
	return rrfMergeWeighted(allLists, weights, topK, 60.0), degraded, nil
}

// GetDocText returns the original text for docID.
// Checks the in-memory LRU cache first, then scans segment stored fields.
func (m *ShardManager) GetDocText(docID string) (string, bool) {
	if text, ok := m.docCache.Get(docID); ok {
		return text, true
	}
	// Cache miss: check shard's in-memory buffer texts, then segments.
	shardIdx := m.shardFor(docID)
	sm := m.shards[shardIdx]
	sm.mu.RLock()
	segs := make([]*segment.Segment, len(sm.segments))
	copy(segs, sm.segments)
	bufText, hasBufText := sm.bufferTexts[docID]
	sm.mu.RUnlock()

	if hasBufText {
		m.docCache.Add(docID, bufText)
		return bufText, true
	}
	// Search segments in reverse order (newest first).
	for i := len(segs) - 1; i >= 0; i-- {
		if segs[i] == nil {
			continue
		}
		if text, ok := segs[i].GetText(docID); ok {
			m.docCache.Add(docID, text)
			return text, true
		}
	}
	return "", false
}

// attachSnippets fetches text from the segment stored-fields cache and builds
// snippet+metadata for each scored document.
// When skipSnippet is true, stored-field lookups and snippet extraction are
// skipped entirely, eliminating all per-result disk I/O.
// termIDFs, when non-nil, enables IDF-weighted window scoring (Tier 2).
func (m *ShardManager) attachSnippets(docs []types.ScoredDocument, tokens []string, termIDFs map[string]float64, skipSnippet bool) []types.SearchResult {
	results := make([]types.SearchResult, 0, len(docs))
	for _, scored := range docs {
		snippet := ""
		if !skipSnippet {
			if text, ok := m.GetDocText(scored.DocID); ok && text != "" {
				snippet = ExtractSnippetWithIDF(text, tokens, termIDFs)
			}
		}
		results = append(results, types.SearchResult{
			DocID:   scored.DocID,
			Score:   scored.Score,
			Rank:    scored.Rank,
			Snippet: snippet,
		})
	}
	return results
}

// SegmentRecords returns all active segment records across all shards,
// sorted by shard then flush sequence.
func (m *ShardManager) SegmentRecords() []storage.SegmentRecord {
	var all []storage.SegmentRecord
	for _, sm := range m.shards {
		all = append(all, sm.SegmentRecords()...)
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].ShardID != all[j].ShardID {
			return all[i].ShardID < all[j].ShardID
		}
		return all[i].FlushSeq < all[j].FlushSeq
	})
	return all
}

// Reset wipes all data across every shard (segments, WAL, segment manifest)
// and clears the query cache. The ShardManager stays running; new
// documents can be indexed immediately after Reset returns.
func (m *ShardManager) Reset(ctx context.Context) error {
	m.cache.Invalidate()
	m.docCache.Purge()

	return m.eachShard(func(sm *SegmentManager) error { return sm.Reset(ctx) })
}

// TriggerMerge signals all shard merge goroutines to check for merge candidates.
func (m *ShardManager) TriggerMerge() {
	for _, sm := range m.shards {
		select {
		case sm.mergeCh <- struct{}{}:
		default:
		}
	}
}

// ForceMerge merges all shards concurrently until each has at most maxSegments
// segments. Unlike the background tiered merge, it bypasses the merge policy
// and directly batches candidates. Blocks until complete.
func (m *ShardManager) ForceMerge(ctx context.Context, maxSegments int) error {
	slog.Info("force merge started", "max_segments_per_shard", maxSegments)
	t0 := time.Now()
	if err := m.eachShard(func(sm *SegmentManager) error { return sm.forceMerge(ctx, maxSegments) }); err != nil {
		return err
	}
	slog.Info("force merge done", "elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}

// WarmupSegments eagerly loads all segments across all shards concurrently so
// the first queries are served from RAM without cold-load latency spikes.
func (m *ShardManager) WarmupSegments(ctx context.Context) error {
	slog.Info("warming up all shards")
	t0 := time.Now()
	if err := m.eachShard(func(sm *SegmentManager) error { return sm.WarmupSegments(ctx) }); err != nil {
		return err
	}
	slog.Info("all shards warm", "elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}

// WarmTopTerms warms the top-k highest-DF posting lists across all shards
// concurrently. If mlock is true, attempts to pin warmed pages in RAM.
// A single shared semaphore (capacity warmMaxParallelSegments) is injected
// into the context so the total number of concurrently-touching goroutines
// is bounded globally across all shards.
func (m *ShardManager) WarmTopTerms(ctx context.Context, k int, mlock bool) error {
	slog.Info("warming top terms across all shards", "top_k", k)
	t0 := time.Now()
	// Shared semaphore: all shards × all segments compete for the same N slots,
	// keeping total concurrent sequential I/O goroutines bounded.
	sem := make(chan struct{}, warmMaxParallelSegments)
	ctx = context.WithValue(ctx, warmConcKey{}, sem)
	if err := m.eachShard(func(sm *SegmentManager) error { return sm.WarmTopTerms(ctx, k, mlock) }); err != nil {
		return err
	}
	slog.Info("all shards top terms warm", "elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}

// WarmTiered applies tiered warmup across all shards concurrently.
// Segments <= thresholdBytes get a full sequential page scan; larger segments
// get targeted top-K DF warmup. If mlock is true, pages are pinned in RAM.
// A single shared semaphore (capacity warmMaxParallelSegments) is injected
// into the context so the total number of concurrently-touching goroutines
// is bounded globally across all shards, preserving sequential I/O throughput.
func (m *ShardManager) WarmTiered(ctx context.Context, thresholdBytes int64, topK int, mlock bool) error {
	threshMB := thresholdBytes / (1024 * 1024)
	slog.Info("warming tiered across all shards", "threshold_mb", threshMB, "top_k", topK)
	t0 := time.Now()
	// Shared semaphore: all shards × all segments compete for the same N slots.
	sem := make(chan struct{}, warmMaxParallelSegments)
	ctx = context.WithValue(ctx, warmConcKey{}, sem)
	if err := m.eachShard(func(sm *SegmentManager) error { return sm.WarmTiered(ctx, thresholdBytes, topK, mlock) }); err != nil {
		return err
	}
	slog.Info("all shards tiered warm", "elapsed", time.Since(t0).Round(time.Millisecond))
	return nil
}

// eachShard runs fn on every shard concurrently and returns the first error in
// shard order (nil if all succeed).
func (m *ShardManager) eachShard(fn func(sm *SegmentManager) error) error {
	errs := make([]error, m.nShards)
	var wg sync.WaitGroup
	for i, sm := range m.shards {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[i] = fn(sm)
		}()
	}
	wg.Wait()
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}

// NumShards returns the configured shard count.
func (m *ShardManager) NumShards() int { return m.nShards }

// ShardStatuses returns per-shard operational metrics for all shards.
func (m *ShardManager) ShardStatuses() []ShardStatus {
	out := make([]ShardStatus, m.nShards)
	for i, sm := range m.shards {
		out[i] = sm.Status()
	}
	return out
}

// PerShardDocCount returns the live document count for each shard by index (0..N-1).
func (m *ShardManager) PerShardDocCount() []int {
	out := make([]int, m.nShards)
	for i, sm := range m.shards {
		out[i] = sm.DocCount()
	}
	return out
}

// GC recycles the builder temp files across all shards in parallel.
// See SegmentManager.GC for details.
func (m *ShardManager) GC() error {
	return m.eachShard(func(sm *SegmentManager) error { return sm.GC() })
}
