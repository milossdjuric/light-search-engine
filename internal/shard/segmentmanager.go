package shard

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	bloom "github.com/bits-and-blooms/bloom/v3"
	"golang.org/x/time/rate"

	"search-eval-platform/internal/analysis"
	"search-eval-platform/internal/index"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/segment"
	"search-eval-platform/internal/storage"
	"search-eval-platform/pkg/types"
)

const (
	opIndex  = "index"
	opDelete = "delete"
	// opReset is never written to a WAL file (Reset wipes the WAL); it is
	// only delivered to the WAL hook so replicas can wipe their copy too.
	opReset = "reset"
)

const (
	defaultMemThresholdMB = 256
	defaultMaxBufferDocs  = 50_000
)

// SegmentManager is the per-shard lifecycle manager.
// It owns the WAL, the in-memory buffer, the on-disk segment files, and the
// background merge goroutine for one shard.
type SegmentManager struct {
	shardID           string
	meta              *storage.ShardMeta
	dataDir           string
	store             *storage.LocalStore // pluggable segment / bloom storage
	tmpSegDir         string              // temp dir for writing before PutFile
	mergePolicy       *TieredMergePolicy
	memThreshold      int64 // bytes
	maxBufferDocs     int
	scorer            scoring.Scorer // default scorer for search (BM25 recommended)
	tokenizer         *analysis.Tokenizer
	synonyms          *analysis.SynonymMap  // optional query-time synonym expansion
	bloomFPRate       float64               // 0 = disabled
	bm25fFields       []scoring.FieldConfig // non-empty → BM25F scoring mode
	maxMergeSizeBytes int64                 // 0 = unlimited; caps total input bytes per merge
	mergeLimiter      *rate.Limiter         // nil = unlimited; token-bucket for merge I/O bytes
	skipStoredFields  bool                  // when true, omit .seg.fld sidecars; disables snippets
	phraseBoost       float64               // resolved phrase boost (0 = phrases not boosted)

	mu             sync.RWMutex
	buffer         *index.IndexBuilder
	bufferDocs     int
	bufferDirty    bool                 // buffer changed since bufferIdx was built
	bufferIdx      *index.InvertedIndex // cached read snapshot; see needsSnapshotLocked
	snapshotAt     time.Time            // when bufferIdx was last built
	bufferDocIDs   map[string]struct{}  // cached buffer membership set; rebuilt alongside bufferIdx
	bufferTexts    map[string]string    // docID → original text for in-flight buffer docs
	nextSeq        int64
	segments       []*segment.Segment // loaded segment objects (parallel to segRecords)
	segRecords     []storage.SegmentRecord
	tombstones     map[string]struct{} // deleted docIDs
	tombstoneBloom *bloom.BloomFilter  // pre-screen: definitely-not-deleted fast path

	mergeMu sync.Mutex  // serializes concurrent executeMerge calls (tryMerge vs forceMerge race)
	merging atomic.Bool // true while executeMerge is running
	warming atomic.Bool // true while WarmTopTerms or WarmTiered is running

	walMu             sync.Mutex
	walFile           *os.File
	walBuf            *bufio.Writer // buffers walFile writes to reduce syscalls
	walDir            string        // directory that holds all WAL files for this shard
	walSeqNum         int           // monotonic file counter; current file = shard_NNNN.wal
	walDurability     string
	walSyncIntervalMs int
	walScratch        []byte         // reusable scratch buffer for binary WAL encoding
	walHook           func(WALEntry) // called under walMu, in seq order, for every WAL write (see SetWALHook)
	walCompress       bool           // true → LZ4-compress each WAL payload before writing
	walCompressBuf    []byte         // reusable LZ4 output buffer

	idleFlushSecs int
	// refreshInterval bounds how often search rebuilds the buffer snapshot
	// after writes: 0 = on the next search, >0 = at most once per interval,
	// <0 = only after a flush.
	refreshInterval time.Duration
	lastWriteTime   atomic.Int64 // unix nanoseconds of last IndexDocument* call

	mergeCh chan struct{}
	stopCh  chan struct{}
	wg      sync.WaitGroup

	// Background-flush pipeline.
	// freeBuffers is a pool of N ready-to-use zeroed IndexBuilders (default N=6,
	// configurable via SegmentManagerOptions.BufferPoolSize).
	// Up to N background flushes can run concurrently; flushWg tracks them all.
	// WAL rotation is intentionally skipped during background flushes to keep
	// manifest flush_seq ordering strictly monotonic.
	freeBuffers chan *index.IndexBuilder
	flushWg     sync.WaitGroup
}

// SegmentManagerOptions holds optional configuration for NewSegmentManager.
type SegmentManagerOptions struct {
	Tokenizer           *analysis.Tokenizer
	Synonyms            *analysis.SynonymMap  // optional query-time synonym expansion
	BloomFPRate         float64               // 0 = disabled
	Store               *storage.LocalStore   // nil = LocalStore(dataDir)
	BM25FFields         []scoring.FieldConfig // non-empty → BM25F scoring mode
	BufferPoolSize      int                   // number of pre-allocated swap buffers (default 4)
	MaxBufferDocs       int                   // 0 = use default (200000)
	MemThresholdMB      int                   // 0 = use default (256)
	SkipStoredFields    bool                  // omit .seg.fld sidecars; disables snippet generation
	PhraseBoost         float64               // per-matched-phrase score multiplier minus 1; 0 = DefaultPhraseBoost, <0 = off
	WALDurability       string                // "async" | "sync" | "off"  default "async"
	WALSyncIntervalMs   int                   // 0 = use default (1000ms)
	WALCompression      string                // "" | "lz4"  — compress WAL payloads with LZ4
	IdleFlushSecs       int                   // flush buffer if no writes for N seconds; 0 = disabled
	RefreshInterval     time.Duration         // min gap between buffer-snapshot rebuilds after writes; 0 = next search, <0 = only after flush
	MaxMergeSizeMB      int64                 // cap total merge input size in MB; 0 = disabled
	MergeRateLimitMBps  float64               // MB/s token-bucket limit on merge reads; 0 = unlimited
	MergeIOBurstMB      int                   // token-bucket burst in MB; 0 = auto (rate×0.5s)
	MergeDeletionWeight float64               // α for deletion-aware candidate scoring; 0 = use policy default
	FloorSegmentMB      int64                 // floor segment size for tiering math; 0 = use policy default
	ExpungeDeletesPct   float64               // opportunistic expunge threshold; 0 = use policy default
}

// NewSegmentManager creates a SegmentManager. Call LoadFromMeta to restore state
// from a previous run, then Start to begin background merging.
func NewSegmentManager(
	shardID, dataDir string,
	scorer scoring.Scorer,
	policy *TieredMergePolicy,
	opts ...SegmentManagerOptions,
) (*SegmentManager, error) {
	if policy == nil {
		policy = DefaultTieredMergePolicy()
	}

	var smOpts SegmentManagerOptions
	if len(opts) > 0 {
		smOpts = opts[0]
	}
	if smOpts.Tokenizer == nil {
		smOpts.Tokenizer = analysis.NewTokenizer(analysis.TokenizerConfig{})
	}

	walDir := filepath.Join(dataDir, "wal")
	if err := os.MkdirAll(walDir, 0o755); err != nil {
		return nil, fmt.Errorf("SegmentManager: mkdir wal dir: %w", err)
	}
	meta, err := storage.LoadShardMeta(walDir, shardID)
	if err != nil {
		return nil, fmt.Errorf("SegmentManager: load manifest: %w", err)
	}
	// Keep the segments dir for backward-compat with LocalStore (absolute-path legacy records).
	if err := os.MkdirAll(filepath.Join(dataDir, "segments"), 0o755); err != nil {
		return nil, fmt.Errorf("SegmentManager: mkdir segments dir: %w", err)
	}

	tmpSegDir := filepath.Join(dataDir, "tmp", "segments")
	if err := os.MkdirAll(tmpSegDir, 0o755); err != nil {
		return nil, fmt.Errorf("SegmentManager: mkdir tmp segments dir: %w", err)
	}

	store := smOpts.Store
	if store == nil {
		var storeErr error
		store, storeErr = storage.NewLocalStore(dataDir)
		if storeErr != nil {
			return nil, fmt.Errorf("SegmentManager: create local store: %w", storeErr)
		}
	}

	walSeqNum, err := findHighestWALSeqNum(walDir, shardID)
	if err != nil {
		return nil, fmt.Errorf("SegmentManager: scan wal dir: %w", err)
	}
	if walSeqNum == 0 {
		walSeqNum = 1
	}
	walPath := walFileName(walDir, shardID, walSeqNum)
	f, err := os.OpenFile(walPath, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return nil, fmt.Errorf("SegmentManager: open wal: %w", err)
	}

	poolSize := smOpts.BufferPoolSize
	if poolSize <= 0 {
		poolSize = 4
	}
	freeBuffers := make(chan *index.IndexBuilder, poolSize)
	for i := 0; i < poolSize; i++ {
		freeBuffers <- index.NewIndexBuilder() // pre-populated, ready for immediate swap
	}

	sm := &SegmentManager{
		shardID:          shardID,
		meta:             meta,
		dataDir:          dataDir,
		store:            store,
		tmpSegDir:        tmpSegDir,
		mergePolicy:      policy,
		memThreshold:     defaultMemThresholdMB * 1024 * 1024,
		maxBufferDocs:    defaultMaxBufferDocs,
		scorer:           scorer,
		tokenizer:        smOpts.Tokenizer,
		synonyms:         smOpts.Synonyms,
		bloomFPRate:      smOpts.BloomFPRate,
		skipStoredFields: smOpts.SkipStoredFields,
		phraseBoost:      resolvePhraseBoost(smOpts.PhraseBoost),
		bm25fFields:      smOpts.BM25FFields,
		buffer:           index.NewIndexBuilder(),
		bufferTexts:      make(map[string]string, defaultMaxBufferDocs),
		tombstones:       make(map[string]struct{}),
		tombstoneBloom:   bloom.NewWithEstimates(1000, 0.001),
		nextSeq:          1,
		walFile:          f,
		walBuf:           bufio.NewWriterSize(f, 256*1024),
		walDir:           walDir,
		walSeqNum:        walSeqNum,
		mergeCh:          make(chan struct{}, 1),
		stopCh:           make(chan struct{}),
		freeBuffers:      freeBuffers,
	}

	if smOpts.MaxBufferDocs > 0 {
		sm.maxBufferDocs = smOpts.MaxBufferDocs
	}
	if smOpts.MemThresholdMB > 0 {
		sm.memThreshold = int64(smOpts.MemThresholdMB) * 1024 * 1024
	}
	sm.walDurability = smOpts.WALDurability
	if sm.walDurability == "" {
		sm.walDurability = "async"
	}
	sm.walSyncIntervalMs = smOpts.WALSyncIntervalMs
	if sm.walSyncIntervalMs == 0 {
		sm.walSyncIntervalMs = 1000
	}
	sm.idleFlushSecs = smOpts.IdleFlushSecs
	sm.refreshInterval = smOpts.RefreshInterval
	sm.walScratch = make([]byte, 0, 4096)
	if smOpts.WALCompression == "lz4" {
		sm.walCompress = true
		sm.walCompressBuf = make([]byte, 0, 8192)
	}

	if smOpts.MaxMergeSizeMB > 0 {
		sm.maxMergeSizeBytes = smOpts.MaxMergeSizeMB * 1024 * 1024
		sm.mergePolicy.MaxMergeSizeBytes = sm.maxMergeSizeBytes
	}
	if smOpts.MergeDeletionWeight > 0 {
		sm.mergePolicy.DeletionWeight = smOpts.MergeDeletionWeight
	}
	if smOpts.FloorSegmentMB > 0 {
		sm.mergePolicy.FloorSegmentMB = smOpts.FloorSegmentMB
	}
	if smOpts.ExpungeDeletesPct > 0 {
		sm.mergePolicy.ExpungeDeletesThreshold = smOpts.ExpungeDeletesPct
	}
	if smOpts.MergeRateLimitMBps > 0 {
		limitBps := rate.Limit(smOpts.MergeRateLimitMBps * 1024 * 1024)
		burstMB := smOpts.MergeIOBurstMB
		if burstMB <= 0 {
			burstMB = int(smOpts.MergeRateLimitMBps * 0.5)
			if burstMB < 16 {
				burstMB = 16 // floor: at least one meaningful token-bucket tick
			}
		}
		sm.mergeLimiter = rate.NewLimiter(limitBps, burstMB*1024*1024)
	}

	// Write binary WAL file header if this is a new empty file.
	walInfo, _ := f.Stat()
	if walInfo.Size() == 0 {
		_, _ = sm.walBuf.WriteString(walBinaryMagic)
		if sm.walCompress {
			_ = sm.walBuf.WriteByte(walBinaryVersionLZ4)
		} else {
			_ = sm.walBuf.WriteByte(walBinaryVersion)
		}
		_ = sm.walBuf.Flush()
	}

	return sm, nil
}

// Start launches the background merge goroutine.
func (sm *SegmentManager) Start() {
	sm.wg.Add(1)
	go sm.runMerge()
	if sm.walDurability == "async" && sm.walSyncIntervalMs > 0 {
		sm.wg.Add(1)
		go sm.runWALFlusher()
	}
	if sm.idleFlushSecs > 0 {
		sm.wg.Add(1)
		go sm.runIdleFlusher()
	}
}

// Close flushes the buffer, stops the merge goroutine, and closes the WAL.
func (sm *SegmentManager) Close() error {
	close(sm.stopCh)
	sm.wg.Wait()

	// Wait for any in-flight background flush to complete before the final
	// synchronous flush.  The background goroutine needs mu to update
	// sm.segments, so we must not hold mu while waiting.
	sm.flushWg.Wait()

	// Drain the free-buffer pool. Call Close() on each to release temp files.
	for len(sm.freeBuffers) > 0 {
		b := <-sm.freeBuffers
		b.Close()
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.bufferDocs > 0 {
		slog.Info("flushing shard buffer", "shard", sm.shardID, "docs", sm.bufferDocs)
		if err := sm.flushLocked(); err != nil {
			return err
		}
		slog.Info("shard buffer flushed", "shard", sm.shardID)
	} else {
		slog.Info("shard buffer already empty, skipping flush", "shard", sm.shardID)
	}
	sm.walMu.Lock()
	sm.walBuf.Flush() //nolint:errcheck — file close follows
	sm.walMu.Unlock()
	return sm.walFile.Close()
}

// Reset wipes all state for this shard: in-memory buffer, WAL, segment files,
// and manifest rows. The manager stays running — documents can be indexed again
// immediately after Reset returns.
func (sm *SegmentManager) Reset(ctx context.Context) error {
	// Wait for any in-flight background flushes before acquiring locks.
	// Background flushes write segment data without holding mu, then acquire mu
	// briefly to register the result. If we delete segment files while a flush
	// is mid-write, the flush will complete and re-register a now-deleted file
	// into sm.segments, leaving the server in a corrupt state.
	sm.flushWg.Wait()

	// Acquire in the same order as flushLocked: mu first, then walMu.
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.walMu.Lock()
	defer sm.walMu.Unlock()

	keys, err := sm.store.List(ctx, "segments/"+sm.shardID+"_")
	if err != nil {
		return fmt.Errorf("Reset list segments: %w", err)
	}
	for _, key := range keys {
		if err := sm.store.Delete(ctx, key); err != nil {
			return fmt.Errorf("Reset delete %s: %w", key, err)
		}
	}

	if sm.walFile != nil {
		sm.walBuf.Flush() //nolint:errcheck
		sm.walFile.Close()
		sm.walFile = nil
	}
	existing, _ := listWALFiles(sm.walDir, sm.shardID)
	for _, p := range existing {
		os.Remove(p)
	}
	sm.walSeqNum = 1
	f, err := os.OpenFile(walFileName(sm.walDir, sm.shardID, 1), os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("Reset open WAL: %w", err)
	}
	sm.walFile = f
	sm.walBuf = bufio.NewWriterSize(f, 256*1024)
	_, _ = sm.walBuf.WriteString(walBinaryMagic)
	if sm.walCompress {
		_ = sm.walBuf.WriteByte(walBinaryVersionLZ4)
	} else {
		_ = sm.walBuf.WriteByte(walBinaryVersion)
	}
	_ = sm.walBuf.Flush()

	// Wipe manifest catalog rows for this shard. Sequence numbers keep
	// counting up (never rewind to 1): the reset itself takes the next seq,
	// which is persisted as the manifest's SeqFloor so a restart resumes
	// above it too. Replicas get the reset through the WAL hook.
	resetSeq := sm.nextSeq
	sm.nextSeq++
	if err := sm.meta.Reset(); err != nil {
		return fmt.Errorf("Reset manifest: %w", err)
	}
	if err := sm.meta.SetSeqFloor(resetSeq); err != nil {
		return fmt.Errorf("Reset manifest seq floor: %w", err)
	}
	if sm.walHook != nil {
		sm.walHook(WALEntry{Seq: resetSeq, Op: opReset, TS: time.Now().UTC()})
	}

	// Reset in-memory state. Reuse the buffer's temp file via Reset().
	sm.buffer.Reset()
	sm.bufferDocs = 0
	sm.bufferDirty = false
	sm.bufferIdx = nil
	sm.bufferDocIDs = nil
	sm.bufferTexts = make(map[string]string, sm.maxBufferDocs)
	sm.segments = nil
	sm.segRecords = nil
	sm.tombstones = make(map[string]struct{})
	sm.tombstoneBloom = bloom.NewWithEstimates(1000, 0.001)
	return nil
}

// GC reclaims disk space used by the builder temp files in the free-buffer pool.
// It drains all idle pool buffers, closes each one (which removes its temp file),
// then refills the pool with fresh builders. Call after a large ingest to free
// the OS blocks that accumulate when temp files grow during flush cycles.
// GC must not be called concurrently with indexing operations.
func (sm *SegmentManager) GC() error {
	// Wait for any in-flight background flushes so all buffers are in the pool.
	sm.flushWg.Wait()

	// Drain the pool, collect how many we removed.
	var drained []*index.IndexBuilder
	for {
		select {
		case b := <-sm.freeBuffers:
			drained = append(drained, b)
		default:
			goto drain_done
		}
	}
drain_done:
	// Close each buffer — removes its temp file from disk.
	for _, b := range drained {
		b.Close()
	}

	// Refill the pool with fresh builders (new temp files, zero size).
	for range drained {
		sm.freeBuffers <- index.NewIndexBuilder()
	}

	slog.Info("GC complete", "shard", sm.shardID, "pool_recycled", len(drained))
	return nil
}

// LoadFromMeta restores active segments and tombstones from the JSON manifest,
// then replays any WAL entries with seq > maxCoveredFlushSeq across all WAL files.
func (sm *SegmentManager) LoadFromMeta() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Register active segment records without loading files into memory.
	// Segments are loaded lazily on first search via loadNilSegments, so
	// startup is instant regardless of index size.
	recs := sm.meta.LoadActiveSegments()
	if len(recs) > 0 {
		slog.Info("restoring segment records (lazy load)", "shard", sm.shardID, "count", len(recs))
	}
	for _, rec := range recs {
		sm.segments = append(sm.segments, nil) // loaded on first search
		sm.segRecords = append(sm.segRecords, rec)
	}

	// Load tombstones from manifest.
	stones := sm.meta.LoadTombstones()
	for _, s := range stones {
		sm.tombstones[s.DocID] = struct{}{}
	}

	// Everything at or below the high-water mark is already in a segment
	// (MaxFlushSeq) or was discarded by a Reset (SeqFloor).
	highWater := sm.meta.SeqHighWater()

	// Replay all WAL files, GC those fully covered by segments.
	// WAL replay may add more tombstones.
	if err := sm.replayWALLocked(highWater); err != nil {
		return fmt.Errorf("LoadFromMeta replayWAL: %w", err)
	}

	// Rebuild tombstone bloom from the complete tombstone set (manifest + WAL replay).
	if len(sm.tombstones) > 0 {
		n := len(sm.tombstones)
		if n < 1000 {
			n = 1000
		}
		sm.tombstoneBloom = bloom.NewWithEstimates(uint(n), 0.001)
		for docID := range sm.tombstones {
			sm.tombstoneBloom.AddString(docID)
		}
	}

	return nil
}

// addDocToBufferLocked adds doc to the in-memory buffer. Called with sm.mu held.
func (sm *SegmentManager) addDocToBufferLocked(doc types.Document, tokens []string, fieldTokens map[string][]string) {
	if _, wasTombstoned := sm.tombstones[doc.ID]; wasTombstoned {
		delete(sm.tombstones, doc.ID)
		_ = sm.meta.RemoveTombstone(sm.shardID, doc.ID)
	}
	if fieldTokens != nil {
		sm.buffer.AddFields(doc.ID, fieldTokens)
	} else {
		sm.buffer.Add(doc.ID, tokens)
	}
	if !sm.skipStoredFields {
		sm.bufferTexts[doc.ID] = doc.Text
	}
	sm.bufferDocs++
}

// IndexDocument adds a document to the buffer, writing a WAL entry first.
// Triggers a flush if memory or count thresholds are exceeded.
func (sm *SegmentManager) IndexDocument(doc types.Document) error {
	return sm.indexDocs([]types.Document{doc}, true)
}

// indexDocs tokenizes docs (outside any lock), optionally logs them to the
// WAL under one walMu hold, then adds them to the buffer under one mu hold and
// triggers a flush when a threshold is crossed — two lock acquisitions per
// batch instead of two per document. withWAL=false is the /index/bulk?wal=off
// path for bulk ingest that can simply be re-run on failure.
func (sm *SegmentManager) indexDocs(docs []types.Document, withWAL bool) error {
	if len(docs) == 0 {
		return nil
	}

	type tokenizedDoc struct {
		tokens      []string
		fieldTokens map[string][]string
	}
	tokenized := make([]tokenizedDoc, len(docs))
	for i, doc := range docs {
		if len(sm.bm25fFields) > 0 {
			tokenized[i] = tokenizedDoc{fieldTokens: sm.resolveFieldTokens(doc)}
		} else {
			tokenized[i] = tokenizedDoc{tokens: sm.tokenizer.Tokenize(doc.Text)}
		}
	}

	if withWAL {
		sm.walMu.Lock()
		now := time.Now().UTC()
		for _, doc := range docs {
			entry := WALEntry{
				Op:       opIndex,
				DocID:    doc.ID,
				Text:     doc.Text,
				Fields:   doc.Fields,
				Metadata: doc.Metadata,
				TS:       now,
			}
			if err := sm.writeWALEntryLocked(&entry); err != nil {
				sm.walMu.Unlock()
				return err
			}
		}
		err := sm.syncWALLocked()
		sm.walMu.Unlock()
		if err != nil {
			return err
		}
	}

	sm.mu.Lock()
	for i, doc := range docs {
		sm.addDocToBufferLocked(doc, tokenized[i].tokens, tokenized[i].fieldTokens)
	}
	sm.bufferDirty = true
	sm.lastWriteTime.Store(time.Now().UnixNano())

	if sm.buffer.MemoryEstimate() >= sm.memThreshold || sm.bufferDocs >= sm.maxBufferDocs {
		return sm.flushOrSwap() // always unlocks mu
	}
	sm.mu.Unlock()
	return nil
}

// DeleteDocument records a tombstone for docID, writing a WAL entry first.
func (sm *SegmentManager) DeleteDocument(docID string) error {
	entry := WALEntry{
		Op:    opDelete,
		DocID: docID,
		TS:    time.Now().UTC(),
	}
	if err := sm.appendWAL(&entry); err != nil {
		return err
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.tombstones[docID] = struct{}{}
	sm.tombstoneBloom.AddString(docID)
	// Update in-memory DeletedDocs count for the segment that owns this docID.
	sm.incrementDeletedDocsLocked(docID)
	if err := sm.meta.AddTombstone(sm.shardID, docID, entry.TS); err != nil {
		return err
	}
	return nil
}

// incrementDeletedDocsLocked scans loaded segments to find the one containing
// docID and increments its DeletedDocs counter. Called with sm.mu held (write).
func (sm *SegmentManager) incrementDeletedDocsLocked(docID string) {
	for i, seg := range sm.segments {
		if seg == nil {
			continue
		}
		if seg.HasDoc(docID) {
			sm.segRecords[i].DeletedDocs++
			return
		}
	}
}

// Flush forces the current buffer to be written to a segment file.
func (sm *SegmentManager) Flush() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sm.bufferDocs == 0 {
		return nil
	}
	return sm.flushLocked()
}

// ShardID returns this shard's identifier (e.g. "shard0").
func (sm *SegmentManager) ShardID() string { return sm.shardID }

// LastSeq returns the latest seq this shard's WAL has assigned (0 if none).
func (sm *SegmentManager) LastSeq() int64 {
	sm.walMu.Lock()
	defer sm.walMu.Unlock()
	return sm.nextSeq - 1
}

// SeqHighWater returns the highest WAL seq already durable outside the WAL
// (flushed into a segment, or discarded by a Reset).
func (sm *SegmentManager) SeqHighWater() int64 { return sm.meta.SeqHighWater() }

// DocCount returns the total number of indexed documents (segments + buffer).
func (sm *SegmentManager) DocCount() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	n := sm.bufferDocs
	for i, seg := range sm.segments {
		if seg != nil {
			n += seg.DocCount()
		} else {
			n += sm.segRecords[i].DocCount // use record metadata if not yet loaded
		}
	}
	return n
}

// SegmentRecords returns a snapshot of the current active segment metadata.
func (sm *SegmentManager) SegmentRecords() []storage.SegmentRecord {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	out := make([]storage.SegmentRecord, len(sm.segRecords))
	copy(out, sm.segRecords)
	return out
}

// ShardStatus holds per-shard operational metrics exposed via GET /shards.
type ShardStatus struct {
	State       string // "STARTED" | "MERGING" | "WARMING"
	DeletedDocs int    // tombstone count
	BufferDocs  int    // unflushed in-memory doc count
	WALSeq      int64  // current WAL sequence number
	MergeDebt   int    // approximate number of pending segment merges
}

// Status returns a snapshot of the shard's operational metrics.
func (sm *SegmentManager) Status() ShardStatus {
	sm.mu.RLock()
	deletedDocs := len(sm.tombstones)
	bufferDocs := sm.bufferDocs
	segRecordsCopy := make([]storage.SegmentRecord, len(sm.segRecords))
	copy(segRecordsCopy, sm.segRecords)
	sm.mu.RUnlock()

	sm.walMu.Lock()
	walSeq := sm.nextSeq
	sm.walMu.Unlock()

	// Approximate merge debt: max(0, total_segments - SegmentsPerTier)
	mergeDebt := len(segRecordsCopy) - sm.mergePolicy.SegmentsPerTier
	if mergeDebt < 0 {
		mergeDebt = 0
	}

	state := "STARTED"
	switch {
	case sm.warming.Load():
		state = "WARMING"
	case sm.merging.Load():
		state = "MERGING"
	}

	return ShardStatus{
		State:       state,
		DeletedDocs: deletedDocs,
		BufferDocs:  bufferDocs,
		WALSeq:      walSeq,
		MergeDebt:   mergeDebt,
	}
}

// resolveFieldTokens builds field→tokens map for a document.
// In BM25F mode: tokenizes each field in doc.Fields.
// Fallback: treats doc.Text as the "body" field if doc.Fields is empty.
func (sm *SegmentManager) resolveFieldTokens(doc types.Document) map[string][]string {
	if len(doc.Fields) > 0 {
		result := make(map[string][]string, len(doc.Fields))
		for fname, ftext := range doc.Fields {
			result[fname] = sm.tokenizer.Tokenize(ftext)
		}
		return result
	}
	// Fallback: treat Text as "body" field.
	return map[string][]string{"body": sm.tokenizer.Tokenize(doc.Text)}
}
