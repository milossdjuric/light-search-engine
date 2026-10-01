package index

import (
	"bufio"
	"encoding/binary"
	"os"
	"sync"

	"search-eval-platform/internal/analysis"
	"search-eval-platform/internal/codec"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/pkg/types"
)

// rawPosting is a single posting entry in the flat accumulator.
// Using integer IDs (not strings) keeps the struct at 12 bytes with no pointers.
// On-disk layout: [TermID uint32][DocID uint32][TF uint32] = 12 bytes, little-endian.
type rawPosting struct {
	TermID uint32 // ingest-order term ID (index into IndexBuilder.termList)
	DocID  uint32 // ingest-order doc ID  (index into IndexBuilder.docList)
	TF     uint32 // term frequency for this (term, doc) pair
}

// PostingEntry records a single occurrence of a term in a document.
type PostingEntry struct {
	DocID string
	TF    int
}

// IndexBuilder is a mutable structure for accumulating documents before building
// an immutable InvertedIndex via Build().
//
// Postings are accumulated by writing 12-byte rawPosting records (TermID uint32 +
// DocID uint32 + TF uint32) in per-document batches to a temporary file via a
// bufio.Writer. This keeps the posting data off the Go heap during ingest,
// eliminating GC scan overhead at high buffer sizes and letting maxBufferDocs
// grow without OOM pressure.
//
// Terms are sorted at Build() time; postings are sorted by (TermID, DocID)
// so each term's posting list is extracted with a single contiguous slice.
type IndexBuilder struct {
	// Term interning: term string → insertion-order uint32 ID.
	termToID map[string]uint32
	termList []string // ID → term string (for Build sort and FST)

	// Doc interning: docID string → insertion-order uint32 ID.
	docToID map[string]uint32
	docList []string // ID → docID string
	docLens []int    // ID → document length (tokens)
	// docLive[id] is false for a doc slot superseded by a later internDoc()
	// call for the same docID (re-Add() before Build(), i.e. an upsert within
	// one flush window). Its earlier postings are dropped at Build() time so
	// the last Add() wins, matching docLens being overwritten in place.
	docLive           []bool
	hasSupersededDocs bool

	// External posting accumulator: rawPosting records written to a temp file.
	// One bufio.Write per document (all its terms concatenated) amortises I/O
	// overhead to O(1) syscalls per doc regardless of vocabulary size.
	postFile   *os.File      // underlying temp file (stays open for the builder's lifetime)
	postBuf    *bufio.Writer // buffered writer over postFile
	postCount  int64         // total rawPosting records written (== file size / 12)
	addScratch []byte        // per-doc scratch buffer for batch encoding before Write
	lastErr    error         // sticky write error; surfaced in Build()

	totalTokens int64
	N           int
	memEstimate int64

	// docLengths and fieldTFs/fieldLens are populated only in BM25F mode
	// (AddFields calls). Not used by Add/AddTermFreqs/Build.
	docLengths map[string]int
	fieldTFs   map[string]map[string]map[string]int
	fieldLens  map[string]map[string]int
}

// NewIndexBuilder creates an empty IndexBuilder backed by a temp file for
// posting accumulation. Panics if the OS cannot create the temp file (fatal
// OS-level error); callers do not need to check the error.
//
// Call Close() when the builder is no longer needed to release the temp file.
func NewIndexBuilder() *IndexBuilder {
	f, err := os.CreateTemp("", "idx-post-*.bin")
	if err != nil {
		panic("NewIndexBuilder: create temp file: " + err.Error())
	}
	return &IndexBuilder{
		termToID:   make(map[string]uint32, 131072), // 2^17: typical 50-100k unique terms
		termList:   make([]string, 0, 131072),
		docToID:    make(map[string]uint32, 65536), // 2^16: typical 50k docs
		docList:    make([]string, 0, 65536),
		docLens:    make([]int, 0, 65536),
		docLive:    make([]bool, 0, 65536),
		postFile:   f,
		postBuf:    bufio.NewWriterSize(f, 1<<20), // 1 MiB write buffer
		addScratch: make([]byte, 0, 128*12),       // pre-size for a typical ~30-term doc
	}
}

// Reset clears all accumulated state so the builder can be reused.
// The underlying temp file is truncated and seeked back to the start;
// the file descriptor is kept open for continued use.
func (b *IndexBuilder) Reset() {
	clear(b.termToID)
	b.termList = b.termList[:0]
	clear(b.docToID)
	b.docList = b.docList[:0]
	b.docLens = b.docLens[:0]
	b.docLive = b.docLive[:0]
	b.hasSupersededDocs = false
	b.totalTokens = 0
	b.N = 0
	b.memEstimate = 0
	b.docLengths = nil
	b.fieldTFs = nil
	b.fieldLens = nil
	b.lastErr = nil
	// Truncate the temp file and reset the buffered writer over it.
	_ = b.postBuf.Flush()
	_ = b.postFile.Truncate(0)
	_, _ = b.postFile.Seek(0, 0)
	b.postBuf.Reset(b.postFile)
	b.postCount = 0
	b.addScratch = b.addScratch[:0]
}

// Close flushes any buffered data, closes, and removes the underlying temp file.
// After Close, the builder must not be used.
func (b *IndexBuilder) Close() {
	_ = b.postBuf.Flush()
	name := b.postFile.Name()
	_ = b.postFile.Close()
	_ = os.Remove(name)
}

// internDoc returns the insertion-order numeric ID for docID, creating one if
// needed. If docID already exists, its doc length is updated.
func (b *IndexBuilder) internDoc(docID string, docLen int) uint32 {
	did, ok := b.docToID[docID]
	if !ok {
		did = uint32(len(b.docList))
		b.docToID[docID] = did
		b.docList = append(b.docList, docID)
		b.docLens = append(b.docLens, docLen)
		b.docLive = append(b.docLive, true)
		b.memEstimate += int64(len(docID)) + 8
		return did
	}
	// Re-added before Build() (upsert within one flush window): the old did's
	// rawPosting records are already written to postFile and can't be cheaply
	// erased, so mark that slot superseded — buildInternal drops its postings
	// and excludes it from N/totalTokens — and allocate a fresh slot for the
	// new content, matching docLens being overwritten in place.
	b.docLive[did] = false
	b.hasSupersededDocs = true
	newDid := uint32(len(b.docList))
	b.docToID[docID] = newDid
	b.docList = append(b.docList, docID)
	b.docLens = append(b.docLens, docLen)
	b.docLive = append(b.docLive, true)
	return newDid
}

// internTerm returns the insertion-order numeric ID for term, creating one if needed.
func (b *IndexBuilder) internTerm(term string) uint32 {
	tid, ok := b.termToID[term]
	if !ok {
		tid = uint32(len(b.termList))
		b.termToID[term] = tid
		b.termList = append(b.termList, term)
		b.memEstimate += int64(len(term)) + 8
	}
	return tid
}

// tfPool recycles the per-document TF map to reduce GC pressure at high
// ingest rates (peak ~57k docs/s → 57k map allocations per second avoided).
var tfPool = sync.Pool{New: func() any { return make(map[string]int, 32) }}

// Add indexes a single document given its ID and token stream.
// TF is accumulated per term; duplicates in tokens count multiple times.
// All rawPosting records for this document are encoded into addScratch then
// written in a single bufio.Write call — O(1) syscalls per document.
func (b *IndexBuilder) Add(docID string, tokens []string) {
	tf := tfPool.Get().(map[string]int)
	for _, t := range tokens {
		tf[t]++
	}
	did := b.internDoc(docID, len(tokens))
	b.totalTokens += int64(len(tokens))
	b.N++

	// Build the per-doc batch: encode all rawPostings into addScratch (12 bytes each).
	need := len(tf) * 12
	if cap(b.addScratch) < need {
		b.addScratch = make([]byte, 0, need+128)
	}
	b.addScratch = b.addScratch[:0]
	for term, count := range tf {
		tid := b.internTerm(term)
		b.addScratch = binary.LittleEndian.AppendUint32(b.addScratch, tid)
		b.addScratch = binary.LittleEndian.AppendUint32(b.addScratch, did)
		b.addScratch = binary.LittleEndian.AppendUint32(b.addScratch, uint32(count))
	}
	if len(b.addScratch) > 0 {
		if _, err := b.postBuf.Write(b.addScratch); err != nil && b.lastErr == nil {
			b.lastErr = err
		}
		b.postCount += int64(len(tf))
		b.memEstimate += int64(len(b.addScratch)) // track temp file bytes so mem_threshold fires correctly
	}

	clear(tf)
	tfPool.Put(tf)
}

// MemoryEstimate returns an approximate number of bytes used by the builder.
func (b *IndexBuilder) MemoryEstimate() int64 {
	return b.memEstimate
}

// Build freezes the builder into an immutable InvertedIndex.
// It flushes the temp file, reads all accumulated rawPosting records back into
// memory, sorts them, and encodes posting lists using the format selected by opts.
func (b *IndexBuilder) Build() *InvertedIndex {
	return b.buildInternal()
}

// buildInternal builds the BM25 index (full blocks FOR-delta packed via codec.PackFOR32).
func (b *IndexBuilder) buildInternal() *InvertedIndex {
	if b.N == 0 {
		return &InvertedIndex{
			termIndex:  make(map[string]int),
			docIDs:     []string{},
			docIDIndex: make(map[string]uint64),
			docLengths: make(map[string]int),
		}
	}
	if b.lastErr != nil {
		// Surface any write error encountered during Add / AddTermFreqs.
		panic("IndexBuilder.Build: posting write error: " + b.lastErr.Error())
	}

	// Flush buffered writes to the temp file.
	if err := b.postBuf.Flush(); err != nil {
		panic("IndexBuilder.Build: flush: " + err.Error())
	}

	// Read all rawPosting records back from the temp file.
	fileBytes := b.postCount * 12
	raw := make([]byte, fileBytes)
	if _, err := b.postFile.ReadAt(raw, 0); err != nil {
		panic("IndexBuilder.Build: ReadAt: " + err.Error())
	}
	postings := make([]rawPosting, b.postCount)
	for i := range postings {
		off := i * 12
		postings[i].TermID = binary.LittleEndian.Uint32(raw[off:])
		postings[i].DocID = binary.LittleEndian.Uint32(raw[off+4:])
		postings[i].TF = binary.LittleEndian.Uint32(raw[off+8:])
	}

	// Drop postings for doc slots superseded by a re-Add() before Build()
	// (see internDoc), and compact numeric doc IDs to exclude them, so a
	// re-indexed doc's earlier content doesn't survive as duplicate postings
	// alongside its latest content. Builds into local variables rather than
	// mutating b.docList/b.docLens/b.N/b.totalTokens in place: Snapshot()
	// documents that it can be called repeatedly "without clearing the
	// builder", and postFile (read again from scratch on every call, since
	// b.postCount is never trimmed) always yields the same raw postings —
	// including the superseded ones — so this filtering must be redone
	// identically on every call, not just the first.
	docList := b.docList
	docLens := b.docLens
	docCount := b.N
	totalTokens := b.totalTokens
	if b.hasSupersededDocs {
		finalID := make([]int32, len(b.docList))
		liveDocList := make([]string, 0, len(b.docList))
		liveDocLens := make([]int, 0, len(b.docList))
		for i, live := range b.docLive {
			if !live {
				finalID[i] = -1
				continue
			}
			finalID[i] = int32(len(liveDocList))
			liveDocList = append(liveDocList, b.docList[i])
			liveDocLens = append(liveDocLens, b.docLens[i])
		}
		docList = liveDocList
		docLens = liveDocLens

		kept := make([]rawPosting, 0, len(postings))
		for _, p := range postings {
			if fid := finalID[p.DocID]; fid >= 0 {
				p.DocID = uint32(fid)
				kept = append(kept, p)
			}
		}
		postings = kept

		docCount = len(liveDocList)
		var total int64
		for _, l := range liveDocLens {
			total += int64(l)
		}
		totalTokens = total
	}

	avgDocLen := float64(totalTokens) / float64(docCount)
	nTerms := len(b.termList)
	nDocs := len(docList)

	// 1. Assign numeric docIDs in insertion order (no sort needed).
	allDocIDs := make([]string, nDocs)
	copy(allDocIDs, docList)
	docIDIndex := make(map[string]uint64, nDocs)
	for i, id := range allDocIDs {
		docIDIndex[id] = uint64(i)
	}

	// 2. Sort terms for FST (vellum requires ascending key order).
	terms := make([]string, nTerms)
	copy(terms, b.termList)
	radixSortStrings(terms)
	termIndex := make(map[string]int, nTerms)
	for i, t := range terms {
		termIndex[t] = i
	}

	// 3. Build reverse map: term string → ingest-order term ID.
	termIngestID := make(map[string]uint32, nTerms)
	for i, t := range b.termList {
		termIngestID[t] = uint32(i)
	}

	// 4. Compute per-term posting group boundaries via counting sort pre-pass.
	termStart := make([]int, nTerms+1)
	for _, p := range postings {
		termStart[p.TermID+1]++
	}
	for i := 1; i <= nTerms; i++ {
		termStart[i] += termStart[i-1]
	}

	// 5. Sort flat postings by (TermID, DocID) using 2-pass LSD radix sort — O(N).
	radixSortPostings(postings)

	// 6. Encode posting lists in sorted term order.
	offsets := make([]int64, nTerms+1)
	dfs := make([]int, nTerms)
	skipLists := make([]*SkipList, nTerms)
	ubs := make([]float64, nTerms)
	var data []byte

	bm25 := scoring.NewDefaultBM25()

	// packBuf must hold the largest possible encoded block.
	packBuf := make([]byte, codec.PackFOR32BufSize)
	docDeltas := make([]uint64, codec.BlockSize)
	tfVals := make([]uint64, codec.BlockSize)
	// uint32 versions for codec.PackFOR32 encoding.
	docDeltasU32 := make([]uint32, codec.BlockSize)
	tfValsU32 := make([]uint32, codec.BlockSize)

	for termOrd, term := range terms {
		ingestID := termIngestID[term]
		termPostings := postings[termStart[ingestID]:termStart[ingestID+1]]

		df := len(termPostings)
		dfs[termOrd] = df

		startOff := int64(len(data))

		var l0DocIDs []uint64
		var l0Offsets []int
		var l0Bases []uint64
		var l0Impact []float32
		var maxUB float64

		nBlocks := df / codec.BlockSize
		tail := df % codec.BlockSize

		var prevDocID uint64
		blockBase := int64(len(data))

		for blk := 0; blk < nBlocks; blk++ {
			blockStart := blk * codec.BlockSize
			blockStartOff := int(int64(len(data)) - blockBase)
			var blockMaxImpact float64

			for j := 0; j < codec.BlockSize; j++ {
				p := termPostings[blockStart+j]
				numID := uint64(p.DocID)
				delta := numID - prevDocID
				prevDocID = numID
				docDeltas[j] = delta
				tfVals[j] = uint64(p.TF)

				dl := docLens[p.DocID]
				score := bm25.ScoreTerm(int(p.TF), df, dl, avgDocLen, docCount)
				if score > blockMaxImpact {
					blockMaxImpact = score
				}
				if score > maxUB {
					maxUB = score
				}
			}

			var n int
			for j := range docDeltasU32 {
				docDeltasU32[j] = uint32(docDeltas[j])
				tfValsU32[j] = uint32(tfVals[j])
			}
			n = codec.PackFOR32(docDeltasU32, packBuf)
			data = append(data, packBuf[:n]...)
			n = codec.PackFOR32(tfValsU32, packBuf)
			data = append(data, packBuf[:n]...)

			l0DocIDs = append(l0DocIDs, prevDocID)
			l0Offsets = append(l0Offsets, blockStartOff)
			if blk == 0 {
				l0Bases = append(l0Bases, 0)
			} else {
				l0Bases = append(l0Bases, l0DocIDs[blk-1])
			}
			l0Impact = append(l0Impact, float32(blockMaxImpact))
		}

		if tail > 0 {
			tailStart := nBlocks * codec.BlockSize
			for j := 0; j < tail; j++ {
				p := termPostings[tailStart+j]
				numID := uint64(p.DocID)
				delta := numID - prevDocID
				prevDocID = numID
				data = codec.AppendVarint(data, delta)
				data = codec.AppendVarint(data, uint64(p.TF))

				dl := docLens[p.DocID]
				score := bm25.ScoreTerm(int(p.TF), df, dl, avgDocLen, docCount)
				if score > maxUB {
					maxUB = score
				}
			}
		}

		offsets[termOrd] = startOff
		ubs[termOrd] = maxUB

		if len(l0DocIDs) > 0 {
			skipLists[termOrd] = BuildSkipList(l0DocIDs, l0Offsets, l0Bases, l0Impact)
		}
	}
	offsets[nTerms] = int64(len(data))

	// Build docLengths map for the InvertedIndex.
	docLengthsCopy := make(map[string]int, nDocs)
	for i, id := range docList {
		docLengthsCopy[id] = docLens[i]
	}

	return &InvertedIndex{
		termIndex:  termIndex,
		terms:      terms,
		data:       data,
		offsets:    offsets,
		dfs:        dfs,
		skipLists:  skipLists,
		ub:         ubs,
		docIDs:     allDocIDs,
		docIDIndex: docIDIndex,
		docLengths: docLengthsCopy,
		avgDocLen:  avgDocLen,
		N:          docCount,
	}
}

// NewInvertedIndex is a convenience wrapper that builds an InvertedIndex
// directly from a slice of Documents, preserving backward compatibility.
func NewInvertedIndex(docs []types.Document) *InvertedIndex {
	b := NewIndexBuilder()
	defer b.Close()
	for _, d := range docs {
		b.Add(d.ID, analysis.Tokenize(d.Text))
	}
	return b.Build()
}

// PreRegisterDoc registers a document and its length for the term-major merge path.
// Must be called exactly once per document before any WritePosting calls for that document.
func (b *IndexBuilder) PreRegisterDoc(docID string, docLen int) {
	b.internDoc(docID, docLen)
	b.N++
	b.totalTokens += int64(docLen)
}

// WritePosting writes a single (term, docID, TF) tuple directly to the external sort
// file. The document must have been pre-registered via PreRegisterDoc.
// Used by the term-major merge path to avoid building a doc-major docTermMap.
func (b *IndexBuilder) WritePosting(term, docID string, tf int) {
	tid := b.internTerm(term)
	did, ok := b.docToID[docID]
	if !ok {
		return // caller must PreRegisterDoc first
	}
	var buf [12]byte
	binary.LittleEndian.PutUint32(buf[0:], tid)
	binary.LittleEndian.PutUint32(buf[4:], did)
	binary.LittleEndian.PutUint32(buf[8:], uint32(tf))
	if _, err := b.postBuf.Write(buf[:]); err != nil && b.lastErr == nil {
		b.lastErr = err
	}
	b.postCount++
	b.memEstimate += 12
}

// AddFields accumulates a document indexed by named fields for BM25F scoring.
// Must not be mixed with Add on the same builder instance.
// fieldTokens maps field name → already-tokenized token slice.
func (b *IndexBuilder) AddFields(docID string, fieldTokens map[string][]string) {
	if b.fieldTFs == nil {
		b.fieldTFs = make(map[string]map[string]map[string]int)
		b.fieldLens = make(map[string]map[string]int)
		b.docLengths = make(map[string]int)
	}
	b.fieldTFs[docID] = make(map[string]map[string]int, len(fieldTokens))
	b.fieldLens[docID] = make(map[string]int, len(fieldTokens))

	totalLen := 0
	for field, tokens := range fieldTokens {
		tfs := make(map[string]int, len(tokens)/2+1)
		for _, t := range tokens {
			tfs[t]++
		}
		b.fieldTFs[docID][field] = tfs
		b.fieldLens[docID][field] = len(tokens)
		totalLen += len(tokens)
	}

	b.docLengths[docID] = totalLen
	b.totalTokens += int64(totalLen)
	b.N++
	// Rough memory estimate: per-doc overhead + tokens.
	b.memEstimate += int64(len(docID)+48) + int64(totalLen)*8
}

// BuildWithOptions dispatches to the correct build path based on opts.
func (b *IndexBuilder) BuildWithOptions(opts BuildOptions) *InvertedIndex {
	switch {
	case len(opts.Fields) > 0 && b.fieldTFs != nil:
		return b.BuildBM25F(opts.Fields)
	case opts.BM25FScaled:
		k1 := opts.BM25FK1
		if k1 <= 0 {
			k1 = 1.2
		}
		return b.buildScaledPseudoTF(k1)
	default:
		return b.buildInternal()
	}
}

// BuildOptions controls how IndexBuilder.BuildWithOptions dispatches.
type BuildOptions struct {
	// Fields, when non-empty, switches to BM25F mode: pseudo-TFs are computed
	// from per-field data accumulated via AddFields. Mutually exclusive with BM25FScaled.
	Fields []scoring.FieldConfig

	// BM25FScaled, when true, means the postings already contain scaled pseudo-TFs
	// (BM25FScaleFactor * pseudoTF as integers). Used during segment merge to preserve
	// BM25F scoring. Mutually exclusive with Fields.
	BM25FScaled bool

	// BM25FK1 is the saturation parameter used when BM25FScaled is true.
	BM25FK1 float64
}
