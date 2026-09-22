package index

import (
	"bufio"
	"encoding/binary"
	"math"
	"os"
	"sort"
	"sync"
	"unicode"
	"unicode/utf8"

	"search-eval-platform/pkg/types"
)


// Flat posting accumulator

// rawPosting is a single posting entry in the flat accumulator.
// Using integer IDs (not strings) keeps the struct at 12 bytes with no pointers.
// On-disk layout: [TermID uint32][DocID uint32][TF uint32] = 12 bytes, little-endian.
type rawPosting struct {
	TermID uint32 // ingest-order term ID (index into IndexBuilder.termList)
	DocID  uint32 // ingest-order doc ID  (index into IndexBuilder.docList)
	TF     uint32 // term frequency for this (term, doc) pair
}


// In-memory skip list: sorted term index


// IndexReader is the read-only interface that MaxScoreSearcher and other
// consumers use to query an index. Both InvertedIndex and the Segment adapter
// implement it, enabling search across both in-memory and on-disk indexes.
type IndexReader interface {
	Iterator(term string) *PostingIter
	TermUB(term string) float64
	DF(term string) int
	DocCount() int
	AvgDocLen() float64
	DocLen(docID string) int
	DocStringID(numID uint64) string
}

// NewPostingIter creates a PostingIter from raw posting bytes without a skip
// list. Prefer NewPostingIterWithSkip when skip bytes are available.
// useFOR32 selects the block decoder: true → UnpackFOR32Into (v6 segments),
// false → UnpackBlockWithScratch (v2/v4 segments, in-memory index).
func NewPostingIter(data []byte, df int, useFOR32 bool) *PostingIter {
	it := &PostingIter{
		data:     data,
		df:       df,
		blockIdx: -1,
	}
	it.decodeBlock = pickDecodeBlock(useFOR32)
	return it
}

// NewPostingIterWithSkip creates a PostingIter with an attached SkipList,
// enabling block-max MaxScore skipping on on-disk segments.
// useFOR32 selects the block decoder: true → UnpackFOR32Into (v6 segments),
// false → UnpackBlockWithScratch (v2/v4 segments, in-memory index).
func NewPostingIterWithSkip(data []byte, skip *SkipList, df int, useFOR32 bool) *PostingIter {
	it := &PostingIter{
		data:     data,
		skip:     skip,
		df:       df,
		blockIdx: -1,
	}
	it.decodeBlock = pickDecodeBlock(useFOR32)
	return it
}

// WithBlockCache attaches a shared block cache to the iterator, enabling
// decoded-block reuse across repeated accesses to the same posting list block.
// termOrd is the term's ordinal within the segment and forms half of the cache key.
func (it *PostingIter) WithBlockCache(cache *BlockCache, termOrd int) *PostingIter {
	it.blockCache = cache
	it.termOrd = termOrd
	return it
}

// pickDecodeBlock returns the correct full-block decode function for the
// given format. intcompDecode uses scratch to avoid per-block allocation.
func pickDecodeBlock(useFOR32 bool) func(src []byte, out []uint64, scratch []uint64) int {
	if useFOR32 {
		return func(src []byte, out []uint64, _ []uint64) int {
			return UnpackFOR32Into(src, out)
		}
	}
	return func(src []byte, out []uint64, scratch []uint64) int {
		return UnpackBlockWithScratch(src, out, scratch)
	}
}

// LoadSkipFromBytes deserializes a SkipList from the binary format produced
// by InvertedIndex.RawSkipBytes.
//
// Format: [4 nL0][4 nL1][nL0 * 28 bytes][nL1 * 16 bytes]
// Per L0 entry (28 bytes): [8 lastDocID][8 byteOffset][8 deltaBase][4 blockMaxImpact]
// The stored L1 data is ignored; L1 is rebuilt from L0 via BuildSkipList.
func LoadSkipFromBytes(data []byte) *SkipList {
	if len(data) < 8 {
		return nil
	}
	nL0 := int(binary.LittleEndian.Uint32(data[0:4]))
	if nL0 == 0 {
		return nil
	}
	const l0EntrySize = 28
	if len(data) < 8+nL0*l0EntrySize {
		return nil
	}

	l0DocIDs := make([]uint64, nL0)
	l0Offsets := make([]int, nL0)
	l0Bases := make([]uint64, nL0)
	l0Impact := make([]float32, nL0)

	off := 8
	for i := 0; i < nL0; i++ {
		l0DocIDs[i] = binary.LittleEndian.Uint64(data[off:])
		off += 8
		l0Offsets[i] = int(binary.LittleEndian.Uint64(data[off:]))
		off += 8
		l0Bases[i] = binary.LittleEndian.Uint64(data[off:])
		off += 8
		l0Impact[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[off:]))
		off += 4
	}
	return BuildSkipList(l0DocIDs, l0Offsets, l0Bases, l0Impact)
}

// Tokenize lowercases text and splits on non-letter, non-digit characters.
// Uses a fast byte-level path for pure ASCII text (the common case for English
// documents), falling back to full Unicode handling for multi-byte input.
func Tokenize(text string) []string {
	if isASCII(text) {
		return tokenizeASCII(text)
	}
	return tokenizeUnicode(text)
}

// isASCII reports whether every byte in s is in the ASCII range.
// The Go compiler auto-vectorizes this simple byte-scan loop on amd64.
func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// tokenizeASCII is the fast path for pure ASCII input.
// Avoids rune decoding, []rune allocation, and Unicode table lookups.
// Lowercase is a single bit-or: 'A'|0x20 == 'a'.
func tokenizeASCII(text string) []string {
	tokens := make([]string, 0, 8)
	buf := make([]byte, 0, 32)
	for i := 0; i < len(text); i++ {
		b := text[i]
		switch {
		case b >= 'a' && b <= 'z', b >= '0' && b <= '9':
			buf = append(buf, b)
		case b >= 'A' && b <= 'Z':
			buf = append(buf, b|0x20) // lowercase: single bit flip
		default:
			if len(buf) > 0 {
				tokens = append(tokens, string(buf))
				buf = buf[:0]
			}
		}
	}
	if len(buf) > 0 {
		tokens = append(tokens, string(buf))
	}
	return tokens
}

// tokenizeUnicode is the fallback for text containing multi-byte UTF-8 characters.
func tokenizeUnicode(text string) []string {
	var tokens []string
	var cur []rune
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			cur = append(cur, unicode.ToLower(r))
		} else {
			if len(cur) > 0 {
				tokens = append(tokens, string(cur))
				cur = cur[:0]
			}
		}
	}
	if len(cur) > 0 {
		tokens = append(tokens, string(cur))
	}
	return tokens
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
	postFile  *os.File       // underlying temp file (stays open for the builder's lifetime)
	postBuf   *bufio.Writer  // buffered writer over postFile
	postCount int64          // total rawPosting records written (== file size / 12)
	addScratch []byte        // per-doc scratch buffer for batch encoding before Write
	lastErr   error          // sticky write error; surfaced in Build()

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

// Snapshot builds and returns an InvertedIndex from the current state of the
// builder without clearing the builder. Safe to call concurrently with reads
// (but not with concurrent Add calls).
func (b *IndexBuilder) Snapshot() *InvertedIndex {
	return b.Build()
}

// bm25Score computes the BM25 score contribution of one term occurrence.
// idf = ln((N - df + 0.5) / (df + 0.5) + 1)
// tfNorm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgDocLen))
func bm25Score(tf, df, dl int, avgDocLen float64, N int, k1, bParam float64) float64 {
	idf := math.Log(float64(N-df+1)/float64(df+1) + 1) // simplified but matches spec intent
	// Use spec formula: ln((N - df + 0.5) / (df + 0.5) + 1)
	idf = math.Log(float64(N-df+1)*0 + float64(N-df) + 0.5)
	// Redo correctly per spec.
	idf = math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
	tfNorm := float64(tf) * (k1 + 1) / (float64(tf) + k1*(1-bParam+bParam*float64(dl)/avgDocLen))
	return idf * tfNorm
}

// radixSortStrings sorts ss in-place using LSD (least-significant-digit) radix
// sort with byte-level buckets. Complexity: O(L × N) where L = max string
// length and N = len(ss). For the typical vocabulary (short words, large N)
// this outperforms the O(N log N) comparison sort by avoiding string comparisons.
// Falls back to sort.Strings for small inputs where the overhead isn't worth it.
func radixSortStrings(ss []string) {
	const threshold = 32
	if len(ss) < threshold {
		sort.Strings(ss)
		return
	}

	maxLen := 0
	for _, s := range ss {
		if len(s) > maxLen {
			maxLen = len(s)
		}
	}
	if maxLen == 0 {
		return
	}

	tmp := make([]string, len(ss))
	count := make([]int, 257) // bucket 0 = "no byte" (shorter strings); 1–256 = byte value + 1

	src, dst := ss, tmp
	for pos := maxLen - 1; pos >= 0; pos-- {
		// Zero count array.
		for i := range count {
			count[i] = 0
		}
		// Frequency count.
		for _, s := range src {
			b := 0
			if pos < len(s) {
				b = int(s[pos]) + 1
			}
			count[b]++
		}
		// Prefix sums → starting positions.
		for i := 1; i <= 256; i++ {
			count[i] += count[i-1]
		}
		// Distribute into dst (right-to-left for stability).
		for i := len(src) - 1; i >= 0; i-- {
			b := 0
			if pos < len(src[i]) {
				b = int(src[i][pos]) + 1
			}
			count[b]--
			dst[count[b]] = src[i]
		}
		src, dst = dst, src
	}

	// If result ended up in tmp (src was swapped to tmp), copy back to ss.
	if len(ss) > 0 && len(src) > 0 && &src[0] != &ss[0] {
		copy(ss, src)
	}
}

// radixSortPostings sorts ps in-place by (TermID, DocID) using a 2-pass LSD
// radix sort on the packed key uint64(TermID)<<32 | uint64(DocID).
// Complexity: O(2N) — two counting passes, each O(N+65536).
// For large posting arrays (millions of entries) this is ~50× faster than
// sort.Slice because it avoids comparison overhead and is cache-friendly.
//
// Falls back to sort.Slice for small inputs where the fixed overhead dominates.
func radixSortPostings(ps []rawPosting) {
	const threshold = 512
	if len(ps) < threshold {
		sort.Slice(ps, func(i, j int) bool {
			pi, pj := ps[i], ps[j]
			if pi.TermID != pj.TermID {
				return pi.TermID < pj.TermID
			}
			return pi.DocID < pj.DocID
		})
		return
	}

	tmp := make([]rawPosting, len(ps))

	// Pass 1: sort by lower 32 bits (DocID).
	var count1 [65536]int
	for i := range ps {
		count1[ps[i].DocID&0xffff]++
	}
	// Prefix-sum → start positions.
	var sum1 int
	for i := range count1 {
		count1[i], sum1 = sum1, sum1+count1[i]
	}
	for i := range ps {
		bucket := ps[i].DocID & 0xffff
		tmp[count1[bucket]] = ps[i]
		count1[bucket]++
	}

	var count2 [65536]int
	for i := range tmp {
		count2[tmp[i].DocID>>16]++
	}
	var sum2 int
	for i := range count2 {
		count2[i], sum2 = sum2, sum2+count2[i]
	}
	for i := range tmp {
		bucket := tmp[i].DocID >> 16
		ps[count2[bucket]] = tmp[i]
		count2[bucket]++
	}

	// Pass 2: sort by upper 32 bits (TermID), stable — preserves DocID order.
	var count3 [65536]int
	for i := range ps {
		count3[ps[i].TermID&0xffff]++
	}
	var sum3 int
	for i := range count3 {
		count3[i], sum3 = sum3, sum3+count3[i]
	}
	for i := range ps {
		bucket := ps[i].TermID & 0xffff
		tmp[count3[bucket]] = ps[i]
		count3[bucket]++
	}

	var count4 [65536]int
	for i := range tmp {
		count4[tmp[i].TermID>>16]++
	}
	var sum4 int
	for i := range count4 {
		count4[i], sum4 = sum4, sum4+count4[i]
	}
	for i := range tmp {
		bucket := tmp[i].TermID >> 16
		ps[count4[bucket]] = tmp[i]
		count4[bucket]++
	}
}

// Build freezes the builder into an immutable InvertedIndex.
// It flushes the temp file, reads all accumulated rawPosting records back into
// memory, sorts them, and encodes posting lists using the format selected by opts.
func (b *IndexBuilder) Build() *InvertedIndex {
	return b.buildInternal(false)
}

// buildInternal is the shared implementation for Build() and buildFOR32().
// useFOR32=true switches block encoding from intcomp to PackFOR32.
func (b *IndexBuilder) buildInternal(useFOR32 bool) *InvertedIndex {
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

	const k1Default = 1.2
	const bDefault = 0.75

	// packBuf must hold the largest possible encoded block.
	packBufSize := blockPackBufSize
	if useFOR32 {
		packBufSize = PackFOR32BufSize
	}
	packBuf := make([]byte, packBufSize)
	docDeltas := make([]uint64, BlockSize)
	tfVals := make([]uint64, BlockSize)
	// uint32 versions for PackFOR32 encoding.
	docDeltasU32 := make([]uint32, BlockSize)
	tfValsU32 := make([]uint32, BlockSize)

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

		nBlocks := df / BlockSize
		tail := df % BlockSize

		var prevDocID uint64
		blockBase := int64(len(data))

		for blk := 0; blk < nBlocks; blk++ {
			blockStart := blk * BlockSize
			blockStartOff := int(int64(len(data)) - blockBase)
			var blockMaxImpact float64

			for j := 0; j < BlockSize; j++ {
				p := termPostings[blockStart+j]
				numID := uint64(p.DocID)
				delta := numID - prevDocID
				prevDocID = numID
				docDeltas[j] = delta
				tfVals[j] = uint64(p.TF)

				dl := docLens[p.DocID]
				score := bm25Score(int(p.TF), df, dl, avgDocLen, docCount, k1Default, bDefault)
				if score > blockMaxImpact {
					blockMaxImpact = score
				}
				if score > maxUB {
					maxUB = score
				}
			}

			var n int
			if useFOR32 {
				for j := range docDeltasU32 {
					docDeltasU32[j] = uint32(docDeltas[j])
					tfValsU32[j] = uint32(tfVals[j])
				}
				n = PackFOR32(docDeltasU32, packBuf)
				data = append(data, packBuf[:n]...)
				n = PackFOR32(tfValsU32, packBuf)
				data = append(data, packBuf[:n]...)
			} else {
				n = PackBlock(docDeltas, packBuf)
				data = append(data, packBuf[:n]...)
				n = PackBlock(tfVals, packBuf)
				data = append(data, packBuf[:n]...)
			}

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
			tailStart := nBlocks * BlockSize
			for j := 0; j < tail; j++ {
				p := termPostings[tailStart+j]
				numID := uint64(p.DocID)
				delta := numID - prevDocID
				prevDocID = numID
				data = AppendVarint(data, delta)
				data = AppendVarint(data, uint64(p.TF))

				dl := docLens[p.DocID]
				score := bm25Score(int(p.TF), df, dl, avgDocLen, docCount, k1Default, bDefault)
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
		useFOR32:   useFOR32,
	}
}

// InvertedIndex is an immutable, lock-free inverted index.
type InvertedIndex struct {
	termIndex  map[string]int // term → ordinal
	terms      []string
	data       []byte
	offsets    []int64   // offsets[i]:offsets[i+1] = posting data for ordinal i
	dfs        []int     // document frequency per term ordinal
	skipLists  []*SkipList
	ub         []float64 // precomputed max BM25 UBt per ordinal
	docIDs     []string  // numeric uint64 → string docID
	docIDIndex map[string]uint64
	docLengths map[string]int
	avgDocLen  float64
	N          int
	useFOR32   bool // true → data blocks encoded with PackFOR32 (v6); false → intcomp (v2/v4)
}

// termOrdinalFor returns the ordinal for term, or false if not found.
func (idx *InvertedIndex) termOrdinalFor(term string) (int, bool) {
	ord, ok := idx.termIndex[term]
	return ord, ok
}

// TermUB returns the precomputed upper-bound BM25 score for term.
// Returns 0 if the term is not in the index.
func (idx *InvertedIndex) TermUB(term string) float64 {
	ord, ok := idx.termOrdinalFor(term)
	if !ok {
		return 0
	}
	return idx.ub[ord]
}

// DocCount returns the number of documents in the index.
func (idx *InvertedIndex) DocCount() int {
	return idx.N
}

// AvgDocLen returns the average document length.
func (idx *InvertedIndex) AvgDocLen() float64 {
	return idx.avgDocLen
}

// DocLen returns the length of the document with the given string ID.
func (idx *InvertedIndex) DocLen(docID string) int {
	return idx.docLengths[docID]
}

// DocStringID converts a numeric docID to its string form.
func (idx *InvertedIndex) DocStringID(numID uint64) string {
	if int(numID) >= len(idx.docIDs) {
		return ""
	}
	return idx.docIDs[numID]
}

// Iterator returns a PostingIter for the given term, or nil if the term is absent.
func (idx *InvertedIndex) Iterator(term string) *PostingIter {
	ord, ok := idx.termOrdinalFor(term)
	if !ok {
		return nil
	}
	return idx.iteratorByOrdinal(ord)
}

// iteratorByOrdinal returns a PostingIter for the term at the given ordinal.
func (idx *InvertedIndex) iteratorByOrdinal(ord int) *PostingIter {
	start := idx.offsets[ord]
	end := idx.offsets[ord+1]
	it := &PostingIter{
		data:     idx.data[start:end],
		skip:     idx.skipLists[ord],
		df:       idx.dfs[ord],
		bufPos:   0,
		bufLen:   0,
		blockIdx: -1,
	}
	it.decodeBlock = pickDecodeBlock(idx.useFOR32)
	return it
}

// PostingIter iterates over the posting list for one term.
type PostingIter struct {
	data    []byte
	skip    *SkipList
	pos     int    // current read position in data
	base    uint64 // delta base for current block
	currDoc uint64 // current docID (numeric)
	currTF  int    // current TF
	df      int    // total entries in this posting list
	read    int    // number of entries decoded so far

	blockIdx int // current block index (-1 = not started)
	docBuf   [BlockSize]uint64
	tfBuf    [BlockSize]uint64
	bufPos   int // position within current decoded block
	bufLen   int // number of valid entries in current decoded block

	// decodeBlock is the full-block decoder selected at construction time:
	//   intcomp  → UnpackBlockWithScratch (v2/v4 segments, in-memory index)
	//   FOR-delta → UnpackFOR32Into (v6 segments)
	// Using a function pointer here avoids a per-block conditional branch.
	decodeBlock func(src []byte, out []uint64, scratch []uint64) int

	// compScratch is a stack-allocated scratch buffer passed to
	// UnpackBlockWithScratch to avoid a heap allocation per block decode.
	// Must be >= compBlockMaxWords (136) elements.
	compScratch [compBlockMaxWords]uint64

	// blockCache, when non-nil, caches decoded full blocks to avoid redundant
	// FOR-delta/intcomp decoding on repeated accesses to the same block.
	blockCache *BlockCache
	termOrd    int // term ordinal within the segment (block cache key component)
}

// loadBlock decodes the next block into docBuf/tfBuf using the format-specific
// decoder function (decodeBlock), which is set at PostingIter construction.
func (it *PostingIter) loadBlock() {
	remaining := it.df - it.read
	if remaining <= 0 {
		it.bufPos = 0
		it.bufLen = 0
		return
	}

	if remaining >= BlockSize {
		// Full block: check cache first, then fall back to format-specific decoder.
		if it.blockCache != nil {
			if entry, ok := it.blockCache.get(it.termOrd, it.blockIdx); ok {
				it.docBuf = entry.docBuf
				it.tfBuf = entry.tfBuf
				it.pos += entry.docBytes + entry.tfBytes
				it.bufLen = BlockSize
			} else {
				docN := it.decodeBlock(it.data[it.pos:], it.docBuf[:], it.compScratch[:])
				it.pos += docN
				tfN := it.decodeBlock(it.data[it.pos:], it.tfBuf[:], it.compScratch[:])
				it.pos += tfN
				it.bufLen = BlockSize
				it.blockCache.set(it.termOrd, it.blockIdx, &cachedBlock{
					docBuf:   it.docBuf,
					tfBuf:    it.tfBuf,
					docBytes: docN,
					tfBytes:  tfN,
				})
			}
		} else {
			docN := it.decodeBlock(it.data[it.pos:], it.docBuf[:], it.compScratch[:])
			it.pos += docN
			it.pos += it.decodeBlock(it.data[it.pos:], it.tfBuf[:], it.compScratch[:])
			it.bufLen = BlockSize
		}
	} else {
		// Partial tail: interleaved LEB128 (delta, tf) pairs — same for all versions.
		for i := 0; i < remaining; i++ {
			delta, n := ReadVarint(it.data, it.pos)
			it.pos += n
			tf, n := ReadVarint(it.data, it.pos)
			it.pos += n
			it.docBuf[i] = delta
			it.tfBuf[i] = tf
		}
		it.bufLen = remaining
	}
	it.bufPos = 0
}

// Next advances the iterator to the next posting.
// Returns false when exhausted.
func (it *PostingIter) Next() bool {
	if it.bufPos >= it.bufLen {
		if it.read >= it.df {
			return false
		}
		it.blockIdx++
		it.loadBlock()
		if it.bufLen == 0 {
			return false
		}
	}

	delta := it.docBuf[it.bufPos]
	it.currDoc = it.base + delta
	it.base = it.currDoc
	it.currTF = int(it.tfBuf[it.bufPos])
	it.bufPos++
	it.read++
	return true
}

// DocID returns the current numeric docID.
func (it *PostingIter) DocID() uint64 {
	return it.currDoc
}

// TF returns the current term frequency.
func (it *PostingIter) TF() int {
	return it.currTF
}

// BlockIdx returns the current block index (0-based).
func (it *PostingIter) BlockIdx() int {
	return it.blockIdx
}

// BlockMaxImpact returns the precomputed max BM25 impact for the current block.
// Returns 0 if no skip list is attached or the block is out of range.
func (it *PostingIter) BlockMaxImpact() float32 {
	return it.skip.BlockMaxImpact(it.blockIdx)
}

// SkipLastDocInBlock returns the last docID in blockIdx and true, delegating
// to the attached SkipList. Returns (0, false) if unavailable.
func (it *PostingIter) SkipLastDocInBlock(blockIdx int) (uint64, bool) {
	return it.skip.LastDocIDInBlock(blockIdx)
}

// SkipTo advances the iterator to the first entry with docID >= target.
// Returns false if no such entry exists.
func (it *PostingIter) SkipTo(target uint64) bool {
	if it.currDoc >= target && it.read > 0 {
		return true
	}

	// Try skip list for a coarse jump.
	if it.skip != nil && it.read < it.df {
		offset, base, blkIdx := it.skip.SkipTo(target)
		if offset > it.pos || (offset == 0 && blkIdx == 0 && it.read == 0) {
			// Only jump forward.
			if offset > it.pos {
				it.pos = offset
				it.base = base
				it.blockIdx = blkIdx - 1 // will be incremented by Next() → loadBlock()
				it.read = blkIdx * BlockSize
				it.bufPos = 0
				it.bufLen = 0
			}
		}
	}

	// Linear scan forward.
	for {
		if it.read >= it.df {
			return false
		}
		if !it.Next() {
			return false
		}
		if it.currDoc >= target {
			return true
		}
	}
}

// NewInvertedIndex is a convenience wrapper that builds an InvertedIndex
// directly from a slice of Documents, preserving backward compatibility.
func NewInvertedIndex(docs []types.Document) *InvertedIndex {
	b := NewIndexBuilder()
	defer b.Close()
	for _, d := range docs {
		b.Add(d.ID, Tokenize(d.Text))
	}
	return b.Build()
}

// AddTermFreqs adds a document by its pre-computed (term → TF) map and doc length.
// Used by MergeSegments to reconstruct documents from posting data without retokenizing.
func (b *IndexBuilder) AddTermFreqs(docID string, termTFs map[string]int, docLen int) {
	did := b.internDoc(docID, docLen)
	b.totalTokens += int64(docLen)
	b.N++

	need := len(termTFs) * 12
	if cap(b.addScratch) < need {
		b.addScratch = make([]byte, 0, need+128)
	}
	b.addScratch = b.addScratch[:0]
	for term, tf := range termTFs {
		tid := b.internTerm(term)
		b.addScratch = binary.LittleEndian.AppendUint32(b.addScratch, tid)
		b.addScratch = binary.LittleEndian.AppendUint32(b.addScratch, did)
		b.addScratch = binary.LittleEndian.AppendUint32(b.addScratch, uint32(tf))
	}
	if len(b.addScratch) > 0 {
		if _, err := b.postBuf.Write(b.addScratch); err != nil && b.lastErr == nil {
			b.lastErr = err
		}
		b.postCount += int64(len(termTFs))
	}
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
		return b.buildInternal(opts.UseFOR32)
	}
}

// BuildBM25F builds an InvertedIndex from per-field data accumulated via AddFields.
// Pseudo-TF = Σ_f w_f * tf(t,d,f) / (1 - b_f + b_f * len_f(d) / avglen_f) is
// computed here and stored scaled as an integer. Block-max impacts are computed
// using the BM25F saturation formula.
func (b *IndexBuilder) BuildBM25F(fields []FieldConfig) *InvertedIndex {
	const k1 = 1.2

	if b.N == 0 || b.fieldTFs == nil {
		return b.Build()
	}

	// 1. Compute avgFieldLen per field across all docs.
	fieldTotalLen := make(map[string]int64, len(fields))
	fieldDocCount := make(map[string]int, len(fields))
	for _, fieldMap := range b.fieldLens {
		for field, l := range fieldMap {
			fieldTotalLen[field] += int64(l)
			fieldDocCount[field]++
		}
	}
	avgFieldLen := make(map[string]float64, len(fields))
	for _, fc := range fields {
		cnt := fieldDocCount[fc.Name]
		if cnt > 0 {
			avgFieldLen[fc.Name] = float64(fieldTotalLen[fc.Name]) / float64(cnt)
		} else {
			avgFieldLen[fc.Name] = 1.0
		}
	}

	// 2. Build synthetic postings with scaled pseudo-TF.
	synPostings := make(map[string][]PostingEntry)
	for docID, fieldMap := range b.fieldTFs {
		// Collect all terms appearing in any configured field.
		allTerms := make(map[string]struct{})
		for _, fc := range fields {
			for term := range fieldMap[fc.Name] {
				allTerms[term] = struct{}{}
			}
		}

		for term := range allTerms {
			pseudoTF := 0.0
			for _, fc := range fields {
				tf := fieldMap[fc.Name][term]
				if tf == 0 {
					continue
				}
				fieldLen := float64(b.fieldLens[docID][fc.Name])
				avgLen := avgFieldLen[fc.Name]
				if avgLen <= 0 {
					avgLen = 1.0
				}
				normTF := float64(tf) / (1 - fc.B + fc.B*fieldLen/avgLen)
				pseudoTF += fc.Weight * normTF
			}
			scaledTF := int(math.Round(pseudoTF * BM25FScaleFactor))
			if scaledTF < 1 {
				scaledTF = 1
			}
			synPostings[term] = append(synPostings[term], PostingEntry{DocID: docID, TF: scaledTF})
		}
	}

	// 3. Sort docIDs, assign numeric IDs.
	allDocIDs := make([]string, 0, b.N)
	for docID := range b.docLengths {
		allDocIDs = append(allDocIDs, docID)
	}
	radixSortStrings(allDocIDs)
	docIDIndex := make(map[string]uint64, len(allDocIDs))
	for i, id := range allDocIDs {
		docIDIndex[id] = uint64(i)
	}

	// 4. Sort terms, assign ordinals.
	terms := make([]string, 0, len(synPostings))
	for term := range synPostings {
		terms = append(terms, term)
	}
	radixSortStrings(terms)
	termIndex := make(map[string]int, len(terms))
	for i, t := range terms {
		termIndex[t] = i
	}

	// 5. Encode posting lists using BM25F impact formula for block-max scores.
	nTerms := len(terms)
	offsets := make([]int64, nTerms+1)
	dfs := make([]int, nTerms)
	skipLists := make([]*SkipList, nTerms)
	ubs := make([]float64, nTerms)
	var data []byte

	packBuf := make([]byte, blockPackBufSize)

	for termOrd, term := range terms {
		entries := synPostings[term]
		sort.Slice(entries, func(i, j int) bool {
			return docIDIndex[entries[i].DocID] < docIDIndex[entries[j].DocID]
		})

		df := len(entries)
		dfs[termOrd] = df
		startOff := int64(len(data))

		var l0DocIDs []uint64
		var l0Offsets []int
		var l0Bases []uint64
		var l0Impact []float32
		var maxUB float64

		nBlocks := df / BlockSize
		tail := df % BlockSize
		var prevDocID uint64
		blockBase := int64(len(data))

		for blk := 0; blk < nBlocks; blk++ {
			blockStart := blk * BlockSize
			blockStartOff := int(int64(len(data)) - blockBase)

			docDeltas := make([]uint64, BlockSize)
			tfVals := make([]uint64, BlockSize)
			var blockMaxImpact float64

			for j := 0; j < BlockSize; j++ {
				entry := entries[blockStart+j]
				numID := docIDIndex[entry.DocID]
				docDeltas[j] = numID - prevDocID
				prevDocID = numID
				tfVals[j] = uint64(entry.TF)

				score := bm25fImpact(uint64(entry.TF), df, b.N, k1)
				if score > blockMaxImpact {
					blockMaxImpact = score
				}
				if score > maxUB {
					maxUB = score
				}
			}

			n := PackBlock(docDeltas, packBuf)
			data = append(data, packBuf[:n]...)
			n = PackBlock(tfVals, packBuf)
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
			tailStart := nBlocks * BlockSize
			for j := 0; j < tail; j++ {
				entry := entries[tailStart+j]
				numID := docIDIndex[entry.DocID]
				delta := numID - prevDocID
				prevDocID = numID
				data = AppendVarint(data, delta)
				data = AppendVarint(data, uint64(entry.TF))

				score := bm25fImpact(uint64(entry.TF), df, b.N, k1)
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

	docLengthsCopy := make(map[string]int, len(b.docLengths))
	for k, v := range b.docLengths {
		docLengthsCopy[k] = v
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
		avgDocLen:  float64(b.totalTokens) / float64(b.N),
		N:          b.N,
	}
}

// buildScaledPseudoTF builds an InvertedIndex from postings that already
// contain scaled pseudo-TFs (used during segment merge of BM25F segments).
// Block-max impacts are computed using the BM25F saturation formula.
func (b *IndexBuilder) buildScaledPseudoTF(k1 float64) *InvertedIndex {
	if b.N == 0 {
		return &InvertedIndex{
			termIndex:  make(map[string]int),
			docIDs:     []string{},
			docIDIndex: make(map[string]uint64),
			docLengths: make(map[string]int),
		}
	}
	if b.lastErr != nil {
		panic("IndexBuilder.buildScaledPseudoTF: posting write error: " + b.lastErr.Error())
	}

	// Flush and read back postings from the temp file.
	if err := b.postBuf.Flush(); err != nil {
		panic("IndexBuilder.buildScaledPseudoTF: flush: " + err.Error())
	}
	fileBytes := b.postCount * 12
	raw := make([]byte, fileBytes)
	if _, err := b.postFile.ReadAt(raw, 0); err != nil {
		panic("IndexBuilder.buildScaledPseudoTF: ReadAt: " + err.Error())
	}
	postings := make([]rawPosting, b.postCount)
	for i := range postings {
		off := i * 12
		postings[i].TermID = binary.LittleEndian.Uint32(raw[off:])
		postings[i].DocID = binary.LittleEndian.Uint32(raw[off+4:])
		postings[i].TF = binary.LittleEndian.Uint32(raw[off+8:])
	}

	avgDocLen := float64(b.totalTokens) / float64(b.N)
	nTerms := len(b.termList)
	nDocs := len(b.docList)

	// Assign numeric docIDs in insertion order.
	allDocIDs := make([]string, nDocs)
	copy(allDocIDs, b.docList)
	docIDIndex := make(map[string]uint64, nDocs)
	for i, id := range allDocIDs {
		docIDIndex[id] = uint64(i)
	}

	// Sort terms for FST.
	terms := make([]string, nTerms)
	copy(terms, b.termList)
	radixSortStrings(terms)
	termIndex := make(map[string]int, nTerms)
	for i, t := range terms {
		termIndex[t] = i
	}

	termIngestID := make(map[string]uint32, nTerms)
	for i, t := range b.termList {
		termIngestID[t] = uint32(i)
	}

	// Count sort pre-pass for group boundaries.
	termStart := make([]int, nTerms+1)
	for _, p := range postings {
		termStart[p.TermID+1]++
	}
	for i := 1; i <= nTerms; i++ {
		termStart[i] += termStart[i-1]
	}

	// Sort postings by (TermID, DocID) — O(N) radix sort.
	radixSortPostings(postings)

	offsets := make([]int64, nTerms+1)
	dfs := make([]int, nTerms)
	skipLists := make([]*SkipList, nTerms)
	ubs := make([]float64, nTerms)
	var data []byte

	packBuf := make([]byte, blockPackBufSize)
	docDeltas := make([]uint64, BlockSize)
	tfVals := make([]uint64, BlockSize)

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

		nBlocks := df / BlockSize
		tail := df % BlockSize
		var prevDocID uint64
		blockBase := int64(len(data))

		for blk := 0; blk < nBlocks; blk++ {
			blockStart := blk * BlockSize
			blockStartOff := int(int64(len(data)) - blockBase)

			var blockMaxImpact float64

			for j := 0; j < BlockSize; j++ {
				p := termPostings[blockStart+j]
				numID := uint64(p.DocID)
				docDeltas[j] = numID - prevDocID
				prevDocID = numID
				tfVals[j] = uint64(p.TF)

				score := bm25fImpact(uint64(p.TF), df, b.N, k1)
				if score > blockMaxImpact {
					blockMaxImpact = score
				}
				if score > maxUB {
					maxUB = score
				}
			}

			n := PackBlock(docDeltas, packBuf)
			data = append(data, packBuf[:n]...)
			n = PackBlock(tfVals, packBuf)
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
			tailStart := nBlocks * BlockSize
			for j := 0; j < tail; j++ {
				p := termPostings[tailStart+j]
				numID := uint64(p.DocID)
				delta := numID - prevDocID
				prevDocID = numID
				data = AppendVarint(data, delta)
				data = AppendVarint(data, uint64(p.TF))

				score := bm25fImpact(uint64(p.TF), df, b.N, k1)
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

	docLengthsCopy := make(map[string]int, nDocs)
	for i, id := range b.docList {
		docLengthsCopy[id] = b.docLens[i]
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
		N:          b.N,
	}
}

// Terms returns the sorted term list of the index.
func (idx *InvertedIndex) Terms() []string {
	return idx.terms
}

// DF returns the document frequency for the given term, or 0 if absent.
func (idx *InvertedIndex) DF(term string) int {
	ord, ok := idx.termOrdinalFor(term)
	if !ok {
		return 0
	}
	return idx.dfs[ord]
}

// RawPostingBytes returns the raw posting bytes for the term at the given ordinal.
// The bytes are in the Phase 1 in-memory format (FOR-delta blocks + LEB128 tail).
func (idx *InvertedIndex) RawPostingBytes(ord int) []byte {
	if ord < 0 || ord >= len(idx.terms) {
		return nil
	}
	start := idx.offsets[ord]
	end := idx.offsets[ord+1]
	return idx.data[start:end]
}

// HasTerm reports whether the document with the given string docID contains term.
// Uses SkipTo on the posting list for an O(log N) membership check.
func (idx *InvertedIndex) HasTerm(term, docID string) bool {
	numID, ok := idx.docIDIndex[docID]
	if !ok {
		return false
	}
	it := idx.Iterator(term)
	if it == nil {
		return false
	}
	it.SkipTo(numID)
	return it.DocID() == numID
}

// RawSkipBytes serialises the skip list for the given ordinal into a compact
// binary representation for embedding in .seg files.
//
// Format per L0 entry (28 bytes):
//
//	[8] lastDocID
//	[8] byteOffset
//	[8] deltaBase
//	[4] blockMaxImpact (float32)
//
// Followed by L1 entries (16 bytes each):
//
//	[8] lastDocID
//	[8] l0Index
//
// Layout: [4 nL0][4 nL1][nL0 * 28 bytes][nL1 * 16 bytes]
func (idx *InvertedIndex) RawSkipBytes(ord int) []byte {
	if ord < 0 || ord >= len(idx.skipLists) || idx.skipLists[ord] == nil {
		return nil
	}
	sl := idx.skipLists[ord]
	nL0 := len(sl.l0DocIDs)
	nL1 := len(sl.l1DocIDs)

	buf := make([]byte, 8+nL0*28+nL1*16)
	off := 0
	putU32 := func(v uint32) { binary.LittleEndian.PutUint32(buf[off:], v); off += 4 }
	putU64 := func(v uint64) { binary.LittleEndian.PutUint64(buf[off:], v); off += 8 }
	putF32 := func(v float32) {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(v))
		off += 4
	}

	putU32(uint32(nL0))
	putU32(uint32(nL1))
	for i := 0; i < nL0; i++ {
		putU64(sl.l0DocIDs[i])
		putU64(uint64(sl.l0Offsets[i]))
		putU64(sl.l0Bases[i])
		putF32(sl.l0Impact[i])
	}
	for i := 0; i < nL1; i++ {
		putU64(sl.l1DocIDs[i])
		putU64(uint64(sl.l1L0Idx[i]))
	}
	return buf
}
