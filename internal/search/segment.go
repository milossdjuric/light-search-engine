package search

import (
	"bufio"
	"bytes"
	"container/heap"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"

	bloom "github.com/bits-and-blooms/bloom/v3"
	"github.com/blevesearch/vellum"
	lz4 "github.com/pierrec/lz4/v4"

	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/pkg/types"
)

// segMagic is the 4-byte magic number at the start of every .seg file.
var segMagic = [4]byte{'S', 'E', 'G', '2'}

// segVersion is the current (uncompressed) format version.
const segVersion = uint32(2)

// segVersionLZ4 is the format version for LZ4-compressed segments.
// Layout identical to v2 except PostingData is a single LZ4 frame.
// LZ4 decompresses at ~3 GB/s — preferred for latency-sensitive segment loading.
const segVersionLZ4 = uint32(4)

// segVersionFOR32 is the format version for FOR-delta uint32 encoded segments.
// Full blocks use PackFOR32 (bit-packed uint32 deltas/tfs) instead of intcomp.
// Tail entries remain LEB128-encoded (same as v2/v4).
// Decompresses faster than intcomp on modern CPUs, especially with SIMD.
const segVersionFOR32 = uint32(6)

// SegmentWriteOptions controls optional features when writing a segment file.
type SegmentWriteOptions struct {
	// Compression is "" for no compression, or "lz4" for LZ4 (~3 GB/s decompress).
	Compression string
	// BloomFPRate is the desired false-positive rate for the bloom filter sidecar.
	// 0 disables bloom filter writing.
	BloomFPRate float64
	// BuildOpts controls BM25F vs plain build path when building from an IndexBuilder.
	BuildOpts index.BuildOptions
	// UseFOR32, when true, writes a v6 segment using PackFOR32 for full blocks.
	// Sets BuildOpts.UseFOR32 internally before calling BuildWithOptions.
	UseFOR32 bool
	// SkipStoredFields, when true, omits the .seg.fld sidecar even if source segments have one.
	SkipStoredFields bool
}

// SegmentHeader is the 37-byte fixed header at byte 0 of a .seg file.
//
//	[4]  magic
//	[4]  version
//	[8]  docCount
//	[8]  fstOffset    — byte offset of the FSTSection from file start
//	[8]  termOffset   — byte offset of the TermInfoStore
//	[8]  postOffset   — byte offset of the PostingData
//	[4]  (reserved / checksum placeholder)
//
// Note: skipOffset is embedded inside PostingData; each posting list
// stores its skip data immediately after the posting bytes.
const headerSize = 4 + 4 + 8 + 8 + 8 + 8 + 4 // 40 bytes

// TermInfo holds per-term metadata stored in the TermInfoStore section.
type TermInfo struct {
	DF             uint32
	UB             float32 // per-term BM25 upper bound
	PostingsOffset uint64  // offset within PostingData section
	PostingsLen    uint64  // length of posting data (including inline skip)
}

// termInfoSize is the fixed size of one serialised TermInfo entry.
const termInfoSize = 4 + 4 + 8 + 8 // 24 bytes

// storedFieldEntry is one entry in the .seg.fld index section.
type storedFieldEntry struct {
	docID      string
	dataOffset uint32 // byte offset within the data section
	textLen    uint32
}

// fldMagic is the 4-byte magic for .seg.fld sidecar files.
var fldMagic = [4]byte{'S', 'F', 'L', 'D'}

const fldVersion = byte(1)

// Segment is an immutable, on-disk inverted index segment.
type Segment struct {
	docCount    uint64
	version     uint32             // segment format version: 2 (plain), 4 (lz4), 6 (for32)
	fst         *vellum.FST
	terms       []string   // term ordinal → string
	infos       []TermInfo
	docLens     []uint32   // docLens[numericID] = token count
	docIDs      []string   // docIDs[numericID] = string doc ID
	docIDToNum  map[string]uint64
	postData    []byte
	totalTokens uint64             // sum of all docLens; used for AvgDocLen
	bloom       *bloom.BloomFilter // nil if no sidecar or disabled

	fldPath      string             // path to .seg.fld sidecar; empty if absent
	fldIndex     []storedFieldEntry // sorted by docID; loaded at segment load time
	fldDataStart int64              // byte offset within .seg.fld where data section begins

	// mmapData is the full mmap'd file backing postData and fst bytes.
	// Non-nil only on Unix. A GC finalizer calls munmapFile when the segment
	// becomes unreachable, so no explicit Close() is needed at merge time —
	// in-flight searches that still hold a *Segment reference keep it alive.
	mmapData []byte

	// blockCache caches decoded posting list blocks to avoid redundant
	// FOR-delta/intcomp decoding on repeated accesses by concurrent queries.
	blockCache *index.BlockCache

	// idfCache caches IDF values keyed by "term\x00scorerType". Since segments
	// are immutable (df and N are fixed after flush), IDF is constant and never
	// needs invalidation.
	idfCache sync.Map
}

// Prefetch advises the kernel to load all segment pages into the page cache.
// The call is non-blocking; actual prefetch happens asynchronously. Call after
// LoadSegment during startup warmup to avoid cold-load latency on first search.
func (s *Segment) Prefetch() {
	prefetchSegmentData(s.mmapData)
}

// WarmTopKTerms touches the posting data pages for the k highest-DF terms,
// ensuring they are resident in the OS page cache before queries arrive.
// This is the targeted Lucene-style warmup: high-DF terms appear in most
// queries and cover the hot posting lists with minimal I/O.
func (s *Segment) WarmTopKTerms(k int) {
	if k <= 0 || len(s.infos) == 0 || len(s.postData) == 0 {
		return
	}
	// Build a slice of (df, ordinal) pairs and partial-sort to find top-k.
	type dfOrd struct {
		df  uint32
		ord int
	}
	order := make([]dfOrd, len(s.infos))
	for i, info := range s.infos {
		order[i] = dfOrd{df: info.DF, ord: i}
	}
	// Sort descending by DF; take only the top-k.
	sort.Slice(order, func(i, j int) bool { return order[i].df > order[j].df })
	if len(order) > k {
		order = order[:k]
	}
	// Touch every 4 KB page of each top-k term's posting list.
	var acc byte
	for _, di := range order {
		info := s.infos[di.ord]
		end := info.PostingsOffset + info.PostingsLen
		if end > uint64(len(s.postData)) {
			continue
		}
		data := s.postData[info.PostingsOffset:end]
		for i := 0; i < len(data); i += 4096 {
			acc ^= data[i]
		}
	}
	_ = acc
}


// WarmFull warms the entire segment into the OS page cache.
//
// It covers the full mmapData region — FST, term metadata, skip lists, doc
// lengths, and posting data — not just postData. This matters because queries
// access all of these on every search, not just the posting lists.
//
// Strategy: call MADV_WILLNEED first so the kernel issues async prefetch at
// full SSD sequential bandwidth (~500 MB/s), then stride-scan to
// synchronously wait for all pages to land before returning. The stride scan
// after WILLNEED is ~10× faster than a cold stride scan because the kernel
// has already queued the reads.
func (s *Segment) WarmFull() {
	if len(s.mmapData) == 0 {
		return
	}
	// Switch to sequential mode so the kernel issues aggressive read-ahead
	// during the stride scan — fewer individual I/O ops than random mode.
	setMadviseSequential(s.mmapData)
	prefetchSegmentData(s.mmapData) // MADV_WILLNEED: kick off async prefetch now
	var acc byte
	for i := 0; i < len(s.mmapData); i += 4096 {
		acc ^= s.mmapData[i]
	}
	_ = acc
	// All pages are now in RAM. Restore random-access mode for query serving,
	// then synchronously collapse to 2 MB huge pages (Linux 6.1+, silent no-op
	// on older kernels). Pages are physically contiguous after sequential load,
	// so collapse succeeds quickly and all subsequent TLB misses hit 2 MB entries.
	setMadviseRandom(s.mmapData)
	collapseHugepages(s.mmapData)
}

// SizeBytes returns the size of the mmap'd backing data in bytes.
// Returns 0 for in-memory-only (non-mmap'd) segments.
func (s *Segment) SizeBytes() int64 {
	return int64(len(s.mmapData))
}

// TryMlock attempts to lock the segment's mmap pages in RAM using mlock(2).
// Prevents OS eviction under memory pressure. Requires CAP_IPC_LOCK or
// adequate RLIMIT_MEMLOCK; logs a debug message and returns nil on EPERM.
func (s *Segment) TryMlock() error {
	if err := tryMlockData(s.mmapData); err != nil {
		slog.Debug("mlock unavailable, pages may be evicted under memory pressure",
			"err", err)
		return nil // non-fatal
	}
	return nil
}


// WriteSegment

// WriteSegment serialises an IndexBuilder snapshot to path as a .seg v2 file.
// It calls Build() internally; the builder must not be used after this call.
func WriteSegment(path string, b *index.IndexBuilder) error {
	return WriteSegmentWithOptions(path, b, SegmentWriteOptions{})
}

// WriteSegmentWithOptions serialises an IndexBuilder snapshot to path with the
// given options (compression, bloom filter). BuildWithOptions() is called internally,
// allowing BM25F pseudo-TF encoding when opts.BuildOpts.Fields is non-empty.
// When opts.UseFOR32 is true, BuildOpts.UseFOR32 is set and a v6 segment is written.
func WriteSegmentWithOptions(path string, b *index.IndexBuilder, opts SegmentWriteOptions) error {
	_, err := WriteSegmentWithIndex(path, b, opts)
	return err
}

// WriteSegmentWithIndex is like WriteSegmentWithOptions but also returns the
// built InvertedIndex so callers can reuse it without a second file read.
func WriteSegmentWithIndex(path string, b *index.IndexBuilder, opts SegmentWriteOptions) (*index.InvertedIndex, error) {
	if opts.UseFOR32 && opts.Compression != "lz4" {
		opts.BuildOpts.UseFOR32 = true
	}
	idx := b.BuildWithOptions(opts.BuildOpts)
	return idx, writeFromIndexWithOptions(path, idx, opts)
}


// writeFromIndexWithOptions writes an already-built InvertedIndex with options.
func writeFromIndexWithOptions(path string, idx *index.InvertedIndex, opts SegmentWriteOptions) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("WriteSegment create %s: %w", path, err)
	}
	defer f.Close()

	// 1. Build FST from sorted terms.
	// vellum.Builder requires strictly ascending key order; Build() already
	// sorts terms, so we just iterate.
	var fstBuf bytes.Buffer
	fstBuilder, err := vellum.New(&fstBuf, nil)
	if err != nil {
		return fmt.Errorf("WriteSegment vellum.New: %w", err)
	}
	terms := idx.Terms()
	for ord, term := range terms {
		if err := fstBuilder.Insert([]byte(term), uint64(ord)); err != nil {
			return fmt.Errorf("WriteSegment fst.Insert %q: %w", term, err)
		}
	}
	if err := fstBuilder.Close(); err != nil {
		return fmt.Errorf("WriteSegment fst.Close: %w", err)
	}
	fstData := fstBuf.Bytes()

	// 2. Encode DocLengths section.
	docCount := uint64(idx.DocCount())
	docLenBuf := make([]byte, 8+docCount*4) // 8-byte count header + 4 bytes/doc
	binary.LittleEndian.PutUint64(docLenBuf[:8], docCount)
	for i := uint64(0); i < docCount; i++ {
		docStr := idx.DocStringID(i)
		dl := uint32(idx.DocLen(docStr))
		binary.LittleEndian.PutUint32(docLenBuf[8+i*4:], dl)
	}

	// 3. Encode DocID strings section.
	// Format: [8 count][for each doc: 4-byte len + utf8 bytes]
	var docIDBuf bytes.Buffer
	writeUint64(&docIDBuf, docCount)
	for i := uint64(0); i < docCount; i++ {
		s := idx.DocStringID(i)
		writeUint32(&docIDBuf, uint32(len(s)))
		docIDBuf.WriteString(s)
	}
	docIDData := docIDBuf.Bytes()

	// 4. Encode posting data + build TermInfo table.
	nTerms := len(terms)
	infos := make([]TermInfo, nTerms)
	var postBuf bytes.Buffer

	for ord, term := range terms {
		iter := idx.Iterator(term)
		if iter == nil {
			continue
		}
		off := uint64(postBuf.Len())
		df := idx.DF(term)
		ub := idx.TermUB(term)

		// Re-use raw posting bytes from the in-memory index directly.
		rawPost := idx.RawPostingBytes(ord)
		skipData := idx.RawSkipBytes(ord)

		// Inline format: [8 skipLen][skipData][postData]
		// This lets LoadSegment split skip from posting bytes easily.
		skipLen := uint64(len(skipData))
		var lenBuf [8]byte
		binary.LittleEndian.PutUint64(lenBuf[:], skipLen)
		postBuf.Write(lenBuf[:])
		postBuf.Write(skipData)
		postBuf.Write(rawPost)

		infos[ord] = TermInfo{
			DF:             uint32(df),
			UB:             float32(ub),
			PostingsOffset: off,
			PostingsLen:    uint64(8 + len(skipData) + len(rawPost)),
		}
	}
	postData := postBuf.Bytes()

	// 5a. Optionally compress postData.
	// "lz4" → LZ4 frame format (v4): ~3 GB/s decompress, ~1.3× ratio
	useLZ4 := opts.Compression == "lz4"
	if useLZ4 {
		var buf bytes.Buffer
		w := lz4.NewWriter(&buf)
		if _, err := w.Write(postData); err != nil {
			return fmt.Errorf("WriteSegment lz4 write: %w", err)
		}
		if err := w.Close(); err != nil {
			return fmt.Errorf("WriteSegment lz4 close: %w", err)
		}
		postData = buf.Bytes()
	}

	// 5b. Compute section offsets.
	// Layout:
	//   [headerSize bytes]  Header
	//   [4 + fstLen]        FSTSection (4-byte length prefix + fstData)
	//   [nTerms*24]         TermInfoStore
	//   [docLenBuf]         DocLengths
	//   [docIDData]         DocID strings
	//   [postData]          PostingData (possibly lz4-compressed)
	fstOff := uint64(headerSize)
	termOff := fstOff + 4 + uint64(len(fstData))
	docLenOff := termOff + uint64(nTerms)*termInfoSize
	docIDOff := docLenOff + uint64(len(docLenBuf))
	postOff := docIDOff + uint64(len(docIDData))

	// 6. Write header.
	hdr := make([]byte, headerSize)
	copy(hdr[0:4], segMagic[:])
	fileVersion := segVersion
	if useLZ4 {
		fileVersion = segVersionLZ4
	} else if opts.UseFOR32 {
		fileVersion = segVersionFOR32
	}
	binary.LittleEndian.PutUint32(hdr[4:8], fileVersion)
	binary.LittleEndian.PutUint64(hdr[8:16], docCount)
	binary.LittleEndian.PutUint64(hdr[16:24], fstOff)
	binary.LittleEndian.PutUint64(hdr[24:32], termOff)
	binary.LittleEndian.PutUint64(hdr[32:40], postOff)
	// The version field (bytes [4:8]) encodes the compression type:
	//   v2 = uncompressed, v4 = lz4.

	// Write everything in order.
	w := io.Writer(f)
	if _, err := w.Write(hdr[:headerSize]); err != nil {
		return err
	}
	// FST section
	var fstLenBuf [4]byte
	binary.LittleEndian.PutUint32(fstLenBuf[:], uint32(len(fstData)))
	if _, err := f.Write(fstLenBuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(fstData); err != nil {
		return err
	}
	// TermInfoStore
	for _, ti := range infos {
		var tibuf [termInfoSize]byte
		binary.LittleEndian.PutUint32(tibuf[0:4], ti.DF)
		binary.LittleEndian.PutUint32(tibuf[4:8], math.Float32bits(ti.UB))
		binary.LittleEndian.PutUint64(tibuf[8:16], ti.PostingsOffset)
		binary.LittleEndian.PutUint64(tibuf[16:24], ti.PostingsLen)
		if _, err := f.Write(tibuf[:]); err != nil {
			return err
		}
	}
	// DocLengths
	if _, err := f.Write(docLenBuf); err != nil {
		return err
	}
	// DocID strings
	if _, err := f.Write(docIDData); err != nil {
		return err
	}
	// PostingData
	if _, err := f.Write(postData); err != nil {
		return err
	}

	_ = docLenOff
	_ = docIDOff
	_ = postOff
	_ = termOff

	if err := f.Sync(); err != nil {
		return err
	}

	// 8. Write bloom filter sidecar.
	if opts.BloomFPRate > 0 && len(terms) > 0 {
		bf := bloom.NewWithEstimates(uint(len(terms)), opts.BloomFPRate)
		for _, term := range terms {
			bf.AddString(term)
		}
		bloomPath := path + ".bloom"
		bf2, err2 := os.Create(bloomPath)
		if err2 == nil {
			_, _ = bf.WriteTo(bf2)
			_ = bf2.Close()
		}
		// Bloom sidecar errors are non-fatal — search still works without it.
	}
	return nil
}

// LoadSegment reads a .seg file from path into a Segment.
// It also loads the bloom filter sidecar (<path>.bloom) if present.
func LoadSegment(path string) (*Segment, error) {
	// Map the segment file into the OS page cache instead of copying it to the
	// Go heap. On Unix this uses mmap(MAP_SHARED|PROT_READ); on other platforms
	// it falls back to os.ReadFile. The mapping is released by a GC finalizer
	// when the *Segment becomes unreachable — this is safe because in-flight
	// searches hold a *Segment reference that keeps the mapping alive.
	data, err := mmapFile(path)
	if err != nil {
		return nil, fmt.Errorf("LoadSegment %s: %w", path, err)
	}
	seg, err := parseSegment(data)
	if err != nil {
		_ = munmapFile(data)
		return nil, err
	}
	if len(data) > 0 {
		seg.mmapData = data
		// parseSegment already did a sequential read-through (header, FST, term
		// metadata). Now switch to random-access mode so posting list traversals
		// (which jump around the file) don't trigger useless kernel read-ahead.
		setMadviseRandom(data)
		hintHugepage(data)  // 2 MB THP pages — reduces TLB pressure on posting list traversal
		hintDontDump(data)  // exclude from core dumps; data is on disk and reloadable
		runtime.SetFinalizer(seg, func(s *Segment) {
			_ = munmapFile(s.mmapData)
		})
	}
	// Load bloom sidecar if it exists (non-fatal if missing).
	if f, ferr := os.Open(path + ".bloom"); ferr == nil {
		var bf bloom.BloomFilter
		if _, rerr := bf.ReadFrom(f); rerr == nil {
			seg.bloom = &bf
		}
		f.Close()
	}
	// Attempt to load stored fields sidecar (optional; absent for old segments).
	_ = LoadStoredFields(seg, strings.TrimSuffix(path, ".seg")+".seg.fld")
	seg.blockCache = index.NewBlockCache(0) // 0 = default size (4096 entries ≈ 8 MB)
	return seg, nil
}

// GetText returns the original text for docID from the stored fields sidecar.
// Returns ("", false) if no sidecar exists or the doc is not found.
func (s *Segment) GetText(docID string) (string, bool) {
	if len(s.fldIndex) == 0 {
		return "", false
	}
	i := sort.Search(len(s.fldIndex), func(i int) bool {
		return s.fldIndex[i].docID >= docID
	})
	if i >= len(s.fldIndex) || s.fldIndex[i].docID != docID {
		return "", false
	}
	entry := s.fldIndex[i]
	if entry.textLen == 0 {
		return "", true
	}

	f, err := os.Open(s.fldPath)
	if err != nil {
		return "", false
	}
	defer f.Close()

	buf := make([]byte, entry.textLen)
	if _, err := f.ReadAt(buf, s.fldDataStart+int64(entry.dataOffset)); err != nil {
		return "", false
	}
	return string(buf), true
}

// WriteStoredFields writes a .seg.fld sidecar file mapping docID → original text.
// texts is a map from docID to document text.
// The index section is sorted by docID to allow binary search.
func WriteStoredFields(path string, texts map[string]string) error {
	if len(texts) == 0 {
		return nil
	}

	// Sort docIDs for binary-searchable index.
	docIDs := make([]string, 0, len(texts))
	for id := range texts {
		docIDs = append(docIDs, id)
	}
	sort.Strings(docIDs)

	// Compute per-entry data offsets.
	type entry struct {
		docID      string
		dataOffset uint32
		textLen    uint32
	}
	entries := make([]entry, len(docIDs))
	var offset uint32
	for i, id := range docIDs {
		t := texts[id]
		entries[i] = entry{id, offset, uint32(len(t))}
		offset += uint32(len(t))
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("WriteStoredFields create %s: %w", path, err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1<<20)

	// Header: magic + version + numEntries.
	if _, err := w.Write(fldMagic[:]); err != nil {
		return err
	}
	if err := w.WriteByte(fldVersion); err != nil {
		return err
	}
	var tmp [4]byte
	binary.LittleEndian.PutUint32(tmp[:], uint32(len(entries)))
	if _, err := w.Write(tmp[:]); err != nil {
		return err
	}

	// Index section.
	for _, e := range entries {
		if len(e.docID) > 255 {
			return fmt.Errorf("WriteStoredFields: docID too long (%d > 255): %s", len(e.docID), e.docID)
		}
		if err := w.WriteByte(byte(len(e.docID))); err != nil {
			return err
		}
		if _, err := w.WriteString(e.docID); err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(tmp[:], e.dataOffset)
		if _, err := w.Write(tmp[:]); err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(tmp[:], e.textLen)
		if _, err := w.Write(tmp[:]); err != nil {
			return err
		}
	}

	// Data section: concatenated texts.
	for _, id := range docIDs {
		if _, err := w.WriteString(texts[id]); err != nil {
			return err
		}
	}

	return w.Flush()
}

// LoadStoredFields reads the index section of a .seg.fld sidecar into memory.
// The data section is read on demand in GetText via ReadAt.
// Returns nil error if path does not exist (sidecar is optional).
func LoadStoredFields(seg *Segment, path string) error {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil // sidecar absent — older segment or stored fields disabled
	}
	if err != nil {
		return fmt.Errorf("LoadStoredFields open %s: %w", path, err)
	}
	defer f.Close()

	// Read header: magic(4) + version(1) + numEntries(4) = 9 bytes.
	var hdr [9]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		return fmt.Errorf("LoadStoredFields read header %s: %w", path, err)
	}
	if hdr[0] != fldMagic[0] || hdr[1] != fldMagic[1] || hdr[2] != fldMagic[2] || hdr[3] != fldMagic[3] {
		return fmt.Errorf("LoadStoredFields %s: bad magic", path)
	}
	// hdr[4] is version — ignore for forward compat
	n := int(binary.LittleEndian.Uint32(hdr[5:9]))

	// Read index section.
	fldIdx := make([]storedFieldEntry, n)
	for i := 0; i < n; i++ {
		// docIDLen uint8
		lenBuf := make([]byte, 1)
		if _, err := io.ReadFull(f, lenBuf); err != nil {
			return fmt.Errorf("LoadStoredFields read docIDLen %s[%d]: %w", path, i, err)
		}
		docIDLen := int(lenBuf[0])
		docIDBuf := make([]byte, docIDLen)
		if _, err := io.ReadFull(f, docIDBuf); err != nil {
			return fmt.Errorf("LoadStoredFields read docID %s[%d]: %w", path, i, err)
		}
		var offLen [8]byte
		if _, err := io.ReadFull(f, offLen[:]); err != nil {
			return fmt.Errorf("LoadStoredFields read offLen %s[%d]: %w", path, i, err)
		}
		fldIdx[i] = storedFieldEntry{
			docID:      string(docIDBuf),
			dataOffset: binary.LittleEndian.Uint32(offLen[0:4]),
			textLen:    binary.LittleEndian.Uint32(offLen[4:8]),
		}
	}

	// Data section starts at current file position.
	dataStart, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("LoadStoredFields seek %s: %w", path, err)
	}

	seg.fldPath = path
	seg.fldIndex = fldIdx
	seg.fldDataStart = dataStart
	return nil
}


func parseSegment(data []byte) (*Segment, error) {
	if len(data) < headerSize {
		return nil, errors.New("segment file too small")
	}
	if [4]byte(data[0:4]) != segMagic {
		return nil, errors.New("invalid segment magic")
	}
	ver := binary.LittleEndian.Uint32(data[4:8])
	if ver != segVersion && ver != segVersionLZ4 && ver != segVersionFOR32 {
		return nil, fmt.Errorf("unsupported segment version %d", ver)
	}
	docCount := binary.LittleEndian.Uint64(data[8:16])
	fstOff := binary.LittleEndian.Uint64(data[16:24])
	termOff := binary.LittleEndian.Uint64(data[24:32])
	postOff := binary.LittleEndian.Uint64(data[32:40])

	// FST section.
	fstLen := binary.LittleEndian.Uint32(data[fstOff : fstOff+4])
	fstData := data[fstOff+4 : fstOff+4+uint64(fstLen)]
	fst, err := vellum.Load(fstData)
	if err != nil {
		return nil, fmt.Errorf("LoadSegment vellum.Load: %w", err)
	}

	// Enumerate all terms from FST to build ordinal map.
	var terms []string
	itr, err := fst.Iterator(nil, nil)
	for err == nil {
		key, _ := itr.Current()
		terms = append(terms, string(key))
		err = itr.Next()
	}
	if !errors.Is(err, vellum.ErrIteratorDone) && err != nil {
		return nil, fmt.Errorf("LoadSegment fst iterate: %w", err)
	}
	nTerms := len(terms)

	// TermInfoStore.
	infos := make([]TermInfo, nTerms)
	for i := 0; i < nTerms; i++ {
		off := termOff + uint64(i)*termInfoSize
		infos[i] = TermInfo{
			DF:             binary.LittleEndian.Uint32(data[off : off+4]),
			UB:             math.Float32frombits(binary.LittleEndian.Uint32(data[off+4 : off+8])),
			PostingsOffset: binary.LittleEndian.Uint64(data[off+8 : off+16]),
			PostingsLen:    binary.LittleEndian.Uint64(data[off+16 : off+24]),
		}
	}

	// DocLengths + DocIDs.
	docLenOff := termOff + uint64(nTerms)*termInfoSize
	if docLenOff+8 > uint64(len(data)) {
		return nil, errors.New("segment: truncated doclen section")
	}
	storedDocCount := binary.LittleEndian.Uint64(data[docLenOff : docLenOff+8])
	if storedDocCount != docCount {
		return nil, errors.New("segment: docCount mismatch")
	}
	docLens := make([]uint32, docCount)
	for i := uint64(0); i < docCount; i++ {
		base := docLenOff + 8 + i*4
		docLens[i] = binary.LittleEndian.Uint32(data[base : base+4])
	}

	docIDOff := docLenOff + 8 + docCount*4
	if docIDOff+8 > uint64(len(data)) {
		return nil, errors.New("segment: truncated docid section")
	}
	storedDocCount2 := binary.LittleEndian.Uint64(data[docIDOff : docIDOff+8])
	if storedDocCount2 != docCount {
		return nil, errors.New("segment: docID count mismatch")
	}
	docIDs := make([]string, docCount)
	docIDToNum := make(map[string]uint64, docCount)
	pos := docIDOff + 8
	var totalTokens uint64
	for i := uint64(0); i < docCount; i++ {
		slen := uint64(binary.LittleEndian.Uint32(data[pos : pos+4]))
		pos += 4
		docIDs[i] = string(data[pos : pos+slen])
		docIDToNum[docIDs[i]] = i
		pos += slen
		totalTokens += uint64(docLens[i])
	}

	// PostingData.
	rawPost := data[postOff:]
	var postData []byte
	switch ver {
	case segVersionLZ4:
		r := lz4.NewReader(bytes.NewReader(rawPost))
		decompressed, derr := io.ReadAll(r)
		if derr != nil {
			return nil, fmt.Errorf("LoadSegment lz4 decompress: %w", derr)
		}
		postData = decompressed
	default:
		postData = rawPost
	}

	return &Segment{
		docCount:    docCount,
		version:     ver,
		fst:         fst,
		terms:       terms,
		infos:       infos,
		docLens:     docLens,
		docIDs:      docIDs,
		docIDToNum:  docIDToNum,
		postData:    postData,
		totalTokens: totalTokens,
	}, nil
}


// Segment query helpers

// DocCount returns the number of documents in this segment.
func (s *Segment) DocCount() int { return int(s.docCount) }

// ordinalFor returns the ordinal for term via FST lookup.
func (s *Segment) ordinalFor(term string) (int, bool) {
	ord, exists, err := s.fst.Get([]byte(term))
	if err != nil || !exists {
		return 0, false
	}
	return int(ord), true
}

// TermUB returns the precomputed per-term upper bound, or 0 if absent.
func (s *Segment) TermUB(term string) float64 {
	ord, ok := s.ordinalFor(term)
	if !ok {
		return 0
	}
	return float64(s.infos[ord].UB)
}

// DF returns the document frequency for term, or 0 if absent.
func (s *Segment) DF(term string) int {
	ord, ok := s.ordinalFor(term)
	if !ok {
		return 0
	}
	return int(s.infos[ord].DF)
}


// MergeSegments  — k-way merge via min-heap over sorted term iterators

// mergeTermIter is one input stream for the k-way term merge.
type mergeTermIter struct {
	seg     *Segment
	segIdx  int    // index in the input slice (used for duplicate docID tie-break)
	terms   []string
	termPos int
}

func (m *mergeTermIter) done() bool        { return m.termPos >= len(m.terms) }
func (m *mergeTermIter) currentTerm() string { return m.terms[m.termPos] }
func (m *mergeTermIter) advance()            { m.termPos++ }

// mergeHeapItem is pushed onto the k-way merge heap.
type mergeHeapItem struct {
	term    string
	iterIdx int
}

type mergeHeap []mergeHeapItem

func (h mergeHeap) Len() int      { return len(h) }
func (h mergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h mergeHeap) Less(i, j int) bool {
	if h[i].term != h[j].term {
		return h[i].term < h[j].term
	}
	return h[i].iterIdx < h[j].iterIdx
}
func (h *mergeHeap) Push(x interface{}) { *h = append(*h, x.(mergeHeapItem)) }
func (h *mergeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// MergeSegments merges segments into a single new .seg file at outPath.
// For duplicate docIDs (from crash-replay), the posting from the highest
// segIdx is kept.
func MergeSegments(outPath string, segments []*Segment) error {
	return MergeSegmentsWithOptions(outPath, segments, SegmentWriteOptions{})
}

// MergeSegmentsWithOptions merges segments with the given write options.
func MergeSegmentsWithOptions(outPath string, segments []*Segment, opts SegmentWriteOptions) error {
	if len(segments) == 0 {
		return errors.New("MergeSegments: no input segments")
	}

	// Phase 1: build unified doc ID space.
	// For duplicate docIDs (crash-replay), the highest-segIdx segment wins.
	type docMeta struct {
		docLen uint32
		segIdx int
	}
	totalDocHint := 0
	for _, seg := range segments {
		totalDocHint += int(seg.docCount)
	}
	docMap := make(map[string]docMeta, totalDocHint)
	for sIdx, seg := range segments {
		for i := uint64(0); i < seg.docCount; i++ {
			strID := seg.docIDs[i]
			if existing, ok := docMap[strID]; !ok || sIdx > existing.segIdx {
				docMap[strID] = docMeta{docLen: seg.docLens[i], segIdx: sIdx}
			}
		}
	}

	// Phase 2: pre-register all unique docs.
	builder := index.NewIndexBuilder()
	defer builder.Close()
	for strID, meta := range docMap {
		builder.PreRegisterDoc(strID, int(meta.docLen))
	}

	// Phase 3: term-major k-way merge via min-heap.
	// Allocate scratch once; reuse across all decodePostings calls.
	docScratch := make([]uint64, index.BlockSize)
	tfScratch := make([]uint64, index.BlockSize)

	iters := make([]*mergeTermIter, 0, len(segments))
	for sIdx, seg := range segments {
		if len(seg.terms) > 0 {
			iters = append(iters, &mergeTermIter{
				seg:    seg,
				segIdx: sIdx,
				terms:  seg.terms,
			})
		}
	}

	h := make(mergeHeap, 0, len(iters))
	for i, it := range iters {
		if !it.done() {
			heap.Push(&h, mergeHeapItem{term: it.currentTerm(), iterIdx: i})
		}
	}

	for h.Len() > 0 {
		minTerm := h[0].term

		// Collect all iterators positioned at minTerm and emit their postings.
		for h.Len() > 0 && h[0].term == minTerm {
			item := heap.Pop(&h).(mergeHeapItem)
			it := iters[item.iterIdx]
			seg := it.seg

			if ord, ok := seg.ordinalFor(minTerm); ok {
				info := seg.infos[ord]
				postSlice := seg.postData[info.PostingsOffset : info.PostingsOffset+info.PostingsLen]
				if len(postSlice) >= 8 {
					skipLen := binary.LittleEndian.Uint64(postSlice[:8])
					rawPost := postSlice[8+skipLen:]
					useFOR32 := seg.version == segVersionFOR32
					numericIDs, tfs := decodePostings(rawPost, int(info.DF), docScratch, tfScratch, useFOR32)
					for j, numID := range numericIDs {
						if int(numID) >= len(seg.docIDs) {
							continue
						}
						strID := seg.docIDs[numID]
						meta, exists := docMap[strID]
						if !exists || meta.segIdx != it.segIdx {
							continue // duplicate docID; not this segment's doc
						}
						builder.WritePosting(minTerm, strID, tfs[j])
					}
				}
			}

			it.advance()
			if !it.done() {
				heap.Push(&h, mergeHeapItem{term: it.currentTerm(), iterIdx: item.iterIdx})
			}
		}
	}

	if err := WriteSegmentWithOptions(outPath, builder, opts); err != nil {
		return err
	}

	// Merge stored fields sidecars.
	hasFld := false
	for _, seg := range segments {
		if len(seg.fldIndex) > 0 {
			hasFld = true
			break
		}
	}
	if hasFld && !opts.SkipStoredFields {
		mergedTexts := make(map[string]string)
		for _, seg := range segments {
			for _, entry := range seg.fldIndex {
				if _, already := mergedTexts[entry.docID]; !already {
					if text, ok := seg.GetText(entry.docID); ok {
						mergedTexts[entry.docID] = text
					}
				}
			}
		}
		if len(mergedTexts) > 0 {
			fldOutPath := strings.TrimSuffix(outPath, ".seg") + ".seg.fld"
			_ = WriteStoredFields(fldOutPath, mergedTexts)
		}
	}
	return nil
}

// decodePostings decodes raw posting bytes (FOR-delta blocks + LEB128 tail)
// into slices of numeric docIDs and TF values.
// docScratch and tfScratch must each have capacity >= index.BlockSize; they are
// used as block-decode scratch and avoid per-call allocations in tight loops.
func decodePostings(data []byte, df int, docScratch, tfScratch []uint64, useFOR32 bool) ([]uint64, []int) {
	docIDs := make([]uint64, 0, df)
	tfs := make([]int, 0, df)

	var base uint64
	pos := 0
	read := 0

	docBuf := docScratch[:index.BlockSize]
	tfBuf := tfScratch[:index.BlockSize]

	for read < df {
		remaining := df - read
		if remaining >= index.BlockSize {
			// Full block: intcomp (v2/v4) or FOR-delta (v6).
			var n int
			if useFOR32 {
				n = index.UnpackFOR32Into(data[pos:], docBuf)
				pos += n
				n = index.UnpackFOR32Into(data[pos:], tfBuf)
			} else {
				n = index.UnpackBlock(data[pos:], docBuf)
				pos += n
				n = index.UnpackBlock(data[pos:], tfBuf)
			}
			pos += n
			for i := 0; i < index.BlockSize; i++ {
				base += docBuf[i]
				docIDs = append(docIDs, base)
				tfs = append(tfs, int(tfBuf[i]))
			}
			read += index.BlockSize
		} else {
			// LEB128 tail: interleaved (delta, tf) pairs.
			for i := 0; i < remaining; i++ {
				delta, n := index.ReadVarint(data, pos)
				pos += n
				tf, n := index.ReadVarint(data, pos)
				pos += n
				base += delta
				docIDs = append(docIDs, base)
				tfs = append(tfs, int(tf))
			}
			read += remaining
		}
	}
	return docIDs, tfs
}

// helper writers

func writeUint64(w io.Writer, v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	w.Write(b[:]) //nolint:errcheck
}

func writeUint32(w io.Writer, v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.Write(b[:]) //nolint:errcheck
}

// compile-time interface checks
var _ = (*Segment)(nil)
var _ = MergeSegments

// SearchResult is the per-doc result used by segment search.
type SearchResult = types.ScoredDocument


// SegmentIndexAdapter — implements index.IndexReader over a *Segment

// SegmentIndexAdapter wraps a *Segment so it can be passed to
// index.NewMaxScoreSearcher, giving full MaxScore search on-disk segments.
type SegmentIndexAdapter struct{ seg *Segment }

// IDFCached returns the cached IDF for term+scorer, computing and storing it on
// the first call. Safe for concurrent use; segments are immutable so the cached
// value is always valid.
func (a *SegmentIndexAdapter) IDFCached(term, scorerType string, df, N int, scorer index.Scorer) float64 {
	key := term + "\x00" + scorerType
	if v, ok := a.seg.idfCache.Load(key); ok {
		return v.(float64)
	}
	val := scorer.IDF(df, N)
	a.seg.idfCache.Store(key, val)
	return val
}

// AsIndexReader wraps seg in a SegmentIndexAdapter.
func (s *Segment) AsIndexReader() *SegmentIndexAdapter {
	return &SegmentIndexAdapter{seg: s}
}

// Iterator returns a PostingIter for term with its skip list wired up, or nil
// if not present. Attaching the skip list enables block-max MaxScore skipping
// on on-disk segments. The block decoder is selected based on the segment version:
// v6 (segVersionFOR32) uses UnpackFOR32Into; v2/v4 use UnpackBlockWithScratch.
func (a *SegmentIndexAdapter) Iterator(term string) *index.PostingIter {
	ord, ok := a.seg.ordinalFor(term)
	if !ok {
		return nil
	}
	info := a.seg.infos[ord]
	postSlice := a.seg.postData[info.PostingsOffset : info.PostingsOffset+info.PostingsLen]
	if len(postSlice) < 8 {
		return nil
	}
	skipLen := binary.LittleEndian.Uint64(postSlice[:8])
	skipData := postSlice[8 : 8+skipLen]
	rawPost := postSlice[8+skipLen:]
	sl := index.LoadSkipFromBytes(skipData)
	useFOR32 := a.seg.version == segVersionFOR32
	it := index.NewPostingIterWithSkip(rawPost, sl, int(info.DF), useFOR32)
	if a.seg.blockCache != nil {
		it.WithBlockCache(a.seg.blockCache, ord)
	}
	return it
}

// TermUB returns the per-term upper bound stored in the segment.
func (a *SegmentIndexAdapter) TermUB(term string) float64 {
	return a.seg.TermUB(term)
}

// DF returns the document frequency for term, or 0 if absent.
func (a *SegmentIndexAdapter) DF(term string) int {
	ord, ok := a.seg.ordinalFor(term)
	if !ok {
		return 0
	}
	return int(a.seg.infos[ord].DF)
}

// DocCount returns the number of documents in the segment.
func (a *SegmentIndexAdapter) DocCount() int { return int(a.seg.docCount) }

// AvgDocLen returns the average document length across all docs in the segment.
func (a *SegmentIndexAdapter) AvgDocLen() float64 {
	if a.seg.docCount == 0 {
		return 0
	}
	return float64(a.seg.totalTokens) / float64(a.seg.docCount)
}

// DocLen returns the length (token count) of the document with the given string ID.
func (a *SegmentIndexAdapter) DocLen(docID string) int {
	num, ok := a.seg.docIDToNum[docID]
	if !ok {
		return 0
	}
	return int(a.seg.docLens[num])
}

// DocStringID converts a numeric docID to its string form.
func (a *SegmentIndexAdapter) DocStringID(numID uint64) string {
	if numID >= a.seg.docCount {
		return ""
	}
	return a.seg.docIDs[numID]
}

// HasTerm reports whether the document with the given string docID contains term.
// Uses SkipTo on the on-disk posting list for an O(log N) membership check.
func (s *Segment) HasTerm(term, docID string) bool {
	numID, ok := s.docIDToNum[docID]
	if !ok {
		return false
	}
	it := s.AsIndexReader().Iterator(term)
	if it == nil {
		return false
	}
	it.SkipTo(numID)
	return it.DocID() == numID
}

// Search returns the top-K results for tokens using MaxScoreSearcher.
func (s *Segment) Search(tokens []string, topK int, scorer index.Scorer) []types.ScoredDocument {
	searcher := index.NewMaxScoreSearcher(s.AsIndexReader(), scorer)
	return searcher.Search(tokens, topK)
}

