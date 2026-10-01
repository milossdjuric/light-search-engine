package segment

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	bloom "github.com/bits-and-blooms/bloom/v3"
	"github.com/blevesearch/vellum"

	"search-eval-platform/internal/index"
)

// SegmentWriteOptions controls optional features when writing a segment file.
type SegmentWriteOptions struct {
	// BloomFPRate is the desired false-positive rate for the bloom filter sidecar.
	// 0 disables bloom filter writing.
	BloomFPRate float64
	// BuildOpts controls BM25F vs plain build path when building from an IndexBuilder.
	BuildOpts index.BuildOptions
	// SkipStoredFields, when true, omits the .seg.fld sidecar even if source segments have one.
	SkipStoredFields bool
}

// WriteSegment serialises an IndexBuilder snapshot to path as a .seg file.
// It calls Build() internally; the builder must not be used after this call.
func WriteSegment(path string, b *index.IndexBuilder) error {
	return WriteSegmentWithOptions(path, b, SegmentWriteOptions{})
}

// WriteSegmentWithOptions serialises an IndexBuilder snapshot to path with the
// given options (bloom filter, stored fields). BuildWithOptions() is called
// internally, allowing BM25F pseudo-TF encoding when opts.BuildOpts.Fields is non-empty.
func WriteSegmentWithOptions(path string, b *index.IndexBuilder, opts SegmentWriteOptions) error {
	_, err := WriteSegmentWithIndex(path, b, opts)
	return err
}

// WriteSegmentWithIndex is like WriteSegmentWithOptions but also returns the
// built InvertedIndex so callers can reuse it without a second file read.
func WriteSegmentWithIndex(path string, b *index.IndexBuilder, opts SegmentWriteOptions) (*index.InvertedIndex, error) {
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
	// Buffer all writes below: the TermInfoStore section issues one write
	// call per term, and without buffering that's one write() syscall per
	// term (measured: ~3000 syscalls, ~85ms, for a 3000-term segment).
	bw := bufio.NewWriterSize(f, 256*1024)

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

	// 5. Compute section offsets.
	// Layout:
	//   [headerSize bytes]  Header
	//   [4 + fstLen]        FSTSection (4-byte length prefix + fstData)
	//   [nTerms*24]         TermInfoStore
	//   [docLenBuf]         DocLengths
	//   [docIDData]         DocID strings
	//   [postData]          PostingData
	fstOff := uint64(headerSize)
	termOff := fstOff + 4 + uint64(len(fstData))
	docLenOff := termOff + uint64(nTerms)*termInfoSize
	docIDOff := docLenOff + uint64(len(docLenBuf))
	postOff := docIDOff + uint64(len(docIDData))

	// 6. Write header.
	hdr := make([]byte, headerSize)
	copy(hdr[0:4], segMagic[:])
	binary.LittleEndian.PutUint32(hdr[4:8], segVersion)
	binary.LittleEndian.PutUint64(hdr[8:16], docCount)
	binary.LittleEndian.PutUint64(hdr[16:24], fstOff)
	binary.LittleEndian.PutUint64(hdr[24:32], termOff)
	binary.LittleEndian.PutUint64(hdr[32:40], postOff)

	// Write everything in order.
	if _, err := bw.Write(hdr[:headerSize]); err != nil {
		return err
	}
	// FST section
	var fstLenBuf [4]byte
	binary.LittleEndian.PutUint32(fstLenBuf[:], uint32(len(fstData)))
	if _, err := bw.Write(fstLenBuf[:]); err != nil {
		return err
	}
	if _, err := bw.Write(fstData); err != nil {
		return err
	}
	// TermInfoStore
	for _, ti := range infos {
		var tibuf [termInfoSize]byte
		binary.LittleEndian.PutUint32(tibuf[0:4], ti.DF)
		binary.LittleEndian.PutUint32(tibuf[4:8], math.Float32bits(ti.UB))
		binary.LittleEndian.PutUint64(tibuf[8:16], ti.PostingsOffset)
		binary.LittleEndian.PutUint64(tibuf[16:24], ti.PostingsLen)
		if _, err := bw.Write(tibuf[:]); err != nil {
			return err
		}
	}
	// DocLengths
	if _, err := bw.Write(docLenBuf); err != nil {
		return err
	}
	// DocID strings
	if _, err := bw.Write(docIDData); err != nil {
		return err
	}
	// PostingData
	if _, err := bw.Write(postData); err != nil {
		return err
	}

	_ = docLenOff
	_ = docIDOff
	_ = postOff
	_ = termOff

	if err := bw.Flush(); err != nil {
		return err
	}
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
