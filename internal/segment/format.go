package segment

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"

	bloom "github.com/bits-and-blooms/bloom/v3"
	"github.com/blevesearch/vellum"

	"search-eval-platform/internal/index"
)

// segMagic is the 4-byte magic number at the start of every .seg file.
var segMagic = [4]byte{'S', 'E', 'G', '2'}

// segVersion is the only supported format version (6): full posting blocks
// are FOR-delta bit-packed uint32 deltas/TFs (codec.PackFOR32), tails LEB128.
// The older intcomp formats (v2, and v4 = v2 + LZ4) are no longer readable.
const segVersion = uint32(6)

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

// Segment is an immutable, on-disk inverted index segment.
type Segment struct {
	docCount    uint64
	fst         *vellum.FST
	terms       []string // term ordinal → string
	infos       []TermInfo
	docLens     []uint32 // docLens[numericID] = token count
	docIDs      []string // docIDs[numericID] = string doc ID
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
	// FOR-delta decoding on repeated accesses by concurrent queries.
	blockCache *index.BlockCache

	// idfCache caches IDF values keyed by "term\x00scorerType". Since segments
	// are immutable (df and N are fixed after flush), IDF is constant and never
	// needs invalidation.
	idfCache sync.Map
}

func parseSegment(data []byte) (*Segment, error) {
	if len(data) < headerSize {
		return nil, errors.New("segment file too small")
	}
	if [4]byte(data[0:4]) != segMagic {
		return nil, errors.New("invalid segment magic")
	}
	ver := binary.LittleEndian.Uint32(data[4:8])
	if ver != segVersion {
		return nil, fmt.Errorf("unsupported segment version %d (only v%d is supported; re-index to upgrade older segments)", ver, segVersion)
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

	postData := data[postOff:]

	return &Segment{
		docCount:    docCount,
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
