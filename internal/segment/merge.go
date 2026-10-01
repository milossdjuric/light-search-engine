package segment

import (
	"container/heap"
	"encoding/binary"
	"errors"
	"strings"

	"search-eval-platform/internal/codec"
	"search-eval-platform/internal/index"
)

// mergeTermIter is one input stream for the k-way term merge.
type mergeTermIter struct {
	seg     *Segment
	segIdx  int // index in the input slice (used for duplicate docID tie-break)
	terms   []string
	termPos int
}

func (m *mergeTermIter) done() bool { return m.termPos >= len(m.terms) }

func (m *mergeTermIter) currentTerm() string { return m.terms[m.termPos] }

func (m *mergeTermIter) advance() { m.termPos++ }

// mergeHeapItem is pushed onto the k-way merge heap.
type mergeHeapItem struct {
	term    string
	iterIdx int
}

type mergeHeap []mergeHeapItem

func (h mergeHeap) Len() int { return len(h) }

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

// MergeSegmentsWithOptions merges segments with the given write options.
// tombstones lists docIDs that must be dropped from the merged output
// entirely (deleted documents); nil/empty means don't drop anything.
func MergeSegmentsWithOptions(outPath string, segments []*Segment, opts SegmentWriteOptions, tombstones map[string]struct{}) error {
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
			if _, tombstoned := tombstones[strID]; tombstoned {
				continue
			}
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
	docScratch := make([]uint64, codec.BlockSize)
	tfScratch := make([]uint64, codec.BlockSize)

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
					numericIDs, tfs := decodePostings(rawPost, int(info.DF), docScratch, tfScratch)
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
// docScratch and tfScratch must each have capacity >= codec.BlockSize; they are
// used as block-decode scratch and avoid per-call allocations in tight loops.
func decodePostings(data []byte, df int, docScratch, tfScratch []uint64) ([]uint64, []int) {
	docIDs := make([]uint64, 0, df)
	tfs := make([]int, 0, df)

	var base uint64
	pos := 0
	read := 0

	docBuf := docScratch[:codec.BlockSize]
	tfBuf := tfScratch[:codec.BlockSize]

	for read < df {
		remaining := df - read
		if remaining >= codec.BlockSize {
			// Full block: FOR-delta docID deltas, then TFs.
			pos += codec.UnpackFOR32Into(data[pos:], docBuf)
			pos += codec.UnpackFOR32Into(data[pos:], tfBuf)
			for i := 0; i < codec.BlockSize; i++ {
				base += docBuf[i]
				docIDs = append(docIDs, base)
				tfs = append(tfs, int(tfBuf[i]))
			}
			read += codec.BlockSize
		} else {
			// LEB128 tail: interleaved (delta, tf) pairs.
			for i := 0; i < remaining; i++ {
				delta, n := codec.ReadVarint(data, pos)
				pos += n
				tf, n := codec.ReadVarint(data, pos)
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
