package index

import (
	"encoding/binary"
	"math"
	"sort"

	"search-eval-platform/internal/codec"
	"search-eval-platform/internal/scoring"
)

// BuildBM25F builds an InvertedIndex from per-field data accumulated via AddFields.
// Pseudo-TF = Σ_f w_f * tf(t,d,f) / (1 - b_f + b_f * len_f(d) / avglen_f) is
// computed here and stored scaled as an integer. Block-max impacts are computed
// using the BM25F saturation formula.
func (b *IndexBuilder) BuildBM25F(fields []scoring.FieldConfig) *InvertedIndex {
	bm25f := scoring.NewDefaultBM25F()

	if b.N == 0 || b.fieldTFs == nil {
		return b.buildInternal()
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
			scaledTF := int(math.Round(pseudoTF * scoring.BM25FScaleFactor))
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

	packBuf := make([]byte, codec.PackFOR32BufSize)
	// uint32 versions for codec.PackFOR32 encoding.
	docDeltasU32 := make([]uint32, codec.BlockSize)
	tfValsU32 := make([]uint32, codec.BlockSize)

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

		nBlocks := df / codec.BlockSize
		tail := df % codec.BlockSize
		var prevDocID uint64
		blockBase := int64(len(data))

		for blk := 0; blk < nBlocks; blk++ {
			blockStart := blk * codec.BlockSize
			blockStartOff := int(int64(len(data)) - blockBase)

			docDeltas := make([]uint64, codec.BlockSize)
			tfVals := make([]uint64, codec.BlockSize)
			var blockMaxImpact float64

			for j := 0; j < codec.BlockSize; j++ {
				entry := entries[blockStart+j]
				numID := docIDIndex[entry.DocID]
				docDeltas[j] = numID - prevDocID
				prevDocID = numID
				tfVals[j] = uint64(entry.TF)

				score := bm25f.ScoreTerm(int(entry.TF), df, 0, 0, b.N)
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
				entry := entries[tailStart+j]
				numID := docIDIndex[entry.DocID]
				delta := numID - prevDocID
				prevDocID = numID
				data = codec.AppendVarint(data, delta)
				data = codec.AppendVarint(data, uint64(entry.TF))

				score := bm25f.ScoreTerm(int(entry.TF), df, 0, 0, b.N)
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
	bm25f := scoring.NewBM25F(k1)
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
				docDeltas[j] = numID - prevDocID
				prevDocID = numID
				tfVals[j] = uint64(p.TF)

				score := bm25f.ScoreTerm(int(p.TF), df, 0, 0, b.N)
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

				score := bm25f.ScoreTerm(int(p.TF), df, 0, 0, b.N)
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
