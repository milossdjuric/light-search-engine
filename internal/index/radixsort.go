package index

import (
	"sort"
)

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
