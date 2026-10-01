package segment

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"search-eval-platform/internal/index"
	"search-eval-platform/internal/scoring"
)

// TestMaxScoreOnSegmentIsExact checks block-max MaxScore over a real on-disk
// segment (FOR-delta postings, persisted skip lists and block-max impacts)
// against exhaustive BM25 on a skewed vocabulary — the in-memory index tests
// don't exercise the segment's persisted block maxima.
func TestMaxScoreOnSegmentIsExact(t *testing.T) {
	common := []string{"the", "of", "and", "to", "in", "is", "for", "with"}
	rng := rand.New(rand.NewSource(11))
	b := index.NewIndexBuilder()
	defer b.Close()
	docs := map[string][]string{}
	for i := 0; i < 5000; i++ {
		n := 20 + rng.Intn(80)
		words := make([]string, n)
		for j := range words {
			if rng.Float64() < 0.6 {
				words[j] = common[rng.Intn(len(common))]
			} else {
				words[j] = fmt.Sprintf("w%d", int(math.Pow(rng.Float64(), 3)*3000))
			}
		}
		id := fmt.Sprintf("doc%05d", i)
		docs[id] = words
		b.Add(id, words)
	}
	path := filepath.Join(t.TempDir(), "s.seg")
	if err := WriteSegmentWithOptions(path, b, SegmentWriteOptions{}); err != nil {
		t.Fatal(err)
	}
	seg, err := LoadSegment(path)
	if err != nil {
		t.Fatal(err)
	}
	reader := seg.AsIndexReader()
	scorer := scoring.NewBM25(1.2, 0.75)

	tfs := map[string]map[string]int{} // term → doc → tf
	for id, words := range docs {
		for _, w := range words {
			if tfs[w] == nil {
				tfs[w] = map[string]int{}
			}
			tfs[w][id]++
		}
	}
	naive := func(tokens []string, k int) []float64 {
		N, avg := reader.DocCount(), reader.AvgDocLen()
		scores := map[string]float64{}
		for _, tok := range tokens { // repeats included: query-term frequency
			df := reader.DF(tok)
			for id, tf := range tfs[tok] {
				scores[id] += scorer.ScoreTerm(tf, df, len(docs[id]), avg, N)
			}
		}
		out := make([]float64, 0, len(scores))
		for _, s := range scores {
			out = append(out, s)
		}
		sort.Sort(sort.Reverse(sort.Float64Slice(out)))
		if len(out) > k {
			out = out[:k]
		}
		return out
	}

	// Random queries: 1-12 words mixing common and rare terms, repeats allowed.
	qrng := rand.New(rand.NewSource(99))
	var randomQueries []string
	for i := 0; i < 150; i++ {
		n := 1 + qrng.Intn(12)
		words := make([]string, n)
		for j := range words {
			if qrng.Float64() < 0.5 {
				words[j] = common[qrng.Intn(len(common))]
			} else {
				words[j] = fmt.Sprintf("w%d", int(math.Pow(qrng.Float64(), 2)*3000))
			}
		}
		randomQueries = append(randomQueries, strings.Join(words, " "))
	}

	queries := []string{
		"the of and to in w2500",
		"what is the w900 of the w5 the",
		"the and for with w2999 w1200",
		"in of is w700",
		"the of and to in is for with",
		"w1 w2 w3 the of",
	}
	queries = append(queries, randomQueries...)
	for _, q := range queries {
		tokens := strings.Fields(q)
		all := naive(tokens, 100)
		for _, k := range []int{1, 10, 100} {
			got := index.NewMaxScoreSearcher(reader, scorer).Search(tokens, k)
			want := all
			if len(want) > k {
				want = want[:k]
			}
			if len(got) != len(want) {
				t.Errorf("top%d %q: got %d docs, want %d", k, q, len(got), len(want))
				continue
			}
			for i := range want {
				if math.Abs(got[i].Score-want[i]) > 1e-6 {
					t.Errorf("top%d %q rank %d: MaxScore %.6f, exhaustive %.6f — a document was pruned or mis-scored", k, q, i+1, got[i].Score, want[i])
					break
				}
			}
		}
	}
}
