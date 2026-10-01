package segment_test

import (
	"fmt"
	"path/filepath"
	"testing"

	"search-eval-platform/internal/analysis"
	"search-eval-platform/internal/index"
	"search-eval-platform/internal/segment"
	"search-eval-platform/pkg/types"
)

// BenchmarkWriteSegmentManyTerms measures WriteSegment's wall-clock time for
// a segment with many distinct terms — the TermInfoStore section writes one
// entry per term directly to the raw *os.File (no bufio.Writer), so this is
// dominated by write() syscall count, not actual bytes written. Mirrors the
// strace-measured shape (~3000 distinct terms, one write() syscall each).
func BenchmarkWriteSegmentManyTerms(b *testing.B) {
	const nDocs = 5000
	const nVocab = 3000

	docs := make([]types.Document, nDocs)
	for i := range docs {
		docs[i] = types.Document{
			ID: fmt.Sprintf("doc%d", i),
			Text: fmt.Sprintf("term%d term%d term%d term%d term%d",
				i%nVocab, (i+1)%nVocab, (i+7)%nVocab, (i+13)%nVocab, (i+29)%nVocab),
		}
	}

	builder := index.NewIndexBuilder()
	defer builder.Close()
	for _, d := range docs {
		builder.Add(d.ID, analysis.Tokenize(d.Text))
	}

	dir := b.TempDir()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		path := filepath.Join(dir, fmt.Sprintf("bench%d.seg", i))
		if err := segment.WriteSegment(path, builder); err != nil {
			b.Fatalf("WriteSegment: %v", err)
		}
	}
}
