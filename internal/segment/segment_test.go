package segment_test

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"search-eval-platform/internal/analysis"
	"search-eval-platform/internal/index"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/segment"
	"search-eval-platform/pkg/types"
)

func buildSegment(t *testing.T, dir string, name string, docs []types.Document) string {
	t.Helper()
	b := index.NewIndexBuilder()
	for _, d := range docs {
		b.Add(d.ID, analysis.Tokenize(d.Text))
	}
	path := filepath.Join(dir, name+".seg")
	if err := segment.WriteSegment(path, b); err != nil {
		t.Fatalf("WriteSegment: %v", err)
	}
	return path
}

func TestWriteLoadSegmentRoundTrip(t *testing.T) {
	dir := t.TempDir()
	docs := []types.Document{
		{ID: "doc1", Text: "search engine indexing documents"},
		{ID: "doc2", Text: "fast retrieval with BM25 scoring"},
		{ID: "doc3", Text: "segment merge tiered policy"},
	}
	path := buildSegment(t, dir, "seg0", docs)

	seg, err := segment.LoadSegment(path)
	if err != nil {
		t.Fatalf("LoadSegment: %v", err)
	}
	if seg.DocCount() != len(docs) {
		t.Errorf("DocCount: want %d, got %d", len(docs), seg.DocCount())
	}
}

// Segments in the removed pre-v6 formats (v2 intcomp, v4 = v2 + LZ4) must be
// rejected with an error that says to re-index, not misread.
func TestLoadSegmentRejectsPreV6Format(t *testing.T) {
	dir := t.TempDir()
	path := buildSegment(t, dir, "old", []types.Document{{ID: "d1", Text: "legacy format"}})
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, ver := range []uint32{2, 4} {
		binary.LittleEndian.PutUint32(data[4:8], ver)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := segment.LoadSegment(path)
		if err == nil || !strings.Contains(err.Error(), "re-index") {
			t.Errorf("v%d segment: got err %v, want an unsupported-version error mentioning re-index", ver, err)
		}
	}
}

func TestLoadSegmentCorruptFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corrupt.seg")
	if err := os.WriteFile(path, []byte("this is not a valid segment file"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := segment.LoadSegment(path)
	if err == nil {
		t.Error("expected error loading corrupt segment file")
	}
}

func TestMergeSegmentsUnion(t *testing.T) {
	dir := t.TempDir()

	docsA := []types.Document{
		{ID: "a1", Text: "alpha beta gamma"},
		{ID: "a2", Text: "delta epsilon"},
	}
	docsB := []types.Document{
		{ID: "b1", Text: "gamma zeta eta"},
		{ID: "b2", Text: "theta iota kappa"},
	}

	pathA := buildSegment(t, dir, "segA", docsA)
	pathB := buildSegment(t, dir, "segB", docsB)

	segA, _ := segment.LoadSegment(pathA)
	segB, _ := segment.LoadSegment(pathB)

	outPath := filepath.Join(dir, "merged.seg")
	if err := segment.MergeSegmentsWithOptions(outPath, []*segment.Segment{segA, segB}, segment.SegmentWriteOptions{}, nil); err != nil {
		t.Fatalf("MergeSegments: %v", err)
	}

	merged, err := segment.LoadSegment(outPath)
	if err != nil {
		t.Fatalf("LoadSegment merged: %v", err)
	}
	// Union of 2+2 = 4 documents
	if merged.DocCount() != 4 {
		t.Errorf("merged DocCount: want 4, got %d", merged.DocCount())
	}
}

// TestMergeSegmentsDropsTombstonedDocs verifies that a docID passed in the
// tombstones set is entirely absent from the merged output segment, rather
// than being silently carried forward.
func TestMergeSegmentsDropsTombstonedDocs(t *testing.T) {
	dir := t.TempDir()

	docsA := []types.Document{
		{ID: "keep", Text: "alpha beta gamma"},
		{ID: "victim", Text: "delta epsilon zeta"},
	}
	pathA := buildSegment(t, dir, "segA", docsA)
	segA, _ := segment.LoadSegment(pathA)

	outPath := filepath.Join(dir, "merged.seg")
	tombstones := map[string]struct{}{"victim": {}}
	if err := segment.MergeSegmentsWithOptions(outPath, []*segment.Segment{segA}, segment.SegmentWriteOptions{}, tombstones); err != nil {
		t.Fatalf("MergeSegmentsWithOptions: %v", err)
	}

	merged, err := segment.LoadSegment(outPath)
	if err != nil {
		t.Fatalf("LoadSegment merged: %v", err)
	}
	if merged.DocCount() != 1 {
		t.Errorf("merged DocCount: want 1 (tombstoned doc dropped), got %d", merged.DocCount())
	}
	if merged.HasTerm("delta", "victim") {
		t.Error("merged segment still contains postings for tombstoned doc \"victim\"")
	}
}

func TestWriteLoadSegmentBM25FRoundTrip(t *testing.T) {
	dir := t.TempDir()

	fields := []scoring.FieldConfig{
		{Name: "title", Weight: 2.5, B: 0.45},
		{Name: "body", Weight: 1.0, B: 0.75},
	}

	b := index.NewIndexBuilder()
	// doc1: "retrieval" appears in the high-weight title field.
	b.AddFields("doc1", map[string][]string{
		"title": {"retrieval", "engine"},
		"body":  {"some", "other", "words"},
	})
	// doc2: "retrieval" appears only in the low-weight body field.
	b.AddFields("doc2", map[string][]string{
		"title": {"unrelated", "title"},
		"body":  {"retrieval", "ranking", "scoring"},
	})

	path := filepath.Join(dir, "bm25f.seg")
	opts := segment.SegmentWriteOptions{
		BuildOpts: index.BuildOptions{Fields: fields},
	}
	if err := segment.WriteSegmentWithOptions(path, b, opts); err != nil {
		t.Fatalf("WriteSegmentWithOptions: %v", err)
	}

	seg, err := segment.LoadSegment(path)
	if err != nil {
		t.Fatalf("LoadSegment: %v", err)
	}
	if seg.DocCount() != 2 {
		t.Fatalf("DocCount: want 2, got %d", seg.DocCount())
	}

	scorer := scoring.NewBM25F(1.2)
	results := seg.Search([]string{"retrieval"}, 10, scorer)
	if len(results) != 2 {
		t.Fatalf("Search: want 2 results, got %d", len(results))
	}
	// doc1 has "retrieval" in the high-weight title — must rank first.
	if results[0].DocID != "doc1" {
		t.Errorf("BM25F ranking: want doc1 first (title hit), got %s first", results[0].DocID)
	}
}

func TestMergeSegmentsDuplicateDocID(t *testing.T) {
	dir := t.TempDir()

	// Same doc ID in both segments — highest segmentID (index in slice) wins
	docsA := []types.Document{{ID: "shared", Text: "version one text"}}
	docsB := []types.Document{{ID: "shared", Text: "version two text"}}

	pathA := buildSegment(t, dir, "segA", docsA)
	pathB := buildSegment(t, dir, "segB", docsB)

	segA, _ := segment.LoadSegment(pathA)
	segB, _ := segment.LoadSegment(pathB)

	outPath := filepath.Join(dir, "merged_dup.seg")
	if err := segment.MergeSegmentsWithOptions(outPath, []*segment.Segment{segA, segB}, segment.SegmentWriteOptions{}, nil); err != nil {
		t.Fatalf("MergeSegments: %v", err)
	}

	merged, _ := segment.LoadSegment(outPath)
	// Duplicate docID -> only 1 doc in the merged segment
	if merged.DocCount() != 1 {
		t.Errorf("duplicate docID: want 1 doc, got %d", merged.DocCount())
	}
}
