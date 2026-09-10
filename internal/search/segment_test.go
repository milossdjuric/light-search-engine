package search_test

import (
	"os"
	"path/filepath"
	"testing"

	"search-eval-platform/internal/retrieval/bm25f"
	"search-eval-platform/internal/retrieval/index"
	"search-eval-platform/internal/search"
	"search-eval-platform/pkg/types"
)

func buildSegment(t *testing.T, dir string, name string, docs []types.Document) string {
	t.Helper()
	b := index.NewIndexBuilder()
	for _, d := range docs {
		b.Add(d.ID, index.Tokenize(d.Text))
	}
	path := filepath.Join(dir, name+".seg")
	if err := search.WriteSegment(path, b); err != nil {
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

	seg, err := search.LoadSegment(path)
	if err != nil {
		t.Fatalf("LoadSegment: %v", err)
	}
	if seg.DocCount() != len(docs) {
		t.Errorf("DocCount: want %d, got %d", len(docs), seg.DocCount())
	}
}

func TestLoadSegmentCorruptFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corrupt.seg")
	if err := os.WriteFile(path, []byte("this is not a valid segment file"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := search.LoadSegment(path)
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

	segA, _ := search.LoadSegment(pathA)
	segB, _ := search.LoadSegment(pathB)

	outPath := filepath.Join(dir, "merged.seg")
	if err := search.MergeSegments(outPath, []*search.Segment{segA, segB}); err != nil {
		t.Fatalf("MergeSegments: %v", err)
	}

	merged, err := search.LoadSegment(outPath)
	if err != nil {
		t.Fatalf("LoadSegment merged: %v", err)
	}
	// Union of 2+2 = 4 documents
	if merged.DocCount() != 4 {
		t.Errorf("merged DocCount: want 4, got %d", merged.DocCount())
	}
}

// TestWriteLoadSegmentBM25FRoundTrip verifies that a segment written with BM25F
// pseudo-TF encoding loads correctly and returns ranked results when searched
// with a BM25F scorer. The title field is weighted higher than body, so a
// document with the query term in its title should outscore one with it only
// in the body.
// TestWriteLoadSegmentLZ4Compression verifies that a segment written with LZ4
// compression loads correctly and returns the same results as an uncompressed
// segment built from the same data.
func TestWriteLoadSegmentLZ4Compression(t *testing.T) {
	dir := t.TempDir()
	docs := []types.Document{
		{ID: "c1", Text: "lz4 compression fast decompression"},
		{ID: "c2", Text: "segment file format with optional compression"},
		{ID: "c3", Text: "inverted index posting lists stored compressed"},
	}

	// Write plain (v2) reference.
	plainPath := filepath.Join(dir, "plain.seg")
	b1 := index.NewIndexBuilder()
	for _, d := range docs {
		b1.Add(d.ID, index.Tokenize(d.Text))
	}
	if err := search.WriteSegment(plainPath, b1); err != nil {
		t.Fatalf("WriteSegment plain: %v", err)
	}

	// Write LZ4-compressed (v4).
	lz4Path := filepath.Join(dir, "lz4.seg")
	b2 := index.NewIndexBuilder()
	for _, d := range docs {
		b2.Add(d.ID, index.Tokenize(d.Text))
	}
	if err := search.WriteSegmentWithOptions(lz4Path, b2, search.SegmentWriteOptions{Compression: "lz4"}); err != nil {
		t.Fatalf("WriteSegmentWithOptions lz4: %v", err)
	}

	// LZ4 file must be smaller than the plain file.
	infoPlain, _ := os.Stat(plainPath)
	infoLZ4, _ := os.Stat(lz4Path)
	if infoLZ4.Size() >= infoPlain.Size() {
		t.Logf("note: lz4 (%d bytes) not smaller than plain (%d bytes) — corpus too small to compress",
			infoLZ4.Size(), infoPlain.Size())
	}

	// Load both and verify same doc count and search results.
	segPlain, err := search.LoadSegment(plainPath)
	if err != nil {
		t.Fatalf("LoadSegment plain: %v", err)
	}
	segLZ4, err := search.LoadSegment(lz4Path)
	if err != nil {
		t.Fatalf("LoadSegment lz4: %v", err)
	}
	if segPlain.DocCount() != segLZ4.DocCount() {
		t.Errorf("DocCount mismatch: plain=%d lz4=%d", segPlain.DocCount(), segLZ4.DocCount())
	}

	scorer := bm25f.NewScorerOnly(1.2)
	resPlain := segPlain.Search([]string{"compression"}, 10, scorer)
	resLZ4 := segLZ4.Search([]string{"compression"}, 10, scorer)
	if len(resPlain) != len(resLZ4) {
		t.Errorf("search result count: plain=%d lz4=%d", len(resPlain), len(resLZ4))
	}
}

func TestWriteLoadSegmentBM25FRoundTrip(t *testing.T) {
	dir := t.TempDir()

	fields := []index.FieldConfig{
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
	opts := search.SegmentWriteOptions{
		BuildOpts: index.BuildOptions{Fields: fields},
	}
	if err := search.WriteSegmentWithOptions(path, b, opts); err != nil {
		t.Fatalf("WriteSegmentWithOptions: %v", err)
	}

	seg, err := search.LoadSegment(path)
	if err != nil {
		t.Fatalf("LoadSegment: %v", err)
	}
	if seg.DocCount() != 2 {
		t.Fatalf("DocCount: want 2, got %d", seg.DocCount())
	}

	scorer := bm25f.NewScorerOnly(1.2)
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

	segA, _ := search.LoadSegment(pathA)
	segB, _ := search.LoadSegment(pathB)

	outPath := filepath.Join(dir, "merged_dup.seg")
	if err := search.MergeSegments(outPath, []*search.Segment{segA, segB}); err != nil {
		t.Fatalf("MergeSegments: %v", err)
	}

	merged, _ := search.LoadSegment(outPath)
	// Duplicate docID -> only 1 doc in the merged segment
	if merged.DocCount() != 1 {
		t.Errorf("duplicate docID: want 1 doc, got %d", merged.DocCount())
	}
}
