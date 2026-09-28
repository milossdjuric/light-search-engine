package index_test

import (
	"testing"

	"search-eval-platform/internal/retrieval/index"
)

// TestBuildWithOptions_BM25FMode_HonorsUseFOR32 is a regression test for
// BuildBM25F silently dropping BuildOptions.UseFOR32: the BM25F code path
// (len(opts.Fields) > 0 && b.fieldTFs != nil) hardcoded PackBlock/intcomp
// encoding regardless of the option, unlike the default path which already
// dispatches correctly via buildInternal(opts.UseFOR32).
func TestBuildWithOptions_BM25FMode_HonorsUseFOR32(t *testing.T) {
	fields := []index.FieldConfig{
		{Name: "title", Weight: 2.0, B: 0.75},
		{Name: "body", Weight: 1.0, B: 0.75},
	}

	b := index.NewIndexBuilder()
	defer b.Close()
	b.AddFields("d1", map[string][]string{
		"title": {"alpha", "beta"},
		"body":  {"alpha", "gamma", "delta"},
	})
	b.AddFields("d2", map[string][]string{
		"title": {"gamma"},
		"body":  {"beta", "epsilon"},
	})

	idx := b.BuildWithOptions(index.BuildOptions{Fields: fields, UseFOR32: true})

	if !idx.UsesFOR32() {
		t.Error("BuildWithOptions with Fields set and UseFOR32: true produced an index not using FOR32 encoding")
	}
}

// TestBuildWithOptions_BM25FScaledMode_HonorsUseFOR32 is the same regression
// for buildScaledPseudoTF, used during BM25F segment merge.
func TestBuildWithOptions_BM25FScaledMode_HonorsUseFOR32(t *testing.T) {
	b := index.NewIndexBuilder()
	defer b.Close()
	b.AddTermFreqs("d1", map[string]int{"alpha": 1200, "beta": 800}, 10)
	b.AddTermFreqs("d2", map[string]int{"beta": 500, "gamma": 900}, 8)

	idx := b.BuildWithOptions(index.BuildOptions{BM25FScaled: true, UseFOR32: true})

	if !idx.UsesFOR32() {
		t.Error("BuildWithOptions with BM25FScaled: true and UseFOR32: true produced an index not using FOR32 encoding")
	}
}
