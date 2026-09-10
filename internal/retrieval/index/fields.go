package index

import "math"

// FieldConfig defines BM25F scoring parameters for one document field.
type FieldConfig struct {
	Name   string  // field name, e.g. "title", "body", "url"
	Weight float64 // w_f: contribution multiplier (higher = more important)
	B      float64 // b_f: length normalization strength (0=none, 1=full)
}

// BuildOptions controls how IndexBuilder.BuildWithOptions dispatches.
type BuildOptions struct {
	// Fields, when non-empty, switches to BM25F mode: pseudo-TFs are computed
	// from per-field data accumulated via AddFields. Mutually exclusive with BM25FScaled.
	Fields []FieldConfig

	// BM25FScaled, when true, means the postings already contain scaled pseudo-TFs
	// (BM25FScaleFactor * pseudoTF as integers). Used during segment merge to preserve
	// BM25F scoring. Mutually exclusive with Fields.
	BM25FScaled bool

	// BM25FK1 is the saturation parameter used when BM25FScaled is true.
	BM25FK1 float64

	// UseFOR32, when true, encodes full posting blocks with PackFOR32 (uint32 bit-
	// packing) instead of intcomp (uint64 delta-pack). Produces v6 segments.
	// Faster to decode on modern CPUs; the resulting InvertedIndex.data is in
	// FOR-delta format and PostingIter uses UnpackFOR32Into for full blocks.
	UseFOR32 bool
}

// BM25FScaleFactor converts a float pseudo-TF to an integer for storage.
// Stored TF = round(pseudoTF * BM25FScaleFactor).
const BM25FScaleFactor = 1000.0

// bm25fImpact computes the BM25F score for a term given a scaled pseudo-TF.
// k1 is the saturation parameter (typically 1.2).
// Defined here (not in bm25f package) to break an import cycle during Build.
func bm25fImpact(scaledTF uint64, df, N int, k1 float64) float64 {
	if df == 0 || N == 0 {
		return 0
	}
	pseudoTF := float64(scaledTF) / BM25FScaleFactor
	idf := math.Log((float64(N)-float64(df)+0.5)/(float64(df)+0.5) + 1)
	return idf * (k1 + 1) * pseudoTF / (k1 + pseudoTF)
}
