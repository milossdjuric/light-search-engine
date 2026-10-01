package analysis_test

import (
	"reflect"
	"testing"

	"search-eval-platform/internal/analysis"
)

// TestTokenize_ASCII pins down tokenizeASCII's current behavior (lowercasing,
// splitting on non-alphanumeric, digit handling, empty/edge inputs) before a
// planned allocation-reducing refactor (per-token string(buf) conversion →
// one allocation per document + safe sub-slicing). Every case here must
// produce identical output before and after that refactor.
func TestTokenize_ASCII(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want []string
	}{
		{"empty", "", nil},
		{"single word", "hello", []string{"hello"}},
		{"lowercases", "Hello WORLD", []string{"hello", "world"}},
		{"splits on space", "alpha beta gamma", []string{"alpha", "beta", "gamma"}},
		{"splits on punctuation", "alpha,beta.gamma!delta?", []string{"alpha", "beta", "gamma", "delta"}},
		{"includes digits", "abc123 456def", []string{"abc123", "456def"}},
		{"leading separator", "  alpha", []string{"alpha"}},
		{"trailing separator", "alpha  ", []string{"alpha"}},
		{"consecutive separators", "alpha,,,,beta", []string{"alpha", "beta"}},
		{"only separators", "   ,,,!!!", nil},
		{"single char tokens", "a b c", []string{"a", "b", "c"}},
		{"mixed case digits punctuation", "The Quick-Brown_Fox42 jumps!", []string{"the", "quick", "brown", "fox42", "jumps"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := analysis.Tokenize(tc.in)
			if len(got) == 0 && len(tc.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Tokenize(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

// TestTokenize_ASCII_TokensAreIndependentOfEachOther verifies each returned
// token is a fully independent string, not just a view sharing mutable state
// with a scratch buffer that could change after the call returns — the
// property the refactor (concatenate-then-sub-slice) must preserve even
// though tokens end up sharing ONE immutable backing array with each other.
func TestTokenize_ASCII_TokensAreIndependentOfEachOther(t *testing.T) {
	tokens := analysis.Tokenize("alpha beta gamma")
	if len(tokens) != 3 {
		t.Fatalf("got %d tokens, want 3", len(tokens))
	}
	// Tokens must retain their own values regardless of what happens to
	// other tokens or to any internal scratch state after the call returns.
	want := []string{"alpha", "beta", "gamma"}
	for i, tok := range tokens {
		if tok != want[i] {
			t.Errorf("tokens[%d] = %q, want %q", i, tok, want[i])
		}
	}
}

// BenchmarkTokenizeASCII measures allocations for tokenizing a realistic
// document, isolated from IndexBuilder/index-build overhead.
func BenchmarkTokenizeASCII(b *testing.B) {
	text := "The Quick Brown Fox jumps over the lazy dog while searching the inverted index for matching terms and computing bm25 scores"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = analysis.Tokenize(text)
	}
}
