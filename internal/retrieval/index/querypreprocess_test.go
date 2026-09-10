package index

import "testing"

func TestPreprocessQuery(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		// Trailing punctuation stripped.
		{"what is machine learning?", "what is machine learning"},
		{"who invented the telephone!", "who invented the telephone"},
		{"where is it??", "where is it"},

		// Contraction expansion (case-insensitive; expansion is always lowercase).
		{"don't know", "do not know"},
		{"it's a test", "it is a test"},
		{"can't find it?", "cannot find it"},
		// Mixed case contraction: I'm → i am (contraction lowercased); rest preserved.
		{"I'm not sure", "i am not sure"},
		{"what's the difference?", "what is the difference"},
		// Uppercase contraction.
		{"DON'T know", "do not know"},

		// Non-contraction words preserve case (so NOT/AND/OR remain uppercase).
		{"machine learning NOT survey", "machine learning NOT survey"},
		{"cat AND dog", "cat AND dog"},

		// Whitespace normalization.
		{"  spaced   out  ", "spaced out"},
		{"multiple  spaces  between words", "multiple spaces between words"},

		// Empty and no-op inputs.
		{"", ""},
		{"machine learning", "machine learning"},
		{"42", "42"},

		// Mixed contraction + trailing ?
		{"doesn't it work?", "does not it work"},

		// Period is kept (abbreviations).
		{"U.S. policy", "U.S. policy"},
	}

	for _, tc := range cases {
		got := PreprocessQuery(tc.in)
		if got != tc.want {
			t.Errorf("PreprocessQuery(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
