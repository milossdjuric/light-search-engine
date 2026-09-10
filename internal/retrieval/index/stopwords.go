package index

import (
	"github.com/bbalet/stopwords"
)

func init() {
	// Keep numeric digits — consistent with Tokenize() which treats digits
	// as valid token characters (e.g. "bm25", "mp3", "ipv6").
	stopwords.DontStripDigits()
}

// FilterStopwords removes English stop words from a token slice.
// It rejoins the tokens into a space-separated string, delegates to
// bbalet/stopwords (which supports 24 languages), then re-tokenizes
// the result so the output is consistent with Tokenize().
func FilterStopwords(tokens []string) []string {
	if len(tokens) == 0 {
		return tokens
	}

	// Join tokens into a space-delimited string for the library.
	joined := joinTokens(tokens)

	// CleanString removes stop words and collapses extra whitespace.
	// "en" = English; cleanHTML=false (tokens are already plain text).
	cleaned := stopwords.CleanString(joined, "en", false)

	// Re-tokenize so the result is identical in form to Tokenize() output.
	return Tokenize(cleaned)
}

// joinTokens concatenates tokens with a single space between each.
func joinTokens(tokens []string) string {
	total := len(tokens) - 1
	for _, t := range tokens {
		total += len(t)
	}
	b := make([]byte, 0, total)
	for i, t := range tokens {
		if i > 0 {
			b = append(b, ' ')
		}
		b = append(b, t...)
	}
	return string(b)
}
