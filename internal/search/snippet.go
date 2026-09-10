package search

import (
	"math"
	"regexp"
	"strings"
	"unicode"
)

const snippetMaxLen = 200

// sentenceBoundary matches sentence-ending punctuation followed by whitespace or end of string.
var sentenceBoundary = regexp.MustCompile(`[.!?\n]+`)

// splitSentences splits text into sentences on '.', '!', '?', '\n'.
// Returns a slice of (start, end) byte index pairs.
type sentenceSpan struct{ start, end int }

func splitSentences(text string) []sentenceSpan {
	var spans []sentenceSpan
	indices := sentenceBoundary.FindAllStringIndex(text, -1)
	prev := 0
	for _, loc := range indices {
		end := loc[1]
		if end > prev {
			spans = append(spans, sentenceSpan{prev, end})
		}
		prev = end
	}
	if prev < len(text) {
		spans = append(spans, sentenceSpan{prev, len(text)})
	}
	return spans
}

// sentenceStartBefore returns the start of the sentence that contains or
// immediately precedes position pos, or 0 if pos is in the first sentence.
func sentenceStartAt(spans []sentenceSpan, pos int) int {
	for _, s := range spans {
		if pos >= s.start && pos < s.end {
			return s.start
		}
	}
	return 0
}

// sentenceEndAfter returns the end of the sentence that contains position pos,
// or len(text) if pos is in the last sentence.
func sentenceEndAfter(spans []sentenceSpan, pos int) int {
	for _, s := range spans {
		if pos >= s.start && pos <= s.end {
			return s.end
		}
	}
	return pos
}

// ExtractSnippet returns a query-centred snippet from text.
//
// Tier 1 (sentence-aware):
//   - Prefers windows that start at a sentence boundary.
//   - Trims the snippet to the nearest sentence end so it never cuts mid-sentence.
//
// If no sentence boundary is found near the window, falls back to word-boundary
// alignment (original behaviour).
func ExtractSnippet(text string, tokens []string) string {
	return extractSnippetWithIDF(text, tokens, nil)
}

// ExtractSnippetWithIDF is the IDF-weighted variant (Tier 2).
// termIDFs maps each query token to its IDF weight (log(N/df)).
// If termIDFs is nil the function falls back to equal-weight scoring.
func ExtractSnippetWithIDF(text string, tokens []string, termIDFs map[string]float64) string {
	return extractSnippetWithIDF(text, tokens, termIDFs)
}

func extractSnippetWithIDF(text string, tokens []string, termIDFs map[string]float64) string {
	if len(text) == 0 {
		return ""
	}
	lower := strings.ToLower(text)
	spans := splitSentences(text)

	// Score each token occurrence by its IDF; pick the position with highest score.
	hitPos := -1
	hitScore := -1.0
	for _, tok := range tokens {
		if tok == "" {
			continue
		}
		idf := 1.0
		if termIDFs != nil {
			if v, ok := termIDFs[tok]; ok {
				idf = v
			}
		}
		idx := strings.Index(lower, tok)
		if idx >= 0 {
			score := idf
			if hitPos < 0 || score > hitScore || (score == hitScore && idx < hitPos) {
				hitPos = idx
				hitScore = score
			}
		}
	}

	// Compute window.
	start := 0
	if hitPos > snippetMaxLen/2 {
		start = hitPos - snippetMaxLen/2
	}

	// Tier 1: try to align start to the nearest sentence boundary.
	sentStart := sentenceStartAt(spans, hitPos)
	if sentStart >= start {
		// Sentence start is within our window — prefer it as the snippet start.
		start = sentStart
	} else {
		// Fall back to word-boundary alignment.
		if start > 0 {
			for start < len(text) && !unicode.IsSpace(rune(text[start])) {
				start++
			}
			for start < len(text) && unicode.IsSpace(rune(text[start])) {
				start++
			}
		}
	}

	end := start + snippetMaxLen
	if end > len(text) {
		end = len(text)
	}

	// Tier 1: trim to nearest sentence end.
	sentEnd := sentenceEndAfter(spans, end-1)
	if sentEnd > start && sentEnd <= end {
		end = sentEnd
	} else if end < len(text) {
		// Fall back to word-boundary alignment.
		for end > start && !unicode.IsSpace(rune(text[end-1])) {
			end--
		}
	}

	snippet := strings.TrimSpace(text[start:end])

	// Prefix an ellipsis if we didn't start from the beginning.
	if start > 0 {
		snippet = "…" + snippet
	}
	// Suffix an ellipsis if we didn't reach the end.
	if end < len(text) {
		snippet = snippet + "…"
	}
	return snippet
}

// buildTermIDFs constructs a termIDFs map from a scorer's IDF values.
// N is total doc count; df maps token → doc frequency.
// Returns nil when N or df is empty.
func buildTermIDFs(N int, df map[string]int) map[string]float64 {
	if N <= 0 || len(df) == 0 {
		return nil
	}
	out := make(map[string]float64, len(df))
	for term, d := range df {
		if d <= 0 {
			d = 1
		}
		out[term] = math.Log(float64(N) / float64(d))
	}
	return out
}
