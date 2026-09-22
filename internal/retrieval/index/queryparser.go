package index

import "strings"

// ParsedQuery is the result of parsing a preprocessed query string.
// It separates tokens by their boolean role so the search layer can
// apply appropriate BM25 scoring and post-filtering.
type ParsedQuery struct {
	// Must contains tokens that every result document must contain.
	// Derived from explicit AND clauses and +prefix terms.
	Must []string

	// Should contains tokens scored by BM25; results need not contain all of them.
	// This is the default role for plain terms (OR semantics).
	Should []string

	// Not contains tokens whose documents are excluded from results.
	// Derived from NOT keyword and -prefix terms.
	Not []string

	// Phrases is a list of tokenized phrases extracted from "quoted strings".
	// Each element is an ordered token slice; results must contain the phrase
	// as a contiguous substring in stored field text.
	Phrases [][]string
}

// AllTokens returns a deduplicated union of Must and Should tokens.
// These are the tokens actually scored by MaxScore; Not is excluded.
// Phrase tokens are already merged into Should during parsing.
func (pq *ParsedQuery) AllTokens() []string {
	seen := make(map[string]struct{}, len(pq.Must)+len(pq.Should))
	out := make([]string, 0, len(pq.Must)+len(pq.Should))
	for _, t := range pq.Must {
		if _, ok := seen[t]; !ok {
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}
	for _, t := range pq.Should {
		if _, ok := seen[t]; !ok {
			seen[t] = struct{}{}
			out = append(out, t)
		}
	}
	return out
}

// HasFilters reports whether any post-search filtering is required
// (Must, Not, or Phrase constraints are present).
func (pq *ParsedQuery) HasFilters() bool {
	return len(pq.Must) > 0 || len(pq.Not) > 0 || len(pq.Phrases) > 0
}

// ParseQuery parses a preprocessed query string into a ParsedQuery.
// The tokenizer and optional synonym map are applied to each parsed component.
//
// Supported syntax (after PreprocessQuery lowercases the input):
//
//	plain terms          → Should (default, OR-BM25 semantics)
//	+term                → Must   (doc must contain this token)
//	-term                → Not    (doc must NOT contain this token)
//	NOT term             → Not
//	termA AND termB      → both termA and termB become Must
//	"quoted phrase"      → Phrase + Should (scored AND substring-filtered)
//	OR                   → explicit OR (no-op; already the default)
//
// Unknown boolean operators are treated as plain terms.
func ParseQuery(q string, tok *Tokenizer, synonyms *SynonymMap) *ParsedQuery {
	pq := &ParsedQuery{}
	words := strings.Fields(q)

	nextNot := false
	nextMust := false
	// lastWordTokens holds the token(s) the most recently processed
	// Should-bound word (or phrase) maps to — a single word can expand into
	// several (synonym expansion, or a multi-word phrase) — so AND can
	// promote all of them to Must, not just one. Tracked by value rather
	// than by position in pq.Should: if the word's tokens were already
	// present from an earlier word (appendUniq dedup), it may add nothing
	// new to Should, but the tokens are still sitting in Should from their
	// first occurrence and AND must still find and promote them there.
	var lastWordTokens []string

	for i := 0; i < len(words); {
		w := words[i]

		// Quoted phrase: collect words until the closing quote.
		if strings.HasPrefix(w, `"`) {
			phrase := strings.TrimPrefix(w, `"`)
			for !strings.HasSuffix(phrase, `"`) && i+1 < len(words) {
				i++
				phrase += " " + words[i]
			}
			phrase = strings.TrimSuffix(phrase, `"`)
			tokens := tok.Tokenize(phrase)
			if len(tokens) > 0 {
				switch {
				case nextNot:
					// A preceding "-"/"NOT" excludes docs containing any of
					// these words — the same word-level exclusion -term/NOT
					// term use. Not added to Phrases: that slot is purely an
					// inclusive substring filter, so an excluded phrase must
					// never end up positively phrase-matched there.
					pq.Not = appendUniq(pq.Not, tokens)
				case nextMust:
					pq.Phrases = append(pq.Phrases, tokens)
					lastWordTokens = tokens
					pq.Must = appendUniq(pq.Must, tokens)
				default:
					pq.Phrases = append(pq.Phrases, tokens)
					// Also add phrase tokens to Should so they participate in BM25 scoring.
					lastWordTokens = tokens
					pq.Should = appendUniq(pq.Should, tokens)
				}
			}
			nextNot = false
			nextMust = false
			i++
			continue
		}

		// Boolean keywords — only recognised when the word is explicitly uppercase
		// (user intent). Lowercase "not"/"and"/"or" from natural language or
		// contraction expansion are treated as plain terms.
		switch w {
		case "AND":
			// Retroactively promote every token the previous word maps to —
			// not just the last one, since a single word can expand into
			// several via synonym expansion or a phrase — to Must. Both
			// sides of "A AND B" become required in full. Looked up by
			// value in pq.Should (not by position) so a token still gets
			// promoted even if the word contributed nothing *new* to
			// Should because it duplicated an earlier word.
			for _, tk := range lastWordTokens {
				for idx, s := range pq.Should {
					if s == tk {
						pq.Should = append(pq.Should[:idx], pq.Should[idx+1:]...)
						pq.Must = appendUniq(pq.Must, []string{tk})
						break
					}
				}
			}
			lastWordTokens = nil
			nextMust = true
			i++
			continue
		case "OR":
			nextMust = false
			nextNot = false
			i++
			continue
		case "NOT":
			nextNot = true
			i++
			continue
		}

		// Prefix operators: +term / -term.
		if strings.HasPrefix(w, "-") && len(w) > 1 {
			w = w[1:]
			nextNot = true
		} else if strings.HasPrefix(w, "+") && len(w) > 1 {
			w = w[1:]
			nextMust = true
		}

		tokens := tok.TokenizeWithSynonyms(w, synonyms)

		switch {
		case nextNot:
			pq.Not = appendUniq(pq.Not, tokens)
			nextNot = false
		case nextMust:
			pq.Must = appendUniq(pq.Must, tokens)
			nextMust = false
		default:
			lastWordTokens = tokens
			pq.Should = appendUniq(pq.Should, tokens)
		}

		i++
	}

	return pq
}

// appendUniq appends elements of src to dst, skipping duplicates.
func appendUniq(dst, src []string) []string {
	for _, s := range src {
		found := false
		for _, d := range dst {
			if d == s {
				found = true
				break
			}
		}
		if !found {
			dst = append(dst, s)
		}
	}
	return dst
}
