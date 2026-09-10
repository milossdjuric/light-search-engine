package index

import "strings"

// contractions maps lowercase English contractions to their expansions.
// Checked in PreprocessQuery before the tokenizer runs.
var contractions = map[string]string{
	"don't":   "do not",
	"doesn't": "does not",
	"didn't":  "did not",
	"can't":   "cannot",
	"won't":   "will not",
	"isn't":   "is not",
	"aren't":  "are not",
	"wasn't":  "was not",
	"weren't": "were not",
	"hasn't":  "has not",
	"haven't": "have not",
	"hadn't":  "had not",
	"wouldn't": "would not",
	"couldn't": "could not",
	"shouldn't": "should not",
	"i'm":     "i am",
	"i've":    "i have",
	"i'll":    "i will",
	"i'd":     "i would",
	"you're":  "you are",
	"you've":  "you have",
	"you'll":  "you will",
	"you'd":   "you would",
	"he's":    "he is",
	"she's":   "she is",
	"it's":    "it is",
	"we're":   "we are",
	"we've":   "we have",
	"we'll":   "we will",
	"we'd":    "we would",
	"they're": "they are",
	"they've": "they have",
	"they'll": "they will",
	"they'd":  "they would",
	"what's":  "what is",
	"what're": "what are",
	"where's": "where is",
	"when's":  "when is",
	"who's":   "who is",
	"that's":  "that is",
	"there's": "there is",
	"here's":  "here is",
	"let's":   "let us",
}

// PreprocessQuery normalizes a raw query string before tokenization or parsing.
//
// It applies three transformations in order:
//  1. Contraction expansion — "don't" → "do not", "it's" → "it is", etc.
//     Matching is case-insensitive; expansions are always lowercase so they
//     don't accidentally trigger ParseQuery's uppercase boolean operators
//     (NOT, AND, OR). Case of non-contraction words is preserved so the parser
//     can distinguish user-typed "NOT" from natural-language "not".
//  2. Trailing punctuation strip — removes trailing '?' and '!' that users
//     commonly append to conversational queries.
//  3. Whitespace normalization — collapses multiple spaces to one.
func PreprocessQuery(q string) string {
	q = strings.TrimSpace(q)
	if q == "" {
		return q
	}

	// Expand contractions case-insensitively.
	// We rebuild word-by-word to avoid partial replacements inside longer words.
	words := strings.Fields(q)
	for i, w := range words {
		if exp, ok := contractions[strings.ToLower(w)]; ok {
			words[i] = exp // expansion is lowercase
		}
	}
	q = strings.Join(words, " ")

	// Strip trailing ? or ! (keep . to preserve abbreviations in phrases).
	q = strings.TrimRight(q, "?!")

	// Collapse repeated whitespace.
	q = strings.Join(strings.Fields(q), " ")
	return q
}
