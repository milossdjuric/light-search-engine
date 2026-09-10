package index

import "github.com/kljensen/snowball"

// TokenizerConfig controls the tokenization pipeline.
type TokenizerConfig struct {
	StopwordsEnabled bool
	StemmingEnabled  bool
	StemmingLanguage string // Snowball language name, e.g. "english"
}

// Tokenizer encapsulates the full tokenization pipeline:
// split → [stopwords filter] → [stemming].
//
// The package-level Tokenize() function remains unchanged for backward
// compatibility with the eval runner (cmd/runner). New code should use
// NewTokenizer and call Tokenize on the returned instance.
type Tokenizer struct {
	cfg TokenizerConfig
}

// NewTokenizer returns a Tokenizer configured according to cfg.
func NewTokenizer(cfg TokenizerConfig) *Tokenizer {
	if cfg.StemmingLanguage == "" {
		cfg.StemmingLanguage = "english"
	}
	return &Tokenizer{cfg: cfg}
}

// Tokenize runs the full pipeline on text and returns the resulting tokens.
func (t *Tokenizer) Tokenize(text string) []string {
	tokens := Tokenize(text) // package-level split
	if t.cfg.StopwordsEnabled {
		tokens = FilterStopwords(tokens)
	}
	if t.cfg.StemmingEnabled && len(tokens) > 0 {
		stemmed := make([]string, 0, len(tokens))
		for _, tok := range tokens {
			s, err := snowball.Stem(tok, t.cfg.StemmingLanguage, true)
			if err == nil && s != "" {
				stemmed = append(stemmed, s)
			} else {
				stemmed = append(stemmed, tok)
			}
		}
		return stemmed
	}
	return tokens
}

// TokenizeWithSynonyms runs the full tokenization pipeline and then expands
// each token using the provided SynonymMap. Synonyms are appended after the
// original tokens; duplicates are removed. If synonyms is nil, behaves
// identically to Tokenize.
func (t *Tokenizer) TokenizeWithSynonyms(text string, synonyms *SynonymMap) []string {
	base := t.Tokenize(text)
	if synonyms == nil {
		return base
	}
	seen := make(map[string]struct{}, len(base)*2)
	result := make([]string, 0, len(base)*2)
	for _, tok := range base {
		if _, ok := seen[tok]; !ok {
			seen[tok] = struct{}{}
			result = append(result, tok)
		}
		for _, syn := range synonyms.expand[tok] {
			if _, ok := seen[syn]; !ok {
				seen[syn] = struct{}{}
				result = append(result, syn)
			}
		}
	}
	return result
}
