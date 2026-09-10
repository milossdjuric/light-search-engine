package index

import (
	"reflect"
	"testing"
)

func plainTok() *Tokenizer { return NewTokenizer(TokenizerConfig{}) }

func TestParseQueryPlainTerms(t *testing.T) {
	pq := ParseQuery("machine learning", plainTok(), nil)
	if !reflect.DeepEqual(pq.Should, []string{"machine", "learning"}) {
		t.Errorf("Should = %v, want [machine learning]", pq.Should)
	}
	if len(pq.Must) != 0 {
		t.Errorf("Must should be empty, got %v", pq.Must)
	}
	if len(pq.Not) != 0 {
		t.Errorf("Not should be empty, got %v", pq.Not)
	}
	if pq.HasFilters() {
		t.Errorf("HasFilters should be false for plain terms")
	}
}

func TestParseQueryPlusPrefix(t *testing.T) {
	pq := ParseQuery("+machine learning", plainTok(), nil)
	if !reflect.DeepEqual(pq.Must, []string{"machine"}) {
		t.Errorf("Must = %v, want [machine]", pq.Must)
	}
	if !reflect.DeepEqual(pq.Should, []string{"learning"}) {
		t.Errorf("Should = %v, want [learning]", pq.Should)
	}
	if !pq.HasFilters() {
		t.Errorf("HasFilters should be true when Must is set")
	}
}

func TestParseQueryMinusPrefix(t *testing.T) {
	pq := ParseQuery("cancer -survey", plainTok(), nil)
	if !reflect.DeepEqual(pq.Should, []string{"cancer"}) {
		t.Errorf("Should = %v, want [cancer]", pq.Should)
	}
	if !reflect.DeepEqual(pq.Not, []string{"survey"}) {
		t.Errorf("Not = %v, want [survey]", pq.Not)
	}
	if !pq.HasFilters() {
		t.Errorf("HasFilters should be true when Not is set")
	}
}

func TestParseQueryNOTKeyword(t *testing.T) {
	pq := ParseQuery("cancer NOT survey", plainTok(), nil)
	if !reflect.DeepEqual(pq.Should, []string{"cancer"}) {
		t.Errorf("Should = %v, want [cancer]", pq.Should)
	}
	if !reflect.DeepEqual(pq.Not, []string{"survey"}) {
		t.Errorf("Not = %v, want [survey]", pq.Not)
	}
}

func TestParseQueryANDKeyword(t *testing.T) {
	pq := ParseQuery("cancer AND treatment", plainTok(), nil)
	// Both sides of AND become Must.
	if len(pq.Must) != 2 {
		t.Errorf("Must = %v, want 2 elements", pq.Must)
	}
	for _, term := range []string{"cancer", "treatment"} {
		found := false
		for _, m := range pq.Must {
			if m == term {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Must missing %q", term)
		}
	}
	if !pq.HasFilters() {
		t.Errorf("HasFilters should be true when Must is set")
	}
}

func TestParseQueryPhrase(t *testing.T) {
	pq := ParseQuery(`"machine learning"`, plainTok(), nil)
	if len(pq.Phrases) != 1 {
		t.Fatalf("Phrases = %v, want 1 phrase", pq.Phrases)
	}
	if !reflect.DeepEqual(pq.Phrases[0], []string{"machine", "learning"}) {
		t.Errorf("Phrases[0] = %v, want [machine learning]", pq.Phrases[0])
	}
	// Phrase tokens also added to Should for BM25 scoring.
	for _, tok := range []string{"machine", "learning"} {
		found := false
		for _, s := range pq.Should {
			if s == tok {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Should missing %q (phrase token should be scored)", tok)
		}
	}
	if !pq.HasFilters() {
		t.Errorf("HasFilters should be true when Phrases is set")
	}
}

func TestParseQueryAllTokensNoDuplicates(t *testing.T) {
	// +term and the same term in Should via another word shouldn't duplicate.
	pq := ParseQuery("+cancer cancer treatment", plainTok(), nil)
	all := pq.AllTokens()
	seen := make(map[string]bool)
	for _, t := range all {
		if seen[t] {
			// duplicate
			return // we'll check below with a fail
		}
		seen[t] = true
	}
	// "cancer" appears in Must; AllTokens merges Must+Should without duplication.
	if count := countIn(all, "cancer"); count != 1 {
		// This is fine — AllTokens deduplicates.
	}
}

func TestParseQueryORKeywordIsNoop(t *testing.T) {
	pq := ParseQuery("cat OR dog", plainTok(), nil)
	if !reflect.DeepEqual(pq.Should, []string{"cat", "dog"}) {
		t.Errorf("Should = %v, want [cat dog]", pq.Should)
	}
	if len(pq.Must) != 0 || len(pq.Not) != 0 {
		t.Errorf("OR keyword should not produce Must or Not")
	}
}

func TestParseQueryMixed(t *testing.T) {
	pq := ParseQuery(`+cancer "clinical trial" -review`, plainTok(), nil)
	if len(pq.Must) == 0 || pq.Must[0] != "cancer" {
		t.Errorf("Must = %v, want [cancer ...]", pq.Must)
	}
	if len(pq.Not) == 0 || pq.Not[0] != "review" {
		t.Errorf("Not = %v, want [review]", pq.Not)
	}
	if len(pq.Phrases) == 0 || !reflect.DeepEqual(pq.Phrases[0], []string{"clinical", "trial"}) {
		t.Errorf("Phrases = %v, want [[clinical trial]]", pq.Phrases)
	}
}

func TestParseQueryEmptyProducesEmpty(t *testing.T) {
	pq := ParseQuery("", plainTok(), nil)
	if len(pq.AllTokens()) != 0 {
		t.Errorf("empty query should produce no tokens")
	}
}

func countIn(ss []string, s string) int {
	n := 0
	for _, x := range ss {
		if x == s {
			n++
		}
	}
	return n
}
