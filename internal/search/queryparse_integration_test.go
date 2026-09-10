package search_test

import (
	"context"
	"testing"

	"search-eval-platform/pkg/types"
)

// TestQueryPreprocessContraction checks that a contraction in the query
// matches a document that contains the expanded form.
func TestQueryPreprocessContraction(t *testing.T) {
	sm := newTestShardManager(t, 1)
	ctx := context.Background()

	sm.IndexDoc(ctx, types.Document{ID: "d1", Text: "machine learning does not require manual feature engineering"})
	sm.IndexDoc(ctx, types.Document{ID: "d2", Text: "deep learning and neural networks are powerful"})

	// "doesn't" → "does not" after preprocessing; should match d1.
	results, _, err := sm.Search(ctx, "machine learning doesn't require", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.DocID == "d1" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected d1 in results for contraction query, got %v", results)
	}
}

// TestQueryParserNotFilter checks that -term excludes documents containing
// that term from results.
func TestQueryParserNotFilter(t *testing.T) {
	sm := newTestShardManager(t, 1)
	ctx := context.Background()

	sm.IndexDoc(ctx, types.Document{ID: "survey", Text: "a comprehensive survey of machine learning algorithms"})
	sm.IndexDoc(ctx, types.Document{ID: "paper", Text: "machine learning for medical diagnosis"})

	results, _, err := sm.Search(ctx, "machine learning -survey", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range results {
		if r.DocID == "survey" {
			t.Errorf("document 'survey' should have been excluded by -survey filter")
		}
	}
	found := false
	for _, r := range results {
		if r.DocID == "paper" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("document 'paper' should be in results, got %v", results)
	}
}

// TestQueryParserNOTKeyword checks that NOT keyword excludes documents.
func TestQueryParserNOTKeyword(t *testing.T) {
	sm := newTestShardManager(t, 1)
	ctx := context.Background()

	sm.IndexDoc(ctx, types.Document{ID: "review", Text: "a literature review of deep learning methods"})
	sm.IndexDoc(ctx, types.Document{ID: "study", Text: "empirical study of deep learning on image tasks"})

	results, _, err := sm.Search(ctx, "deep learning NOT review", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range results {
		if r.DocID == "review" {
			t.Errorf("'review' doc should be excluded by NOT review")
		}
	}
}

// TestQueryParserMustFilter checks that +term requires documents to contain
// the term; documents without it are excluded.
func TestQueryParserMustFilter(t *testing.T) {
	sm := newTestShardManager(t, 1)
	ctx := context.Background()

	sm.IndexDoc(ctx, types.Document{ID: "with-cancer", Text: "cancer treatment with immunotherapy shows promise"})
	sm.IndexDoc(ctx, types.Document{ID: "no-cancer", Text: "treatment of autoimmune diseases with biologics"})

	results, _, err := sm.Search(ctx, "+cancer treatment", 10, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range results {
		if r.DocID == "no-cancer" {
			t.Errorf("'no-cancer' doc should be excluded by +cancer filter")
		}
	}
	found := false
	for _, r := range results {
		if r.DocID == "with-cancer" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("'with-cancer' should be in results, got %v", results)
	}
}

// TestQueryParserTrailingQuestionMark checks that a trailing ? does not
// break retrieval and returns the same results as without it.
func TestQueryParserTrailingQuestionMark(t *testing.T) {
	sm := newTestShardManager(t, 1)
	ctx := context.Background()

	sm.IndexDoc(ctx, types.Document{ID: "d1", Text: "how does photosynthesis work in plants"})
	sm.IndexDoc(ctx, types.Document{ID: "d2", Text: "respiration process in living organisms"})

	withQ, _, err := sm.Search(ctx, "photosynthesis?", 10, nil)
	if err != nil {
		t.Fatalf("Search with ?: %v", err)
	}
	withoutQ, _, err := sm.Search(ctx, "photosynthesis", 10, nil)
	if err != nil {
		t.Fatalf("Search without ?: %v", err)
	}
	if len(withQ) != len(withoutQ) {
		t.Errorf("trailing ? changed result count: %d vs %d", len(withQ), len(withoutQ))
	}
}
