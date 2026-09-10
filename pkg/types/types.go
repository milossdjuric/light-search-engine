package types

// Document is a text document with optional metadata.
type Document struct {
	ID       string
	Text     string
	Fields   map[string]string // named fields for BM25F (e.g. "title", "body")
	Metadata map[string]string
}

// ScoredDocument is a retrieval result with a score and rank.
type ScoredDocument struct {
	DocID string
	Score float64
	Rank  int
}

// SearchResult is a single hit returned by the search API.
type SearchResult struct {
	DocID    string            `json:"doc_id"`
	Score    float64           `json:"score"`
	Rank     int               `json:"rank"`
	Snippet  string            `json:"snippet,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}
