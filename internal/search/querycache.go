package search

import (
	lru "github.com/hashicorp/golang-lru/v2"
	"search-eval-platform/pkg/types"
)

// cacheKey is the composite key for the query result cache.
type cacheKey struct {
	query  string
	topK   int
	scorer string
}

// QueryCache caches top-K search results keyed by (query, topK, scorerType).
// It is goroutine-safe. Invalidate() should be called on any write that could
// alter results (index or delete).
type QueryCache struct {
	cache *lru.Cache[cacheKey, []types.SearchResult]
}

// NewQueryCache creates a QueryCache that holds at most size entries.
// Returns nil if size <= 0 (disabled).
func NewQueryCache(size int) *QueryCache {
	if size <= 0 {
		return nil
	}
	c, err := lru.New[cacheKey, []types.SearchResult](size)
	if err != nil {
		return nil
	}
	return &QueryCache{cache: c}
}

// Get returns cached results if present.
func (qc *QueryCache) Get(query string, topK int, scorer string) ([]types.SearchResult, bool) {
	if qc == nil {
		return nil, false
	}
	return qc.cache.Get(cacheKey{query, topK, scorer})
}

// Set stores results in the cache.
func (qc *QueryCache) Set(query string, topK int, scorer string, results []types.SearchResult) {
	if qc == nil {
		return
	}
	qc.cache.Add(cacheKey{query, topK, scorer}, results)
}

// Invalidate purges all cached entries. Call after any write operation.
func (qc *QueryCache) Invalidate() {
	if qc == nil {
		return
	}
	qc.cache.Purge()
}
