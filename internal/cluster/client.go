package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"sync"
	"time"

	"search-eval-platform/pkg/types"
)

const (
	rrfK        = 60
	httpTimeout = 10 * time.Second
)

// Client is the coordinator client that routes requests to shard nodes.
// It supports two search modes:
//   - "score" – single-phase, local IDF, merge by raw score (default; matches ES query_then_fetch)
//   - "rrf"   – single-phase, local IDF, Reciprocal Rank Fusion merge
type Client struct {
	ring           *Ring
	consistentRing *ConsistentRing
	breakers       map[string]*CircuitBreaker // keyed by NodeID
	cbMu           sync.RWMutex
	httpCli        *http.Client
	fusionMode     string // "score" | "rrf"
	totalShards    int
}

// NewClient creates a coordinator Client.
// vnodes is the number of virtual nodes per shard for consistent-hash routing; 0 defaults to 150.
func NewClient(ring *Ring, fusionMode string, vnodes int) *Client {
	return &Client{
		ring:           ring,
		consistentRing: BuildConsistentRing(ring.TotalShards(), vnodes),
		breakers:       make(map[string]*CircuitBreaker),
		httpCli:        &http.Client{Timeout: httpTimeout},
		fusionMode:     fusionMode,
		totalShards:    ring.TotalShards(),
	}
}

// shardFor routes a docID to a shard index via the consistent hash ring.
func (c *Client) shardFor(docID string) int {
	return c.consistentRing.ShardFor(docID)
}

// Breaker returns the CircuitBreaker for nodeID, creating it if needed.
// Exported so the health poller can share the same breaker instances.
func (c *Client) Breaker(nodeID string) *CircuitBreaker {
	return c.breaker(nodeID)
}

// breaker returns (creating if needed) the CircuitBreaker for nodeID.
func (c *Client) breaker(nodeID string) *CircuitBreaker {
	c.cbMu.RLock()
	cb, ok := c.breakers[nodeID]
	c.cbMu.RUnlock()
	if ok {
		return cb
	}
	c.cbMu.Lock()
	if cb, ok = c.breakers[nodeID]; !ok {
		cb = NewCircuitBreaker(3, 30*time.Second)
		c.breakers[nodeID] = cb
	}
	c.cbMu.Unlock()
	return cb
}

// nodeFor returns the node to route shardID's request to, plus the
// breaker's done callback that the caller MUST invoke with the request's
// success/failure once it completes.
//
// Routing is decided via each candidate's Allow(), not a plain IsOpen()
// read: Allow() is the only method that can transition a breaker from Open
// to HalfOpen (once its reset timeout has elapsed) and admit a probe. A
// stale IsOpen() check never does that on its own, so if nodeFor only ever
// consulted IsOpen() and filtered out any Open primary before a real
// request could be attempted against it, nothing would ever call Allow()
// on that breaker again — permanently quarantining a primary even after it
// recovers.
func (c *Client) nodeFor(shardID int) (node *NodeMeta, done func(bool), degraded bool, err error) {
	prim, err := c.ring.Primary(shardID)
	if err != nil {
		return nil, nil, false, err
	}
	if allowed, d := c.breaker(prim.NodeID).Allow(); allowed {
		return prim, d, false, nil
	}
	// Primary denied — try a replica.
	for _, rep := range c.ring.ReplicasForShard(shardID) {
		if allowed, d := c.breaker(rep.NodeID).Allow(); allowed {
			slog.Warn("client: primary unavailable, using replica",
				"shard", shardID, "primary", prim.NodeID, "replica", rep.NodeID)
			return rep, d, true, nil
		}
	}
	return nil, nil, true, fmt.Errorf("client: no available node for shard %d", shardID)
}

// IndexDoc routes a single document to the owning shard.
func (c *Client) IndexDoc(ctx context.Context, doc types.Document) error {
	shard := c.shardFor(doc.ID)
	node, done, _, err := c.nodeFor(shard)
	if err != nil {
		return err
	}
	body, _ := json.Marshal(doc)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"http://"+node.HTTPAddr+"/index", bytes.NewReader(body))
	if err != nil {
		done(false)
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	return c.doWithDone(req, done)
}

// DeleteDoc routes a delete to the owning shard.
func (c *Client) DeleteDoc(ctx context.Context, docID string) error {
	shard := c.shardFor(docID)
	node, done, _, err := c.nodeFor(shard)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete,
		"http://"+node.HTTPAddr+"/index/"+url.PathEscape(docID), nil)
	if err != nil {
		done(false)
		return err
	}
	return c.doWithDone(req, done)
}

// searchNodeResult is an intermediate per-node search result.
type searchNodeResult struct {
	nodeID   string
	docs     []types.SearchResult
	degraded bool
	err      error
}

// shardSearchResponse mirrors the server's SearchResponse JSON.
type shardSearchResponse struct {
	Results []types.SearchResult `json:"results"`
	Total   int                  `json:"total"`
	TookMs  int64                `json:"took_ms"`
}

// Search fans out to all shards.
func (c *Client) Search(ctx context.Context, query string, topK int, retriever, fusion string) ([]types.SearchResult, bool, error) {
	switch c.fusionMode {
	case "rrf":
		return c.searchRRF(ctx, query, topK, retriever, fusion)
	default: // "score"
		return c.searchScore(ctx, query, topK, retriever, fusion)
	}
}

// searchRRF fans out to all nodes in parallel, collects per-node top-K results,
// and merges with Reciprocal Rank Fusion.
func (c *Client) searchRRF(ctx context.Context, query string, topK int, retriever, fusion string) ([]types.SearchResult, bool, error) {
	nodes := c.ring.Nodes()
	results := make(chan searchNodeResult, len(nodes))

	for _, n := range nodes {
		n := n
		go func() {
			docs, degraded, err := c.searchNode(ctx, n, query, topK, retriever, fusion)
			results <- searchNodeResult{nodeID: n.NodeID, docs: docs, degraded: degraded, err: err}
		}()
	}

	var lists [][]types.SearchResult
	degraded := false
	var firstErr error

	for range nodes {
		r := <-results
		if r.degraded {
			degraded = true
		}
		if r.err != nil {
			slog.Warn("client: node search failed", "node", r.nodeID, "err", r.err)
			if firstErr == nil {
				firstErr = r.err
			}
			continue
		}
		if len(r.docs) > 0 {
			lists = append(lists, r.docs)
		}
	}

	if len(lists) == 0 {
		return nil, degraded, firstErr
	}
	return rrfMergeResults(lists, topK), degraded, nil
}

// searchScore fans out to all nodes in parallel, each scoring with its local IDF,
// and merges by raw score. With uniform FNV routing local IDF == global IDF so
// scores are directly comparable across nodes. Matches ES query_then_fetch.
func (c *Client) searchScore(ctx context.Context, query string, topK int, retriever, fusion string) ([]types.SearchResult, bool, error) {
	nodes := c.ring.Nodes()
	fetchK := topK * len(nodes)
	results := make(chan searchNodeResult, len(nodes))

	for _, n := range nodes {
		n := n
		go func() {
			docs, degraded, err := c.searchNode(ctx, n, query, fetchK, retriever, fusion)
			results <- searchNodeResult{nodeID: n.NodeID, docs: docs, degraded: degraded, err: err}
		}()
	}

	var lists [][]types.SearchResult
	degraded := false
	var firstErr error

	for range nodes {
		r := <-results
		if r.degraded {
			degraded = true
		}
		if r.err != nil {
			slog.Warn("client: node search failed", "node", r.nodeID, "err", r.err)
			if firstErr == nil {
				firstErr = r.err
			}
			continue
		}
		if len(r.docs) > 0 {
			lists = append(lists, r.docs)
		}
	}

	if len(lists) == 0 {
		return nil, degraded, firstErr
	}
	return mergeResultsByScore(lists, topK), degraded, nil
}

// searchNode performs a search request to one node, with circuit-breaker
// failover to a replica if needed.
func (c *Client) searchNode(
	ctx context.Context,
	node *NodeMeta,
	query string,
	topK int,
	retriever string,
	fusion string,
) ([]types.SearchResult, bool /*degraded*/, error) {
	cb := c.breaker(node.NodeID)
	allowed, done := cb.Allow()
	if !allowed {
		return nil, true, fmt.Errorf("client: circuit breaker open for %s", node.NodeID)
	}

	params := url.Values{
		"q":         {query},
		"top_k":     {strconv.Itoa(topK)},
		"retriever": {retriever},
	}
	if fusion != "" {
		params.Set("fusion", fusion)
	}

	var resp shardSearchResponse
	err := c.getJSON(ctx, node, "/search?"+params.Encode(), &resp)
	done(err == nil)
	if err != nil {
		return nil, false, err
	}
	return resp.Results, false, nil
}

func (c *Client) postJSONDecode(ctx context.Context, node *NodeMeta, path string, body []byte, dst interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"http://"+node.HTTPAddr+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	return c.doDecodeWithBreaker(node, req, dst)
}

func (c *Client) getJSON(ctx context.Context, node *NodeMeta, pathAndQuery string, dst interface{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		"http://"+node.HTTPAddr+pathAndQuery, nil)
	if err != nil {
		return err
	}

	return c.doDecodeWithBreaker(node, req, dst)
}

// doWithDone performs req and reports its outcome via done, obtained from a
// prior CircuitBreaker.Allow() call (nodeFor) — unlike doWithBreaker, it
// does not call Allow() itself, so a request routed through nodeFor records
// exactly one breaker outcome instead of two.
func (c *Client) doWithDone(req *http.Request, done func(bool)) error {
	resp, err := c.httpCli.Do(req)
	if err != nil {
		done(false)
		return err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	ok := resp.StatusCode < 500
	done(ok)
	if !ok {
		return fmt.Errorf("client: node returned %d", resp.StatusCode)
	}
	return nil
}

func (c *Client) doWithBreaker(node *NodeMeta, req *http.Request) error {
	cb := c.breaker(node.NodeID)
	allowed, done := cb.Allow()
	if !allowed {
		return fmt.Errorf("client: circuit breaker open for %s", node.NodeID)
	}
	resp, err := c.httpCli.Do(req)
	if err != nil {
		done(false)
		return err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	ok := resp.StatusCode < 500
	done(ok)
	if !ok {
		return fmt.Errorf("client: node %s returned %d", node.NodeID, resp.StatusCode)
	}
	return nil
}

func (c *Client) doDecodeWithBreaker(node *NodeMeta, req *http.Request, dst interface{}) error {
	cb := c.breaker(node.NodeID)
	allowed, done := cb.Allow()
	if !allowed {
		return fmt.Errorf("client: circuit breaker open for %s", node.NodeID)
	}
	resp, err := c.httpCli.Do(req)
	if err != nil {
		done(false)
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 500 {
		io.Copy(io.Discard, resp.Body)
		done(false)
		return fmt.Errorf("client: node %s returned %d", node.NodeID, resp.StatusCode)
	}
	err = json.NewDecoder(resp.Body).Decode(dst)
	done(err == nil && resp.StatusCode < 500)
	return err
}

// mergeResultsByScore merges per-node result lists by raw score, keeping topK.
// When routing is uniform, local IDF == global IDF so scores are directly comparable.
func mergeResultsByScore(lists [][]types.SearchResult, topK int) []types.SearchResult {
	scores := make(map[string]float64)
	snippets := make(map[string]string)
	metas := make(map[string]map[string]string)

	for _, list := range lists {
		for _, doc := range list {
			if doc.Score > scores[doc.DocID] {
				scores[doc.DocID] = doc.Score
			}
			if snippets[doc.DocID] == "" && doc.Snippet != "" {
				snippets[doc.DocID] = doc.Snippet
			}
			if metas[doc.DocID] == nil && doc.Metadata != nil {
				metas[doc.DocID] = doc.Metadata
			}
		}
	}

	type entry struct {
		docID string
		score float64
	}
	entries := make([]entry, 0, len(scores))
	for docID, score := range scores {
		entries = append(entries, entry{docID, score})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].score != entries[j].score {
			return entries[i].score > entries[j].score
		}
		return entries[i].docID < entries[j].docID
	})
	if len(entries) > topK {
		entries = entries[:topK]
	}

	out := make([]types.SearchResult, len(entries))
	for i, e := range entries {
		out[i] = types.SearchResult{
			DocID:    e.docID,
			Score:    e.score,
			Snippet:  snippets[e.docID],
			Metadata: metas[e.docID],
		}
	}
	return out
}

func rrfMergeResults(lists [][]types.SearchResult, topK int) []types.SearchResult {
	scores := make(map[string]float64)
	snippets := make(map[string]string)
	metas := make(map[string]map[string]string)

	for _, list := range lists {
		for rank, doc := range list {
			scores[doc.DocID] += 1.0 / float64(rrfK+rank+1)
			if snippets[doc.DocID] == "" && doc.Snippet != "" {
				snippets[doc.DocID] = doc.Snippet
			}
			if metas[doc.DocID] == nil && doc.Metadata != nil {
				metas[doc.DocID] = doc.Metadata
			}
		}
	}

	type entry struct {
		docID string
		score float64
	}
	entries := make([]entry, 0, len(scores))
	for docID, score := range scores {
		entries = append(entries, entry{docID, score})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].score != entries[j].score {
			return entries[i].score > entries[j].score
		}
		return entries[i].docID < entries[j].docID
	})
	if len(entries) > topK {
		entries = entries[:topK]
	}

	out := make([]types.SearchResult, len(entries))
	for i, e := range entries {
		out[i] = types.SearchResult{
			DocID:    e.docID,
			Score:    e.score,
			Rank:     i + 1,
			Snippet:  snippets[e.docID],
			Metadata: metas[e.docID],
		}
	}
	return out
}
