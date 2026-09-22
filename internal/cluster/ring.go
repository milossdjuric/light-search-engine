package cluster

import (
	"fmt"
	"hash/fnv"
	"sort"
	"sync"
)

// NodeMeta describes a cluster node. It is carried in Memberlist metadata
// and also used for static config in standalone/shard modes.
type NodeMeta struct {
	NodeID    string   `json:"node_id"`
	HTTPAddr  string   `json:"http_addr"`
	GRPCAddr  string   `json:"grpc_addr"`
	Shards    []int    `json:"shards"`     // shard IDs owned by this node
	Role      string   `json:"role"`       // "coordinator" | "shard"
}

// Ring maps shard IDs to their primary node and replicas.
// It is rebuilt atomically when membership changes.
type Ring struct {
	mu          sync.RWMutex
	primary     map[int]*NodeMeta   // shardID → primary
	replicas    map[int][]*NodeMeta // shardID → ordered replica list
	nodes       []*NodeMeta
	totalShards int
}

// NewRing creates an empty ring for the given total shard count.
func NewRing(totalShards int) *Ring {
	return &Ring{
		totalShards: totalShards,
		primary:     make(map[int]*NodeMeta),
		replicas:    make(map[int][]*NodeMeta),
	}
}

// Rebuild replaces the ring contents atomically from the provided node list.
// Each node's Shards slice declares which shards it is primary for.
func (r *Ring) Rebuild(nodes []*NodeMeta) {
	primary := make(map[int]*NodeMeta, r.totalShards)
	replicas := make(map[int][]*NodeMeta, r.totalShards)

	for _, n := range nodes {
		n := n
		for _, s := range n.Shards {
			if _, exists := primary[s]; !exists {
				primary[s] = n
			}
		}
	}

	// Replicas: nodes that actually declare shard s in their own Shards list
	// and are not its primary, in node-list order. Being merely "not the
	// primary" is not enough — a node with no data for shard s at all would
	// otherwise be handed failover traffic for it.
	for s := 0; s < r.totalShards; s++ {
		prim := primary[s]
		var reps []*NodeMeta
		for _, n := range nodes {
			if n == prim {
				continue
			}
			if !nodeOwnsShard(n, s) {
				continue
			}
			reps = append(reps, n)
		}
		replicas[s] = reps
	}

	r.mu.Lock()
	r.nodes = nodes
	r.primary = primary
	r.replicas = replicas
	r.mu.Unlock()
}

// nodeOwnsShard reports whether n declares shardID in its Shards list.
func nodeOwnsShard(n *NodeMeta, shardID int) bool {
	for _, s := range n.Shards {
		if s == shardID {
			return true
		}
	}
	return false
}

// Primary returns the primary node for shardID, or an error if unknown.
func (r *Ring) Primary(shardID int) (*NodeMeta, error) {
	r.mu.RLock()
	n := r.primary[shardID]
	r.mu.RUnlock()
	if n == nil {
		return nil, fmt.Errorf("ring: no primary for shard %d", shardID)
	}
	return n, nil
}

// ReplicasForShard returns replica nodes for shardID (excludes the primary).
func (r *Ring) ReplicasForShard(shardID int) []*NodeMeta {
	r.mu.RLock()
	reps := r.replicas[shardID]
	r.mu.RUnlock()
	// Return a copy to avoid races on the slice.
	out := make([]*NodeMeta, len(reps))
	copy(out, reps)
	return out
}

// Nodes returns all known nodes (snapshot).
func (r *Ring) Nodes() []*NodeMeta {
	r.mu.RLock()
	out := make([]*NodeMeta, len(r.nodes))
	copy(out, r.nodes)
	r.mu.RUnlock()
	return out
}

// TotalShards returns the configured total shard count.
func (r *Ring) TotalShards() int { return r.totalShards }

// ConsistentRing routes documents to shards via a virtual-node hash ring.
// Each physical shard owns vnodes tokens spread over [0, 2^32).
// A doc hashes to the shard whose next clockwise token it falls under.
// Adding one shard moves only ~1/(N+1) documents (vs full re-index with modulo).
type ConsistentRing struct {
	tokens []uint32 // sorted token positions
	ids    []int    // ids[i] = physical shard for tokens[i]
}

// BuildConsistentRing constructs a ring for nShards physical shards.
// vnodes is the number of virtual nodes per shard; 0 defaults to 150 (the Cassandra default).
func BuildConsistentRing(nShards, vnodes int) *ConsistentRing {
	if vnodes <= 0 {
		vnodes = 150
	}
	total := nShards * vnodes
	tokens := make([]uint32, total)
	ids := make([]int, total)

	h := fnv.New32a()
	for shard := 0; shard < nShards; shard++ {
		for k := 0; k < vnodes; k++ {
			h.Reset()
			fmt.Fprintf(h, "%d:%d", shard, k)
			idx := shard*vnodes + k
			tokens[idx] = h.Sum32()
			ids[idx] = shard
		}
	}

	r := &ConsistentRing{tokens: tokens, ids: ids}
	sort.Sort(r)
	return r
}

// ShardFor maps docID to a physical shard index via clockwise ring lookup.
func (r *ConsistentRing) ShardFor(docID string) int {
	h := fnv.New32a()
	h.Write([]byte(docID))
	hash := h.Sum32()
	i := sort.Search(len(r.tokens), func(i int) bool {
		return r.tokens[i] >= hash
	})
	if i >= len(r.tokens) {
		i = 0
	}
	return r.ids[i]
}

// sort.Interface so BuildConsistentRing can sort tokens and ids together.
func (r *ConsistentRing) Len() int           { return len(r.tokens) }
func (r *ConsistentRing) Less(i, j int) bool { return r.tokens[i] < r.tokens[j] }
func (r *ConsistentRing) Swap(i, j int) {
	r.tokens[i], r.tokens[j] = r.tokens[j], r.tokens[i]
	r.ids[i], r.ids[j] = r.ids[j], r.ids[i]
}
