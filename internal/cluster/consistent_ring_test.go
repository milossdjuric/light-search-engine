package cluster

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestConsistentRingBalance is a regression test for poorly mixed ring
// hashing: plain FNV-32a over short, near-identical vnode keys ("0:0",
// "0:1", …) clustered the tokens, so with 150 vnodes/shard the largest of 4
// shards got 28% more documents than average (46% with 16 shards) — and
// query latency follows the largest shard.
func TestConsistentRingBalance(t *testing.T) {
	const docs = 171_332
	for _, nShards := range []int{4, 16} {
		r := BuildConsistentRing(nShards, 150)
		counts := make([]int, nShards)
		rng := rand.New(rand.NewSource(1))
		for i := 0; i < docs; i++ {
			counts[r.ShardFor(fmt.Sprintf("%08x", rng.Uint32()))]++
		}
		largest := 0
		for _, c := range counts {
			largest = max(largest, c)
		}
		if ratio := float64(largest) / (float64(docs) / float64(nShards)); ratio > 1.15 {
			t.Errorf("%d shards: largest shard holds %.2f× the average (%v), want ≤ 1.15", nShards, ratio, counts)
		}
	}
}

// TestConsistentRingMovesFewDocsWhenGrowing verifies the point of the vnode
// ring: adding one shard to N re-routes only about 1/(N+1) of documents.
func TestConsistentRingMovesFewDocsWhenGrowing(t *testing.T) {
	const docs = 50_000
	before, after := BuildConsistentRing(4, 150), BuildConsistentRing(5, 150)
	moved := 0
	for i := 0; i < docs; i++ {
		id := fmt.Sprintf("doc-%d", i)
		if before.ShardFor(id) != after.ShardFor(id) {
			moved++
		}
	}
	if frac := float64(moved) / docs; frac > 0.30 {
		t.Errorf("4→5 shards moved %.0f%% of docs, want about 20%% (≤ 30%%)", frac*100)
	}
}
