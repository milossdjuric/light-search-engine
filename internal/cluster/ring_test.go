package cluster

import "testing"

// TestRebuildReplicasOnlyIncludesNodesThatOwnShard verifies that
// ReplicasForShard never returns a node that isn't actually declared to own
// that shard. Rebuild previously treated "not the primary" as sufficient to
// be a replica, so a node with data for an unrelated shard could be handed
// traffic for a shard it has none of.
func TestRebuildReplicasOnlyIncludesNodesThatOwnShard(t *testing.T) {
	nodeA := &NodeMeta{NodeID: "a", Shards: []int{0}}
	nodeB := &NodeMeta{NodeID: "b", Shards: []int{5}} // does not own shard 0
	nodeC := &NodeMeta{NodeID: "c", Shards: []int{0}} // also owns shard 0 -> legit replica

	r := NewRing(8)
	r.Rebuild([]*NodeMeta{nodeA, nodeB, nodeC})

	reps := r.ReplicasForShard(0)
	for _, n := range reps {
		if n.NodeID == "b" {
			t.Fatalf("ReplicasForShard(0) included node %q which does not own shard 0 (Shards=%v): full list %v",
				n.NodeID, n.Shards, reps)
		}
	}
	found := false
	for _, n := range reps {
		if n.NodeID == "c" {
			found = true
		}
	}
	if !found {
		t.Errorf("ReplicasForShard(0) missing node %q which does own shard 0: %v", "c", reps)
	}
}
