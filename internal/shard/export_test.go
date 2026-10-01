package shard

import "context"

// Test-only hooks into SegmentManager internals, visible to the shard_test
// package because this file is compiled only with the tests.

// TriggerMerge signals the background merge goroutine to run one policy check.
func (sm *SegmentManager) TriggerMerge() {
	select {
	case sm.mergeCh <- struct{}{}:
	default:
	}
}

// ForceMerge runs the production force-merge path (the one behind
// POST /index/merge?max_segments=N) on this shard.
func (sm *SegmentManager) ForceMerge(ctx context.Context, maxSegments int) error {
	return sm.forceMerge(ctx, maxSegments)
}
