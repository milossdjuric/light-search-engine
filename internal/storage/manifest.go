// Package storage holds the on-disk bookkeeping for a shard: the per-shard
// JSON manifest (segment + tombstone catalog) and LocalStore, which places
// segment files and their sidecars under the data directory.
//
// Manifest: one JSON file per shard (data/wal/<shardID>.meta.json) stores
// segment records. Writes are atomic: write to <path>.tmp then os.Rename.
package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// SegmentRecord describes one immutable segment file on disk.
type SegmentRecord struct {
	SegmentID   string `json:"segment_id"`
	ShardID     string `json:"shard_id"`
	Level       int    `json:"level"`
	DocCount    int    `json:"doc_count"`
	DeletedDocs int64  `json:"deleted_docs,omitempty"` // tombstoned docs still physically in this segment
	Path        string `json:"path"`
	FlushSeq    int64  `json:"flush_seq"`
	SizeBytes   int64  `json:"size_bytes,omitempty"` // on-disk size; 0 means unknown (legacy records)
}

// TombstoneRecord is a soft-deleted document entry.
type TombstoneRecord struct {
	ShardID   string    `json:"shard_id"`
	DocID     string    `json:"doc_id"`
	DeletedAt time.Time `json:"deleted_at"`
}

// ShardMeta is the persistent catalog for one shard.
// Loaded from <shardID>.meta.json on startup and saved atomically on every mutation.
type ShardMeta struct {
	mu         sync.Mutex
	path       string            // absolute path to the meta.json file
	Segments   []SegmentRecord   `json:"segments"`
	Tombstones []TombstoneRecord `json:"tombstones"`
	// SeqFloor is the highest WAL seq discarded by a Reset. It survives
	// Reset (which wipes Segments) so WAL sequence numbers never rewind —
	// replicas rely on seq only ever increasing.
	SeqFloor int64 `json:"seq_floor,omitempty"`
}

// Load reads the shard manifest from dir/<shardID>.meta.json.
// Returns an empty ShardMeta if the file does not exist (first startup).
func LoadShardMeta(dir, shardID string) (*ShardMeta, error) {
	p := filepath.Join(dir, shardID+".meta.json")
	m := &ShardMeta{path: p}
	data, err := os.ReadFile(p)
	if os.IsNotExist(err) {
		return m, nil
	}
	if err != nil {
		return nil, fmt.Errorf("manifest.Load %s: %w", p, err)
	}
	if err := json.Unmarshal(data, m); err != nil {
		return nil, fmt.Errorf("manifest.Load decode %s: %w", p, err)
	}
	return m, nil
}

// save writes the manifest atomically. Caller must hold m.mu.
func (m *ShardMeta) save() error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("manifest.save marshal: %w", err)
	}
	tmp := m.path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("manifest.save write tmp: %w", err)
	}
	if err := os.Rename(tmp, m.path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("manifest.save rename: %w", err)
	}
	return nil
}

// AddSegment appends a segment record and persists atomically.
func (m *ShardMeta) AddSegment(seg SegmentRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Segments = append(m.Segments, seg)
	return m.save()
}

// AddThenRemoveSegments adds newSeg and removes the listed segment IDs in one
// atomic save. Used by the merge path: output segment registered before inputs
// are removed, so a crash leaves both old and new segments visible (safe).
func (m *ShardMeta) AddThenRemoveSegments(newSeg SegmentRecord, removeIDs []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	set := make(map[string]bool, len(removeIDs))
	for _, id := range removeIDs {
		set[id] = true
	}
	kept := make([]SegmentRecord, 0, len(m.Segments))
	for _, s := range m.Segments {
		if !set[s.SegmentID] {
			kept = append(kept, s)
		}
	}
	kept = append(kept, newSeg)
	m.Segments = kept
	return m.save()
}

// LoadActiveSegments returns a copy of current segment records.
func (m *ShardMeta) LoadActiveSegments() []SegmentRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]SegmentRecord, len(m.Segments))
	copy(out, m.Segments)
	return out
}

// MaxFlushSeq returns the highest flush_seq among active segments (0 if none).
func (m *ShardMeta) MaxFlushSeq() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	var maxSeq int64
	for _, s := range m.Segments {
		if s.FlushSeq > maxSeq {
			maxSeq = s.FlushSeq
		}
	}
	return maxSeq
}

// SeqHighWater returns the highest WAL seq whose effects are already durable
// outside the WAL: max(MaxFlushSeq, SeqFloor). WAL replay skips entries at or
// below it, and new entries must be numbered above it.
func (m *ShardMeta) SeqHighWater() int64 {
	hw := m.MaxFlushSeq()
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.SeqFloor > hw {
		hw = m.SeqFloor
	}
	return hw
}

// SetSeqFloor raises SeqFloor to seq (never lowers it) and persists.
func (m *ShardMeta) SetSeqFloor(seq int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if seq <= m.SeqFloor {
		return nil
	}
	m.SeqFloor = seq
	return m.save()
}

// AddTombstone records a soft-delete, overwriting any prior entry for the same docID.
func (m *ShardMeta) AddTombstone(shardID, docID string, deletedAt time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := m.Tombstones[:0]
	for _, t := range m.Tombstones {
		if t.DocID != docID {
			out = append(out, t)
		}
	}
	m.Tombstones = append(out, TombstoneRecord{ShardID: shardID, DocID: docID, DeletedAt: deletedAt})
	return m.save()
}

// RemoveTombstone deletes a tombstone (e.g., after a document is re-indexed).
// No-op and no save if the docID is not found.
func (m *ShardMeta) RemoveTombstone(shardID, docID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := m.Tombstones[:0]
	changed := false
	for _, t := range m.Tombstones {
		if t.DocID == docID {
			changed = true
			continue
		}
		out = append(out, t)
	}
	if !changed {
		return nil
	}
	m.Tombstones = out
	return m.save()
}

// LoadTombstones returns all tombstone records.
func (m *ShardMeta) LoadTombstones() []TombstoneRecord {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]TombstoneRecord, len(m.Tombstones))
	copy(out, m.Tombstones)
	return out
}

// Reset wipes all segments and tombstones and persists. SeqFloor is kept.
func (m *ShardMeta) Reset() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Segments = nil
	m.Tombstones = nil
	return m.save()
}
