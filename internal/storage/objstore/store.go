// Package objstore provides a pluggable object store abstraction used for
// segment files, bloom sidecars, and vector store snapshots.
//
// Two implementations are provided:
//   - LocalStore: wraps the local filesystem. Zero overhead; current default.
//     Works for bare-metal and EFS deployments.
//   - S3Store: stores objects in AWS S3 with a local read-through LRU disk
//     cache. Immutable segment files are written once and read many times;
//     the cache ensures warm reads cost local-disk µs, not S3 round-trips.
//
// The WAL and segment manifest always stay on local disk regardless of which
// driver is configured — they require append semantics and file-locking that
// object stores cannot provide.
//
// Key format: relative slash-separated paths, e.g.
//
//	"segments/shard0_12_L0.seg"
//	"segments/shard0_12_L0.seg.bloom"
//	"vectors/shard0.vecs"
//
// Manifest SegmentRecord.Path stores the object key (not a filesystem path).
// Legacy records written before objstore was introduced store absolute paths;
// LocalStore.LocalPath handles both forms for backward compatibility.
package objstore

import (
	"context"
	"errors"
)

// ErrNotFound is returned by LocalPath when the object does not exist in
// either the local cache or the remote store.
var ErrNotFound = errors.New("objstore: object not found")

// ObjectStore abstracts segment/bloom/vector file storage behind a uniform
// interface that works for both local filesystem and S3.
type ObjectStore interface {
	// PutFile atomically stores the file at localPath under key, consuming
	// localPath in the process (the caller must not use localPath after this).
	// For LocalStore: atomic rename (or copy+rename if cross-device).
	// For S3Store: stream upload to S3, then move localPath to local cache.
	PutFile(ctx context.Context, key, localPath string) error

	// LocalPath returns a local filesystem path for key, suitable for passing
	// to LoadSegment or similar file-reading code.
	// For LocalStore: returns filepath.Join(root, key) directly.
	// For S3Store: checks the local cache; on miss, downloads from S3 and
	//   caches the result. Returns ErrNotFound when the object does not exist.
	// The returned file is read-only. The cache may evict it after the call.
	LocalPath(ctx context.Context, key string) (string, error)

	// Delete removes the object. No-op if the key does not exist.
	// For S3Store: also evicts the local cache entry.
	Delete(ctx context.Context, key string) error

	// List returns all keys that begin with prefix.
	List(ctx context.Context, prefix string) ([]string, error)
}
