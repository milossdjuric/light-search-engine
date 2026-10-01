package storage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ErrNotFound is returned by LocalPath when the object does not exist.
var ErrNotFound = errors.New("localstore: object not found")

// LocalStore stores segment files and their sidecars as regular files under a
// root directory (the data dir). Keys are relative slash-separated paths:
// key "segments/shard0_1_L0.seg" maps to root/segments/shard0_1_L0.seg, and
// SegmentRecord.Path in the manifest stores the key, not a filesystem path.
// Legacy records written before LocalStore existed hold absolute (or
// data-dir-relative) paths; LocalPath resolves those too.
//
// The WAL and the manifest itself are written directly, not through
// LocalStore: they need append semantics and atomic renames.
type LocalStore struct {
	root string
}

// NewLocalStore creates a LocalStore rooted at root. The directory is created if
// it does not exist.
func NewLocalStore(root string) (*LocalStore, error) {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, fmt.Errorf("localstore: mkdir %s: %w", root, err)
	}
	return &LocalStore{root: root}, nil
}

// PutFile atomically stores the file at localPath under key, consuming
// localPath (the caller must not use it afterwards): an atomic rename, or
// copy+rename when source and destination are on different filesystems.
func (s *LocalStore) PutFile(_ context.Context, key, localPath string) error {
	dst := filepath.Join(s.root, filepath.FromSlash(key))
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return fmt.Errorf("localstore: PutFile mkdir: %w", err)
	}
	// Same-filesystem: atomic rename.
	if err := os.Rename(localPath, dst); err == nil {
		return nil
	}
	// Cross-device fallback: copy then remove source.
	if err := fileCopy(localPath, dst); err != nil {
		return fmt.Errorf("localstore: PutFile copy: %w", err)
	}
	return os.Remove(localPath)
}

// LocalPath returns the filesystem path for key, suitable for LoadSegment.
// Returns ErrNotFound when the object does not exist.
func (s *LocalStore) LocalPath(_ context.Context, key string) (string, error) {
	// Backward-compat: absolute paths stored before LocalStore existed.
	if filepath.IsAbs(key) {
		if _, err := os.Stat(key); err != nil {
			if os.IsNotExist(err) {
				return "", ErrNotFound
			}
			return "", fmt.Errorf("localstore: LocalPath (legacy) %s: %w", key, err)
		}
		return key, nil
	}
	// Backward-compat: relative paths that already include the data dir prefix
	// (e.g. "data/segments/shard0_2_L0.seg") stored before LocalStore existed.
	// Stat the key as-is from the working directory before joining with root.
	if _, err := os.Stat(key); err == nil {
		return key, nil
	}
	p := filepath.Join(s.root, filepath.FromSlash(key))
	if _, err := os.Stat(p); err != nil {
		if os.IsNotExist(err) {
			return "", ErrNotFound
		}
		return "", fmt.Errorf("localstore: LocalPath %s: %w", key, err)
	}
	return p, nil
}

// Delete removes the object. No-op if the key does not exist.
func (s *LocalStore) Delete(_ context.Context, key string) error {
	var p string
	if filepath.IsAbs(key) {
		p = key
	} else {
		p = filepath.Join(s.root, filepath.FromSlash(key))
	}
	err := os.Remove(p)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// List returns all keys that begin with prefix.
func (s *LocalStore) List(_ context.Context, prefix string) ([]string, error) {
	var keys []string
	err := filepath.Walk(s.root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			if os.IsNotExist(err) {
				return nil // file deleted concurrently (e.g. by merge); safe to skip
			}
			return err
		}
		if info.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(s.root, path)
		if err != nil {
			return err
		}
		key := filepath.ToSlash(rel)
		if strings.HasPrefix(key, prefix) {
			keys = append(keys, key)
		}
		return nil
	})
	return keys, err
}

// fileCopy copies src to dst, creating dst atomically via a temp file.
func fileCopy(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	tmp := dst + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}
