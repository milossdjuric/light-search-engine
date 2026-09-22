package main

import (
	"errors"
	"testing"
)

// TestRunShardServeAndCloseAlwaysClosesOnServeError verifies that a
// serveHTTP failure (e.g. a graceful-shutdown timeout) does not skip the
// final shard close: skipping it would leak buffered-but-unflushed
// documents (especially under wal_durability: async) and the WAL file
// descriptor.
func TestRunShardServeAndCloseAlwaysClosesOnServeError(t *testing.T) {
	closeCalled := false
	serveErr := errors.New("shutdown timed out")

	err := runShardServeAndClose(
		func() error { return serveErr },
		func() error { closeCalled = true; return nil },
	)

	if !closeCalled {
		t.Error("close was not called after serve returned an error, want it always called")
	}
	if !errors.Is(err, serveErr) {
		t.Errorf("got err %v, want it to wrap the serve error %v", err, serveErr)
	}
}

// TestRunShardServeAndCloseReturnsNilOnCleanShutdown verifies the normal
// path: serve and close both succeed, no error is returned.
func TestRunShardServeAndCloseReturnsNilOnCleanShutdown(t *testing.T) {
	closeCalled := false

	err := runShardServeAndClose(
		func() error { return nil },
		func() error { closeCalled = true; return nil },
	)

	if !closeCalled {
		t.Error("close was not called on a clean shutdown")
	}
	if err != nil {
		t.Errorf("got err %v, want nil", err)
	}
}
