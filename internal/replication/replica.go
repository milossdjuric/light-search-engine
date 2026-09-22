package replication

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	backoffMin = 1 * time.Second
	backoffMax = 30 * time.Second
	backoffMul = 2.0
)

// IndexFunc is the callback used to apply a WAL entry to the local index
// without re-logging to the WAL.
type IndexFunc func(op, docID, text string, metadata map[string]string) error

// ReplicaApplier streams WAL entries from the primary and applies them
// to the local shard via IndexFunc.
type ReplicaApplier struct {
	shardID    string
	primaryAddr string     // gRPC address of the primary
	apply      IndexFunc

	appliedSeq atomic.Uint64

	stopCh chan struct{}
	doneCh chan struct{}
}

// NewReplicaApplier creates a ReplicaApplier.
func NewReplicaApplier(shardID, primaryAddr string, apply IndexFunc) *ReplicaApplier {
	return &ReplicaApplier{
		shardID:     shardID,
		primaryAddr: primaryAddr,
		apply:       apply,
		stopCh:      make(chan struct{}),
		doneCh:      make(chan struct{}),
	}
}

// AppliedSeq returns the highest successfully applied WAL sequence number.
func (r *ReplicaApplier) AppliedSeq() uint64 {
	return r.appliedSeq.Load()
}

// Start launches the background replication loop.
func (r *ReplicaApplier) Start() {
	go r.run()
}

// Close gracefully stops the applier.
func (r *ReplicaApplier) Close() {
	close(r.stopCh)
	<-r.doneCh
}

// run reconnects with exponential back-off on failure.
func (r *ReplicaApplier) run() {
	defer close(r.doneCh)

	backoff := backoffMin
	for {
		select {
		case <-r.stopCh:
			return
		default:
		}

		if err := r.stream(); err != nil && err != io.EOF {
			slog.Warn("replica: stream error, reconnecting",
				"shard", r.shardID, "addr", r.primaryAddr, "err", err, "backoff", backoff)
		}

		select {
		case <-r.stopCh:
			return
		case <-time.After(backoff):
		}

		backoff = time.Duration(float64(backoff) * backoffMul)
		if backoff > backoffMax {
			backoff = backoffMax
		}
	}
}

// stream connects to the primary and applies entries until error or stop.
func (r *ReplicaApplier) stream() error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Honour stop signal.
	go func() {
		select {
		case <-r.stopCh:
			cancel()
		case <-ctx.Done():
		}
	}()

	conn, err := grpc.NewClient(r.primaryAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()

	cli := NewWALReplicationClient(conn)
	stream, err := cli.StreamWAL(ctx, &StreamRequest{
		ShardId: r.shardID,
		FromSeq: r.appliedSeq.Load(),
	})
	if err != nil {
		return err
	}

	// Reset backoff on successful connect.
	slog.Info("replica: connected to primary", "shard", r.shardID, "addr", r.primaryAddr)

	return r.applyStream(stream)
}

// walEntryReceiver is the subset of WALReplication_StreamWALClient that
// applyStream needs; narrowing it lets tests drive applyStream without a
// live gRPC connection.
type walEntryReceiver interface {
	Recv() (*WALEntry, error)
}

// applyStream reads entries from recv until it errors (including io.EOF) and
// applies each one via r.apply, advancing appliedSeq only for entries that
// applied successfully.
func (r *ReplicaApplier) applyStream(recv walEntryReceiver) error {
	for {
		entry, err := recv.Recv()
		if err != nil {
			return err
		}

		// Heartbeat entries carry Op="heartbeat"; skip.
		if entry.Op == "heartbeat" || entry.Seq == 0 {
			continue
		}

		// A gap here means an entry was dropped upstream (e.g. the primary's
		// fan-out channel was full — see Append's select/default). Applying
		// entry.Seq directly would silently and permanently lose the missing
		// one; force a reconnect instead so run()'s backoff loop re-requests
		// FromSeq: appliedSeq and the primary's catch-up phase (ring or WAL
		// file) resends everything from there, including the dropped entry.
		if want := r.appliedSeq.Load() + 1; entry.Seq != want {
			slog.Warn("replica: sequence gap detected, forcing resync",
				"shard", r.shardID, "want", want, "got", entry.Seq)
			return fmt.Errorf("replica: sequence gap detected: want seq %d, got %d", want, entry.Seq)
		}

		if err := r.apply(entry.Op, entry.DocId, entry.Text, entry.Metadata); err != nil {
			slog.Error("replica: apply error", "seq", entry.Seq, "err", err)
			// Stop consuming this stream rather than skipping ahead: appliedSeq
			// stays at the last successfully-applied entry, so run()'s
			// reconnect-with-backoff loop re-requests from here and the failed
			// entry (and anything after it on this stream) gets retried in
			// order instead of being silently lost.
			return err
		}

		r.appliedSeq.Store(entry.Seq)
	}
}

// Ensure insecure import is used
var _ = insecure.NewCredentials
