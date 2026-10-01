package replication

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

const (
	backoffMin = 1 * time.Second
	backoffMax = 30 * time.Second
	backoffMul = 2.0

	// seqSaveInterval is how often the applied seq is persisted. The value
	// saved is the one observed one interval earlier, giving the replica's
	// own WAL (async, ~1s sync) time to make those entries durable; entries
	// re-applied after a crash are idempotent.
	seqSaveInterval = 2 * time.Second
)

// ApplyFunc applies one replicated WAL entry ("index", "delete" or "reset")
// to the local shard.
type ApplyFunc func(e *WALEntry) error

// ReplicaStatus is a point-in-time view of a replica's progress, for /health.
type ReplicaStatus struct {
	ShardID     string `json:"shard_id"`
	Primary     string `json:"primary"`
	Connected   bool   `json:"connected"`
	AppliedSeq  uint64 `json:"applied_seq"`
	NeedsResync bool   `json:"needs_resync,omitempty"` // primary no longer retains the entries this replica needs
	LastError   string `json:"last_error,omitempty"`
	DeadLetters uint64 `json:"dead_letters"` // non-zero = replica has diverged from its primary
}

// ReplicaApplier streams WAL entries from the primary and applies them
// to the local shard via ApplyFunc.
type ReplicaApplier struct {
	shardID     string
	primaryAddr string // gRPC address of the primary
	apply       ApplyFunc

	appliedSeq  atomic.Uint64
	connected   atomic.Bool
	needsResync atomic.Bool
	gotMessage  atomic.Bool // set when the current stream received anything
	errMu       sync.Mutex
	lastError   string

	// Apply-failure tracking for the entry currently failing (touched only
	// by the run goroutine). See maxApplyAttempts.
	failSeq   uint64
	failCount int

	deadLetterPath  string
	dlMu            sync.Mutex
	deadLetters     []DeadLetter
	deadLetterCount atomic.Uint64

	seqPath string

	stopCh    chan struct{}
	doneCh    chan struct{}
	closeOnce sync.Once
}

// NewReplicaApplier creates a ReplicaApplier.
func NewReplicaApplier(shardID, primaryAddr string, apply ApplyFunc) *ReplicaApplier {
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

// Status returns the replica's current replication state.
func (r *ReplicaApplier) Status() ReplicaStatus {
	r.errMu.Lock()
	lastErr := r.lastError
	r.errMu.Unlock()
	return ReplicaStatus{
		ShardID:     r.shardID,
		Primary:     r.primaryAddr,
		Connected:   r.connected.Load(),
		AppliedSeq:  r.appliedSeq.Load(),
		NeedsResync: r.needsResync.Load(),
		LastError:   lastErr,
		DeadLetters: r.deadLetterCount.Load(),
	}
}

// SetSeqPath sets the file the applied seq is persisted to, so a restarted
// replica resumes where it left off. Call before Start.
func (r *ReplicaApplier) SetSeqPath(path string) { r.seqPath = path }

// Start loads the persisted applied seq (initialSeq if none was saved yet —
// e.g. the primary's flushed seq when the replica was seeded from a copy of
// the primary's data dir) and launches the background replication loop.
func (r *ReplicaApplier) Start(initialSeq uint64) error {
	if err := r.loadAppliedSeq(initialSeq); err != nil {
		return err
	}
	go r.run()
	return nil
}

// Close stops the replication loop. It does not save the applied seq: call
// SaveAppliedSeq once the local shard has durably flushed what was applied.
func (r *ReplicaApplier) Close() {
	r.closeOnce.Do(func() { close(r.stopCh) })
	<-r.doneCh
}

func (r *ReplicaApplier) loadAppliedSeq(initialSeq uint64) error {
	seq := initialSeq
	if r.seqPath != "" {
		data, err := os.ReadFile(r.seqPath)
		switch {
		case err == nil:
			v, perr := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
			if perr != nil {
				return fmt.Errorf("replica: parse %s: %w", r.seqPath, perr)
			}
			seq = v
		case !os.IsNotExist(err):
			return fmt.Errorf("replica: read %s: %w", r.seqPath, err)
		}
	}
	r.appliedSeq.Store(seq)
	return nil
}

// SaveAppliedSeq persists the current applied seq (no-op without SetSeqPath).
func (r *ReplicaApplier) SaveAppliedSeq() error {
	return r.saveSeq(r.appliedSeq.Load())
}

func (r *ReplicaApplier) saveSeq(seq uint64) error {
	if r.seqPath == "" {
		return nil
	}
	tmp := r.seqPath + ".tmp"
	if err := os.MkdirAll(filepath.Dir(r.seqPath), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(tmp, []byte(strconv.FormatUint(seq, 10)+"\n"), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, r.seqPath)
}

// run reconnects with exponential back-off on failure, and periodically
// persists the applied seq.
func (r *ReplicaApplier) run() {
	defer close(r.doneCh)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		r.persistLoop()
	}()
	defer wg.Wait()

	backoff := backoffMin
	for {
		select {
		case <-r.stopCh:
			return
		default:
		}

		r.gotMessage.Store(false)
		err := r.stream()
		r.connected.Store(false)
		if r.gotMessage.Load() {
			backoff = backoffMin // the connection worked; retry promptly
		}
		select {
		case <-r.stopCh:
			return // shutting down; the stream was cancelled on purpose
		default:
		}
		if err != nil {
			slog.Warn("replica: stream ended, reconnecting",
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

// persistLoop saves the applied seq observed one interval earlier.
func (r *ReplicaApplier) persistLoop() {
	ticker := time.NewTicker(seqSaveInterval)
	defer ticker.Stop()
	prev := r.appliedSeq.Load()
	for {
		select {
		case <-r.stopCh:
			return
		case <-ticker.C:
			if err := r.saveSeq(prev); err != nil {
				slog.Error("replica: persist applied seq", "shard", r.shardID, "err", err)
			}
			prev = r.appliedSeq.Load()
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
		r.setLastError(err)
		return err
	}
	defer conn.Close()

	cli := NewWALReplicationClient(conn)
	stream, err := cli.StreamWAL(ctx, &StreamRequest{
		ShardId: r.shardID,
		FromSeq: r.appliedSeq.Load(),
	})
	if err != nil {
		r.setLastError(err)
		return err
	}
	return r.applyStream(stream)
}

func (r *ReplicaApplier) setLastError(err error) {
	r.errMu.Lock()
	r.lastError = err.Error()
	r.errMu.Unlock()
}

// walEntryReceiver is the subset of WALReplication_StreamWALClient that
// applyStream needs; narrowing it lets tests drive applyStream without a
// live gRPC connection.
type walEntryReceiver interface {
	Recv() (*WALEntry, error)
}

// applyStream reads entries from recv until it errors (including io.EOF) and
// applies each one via r.apply, advancing appliedSeq only for entries that
// applied successfully (or were dead-lettered).
func (r *ReplicaApplier) applyStream(recv walEntryReceiver) error {
	for {
		entry, err := recv.Recv()
		if err != nil {
			if status.Code(err) == codes.FailedPrecondition {
				slog.Error("replica: primary no longer retains the entries this replica needs; re-seed it from a copy of the primary's data dir",
					"shard", r.shardID, "applied_seq", r.appliedSeq.Load(), "err", err)
				r.needsResync.Store(true)
			}
			if status.Code(err) != codes.Canceled {
				r.setLastError(err)
			}
			return err
		}
		if !r.gotMessage.Swap(true) {
			r.connected.Store(true)
			slog.Info("replica: streaming from primary", "shard", r.shardID, "addr", r.primaryAddr,
				"from_seq", r.appliedSeq.Load())
		}

		// Heartbeat entries carry Op="heartbeat"; skip.
		if entry.Op == "heartbeat" || entry.Seq == 0 {
			continue
		}

		applied := r.appliedSeq.Load()
		// Overlapping catch-up sources or a resend after reconnect.
		if entry.Seq <= applied {
			continue
		}
		// A gap here means an entry was dropped upstream. Applying entry.Seq
		// directly would silently and permanently lose the missing one;
		// force a reconnect instead so the primary's catch-up resends
		// everything from appliedSeq.
		if want := applied + 1; entry.Seq != want {
			slog.Warn("replica: sequence gap detected, forcing resync",
				"shard", r.shardID, "want", want, "got", entry.Seq)
			return fmt.Errorf("replica: sequence gap detected: want seq %d, got %d", want, entry.Seq)
		}

		if err := r.apply(entry); err != nil {
			if entry.Seq != r.failSeq {
				r.failSeq, r.failCount = entry.Seq, 0
			}
			r.failCount++
			r.setLastError(err)
			if r.failCount < maxApplyAttempts {
				slog.Error("replica: apply error", "seq", entry.Seq, "attempt", r.failCount, "err", err)
				// Stop consuming this stream rather than skipping ahead:
				// appliedSeq stays at the last successfully-applied entry, so
				// run()'s reconnect-with-backoff loop re-requests from here
				// and the failed entry (and anything after it) gets retried
				// in order instead of being silently lost.
				return err
			}
			// Out of attempts: set the entry aside (logged, kept in memory,
			// appended to the dead-letter file) and move past it so one
			// poisoned entry can't stall replication forever.
			r.deadLetter(entry, err, r.failCount)
		}
		if entry.Seq == r.failSeq {
			r.failSeq, r.failCount = 0, 0
		}

		r.appliedSeq.Store(entry.Seq)
		r.needsResync.Store(false)
	}
}
