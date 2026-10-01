package replication

import (
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	ringBufferSize    = 10_000
	heartbeatInterval = 5 * time.Second
	replicaChanSize   = 256
)

// CatchUpFunc calls fn, in seq order, for every entry still retained in the
// primary's WAL files with seq > afterSeq (search.SegmentManager.ScanWAL).
// Entries already flushed into segments are no longer retained.
type CatchUpFunc func(afterSeq uint64, fn func(*WALEntry) error) error

// PrimaryReplicator streams one shard's WAL entries to its replicas.
//
// A replica connecting with fromSeq is caught up from, in order: the WAL
// files on disk (catchUp), the in-memory ring of recent entries, and finally
// the live channel fed by Append. The sources overlap, so every entry at or
// below the last one sent is skipped; a hole means the entries the replica
// needs are gone, reported as FailedPrecondition (needs re-seeding) during
// catch-up, or Unavailable (reconnect and catch up) in the live phase.
type PrimaryReplicator struct {
	shardID string
	catchUp CatchUpFunc
	lastSeq func() uint64 // latest seq assigned by the shard's WAL; nil = unknown

	mu   sync.Mutex
	ring [ringBufferSize]*WALEntry
	head uint64 // highest seq appended + 1

	repMu    sync.RWMutex
	replicas map[uint64]chan *WALEntry // keyed by a per-connection ID
	nextID   atomic.Uint64

	stopCh    chan struct{}
	closeOnce sync.Once
	wg        sync.WaitGroup
}

// NewPrimaryReplicator creates a PrimaryReplicator for shardID. catchUp may be
// nil, in which case only the ring buffer serves catch-up.
func NewPrimaryReplicator(shardID string, catchUp CatchUpFunc) *PrimaryReplicator {
	return &PrimaryReplicator{
		shardID:  shardID,
		catchUp:  catchUp,
		replicas: make(map[uint64]chan *WALEntry),
		stopCh:   make(chan struct{}),
	}
}

// SetLastSeqFunc sets a function returning the latest seq the shard's WAL has
// assigned. With it, a replica behind that seq whose missing entries are
// retained nowhere is told to re-seed even when nothing at all is retained;
// it also backs HeadSeq after a restart, before new writes arrive. Call
// before serving replicas.
func (p *PrimaryReplicator) SetLastSeqFunc(fn func() uint64) { p.lastSeq = fn }

// ShardID returns the shard this replicator serves.
func (p *PrimaryReplicator) ShardID() string { return p.shardID }

// HeadSeq returns the highest seq appended so far (0 if none since start).
func (p *PrimaryReplicator) HeadSeq() uint64 {
	if p.lastSeq != nil {
		return p.lastSeq()
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.head == 0 {
		return 0
	}
	return p.head - 1
}

// ReplicaCount returns the number of currently connected replica streams.
func (p *PrimaryReplicator) ReplicaCount() int {
	p.repMu.RLock()
	defer p.repMu.RUnlock()
	return len(p.replicas)
}

// Append adds a new WAL entry to the ring buffer and fans out to all replicas.
// Must be called in strictly increasing seq order (the WAL hook guarantees
// this). Never blocks: a replica whose channel is full misses the entry and
// is forced to reconnect and catch up (see streamToReplica).
//
// The ring is indexed by the entry's own persistent Seq, which keeps counting
// across restarts and resets, so catch-up lookups stay aligned.
func (p *PrimaryReplicator) Append(entry *WALEntry) {
	p.mu.Lock()
	p.ring[entry.Seq%ringBufferSize] = entry
	if entry.Seq+1 > p.head {
		p.head = entry.Seq + 1
	}
	p.mu.Unlock()

	p.repMu.RLock()
	for id, ch := range p.replicas {
		select {
		case ch <- entry:
		default:
			slog.Warn("primary: replica channel full, dropping (replica will reconnect and catch up)",
				"shard", p.shardID, "replica", id)
		}
	}
	p.repMu.RUnlock()
}

// AddReplica registers a replica stream starting after fromSeq and serves it
// in a background goroutine until the stream ends or the replicator closes.
func (p *PrimaryReplicator) AddReplica(fromSeq uint64, stream WALReplication_StreamWALServer) {
	id, ch := p.register()
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		defer p.unregister(id)
		if err := p.streamToReplica(fromSeq, ch, stream); err != nil {
			slog.Warn("primary: replica stream ended", "shard", p.shardID, "replica", id, "err", err)
		}
	}()
}

func (p *PrimaryReplicator) register() (uint64, chan *WALEntry) {
	id := p.nextID.Add(1)
	ch := make(chan *WALEntry, replicaChanSize)
	p.repMu.Lock()
	p.replicas[id] = ch
	p.repMu.Unlock()
	return id, ch
}

func (p *PrimaryReplicator) unregister(id uint64) {
	p.repMu.Lock()
	delete(p.replicas, id)
	p.repMu.Unlock()
}

// serve registers a replica and streams to it on the calling goroutine.
func (p *PrimaryReplicator) serve(fromSeq uint64, stream WALReplication_StreamWALServer) error {
	id, ch := p.register()
	defer p.unregister(id)
	p.wg.Add(1)
	defer p.wg.Done()
	return p.streamToReplica(fromSeq, ch, stream)
}

// streamToReplica catches the replica up from fromSeq (WAL files, then ring),
// then forwards live entries from ch until the stream or replicator ends.
// ch must be registered before this is called so no entry appended during
// catch-up is missed.
func (p *PrimaryReplicator) streamToReplica(fromSeq uint64, ch <-chan *WALEntry, stream WALReplication_StreamWALServer) error {
	lastSent := fromSeq
	send := func(e *WALEntry, phase string) error {
		if e.Seq <= lastSent {
			return nil // already sent by an earlier, overlapping source
		}
		if e.Seq != lastSent+1 {
			if phase == "live" {
				return status.Errorf(codes.Unavailable,
					"replica fell behind: seq %d dropped (next available %d); reconnect to catch up", lastSent+1, e.Seq)
			}
			return status.Errorf(codes.FailedPrecondition,
				"shard %s: seqs %d..%d are no longer retained by the primary (flushed into segments); replica must be re-seeded",
				p.shardID, lastSent+1, e.Seq-1)
		}
		if err := stream.Send(e); err != nil {
			return err
		}
		lastSent = e.Seq
		return nil
	}

	// Every seq up to lastAtStart must come from the WAL or the ring; later
	// ones arrive on ch, which was registered before this read.
	var lastAtStart uint64
	if p.lastSeq != nil {
		lastAtStart = p.lastSeq()
	}

	// Phase 1: WAL files.
	if p.catchUp != nil {
		if err := p.catchUp(lastSent, func(e *WALEntry) error { return send(e, "wal") }); err != nil {
			return err
		}
	}

	// Phase 2: ring buffer, for entries appended but not yet visible in the
	// WAL scan. A slot holding a different seq was overwritten by a newer
	// entry; the resulting hole is reported by send.
	p.mu.Lock()
	head := p.head
	p.mu.Unlock()
	start := lastSent + 1
	if head > ringBufferSize && start < head-ringBufferSize {
		start = head - ringBufferSize
	}
	for seq := start; seq < head; seq++ {
		p.mu.Lock()
		e := p.ring[seq%ringBufferSize]
		p.mu.Unlock()
		if e == nil || e.Seq != seq {
			continue
		}
		if err := send(e, "ring"); err != nil {
			return err
		}
	}

	if lastSent < lastAtStart {
		return status.Errorf(codes.FailedPrecondition,
			"shard %s: seqs %d..%d are no longer retained by the primary (flushed into segments); replica must be re-seeded",
			p.shardID, lastSent+1, lastAtStart)
	}

	// Phase 3: live entries. Heartbeats keep the stream alive when idle.
	ticker := time.NewTicker(heartbeatInterval)
	defer ticker.Stop()
	for {
		select {
		case <-p.stopCh:
			return nil
		case <-stream.Context().Done():
			return nil
		case e := <-ch:
			if err := send(e, "live"); err != nil {
				return err
			}
		case <-ticker.C:
			if err := stream.Send(&WALEntry{Op: "heartbeat"}); err != nil {
				return err
			}
		}
	}
}

// Close stops all replica streams and waits for them to finish.
func (p *PrimaryReplicator) Close() {
	p.closeOnce.Do(func() { close(p.stopCh) })
	p.wg.Wait()
}

// Server is the WALReplication gRPC service for a node, dispatching each
// StreamWAL call to the PrimaryReplicator of the requested shard.
type Server struct {
	UnimplementedWALReplicationServer

	mu        sync.RWMutex
	primaries map[string]*PrimaryReplicator
}

// NewServer creates an empty Server; add shards with Register.
func NewServer() *Server {
	return &Server{primaries: make(map[string]*PrimaryReplicator)}
}

// Register makes p's shard available to replicas.
func (s *Server) Register(p *PrimaryReplicator) {
	s.mu.Lock()
	s.primaries[p.shardID] = p
	s.mu.Unlock()
}

// StreamWAL implements WALReplicationServer.
func (s *Server) StreamWAL(req *StreamRequest, stream WALReplication_StreamWALServer) error {
	s.mu.RLock()
	p := s.primaries[req.ShardId]
	s.mu.RUnlock()
	if p == nil {
		return status.Errorf(codes.NotFound, "shard %q is not served by this primary", req.ShardId)
	}
	slog.Info("primary: replica connected", "shard", req.ShardId, "from_seq", req.FromSeq)
	return p.serve(req.FromSeq, stream)
}

// PrimaryStatus is a point-in-time view of a primary shard, for /health.
type PrimaryStatus struct {
	ShardID  string `json:"shard_id"`
	HeadSeq  uint64 `json:"head_seq"`
	Replicas int    `json:"replicas"`
}

// Status returns the shard's current replication state.
func (p *PrimaryReplicator) Status() PrimaryStatus {
	return PrimaryStatus{ShardID: p.shardID, HeadSeq: p.HeadSeq(), Replicas: p.ReplicaCount()}
}
