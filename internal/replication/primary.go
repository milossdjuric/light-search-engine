package replication

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"log/slog"
	"os"
	"sync"
	"time"
)

// walFileEntry mirrors the NDJSON format written by SegmentManager.appendWAL.
// Defined locally to avoid an import cycle with internal/search.
type walFileEntry struct {
	Seq      uint64            `json:"seq"`
	Op       string            `json:"op"`
	DocID    string            `json:"doc_id"`
	Text     string            `json:"text,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

const (
	ringBufferSize = 10_000
	heartbeatInterval = 5 * time.Second
)

// PrimaryReplicator streams WAL entries to replicas.
// It maintains a ring buffer of recent entries for fast catch-up;
// older entries are read directly from the WAL file.
type PrimaryReplicator struct {
	shardID  string
	walPath  string

	mu      sync.Mutex
	ring    [ringBufferSize]*WALEntry
	head    uint64 // next write index (monotonically increasing)

	// per-replica send channels (nodeID → channel)
	repMu    sync.RWMutex
	replicas map[string]chan *WALEntry

	stopCh chan struct{}
	wg     sync.WaitGroup
}

// NewPrimaryReplicator creates a PrimaryReplicator for the given shard.
func NewPrimaryReplicator(shardID, walPath string) *PrimaryReplicator {
	return &PrimaryReplicator{
		shardID:  shardID,
		walPath:  walPath,
		replicas: make(map[string]chan *WALEntry),
		stopCh:   make(chan struct{}),
	}
}

// Append adds a new WAL entry to the ring buffer and fans out to all replicas.
//
// The ring is indexed by the entry's own persistent Seq (not an internal
// append counter): Seq is assigned by the WAL and keeps counting across
// process restarts, while a counter that starts at 0 in every new
// PrimaryReplicator would not — using it as the index would misalign catch-up
// lookups (streamToReplica reads p.ring[seq%ringBufferSize] using real Seq
// values) both across restarts and, since Seq starts at 1 rather than 0,
// even within a single process lifetime.
func (p *PrimaryReplicator) Append(entry *WALEntry) {
	p.mu.Lock()
	idx := entry.Seq % ringBufferSize
	p.ring[idx] = entry
	if entry.Seq+1 > p.head {
		p.head = entry.Seq + 1
	}
	p.mu.Unlock()

	p.repMu.RLock()
	for nodeID, ch := range p.replicas {
		select {
		case ch <- entry:
		default:
			slog.Warn("primary: replica channel full, dropping (will catch up)",
				"shard", p.shardID, "replica", nodeID)
		}
	}
	p.repMu.RUnlock()
}

// AddReplica registers a replica and starts a background goroutine that
// streams entries starting from fromSeq via the provided stream sender.
func (p *PrimaryReplicator) AddReplica(nodeID string, fromSeq uint64, stream WALReplication_StreamWALServer) {
	ch := make(chan *WALEntry, 256)

	p.repMu.Lock()
	p.replicas[nodeID] = ch
	p.repMu.Unlock()

	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		defer func() {
			p.repMu.Lock()
			delete(p.replicas, nodeID)
			p.repMu.Unlock()
		}()
		p.streamToReplica(nodeID, fromSeq, ch, stream)
	}()
}

// streamToReplica sends catch-up entries then tails the live channel.
//
// Catch-up has two phases:
//  1. WAL-file phase: if the replica is further behind than the ring buffer
//     covers, read the WAL file on disk and stream entries (fromSeq, ringStart).
//  2. Ring-buffer phase: stream in-memory entries [ringStart, head).
//
// After both phases the goroutine enters the live-stream loop.
func (p *PrimaryReplicator) streamToReplica(
	nodeID string,
	fromSeq uint64,
	ch <-chan *WALEntry,
	stream WALReplication_StreamWALServer,
) {
	p.mu.Lock()
	head := p.head
	p.mu.Unlock()

	// ringStart is the oldest seq still held in the ring buffer.
	ringStart := uint64(0)
	if head > ringBufferSize {
		ringStart = head - ringBufferSize
	}

	// Phase 1: WAL-file catch-up for the gap (fromSeq, ringStart).
	if fromSeq+1 < ringStart {
		slog.Info("primary: replica too far behind ring buffer; reading WAL file",
			"replica", nodeID, "fromSeq", fromSeq, "ringStart", ringStart)
		if err := p.catchUpFromWAL(stream, fromSeq, ringStart); err != nil {
			slog.Warn("primary: WAL file catch-up failed", "replica", nodeID, "err", err)
			return
		}
	}

	// Phase 2: ring-buffer catch-up for entries [max(ringStart, fromSeq+1), head).
	startIdx := fromSeq + 1
	if startIdx < ringStart {
		startIdx = ringStart
	}
	for seq := startIdx; seq < head; seq++ {
		p.mu.Lock()
		entry := p.ring[seq%ringBufferSize]
		p.mu.Unlock()
		if entry == nil {
			continue
		}
		if err := stream.Send(entry); err != nil {
			slog.Warn("primary: send catch-up failed", "replica", nodeID, "err", err)
			return
		}
	}

	// Live-stream loop: forward new entries as they arrive.
	ticker := time.NewTicker(heartbeatInterval)
	defer ticker.Stop()

	for {
		select {
		case <-p.stopCh:
			return
		case <-stream.Context().Done():
			return
		case entry := <-ch:
			if err := stream.Send(entry); err != nil {
				slog.Warn("primary: send entry failed", "replica", nodeID, "err", err)
				return
			}
		case <-ticker.C:
			// Heartbeat keeps the gRPC stream alive across idle periods.
			if err := stream.Send(&WALEntry{Op: "heartbeat"}); err != nil {
				slog.Warn("primary: heartbeat failed", "replica", nodeID, "err", err)
				return
			}
		}
	}
}

// catchUpFromWAL reads the WAL file and streams every entry whose seq is in
// the half-open range (fromSeq, upToSeq). Handles both binary (SWAL\x01) and
// legacy NDJSON formats automatically.
func (p *PrimaryReplicator) catchUpFromWAL(
	stream WALReplication_StreamWALServer,
	fromSeq, upToSeq uint64,
) error {
	f, err := os.Open(p.walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("open WAL %s: %w", p.walPath, err)
	}
	defer f.Close()

	// Detect binary vs JSON by peeking at first 5 bytes.
	var peek [5]byte
	n, _ := io.ReadFull(f, peek[:])
	if n >= 5 && string(peek[:4]) == "SWAL" && peek[4] == 1 {
		return p.catchUpBinaryWAL(stream, bufio.NewReaderSize(f, 1<<20), fromSeq, upToSeq)
	}

	// Legacy JSON format — seek back to beginning.
	if _, serr := f.Seek(0, io.SeekStart); serr != nil {
		return fmt.Errorf("seek WAL %s: %w", p.walPath, serr)
	}
	return p.catchUpJSONWAL(stream, f, fromSeq, upToSeq)
}

// catchUpJSONWAL handles the legacy NDJSON WAL format.
func (p *PrimaryReplicator) catchUpJSONWAL(
	stream WALReplication_StreamWALServer,
	r io.Reader,
	fromSeq, upToSeq uint64,
) error {
	const scanBuf = 4 << 20
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, scanBuf), scanBuf)

	for scanner.Scan() {
		var e walFileEntry
		if err := json.Unmarshal(scanner.Bytes(), &e); err != nil {
			continue
		}
		if e.Seq <= fromSeq || e.Seq >= upToSeq {
			continue
		}
		if err := stream.Send(&WALEntry{
			Seq:      e.Seq,
			Op:       e.Op,
			DocId:    e.DocID,
			Text:     e.Text,
			Metadata: e.Metadata,
		}); err != nil {
			return err
		}
	}
	return scanner.Err()
}

// catchUpBinaryWAL handles the binary WAL format.
func (p *PrimaryReplicator) catchUpBinaryWAL(
	stream WALReplication_StreamWALServer,
	r io.Reader,
	fromSeq, upToSeq uint64,
) error {
	var hdr [8]byte
	for {
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			break // EOF or truncated tail
		}
		payloadLen := binary.LittleEndian.Uint32(hdr[0:4])
		storedCRC := binary.LittleEndian.Uint32(hdr[4:8])

		payload := make([]byte, payloadLen)
		if _, err := io.ReadFull(r, payload); err != nil {
			break
		}
		if crc32.ChecksumIEEE(payload) != storedCRC {
			continue // corrupted record
		}
		e, err := walDecodeReplicationEntry(payload)
		if err != nil {
			continue
		}
		if e.Seq <= fromSeq || e.Seq >= upToSeq {
			continue
		}
		if err := stream.Send(e); err != nil {
			return err
		}
	}
	return nil
}

// walDecodeReplicationEntry decodes a binary WAL payload into a WALEntry proto.
// This is a local copy of the decode logic to avoid an import cycle with internal/search.
func walDecodeReplicationEntry(b []byte) (*WALEntry, error) {
	if len(b) < 17 {
		return nil, fmt.Errorf("binary WAL record too short")
	}
	seq := binary.LittleEndian.Uint64(b[0:8])
	// tsNano at b[8:16] — not used by replication stream
	opByte := b[16]
	b = b[17:]

	e := &WALEntry{}

	// docID (uint16 len prefix)
	if len(b) < 2 {
		return nil, fmt.Errorf("binary WAL: short docID len")
	}
	dlen := int(binary.LittleEndian.Uint16(b[0:2]))
	b = b[2:]
	if len(b) < dlen {
		return nil, fmt.Errorf("binary WAL: short docID data")
	}
	e.DocId = string(b[:dlen])
	b = b[dlen:]
	e.Seq = seq

	switch opByte {
	case 1: // index
		e.Op = "index"
		if len(b) < 4 {
			return nil, fmt.Errorf("binary WAL: short text len")
		}
		tlen := int(binary.LittleEndian.Uint32(b[0:4]))
		b = b[4:]
		if len(b) < tlen {
			return nil, fmt.Errorf("binary WAL: short text data")
		}
		e.Text = string(b[:tlen])
		// skip fields and metadata — not needed for replication stream
	case 2: // delete
		e.Op = "delete"
	default:
		return nil, fmt.Errorf("binary WAL: unknown op %d", opByte)
	}
	return e, nil
}

// Close shuts down the replicator.
func (p *PrimaryReplicator) Close() {
	close(p.stopCh)
	p.wg.Wait()
}

// StreamWAL implements the WALReplication gRPC service method.
func (p *PrimaryReplicator) StreamWAL(req *StreamRequest, stream WALReplication_StreamWALServer) error {
	if req.ShardId != p.shardID {
		return io.ErrUnexpectedEOF
	}
	nodeID := req.ShardId + "-replica"
	p.AddReplica(nodeID, req.FromSeq, stream)
	// Block until the stream context is cancelled.
	<-stream.Context().Done()
	return context.Cause(stream.Context())
}
