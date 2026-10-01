package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"path/filepath"
	"time"

	"google.golang.org/grpc"

	"search-eval-platform/internal/replication"
	"search-eval-platform/internal/shard"
	"search-eval-platform/pkg/types"
)

// replicationConfig selects this shard node's replication role.
type replicationConfig struct {
	Role        string // "" (disabled) | "primary" | "replica"
	ListenAddr  string // primary: gRPC listen address, e.g. ":9090"
	PrimaryAddr string // replica: the primary's gRPC address
	StateDir    string // replica: applied-seq and dead-letter files
}

// replicationNode is the running replication side of a shard node. Every
// local shard is replicated: shard i on a replica follows shard i on its
// primary, so both must run with the same index.num_shards.
type replicationNode struct {
	role      string
	lis       net.Listener
	grpcSrv   *grpc.Server
	primaries []*replication.PrimaryReplicator
	replicas  []*replication.ReplicaApplier
}

// replicationStatus is reported under "replication" in /health.
type replicationStatus struct {
	Role     string                      `json:"role"`
	Primary  []replication.PrimaryStatus `json:"primary,omitempty"`
	Replicas []replication.ReplicaStatus `json:"replicas,omitempty"`
}

// startReplication wires replication for every shard in shards. It must run
// after shards.Start() and before the node accepts writes, so the primary's
// WAL hook sees every write. Returns (nil, nil) when replication is disabled.
func startReplication(rc replicationConfig, shards *shard.ShardManager) (*replicationNode, error) {
	switch rc.Role {
	case "":
		return nil, nil
	case "primary":
		return startPrimary(rc, shards)
	case "replica":
		return startReplica(rc, shards)
	default:
		return nil, fmt.Errorf("cluster.replication_role %q: want \"\", \"primary\" or \"replica\"", rc.Role)
	}
}

func startPrimary(rc replicationConfig, shards *shard.ShardManager) (*replicationNode, error) {
	if rc.ListenAddr == "" {
		return nil, fmt.Errorf("replication primary: cluster.grpc_addr is required")
	}
	lis, err := net.Listen("tcp", rc.ListenAddr)
	if err != nil {
		return nil, fmt.Errorf("replication primary: listen %s: %w", rc.ListenAddr, err)
	}
	n := &replicationNode{role: rc.Role, lis: lis, grpcSrv: grpc.NewServer()}
	srv := replication.NewServer()
	for i := 0; i < shards.NumShards(); i++ {
		sm := shards.Shard(i)
		p := replication.NewPrimaryReplicator(sm.ShardID(), catchUpFrom(sm))
		p.SetLastSeqFunc(func() uint64 { return uint64(sm.LastSeq()) })
		sm.SetWALHook(func(e shard.WALEntry) { p.Append(toReplicationEntry(e)) })
		srv.Register(p)
		n.primaries = append(n.primaries, p)
	}
	replication.RegisterWALReplicationServer(n.grpcSrv, srv)
	go func() {
		if err := n.grpcSrv.Serve(lis); err != nil {
			slog.Error("replication gRPC server stopped", "err", err)
		}
	}()
	slog.Info("replication primary listening", "addr", lis.Addr().String(), "shards", shards.NumShards())
	return n, nil
}

func startReplica(rc replicationConfig, shards *shard.ShardManager) (*replicationNode, error) {
	if rc.PrimaryAddr == "" {
		return nil, fmt.Errorf("replication replica: cluster.primary_addr is required")
	}
	n := &replicationNode{role: rc.Role}
	for i := 0; i < shards.NumShards(); i++ {
		sm := shards.Shard(i)
		a := replication.NewReplicaApplier(sm.ShardID(), rc.PrimaryAddr, applyTo(shards, i))
		a.SetSeqPath(filepath.Join(rc.StateDir, sm.ShardID()+".replica.seq"))
		a.SetDeadLetterPath(filepath.Join(rc.StateDir, sm.ShardID()+".deadletter.ndjson"))
		// Without a saved position, resume from what's already on disk: 0 for
		// an empty replica, or the primary's flushed seq when this data dir
		// was seeded from a copy of the primary's.
		if err := a.Start(uint64(sm.SeqHighWater())); err != nil {
			n.stop()
			return nil, err
		}
		n.replicas = append(n.replicas, a)
	}
	slog.Info("replication replica started", "primary", rc.PrimaryAddr, "shards", shards.NumShards())
	return n, nil
}

// catchUpFrom serves a primary's WAL catch-up from the shard's WAL files.
func catchUpFrom(sm *shard.SegmentManager) replication.CatchUpFunc {
	return func(afterSeq uint64, fn func(*replication.WALEntry) error) error {
		return sm.ScanWAL(int64(afterSeq), func(e shard.WALEntry) error {
			return fn(toReplicationEntry(e))
		})
	}
}

// applyTo applies replicated entries to shard i through the replica's own
// WAL, so what it has applied survives a replica restart.
func applyTo(shards *shard.ShardManager, i int) replication.ApplyFunc {
	return func(e *replication.WALEntry) error {
		switch e.Op {
		case "index":
			return shards.IndexOnShard(i, types.Document{ID: e.DocId, Text: e.Text, Fields: e.Fields, Metadata: e.Metadata})
		case "delete":
			return shards.DeleteOnShard(i, e.DocId)
		case "reset":
			return shards.ResetShard(context.Background(), i)
		default:
			return fmt.Errorf("unknown WAL op %q", e.Op)
		}
	}
}

func toReplicationEntry(e shard.WALEntry) *replication.WALEntry {
	return &replication.WALEntry{
		Seq:      uint64(e.Seq),
		Op:       e.Op,
		DocId:    e.DocID,
		Text:     e.Text,
		Fields:   e.Fields,
		Metadata: e.Metadata,
		TsUnix:   e.TS.UnixNano(),
	}
}

// Addr returns the primary's gRPC listen address ("" on a replica).
func (n *replicationNode) Addr() string {
	if n == nil || n.lis == nil {
		return ""
	}
	return n.lis.Addr().String()
}

// ReadOnly reports whether this node must reject client writes (replicas
// only take writes from their primary).
func (n *replicationNode) ReadOnly() bool { return n != nil && n.role == "replica" }

// Status returns the node's replication state for /health.
func (n *replicationNode) Status() any {
	if n == nil {
		return nil
	}
	st := replicationStatus{Role: n.role}
	for _, p := range n.primaries {
		st.Primary = append(st.Primary, p.Status())
	}
	for _, a := range n.replicas {
		st.Replicas = append(st.Replicas, a.Status())
	}
	return st
}

// stop ends replication before the shards close: replicas stop applying and
// primaries end their streams.
func (n *replicationNode) stop() {
	if n == nil {
		return
	}
	for _, a := range n.replicas {
		a.Close()
	}
	for _, p := range n.primaries {
		p.Close()
	}
	if n.grpcSrv != nil {
		done := make(chan struct{})
		go func() { n.grpcSrv.GracefulStop(); close(done) }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			n.grpcSrv.Stop()
		}
	}
}

// afterClose runs once the shards have closed (flushing everything applied):
// only now is the replicas' current position durable, so save it.
func (n *replicationNode) afterClose() {
	if n == nil {
		return
	}
	for _, a := range n.replicas {
		if err := a.SaveAppliedSeq(); err != nil {
			slog.Error("replica: save applied seq on shutdown", "err", err)
		}
	}
}
