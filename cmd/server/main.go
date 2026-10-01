package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	_ "net/http/pprof" // registers /debug/pprof/* on http.DefaultServeMux
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"search-eval-platform/internal/analysis"
	"search-eval-platform/internal/api"
	"search-eval-platform/internal/cluster"
	"search-eval-platform/internal/config"
	"search-eval-platform/internal/logging"
	"search-eval-platform/internal/scoring"
	"search-eval-platform/internal/shard"
	"search-eval-platform/internal/storage"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// ── Flags ─────────────────────────────────────────────────────────────
	cfgPath := flag.String("config", "", "path to YAML config file (defaults used if empty)")
	modeFlag := flag.String("mode", "", "override server.mode from config (coordinator|shard)")
	pprofAddr := flag.String("pprof", "", "start pprof server on this address (e.g. localhost:6060); empty = disabled")
	flag.Parse()

	// ── Config ────────────────────────────────────────────────────────────
	var cfg *config.Config
	if *cfgPath != "" {
		var err error
		cfg, err = config.Load(*cfgPath)
		if err != nil {
			return err
		}
	} else {
		cfg = config.Default()
		config.ApplyEnv(cfg) // apply SEARCH_* env vars when no config file is provided
	}
	if *modeFlag != "" {
		cfg.Server.Mode = *modeFlag
	}

	// ── Logging ───────────────────────────────────────────────────────────
	logging.Init(cfg.Logging.Level, cfg.Logging.Format)
	slog.Info("starting search engine",
		"mode", cfg.Server.Mode,
		"addr", cfg.Addr(),
	)

	// ── pprof ─────────────────────────────────────────────────────────────
	// Runs on a separate loopback-only port so it is never reachable from the
	// public API listener. Enable with: -pprof localhost:6060
	if *pprofAddr != "" {
		go func() {
			slog.Info("pprof server listening", "addr", *pprofAddr)
			if err := http.ListenAndServe(*pprofAddr, nil); err != nil {
				slog.Error("pprof server exited", "err", err)
			}
		}()
	}

	// ── Storage: ensure data directories exist ────────────────────────────
	if err := os.MkdirAll(cfg.Storage.DataDir, 0o755); err != nil {
		return fmt.Errorf("mkdir data dir: %w", err)
	}

	// ── Route by mode ─────────────────────────────────────────────────────
	switch cfg.Server.Mode {
	case "coordinator":
		return runCoordinator(cfg)
	case "shard":
		return runShard(cfg)
	default:
		return fmt.Errorf("unknown server.mode %q: must be coordinator or shard", cfg.Server.Mode)
	}
}

// runShard runs the server with a local ShardManager (shard mode).
func runShard(cfg *config.Config) error {
	// Build scorer from config.
	var scorer scoring.Scorer
	switch cfg.Index.Scorer {
	case "tfidf":
		scorer = scoring.NewTFIDF()
	case "bm25f":
		scorer = scoring.NewBM25F(cfg.Index.BM25K1)
	default: // "bm25" or ""
		scorer = scoring.NewBM25(cfg.Index.BM25K1, cfg.Index.BM25B)
	}

	// Convert field configs for BM25F.
	var idxFields []scoring.FieldConfig
	for _, f := range cfg.Index.Fields {
		idxFields = append(idxFields, scoring.FieldConfig{
			Name:   f.Name,
			Weight: f.Weight,
			B:      f.B,
		})
	}
	if cfg.Index.Scorer == "bm25f" && len(idxFields) == 0 {
		// Default BM25F: title (boosted, low b) + body (standard b)
		idxFields = []scoring.FieldConfig{
			{Name: "title", Weight: 2.5, B: 0.45},
			{Name: "body", Weight: 1.0, B: 0.75},
		}
	}

	policy := &shard.TieredMergePolicy{
		SegmentsPerTier: cfg.Index.SegmentsPerTier,
		MaxMergeAtOnce:  cfg.Index.MaxMergeAtOnce,
		LevelRatio:      10,
	}

	tok := analysis.NewTokenizer(analysis.TokenizerConfig{
		StopwordsEnabled: cfg.Index.Stopwords.Enabled,
		StemmingEnabled:  cfg.Index.Stemming.Enabled,
		StemmingLanguage: cfg.Index.Stemming.Language,
	})

	store, err := storage.NewLocalStore(cfg.Storage.DataDir)
	if err != nil {
		return fmt.Errorf("create object store: %w", err)
	}

	// Load optional query-time synonym map.
	synonyms, err := analysis.LoadSynonyms(cfg.Index.SynonymFile)
	if err != nil {
		return fmt.Errorf("load synonyms: %w", err)
	}
	if synonyms != nil {
		slog.Info("synonym expansion enabled", "file", cfg.Index.SynonymFile)
	}

	smOpts := shard.ShardManagerOptions{
		Tokenizer:            tok,
		BloomFPRate:          cfg.Index.BloomFPRate,
		QueryCacheSize:       cfg.Search.QueryCacheSize,
		DocCacheSize:         cfg.Search.DocCacheSize,
		Store:                store,
		BM25FFields:          idxFields,
		MaxBufferDocs:        cfg.Index.MaxBufferDocs,
		MemThresholdMB:       cfg.Index.MemThresholdMB,
		WALDurability:        cfg.Index.WALDurability,
		WALSyncIntervalMs:    cfg.Index.WALSyncIntervalMs,
		WALCompression:       cfg.Index.WALCompression,
		IdleFlushSecs:        cfg.Index.IdleFlushSecs,
		RefreshInterval:      time.Duration(cfg.Index.RefreshIntervalMs) * time.Millisecond,
		MaxMergeSizeMB:       cfg.Index.MaxMergeSizeMB,
		MergeRateLimitMBps:   cfg.Index.MergeRateLimitMBps,
		MergeIOBurstMB:       cfg.Index.MergeIOBurstMB,
		MergeDeletionWeight:  cfg.Index.MergeDeletionWeight,
		FloorSegmentMB:       cfg.Index.FloorSegmentMB,
		ExpungeDeletesPct:    cfg.Index.ExpungeDeletesPct,
		GlobalFusion:         cfg.Search.GlobalFusion,
		BufferPoolSize:       cfg.Index.BufferPoolSize,
		SkipStoredFields:     cfg.Index.SkipStoredFields,
		PhraseBoost:          cfg.Search.PhraseBoost,
		VirtualNodesPerShard: cfg.Index.VirtualNodesPerShard,
		Synonyms:             synonyms,
	}
	shards, err := shard.NewShardManager(
		cfg.Index.NumShards,
		cfg.Storage.DataDir,
		scorer,
		policy,
		smOpts,
	)
	if err != nil {
		return fmt.Errorf("create shard manager: %w", err)
	}
	if err := shards.Start(); err != nil {
		return fmt.Errorf("start shard manager: %w", err)
	}

	mux := http.NewServeMux()
	h := api.NewHandler(shards, cfg.Server.Mode, cfg.Index.BM25K1, cfg.Index.BM25B, cfg.Server.APIKey)

	// WAL replication: wire before serving so the primary sees every write.
	repl, err := startReplication(replicationConfig{
		Role:        cfg.Cluster.ReplicationRole,
		ListenAddr:  grpcListenAddr(cfg.Cluster.GRPCAddr),
		PrimaryAddr: cfg.Cluster.PrimaryAddr,
		StateDir:    filepath.Join(cfg.Storage.DataDir, "replication"),
	}, shards)
	if err != nil {
		shards.Close()
		return err
	}
	if repl != nil {
		h.SetReplicationStatus(repl.Status)
		h.SetRequireWAL(cfg.Cluster.ReplicationRole == "primary")
		if repl.ReadOnly() {
			h.SetReadOnly("read-only replica of " + cfg.Cluster.PrimaryAddr)
		}
	}

	if len(cfg.Cluster.LocalShards) > 0 {
		h.SetLocalShards(cfg.Cluster.LocalShards)
	}
	h.Register(mux)
	h.SetDefaultTopK(cfg.Search.DefaultTopK)

	// Shard nodes expose internal endpoints for coordinator requests.
	h.RegisterInternal(mux)

	// Startup warmup: structural load → SetReady → posting warmup →
	// SetWarm → background merge → re-warm → SetWarm again.
	go func() {
		ctx := context.Background()
		topK := cfg.Index.WarmupTopTerms
		mlock := cfg.Index.MLockSegments
		warmupMode := cfg.Index.WarmupMode
		if warmupMode == "" {
			warmupMode = "top_k"
		}
		tierThreshold := int64(cfg.Index.WarmupTierThresholdMB) * 1024 * 1024
		if tierThreshold <= 0 {
			tierThreshold = 256 * 1024 * 1024
		}

		// warmSegments runs the configured warmup strategy on current segments.
		warmSegments := func(label string) {
			if len(shards.SegmentRecords()) == 0 {
				return
			}
			switch warmupMode {
			case "full":
				slog.Info("startup: full segment warmup", "phase", label)
				if err := shards.WarmTiered(ctx, 1<<62, topK, mlock); err != nil {
					slog.Error("full warmup failed, continuing", "err", err)
				}
			case "tiered":
				slog.Info("startup: tiered segment warmup", "phase", label,
					"threshold_mb", cfg.Index.WarmupTierThresholdMB, "top_k", topK)
				if err := shards.WarmTiered(ctx, tierThreshold, topK, mlock); err != nil {
					slog.Error("tiered warmup failed, continuing", "err", err)
				}
			default: // "top_k"
				if topK > 0 {
					slog.Info("startup: warming top terms", "phase", label, "top_k", topK)
					if err := shards.WarmTopTerms(ctx, topK, mlock); err != nil {
						slog.Error("top-term warmup failed, continuing", "err", err)
					}
				}
			}
		}

		// Phase 1: Structural load — parse FST + metadata for all segments.
		// No posting data I/O. SetReady so load balancers can connect immediately.
		if cfg.Index.WarmupOnStartup && len(shards.SegmentRecords()) > 0 {
			slog.Info("startup: loading segment structure")
			if err := shards.WarmupSegments(ctx); err != nil {
				slog.Error("segment structure load failed, continuing", "err", err)
			}
		}
		h.SetReady()
		slog.Info("server ready, accepting search requests")

		// Phase 2: Posting warmup (mode-driven).
		warmSegments("initial")
		h.SetWarm()
		slog.Info("server warm", "mode", warmupMode)

		// Phase 3: Background merge. After merge, re-warm the new segments
		// so the warm flag remains valid.
		if cfg.Index.MergeOnStartup && len(shards.SegmentRecords()) > 0 {
			target := cfg.Index.StartupMergeTarget
			if target <= 0 {
				target = 1
			}
			slog.Info("startup: force-merging segments (background)", "target", target)
			h.ResetWarm()
			if err := shards.ForceMerge(ctx, target); err != nil {
				slog.Error("startup merge failed", "err", err)
			} else {
				slog.Info("background merge complete", "target_segments_per_shard", target)
			}
			warmSegments("post-merge")
			h.SetWarm()
			slog.Info("post-merge warm complete")
		}
	}()

	return runShardServeAndClose(
		func() error { return serveHTTP(cfg, mux) },
		func() error {
			repl.stop() // stop applying / streaming before the shards close
			err := shards.Close()
			repl.afterClose() // replicas' position is durable only after the flush
			return err
		},
	)
}

// grpcListenAddr turns cluster.grpc_addr (the address other nodes dial, e.g.
// "node-1:9090") into a listen address on all interfaces (":9090").
func grpcListenAddr(addr string) string {
	if _, port, err := net.SplitHostPort(addr); err == nil {
		return ":" + port
	}
	return addr
}

// runShardServeAndClose runs serve (the HTTP server) and always attempts
// close (the shard manager's final flush) afterward, regardless of whether
// serve returned an error. A serve error (e.g. a graceful-shutdown timeout)
// used to skip close entirely, leaking buffered-but-unflushed documents
// (especially under wal_durability: async) and the WAL file descriptor.
func runShardServeAndClose(serve func() error, close func() error) error {
	serveErr := serve()

	slog.Info("flushing shards before exit")
	if err := close(); err != nil {
		slog.Error("shard manager close", "err", err)
	}
	slog.Info("shutdown complete")

	if serveErr != nil {
		return fmt.Errorf("http server: %w", serveErr)
	}
	return nil
}

// runCoordinator runs the server in coordinator mode, routing all requests to
// shard nodes via the cluster.Client.
func runCoordinator(cfg *config.Config) error {
	ring := cluster.NewRing(cfg.Cluster.TotalShards)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Static membership: the bootstrap nodes' shard ownership is learned from
	// their /health (local_shards) and kept current by a background refresh.
	var staticNodes []*cluster.NodeMeta
	for _, addr := range cfg.Cluster.BootstrapAddrs {
		staticNodes = append(staticNodes, &cluster.NodeMeta{
			NodeID:   addr,
			HTTPAddr: addr,
			Role:     "shard",
		})
	}
	if len(staticNodes) > 0 {
		members := cluster.NewStaticMemberWatcher(ring, cfg.Cluster.BootstrapAddrs)
		discoverStaticMembers(ctx, members, len(staticNodes), 30*time.Second)
		go members.Run(ctx, 10*time.Second)
	}

	client := cluster.NewClient(ring, cfg.Search.GlobalFusion, cfg.Index.VirtualNodesPerShard)
	client.SetAPIKey(cfg.Server.APIKey)
	if len(staticNodes) > 0 {
		poller := cluster.NewHealthPoller(staticNodes, client.Breaker, 10*time.Second)
		poller.Start()
		defer poller.Stop()
		slog.Info("static ring + health poller active", "nodes", len(staticNodes))
	}

	// K8s-native dynamic membership: watch the headless service's Endpoints.
	// When set, the coordinator discovers shard pods automatically as they
	// start or restart, without needing static bootstrap_addrs.
	if cfg.Cluster.K8sService != "" {
		watcher := cluster.NewK8sMemberWatcher(ring, cluster.K8sMemberConfig{
			Namespace: cfg.Cluster.K8sNamespace,
			Service:   cfg.Cluster.K8sService,
			HTTPPort:  cfg.Cluster.K8sHTTPPort,
			GRPCPort:  cfg.Cluster.K8sGRPCPort,
		})
		go watcher.Run(ctx)
		slog.Info("k8s membership watcher started",
			"service", cfg.Cluster.K8sService,
			"namespace", cfg.Cluster.K8sNamespace)
	}

	mux := http.NewServeMux()
	h := api.NewCoordinatorHandler(client, cfg.Server.Mode, cfg.Index.BM25K1, cfg.Index.BM25B, cfg.Server.APIKey)
	h.Register(mux)
	h.SetDefaultTopK(cfg.Search.DefaultTopK)
	h.SetReady()
	slog.Info("coordinator ready", "fusion", cfg.Search.GlobalFusion,
		"total_shards", cfg.Cluster.TotalShards)
	if err := serveHTTP(cfg, mux); err != nil {
		return fmt.Errorf("http server: %w", err)
	}
	slog.Info("shutdown complete")
	return nil
}

// discoverStaticMembers runs the first membership refresh, retrying until all
// want nodes have reported their shards or the timeout passes (shard nodes may
// still be starting). Routing starts either way; later refreshes pick up the rest.
func discoverStaticMembers(ctx context.Context, m *cluster.StaticMemberWatcher, want int, timeout time.Duration) {
	deadline := time.Now().Add(timeout)
	for {
		n := m.Refresh(ctx)
		if n >= want {
			slog.Info("static membership discovered", "nodes", n)
			return
		}
		if time.Now().After(deadline) || ctx.Err() != nil {
			slog.Warn("static membership incomplete; continuing, will retry in background",
				"nodes", n, "configured", want)
			return
		}
		time.Sleep(time.Second)
	}
}

// serveHTTP wraps a mux with middleware and runs the HTTP server with
// graceful shutdown on SIGINT/SIGTERM.
func serveHTTP(cfg *config.Config, mux *http.ServeMux) error {
	handler := api.Chain(mux,
		api.RecoveryMiddleware,
		api.LoggingMiddleware,
	)

	srv := &http.Server{
		Addr:         cfg.Addr(),
		Handler:      handler,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	serverErr := make(chan error, 1)
	go func() {
		slog.Info("listening", "addr", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
		}
	}()

	select {
	case err := <-serverErr:
		return fmt.Errorf("http server: %w", err)
	case sig := <-quit:
		slog.Info("shutdown signal received", "signal", sig)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		return fmt.Errorf("http shutdown: %w", err)
	}
	slog.Info("http server stopped")
	return nil
}
