package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Config is the top-level configuration structure.
// All fields have sensible defaults (see Default()).
type Config struct {
	Server      ServerConfig      `yaml:"server"`
	Storage     StorageConfig     `yaml:"storage"`
	Index       IndexConfig       `yaml:"index"`
	Search      SearchConfig      `yaml:"search"`
	Logging     LoggingConfig     `yaml:"logging"`
	Cluster     ClusterConfig     `yaml:"cluster"`
}

// ServerConfig controls the HTTP listener.
type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
	// Mode controls the node role: coordinator | shard
	Mode string `yaml:"mode"`
	// APIKey, when non-empty, requires write endpoints to present
	// "Authorization: Bearer <key>". Read endpoints (/search, /health) are unaffected.
	APIKey string `yaml:"api_key"`
}

// StorageConfig selects the storage paths for manifests and data files.
type StorageConfig struct {
	// Path is the file path for the segment manifest (e.g. data/manifest.json)
	Path string `yaml:"path"`
	// DataDir is the root for WAL and segment files
	DataDir string `yaml:"data_dir"`
}

// StemmingConfig controls the optional stemming pipeline stage.
type StemmingConfig struct {
	Enabled  bool   `yaml:"enabled"`
	Language string `yaml:"language"` // "english" (Snowball language name)
}

// StopwordsConfig controls the stopword-filtering pipeline stage.
type StopwordsConfig struct {
	Enabled  bool   `yaml:"enabled"`
	Language string `yaml:"language"` // "en" = English (bbalet/stopwords language code)
}

// FieldConfig defines BM25F scoring parameters for one document field.
// Used only when Scorer is set to "bm25f".
type FieldConfig struct {
	Name   string  `yaml:"name"`   // e.g. "title", "body"
	Weight float64 `yaml:"weight"` // field contribution multiplier
	B      float64 `yaml:"b"`      // length normalization (0=none, 1=full)
}

// IndexConfig tunes the indexing pipeline.
type IndexConfig struct {
	MemThresholdMB     int            `yaml:"mem_threshold_mb"`
	MaxBufferDocs      int            `yaml:"max_buffer_docs"`
	NumShards          int            `yaml:"num_shards"`
	BM25K1             float64        `yaml:"bm25_k1"`
	BM25B              float64        `yaml:"bm25_b"`
	SegmentsPerTier    int            `yaml:"segments_per_tier"`
	MaxMergeAtOnce     int            `yaml:"max_merge_at_once"`
	SegmentCompression string         `yaml:"segment_compression"` // "" | "lz4"
	UseFOR32           bool           `yaml:"use_for32"`           // true → v6 FOR-delta segments (AVX2 SIMD); default true
	BloomFPRate        float64        `yaml:"bloom_fp_rate"`       // 0 = disabled
	Stemming           StemmingConfig  `yaml:"stemming"`
	Stopwords          StopwordsConfig `yaml:"stopwords"`
	Scorer             string          `yaml:"scorer"`  // "bm25" (default) | "tfidf" | "bm25f"
	Fields             []FieldConfig  `yaml:"fields"`  // BM25F field definitions (scorer=bm25f only)
	WALDurability      string         `yaml:"wal_durability"`       // "async" (default) | "sync" | "off"
	WALSyncIntervalMs  int            `yaml:"wal_sync_interval_ms"` // default 1000 (ms between async flushes)
	WALCompression     string         `yaml:"wal_compression"`      // "" | "lz4"
	IdleFlushSecs      int            `yaml:"idle_flush_secs"`      // flush buffer if no writes for N seconds; 0 = disabled
	MaxMergeSizeMB          int64          `yaml:"max_merge_size_mb"`           // cap total input size per merge; 0 = disabled
	MergeRateLimitMBps      float64        `yaml:"merge_rate_limit_mbps"`        // MB/s token-bucket limit on merge I/O; 0 = unlimited
	MergeIOBurstMB          int            `yaml:"merge_io_burst_mb"`            // token-bucket burst in MB; 0 = auto (rate×0.5s)
	MergeDeletionWeight     float64        `yaml:"merge_deletion_weight"`        // α in deletion_score=1+α*(deleted/total); 0 = disable deletion-aware scoring; default 2.0
	FloorSegmentMB          int64          `yaml:"floor_segment_mb"`             // treat segments smaller than this as this size in tier math; default 2 MB
	ExpungeDeletesPct       float64        `yaml:"expunge_deletes_pct"`          // opportunistically merge segments with deletion ratio >= this; 0 = disabled; default 0.25
	BufferPoolSize     int            `yaml:"buffer_pool_size"`      // number of background-flush swap buffers per shard; 0 = default (4)
	SkipStoredFields   bool           `yaml:"skip_stored_fields"`    // omit .seg.fld sidecars; disables snippet generation but cuts segment size ~75%
	MergeOnStartup      bool           `yaml:"merge_on_startup"`       // compact segments at startup before serving; default true
	WarmupOnStartup        bool           `yaml:"warmup_on_startup"`           // load all segments into RAM at startup; default true
	StartupMergeTarget     int            `yaml:"startup_merge_target"`        // target segment count after startup merge; default 3 (0 or 1 = merge to 1)
	WarmupTopTerms         int            `yaml:"warmup_top_terms"`            // top-K DF terms to warm; default 1000; 0 = disabled
	MLockSegments          bool           `yaml:"mlock_segments"`              // attempt mlock after warmup; requires CAP_IPC_LOCK
	WarmupMode             string         `yaml:"warmup_mode"`                 // "top_k" (default) | "full" | "tiered"
	WarmupTierThresholdMB  int            `yaml:"warmup_tier_threshold_mb"`    // tiered mode: segments <= this get full warmup; default 256
	VirtualNodesPerShard   int            `yaml:"virtual_nodes_per_shard"`     // virtual nodes per shard for consistent-hash ring routing; 0 = default (150)
	SynonymFile            string         `yaml:"synonym_file"`                // path to synonyms.txt; "" = disabled
}

// SearchConfig tunes query execution.
type SearchConfig struct {
	// GlobalFusion selects cross-shard result fusion:
	//   "score" – single-phase, local IDF per shard, merge by raw score (default; matches ES query_then_fetch)
	//   "rrf"   – single-phase, local IDF per shard, Reciprocal Rank Fusion merge
	GlobalFusion   string `yaml:"global_fusion"`
	DefaultTopK    int    `yaml:"default_top_k"`
	QueryCacheSize int    `yaml:"query_cache_size"` // 0 = disabled
	DocCacheSize   int    `yaml:"doc_cache_size"`   // LRU doc text cache entries, default 50000
}

// LoggingConfig controls log output.
type LoggingConfig struct {
	// Level is one of: debug | info | warn | error
	Level string `yaml:"level"`
	// Format is one of: text | json
	Format string `yaml:"format"`
}

// ClusterConfig is used only in coordinator/shard modes.
type ClusterConfig struct {
	NodeID         string   `yaml:"node_id"`
	LocalShards    []int    `yaml:"local_shards"`
	TotalShards    int      `yaml:"total_shards"`
	BootstrapAddrs []string `yaml:"bootstrap_addrs"`
	GRPCAddr       string   `yaml:"grpc_addr"`

	// K8s-native membership discovery (coordinator only).
	// When K8sService is set, the coordinator watches the named Endpoints
	// resource instead of (or after) the static BootstrapAddrs list.
	K8sNamespace string `yaml:"k8s_namespace"` // default: reads from in-cluster pod namespace file
	K8sService   string `yaml:"k8s_service"`   // headless service name, e.g. "search-shards"
	K8sHTTPPort  int    `yaml:"k8s_http_port"` // port for /health probes; default 8080
	K8sGRPCPort  int    `yaml:"k8s_grpc_port"` // port for gRPC replication; default 9090
}

// Default returns a Config populated with safe defaults.
func Default() *Config {
	return &Config{
		Server: ServerConfig{
			Host: "0.0.0.0",
			Port: 8080,
			Mode: "shard",
		},
		Storage: StorageConfig{
			Path:    "data/manifest.json",
			DataDir: "data",
		},
		Index: IndexConfig{
			UseFOR32:           true,
			MemThresholdMB:     512,
			MaxBufferDocs:      100_000,
			WALDurability:      "async",
			WALSyncIntervalMs:  1000,
			NumShards:          1,
			BM25K1:             1.2,
			BM25B:              0.75,
			SegmentsPerTier:    15,
			MaxMergeAtOnce:     10,
			MaxMergeSizeMB:          256,
			MergeRateLimitMBps:      80,
			MergeDeletionWeight:     2.0,
			FloorSegmentMB:          2,
			ExpungeDeletesPct:       0.25,
			BufferPoolSize:     4,
			MergeOnStartup:     true,
			WarmupOnStartup:       true,
			StartupMergeTarget:    2,
			WarmupTopTerms:        1000,
			MLockSegments:         false,
			WarmupMode:            "tiered",
			WarmupTierThresholdMB: 600,
			VirtualNodesPerShard:  150,
			SegmentCompression: "",
			BloomFPRate:        0.01,
			Stemming: StemmingConfig{
				Enabled:  false,
				Language: "english",
			},
			Stopwords: StopwordsConfig{
				Enabled:  false,
				Language: "en",
			},
			Scorer: "bm25",
		},
		Search: SearchConfig{
			GlobalFusion:   "score",
			DefaultTopK:    10,
			QueryCacheSize: 1000,
			DocCacheSize:   50_000,
		},
		Logging: LoggingConfig{
			Level:  "info",
			Format: "text",
		},
	}
}

// Load reads a YAML config file and merges it onto the defaults.
// Fields absent from the file keep their default values.
// After decoding, environment variables with the SEARCH_ prefix are applied
// as a final override layer (see ApplyEnv).
func Load(path string) (*Config, error) {
	cfg := Default()
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("config: open %s: %w", path, err)
	}
	defer f.Close()
	if err := yaml.NewDecoder(f).Decode(cfg); err != nil {
		return nil, fmt.Errorf("config: decode %s: %w", path, err)
	}
	ApplyEnv(cfg)
	return cfg, nil
}

// ApplyEnv overrides config fields from SEARCH_* environment variables.
// Priority: env > YAML file > defaults.
//
// Supported variables:
//
//	SEARCH_SERVER_MODE               server.mode
//	SEARCH_SERVER_PORT               server.port
//	SEARCH_SERVER_API_KEY            server.api_key
//	SEARCH_LOG_LEVEL                 logging.level
//	SEARCH_LOG_FORMAT                logging.format
//	SEARCH_INDEX_NUM_SHARDS          index.num_shards
//	SEARCH_INDEX_MEM_THRESHOLD_MB    index.mem_threshold_mb
//	SEARCH_INDEX_BM25_K1             index.bm25_k1
//	SEARCH_INDEX_BM25_B              index.bm25_b
//	SEARCH_INDEX_SEGMENT_COMPRESSION index.segment_compression
//	SEARCH_INDEX_BLOOM_FP_RATE       index.bloom_fp_rate
//	SEARCH_INDEX_WARMUP_MODE               index.warmup_mode
//	SEARCH_INDEX_VIRTUAL_NODES_PER_SHARD   index.virtual_nodes_per_shard
//	SEARCH_SEARCH_GLOBAL_FUSION            search.global_fusion
//	SEARCH_CLUSTER_NODE_ID           cluster.node_id
//	SEARCH_CLUSTER_TOTAL_SHARDS      cluster.total_shards
//	SEARCH_CLUSTER_LOCAL_SHARDS      cluster.local_shards  (comma-separated ints)
//	SEARCH_CLUSTER_BOOTSTRAP         cluster.bootstrap_addrs (comma-separated)
//	SEARCH_CLUSTER_GRPC_ADDR         cluster.grpc_addr
//	SEARCH_CLUSTER_K8S_NAMESPACE     cluster.k8s_namespace
//	SEARCH_CLUSTER_K8S_SERVICE       cluster.k8s_service
//	SEARCH_CLUSTER_K8S_HTTP_PORT     cluster.k8s_http_port
//	SEARCH_CLUSTER_K8S_GRPC_PORT     cluster.k8s_grpc_port
//	SEARCH_STORAGE_PATH              storage.path
//	SEARCH_STORAGE_DATA_DIR          storage.data_dir
func ApplyEnv(cfg *Config) {
	setStr := func(dst *string, key string) {
		if v := os.Getenv(key); v != "" {
			*dst = v
		}
	}
	setInt := func(dst *int, key string) {
		if v := os.Getenv(key); v != "" {
			if n, err := strconv.Atoi(v); err == nil {
				*dst = n
			}
		}
	}
	setFloat := func(dst *float64, key string) {
		if v := os.Getenv(key); v != "" {
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				*dst = f
			}
		}
	}
	setStrSlice := func(dst *[]string, key string) {
		if v := os.Getenv(key); v != "" {
			parts := strings.Split(v, ",")
			result := make([]string, 0, len(parts))
			for _, p := range parts {
				if s := strings.TrimSpace(p); s != "" {
					result = append(result, s)
				}
			}
			*dst = result
		}
	}
	setIntSlice := func(dst *[]int, key string) {
		if v := os.Getenv(key); v != "" {
			parts := strings.Split(v, ",")
			result := make([]int, 0, len(parts))
			for _, p := range parts {
				if n, err := strconv.Atoi(strings.TrimSpace(p)); err == nil {
					result = append(result, n)
				}
			}
			*dst = result
		}
	}

	setStr(&cfg.Server.Mode, "SEARCH_SERVER_MODE")
	setInt(&cfg.Server.Port, "SEARCH_SERVER_PORT")
	setStr(&cfg.Server.APIKey, "SEARCH_SERVER_API_KEY")
	setStr(&cfg.Logging.Level, "SEARCH_LOG_LEVEL")
	setStr(&cfg.Logging.Format, "SEARCH_LOG_FORMAT")
	setInt(&cfg.Index.NumShards, "SEARCH_INDEX_NUM_SHARDS")
	setInt(&cfg.Index.MemThresholdMB, "SEARCH_INDEX_MEM_THRESHOLD_MB")
	setFloat(&cfg.Index.BM25K1, "SEARCH_INDEX_BM25_K1")
	setFloat(&cfg.Index.BM25B, "SEARCH_INDEX_BM25_B")
	setStr(&cfg.Index.SegmentCompression, "SEARCH_INDEX_SEGMENT_COMPRESSION")
	setFloat(&cfg.Index.BloomFPRate, "SEARCH_INDEX_BLOOM_FP_RATE")
	setStr(&cfg.Index.WarmupMode, "SEARCH_INDEX_WARMUP_MODE")
	setInt(&cfg.Index.StartupMergeTarget, "SEARCH_INDEX_STARTUP_MERGE_TARGET")
	setInt(&cfg.Index.VirtualNodesPerShard, "SEARCH_INDEX_VIRTUAL_NODES_PER_SHARD")
	setStr(&cfg.Search.GlobalFusion, "SEARCH_SEARCH_GLOBAL_FUSION")
	setStr(&cfg.Cluster.NodeID, "SEARCH_CLUSTER_NODE_ID")
	setInt(&cfg.Cluster.TotalShards, "SEARCH_CLUSTER_TOTAL_SHARDS")
	setIntSlice(&cfg.Cluster.LocalShards, "SEARCH_CLUSTER_LOCAL_SHARDS")
	setStrSlice(&cfg.Cluster.BootstrapAddrs, "SEARCH_CLUSTER_BOOTSTRAP")

	// Storage
	setStr(&cfg.Storage.Path,    "SEARCH_STORAGE_PATH")
	setStr(&cfg.Storage.DataDir, "SEARCH_STORAGE_DATA_DIR")

	setStr(&cfg.Cluster.GRPCAddr, "SEARCH_CLUSTER_GRPC_ADDR")
	setStr(&cfg.Cluster.K8sNamespace, "SEARCH_CLUSTER_K8S_NAMESPACE")
	setStr(&cfg.Cluster.K8sService, "SEARCH_CLUSTER_K8S_SERVICE")
	setInt(&cfg.Cluster.K8sHTTPPort, "SEARCH_CLUSTER_K8S_HTTP_PORT")
	setInt(&cfg.Cluster.K8sGRPCPort, "SEARCH_CLUSTER_K8S_GRPC_PORT")
}

// Addr returns "host:port" suitable for http.ListenAndServe.
func (c *Config) Addr() string {
	return fmt.Sprintf("%s:%d", c.Server.Host, c.Server.Port)
}
