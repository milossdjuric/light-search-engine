package config_test

import (
	"os"
	"testing"

	"search-eval-platform/internal/config"
)

func TestApplyEnvOverridesDefault(t *testing.T) {
	keys := []string{
		"SEARCH_SERVER_MODE", "SEARCH_SERVER_PORT", "SEARCH_LOG_LEVEL",
		"SEARCH_SEARCH_GLOBAL_FUSION", "SEARCH_CLUSTER_GRPC_ADDR",
		"SEARCH_STORAGE_DATA_DIR",
	}
	t.Cleanup(func() {
		for _, k := range keys {
			os.Unsetenv(k)
		}
	})

	os.Setenv("SEARCH_SERVER_MODE", "coordinator")
	os.Setenv("SEARCH_SERVER_PORT", "9999")
	os.Setenv("SEARCH_LOG_LEVEL", "debug")
	os.Setenv("SEARCH_SEARCH_GLOBAL_FUSION", "rrf")
	os.Setenv("SEARCH_CLUSTER_GRPC_ADDR", "node-1:9090")
	os.Setenv("SEARCH_STORAGE_DATA_DIR", "/mnt/efs/data")

	cfg := config.Default()
	config.ApplyEnv(cfg)

	if cfg.Server.Mode != "coordinator" {
		t.Errorf("Server.Mode: want coordinator, got %s", cfg.Server.Mode)
	}
	if cfg.Server.Port != 9999 {
		t.Errorf("Server.Port: want 9999, got %d", cfg.Server.Port)
	}
	if cfg.Logging.Level != "debug" {
		t.Errorf("Logging.Level: want debug, got %s", cfg.Logging.Level)
	}
	if cfg.Search.GlobalFusion != "rrf" {
		t.Errorf("Search.GlobalFusion: want rrf, got %s", cfg.Search.GlobalFusion)
	}
	if cfg.Cluster.GRPCAddr != "node-1:9090" {
		t.Errorf("Cluster.GRPCAddr: want node-1:9090, got %s", cfg.Cluster.GRPCAddr)
	}
	if cfg.Storage.DataDir != "/mnt/efs/data" {
		t.Errorf("Storage.DataDir: want /mnt/efs/data, got %s", cfg.Storage.DataDir)
	}
}

func TestApplyEnvNoopWhenEnvUnset(t *testing.T) {
	// With no env vars set, ApplyEnv must not change any defaults.
	cfg1 := config.Default()
	cfg2 := config.Default()
	config.ApplyEnv(cfg2)

	if cfg1.Server.Mode != cfg2.Server.Mode {
		t.Errorf("Server.Mode changed: %s → %s", cfg1.Server.Mode, cfg2.Server.Mode)
	}
	if cfg1.Server.Port != cfg2.Server.Port {
		t.Errorf("Server.Port changed: %d → %d", cfg1.Server.Port, cfg2.Server.Port)
	}
	if cfg1.Search.GlobalFusion != cfg2.Search.GlobalFusion {
		t.Errorf("GlobalFusion changed: %s → %s", cfg1.Search.GlobalFusion, cfg2.Search.GlobalFusion)
	}
}
