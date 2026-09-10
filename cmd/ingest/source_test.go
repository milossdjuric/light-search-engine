package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectFormat(t *testing.T) {
	cases := []struct {
		path     string
		explicit string
		want     string
	}{
		{"corpus.parquet", "auto", "parquet"},
		{"corpus.jsonl", "auto", "jsonl"},
		{"corpus.ndjson", "auto", "jsonl"},
		{"corpus.json", "auto", "json"},
		{"corpus.csv", "auto", "csv"},
		{"corpus.CSV", "auto", "csv"}, // case-insensitive
		{"corpus.unknown", "auto", "jsonl"}, // fallback
		{"corpus.json", "csv", "csv"}, // explicit override wins
	}
	for _, c := range cases {
		got := detectFormat(c.path, c.explicit)
		if got != c.want {
			t.Errorf("detectFormat(%q, %q) = %q, want %q", c.path, c.explicit, got, c.want)
		}
	}
}

func TestResolveSource_Gen(t *testing.T) {
	cfg := &Config{Source: "gen://100"}
	files, err := resolveSource(cfg)
	if err != nil {
		t.Fatalf("resolveSource: %v", err)
	}
	if len(files) != 1 {
		t.Fatalf("want 1 sourceFile, got %d", len(files))
	}
	if files[0].format != "gen" || files[0].path != "gen://100" {
		t.Errorf("got %+v", files[0])
	}
}

func TestResolveSource_LocalFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.jsonl")
	if err := os.WriteFile(path, []byte(`{"_id":"1"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{Source: path, Format: "auto"}
	files, err := resolveSource(cfg)
	if err != nil {
		t.Fatalf("resolveSource: %v", err)
	}
	if len(files) != 1 {
		t.Fatalf("want 1 sourceFile, got %d", len(files))
	}
	if files[0].format != "jsonl" || files[0].temp {
		t.Errorf("got %+v", files[0])
	}
}

func TestResolveSource_LocalFileExplicitFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corpus.dat")
	if err := os.WriteFile(path, []byte(`id,text`), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{Source: path, Format: "csv"}
	files, err := resolveSource(cfg)
	if err != nil {
		t.Fatalf("resolveSource: %v", err)
	}
	if files[0].format != "csv" {
		t.Errorf("format: want csv, got %q", files[0].format)
	}
}

// resolveHuggingFace hits the real HuggingFace network API and is not covered
// here; it is exercised indirectly by bench/run_beir.sh in CI-adjacent runs.
