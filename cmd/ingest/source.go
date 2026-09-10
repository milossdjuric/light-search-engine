package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// resolveSource turns cfg.Source into a list of local sourceFiles.
// For hf:// sources it queries the HuggingFace datasets-server API to get
// parquet file URLs, then downloads each to a temp file.
// For local paths it just wraps the file with format detection.
func resolveSource(cfg *Config) ([]sourceFile, error) {
	if strings.HasPrefix(cfg.Source, "hf://") {
		return resolveHuggingFace(cfg)
	}
	if strings.HasPrefix(cfg.Source, "gen://") {
		return []sourceFile{{path: cfg.Source, format: "gen"}}, nil
	}
	// Local file — detect format from extension unless overridden.
	format := detectFormat(cfg.Source, cfg.Format)
	return []sourceFile{{path: cfg.Source, format: format, temp: false}}, nil
}

// ── HuggingFace ────────────────────────────────────────────────────────────────

// hfParquetResp is the response body from the datasets-server /parquet endpoint.
type hfParquetResp struct {
	ParquetFiles []hfParquetFile `json:"parquet_files"`
}

type hfParquetFile struct {
	Dataset  string `json:"dataset"`
	Config   string `json:"config"`
	Split    string `json:"split"`
	URL      string `json:"url"`
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}

// resolveHuggingFace resolves a hf://owner/repo source to downloaded parquet files.
// It calls the public HuggingFace datasets-server API (no auth needed for public
// datasets) to discover the parquet shard URLs for the requested config+split,
// then downloads each shard to a temp file.
func resolveHuggingFace(cfg *Config) ([]sourceFile, error) {
	repo := strings.TrimPrefix(cfg.Source, "hf://")
	if repo == "" {
		return nil, fmt.Errorf("invalid hf:// source: %q", cfg.Source)
	}

	// Query the datasets-server for parquet file list.
	apiURL := "https://datasets-server.huggingface.co/parquet?dataset=" + repo
	log.Printf("querying HuggingFace datasets-server: %s", apiURL)

	cli := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequest("GET", apiURL, nil)
	if cfg.Token != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.Token)
	}

	resp, err := cli.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HuggingFace API request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HuggingFace API returned %d: %s", resp.StatusCode, string(body))
	}

	var apiResp hfParquetResp
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("HuggingFace API decode: %w", err)
	}

	// Filter by requested config + split.
	var matching []hfParquetFile
	for _, pf := range apiResp.ParquetFiles {
		if pf.Config == cfg.HFConfig && pf.Split == cfg.HFSplit {
			matching = append(matching, pf)
		}
	}

	if len(matching) == 0 {
		// Help the user by listing what's available.
		log.Printf("no parquet files found for config=%q split=%q", cfg.HFConfig, cfg.HFSplit)
		log.Printf("available config/split combinations:")
		seen := map[string]bool{}
		for _, pf := range apiResp.ParquetFiles {
			key := pf.Config + "/" + pf.Split
			if !seen[key] {
				seen[key] = true
				log.Printf("  -config %s -split %s", pf.Config, pf.Split)
			}
		}
		return nil, fmt.Errorf("no matching parquet files; see available combinations above")
	}

	log.Printf("found %d parquet shard(s) for %s config=%s split=%s",
		len(matching), repo, cfg.HFConfig, cfg.HFSplit)

	// Download each shard to a temp file.
	var files []sourceFile
	for i, pf := range matching {
		sizeMB := float64(pf.Size) / (1 << 20)
		log.Printf("downloading shard [%d/%d]: %s (%.1f MB)", i+1, len(matching), pf.Filename, sizeMB)

		tmpPath, err := downloadToTemp(pf.URL, pf.Filename, cfg.Token)
		if err != nil {
			// Clean up already-downloaded shards before returning.
			for _, f := range files {
				os.Remove(f.path)
			}
			return nil, fmt.Errorf("download shard %s: %w", pf.Filename, err)
		}
		files = append(files, sourceFile{path: tmpPath, format: "parquet", temp: true})
	}
	return files, nil
}

// downloadToTemp downloads url to an OS temp file and returns its path.
// The caller is responsible for deleting the file when done.
func downloadToTemp(url, filename, token string) (string, error) {
	req, _ := http.NewRequest("GET", url, nil)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	cli := &http.Client{Timeout: 30 * time.Minute}
	resp, err := cli.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	ext := filepath.Ext(filename)
	tmp, err := os.CreateTemp("", "ingest-*"+ext)
	if err != nil {
		return "", fmt.Errorf("create temp file: %w", err)
	}

	n, err := io.Copy(tmp, resp.Body)
	tmp.Close()
	if err != nil {
		os.Remove(tmp.Name())
		return "", fmt.Errorf("write temp file: %w", err)
	}

	log.Printf("  saved %.1f MB → %s", float64(n)/(1<<20), tmp.Name())
	return tmp.Name(), nil
}

// ── format detection ───────────────────────────────────────────────────────────

// detectFormat returns the format string for path.
// If explicit is not "auto" it is returned as-is; otherwise the extension is used.
func detectFormat(path, explicit string) string {
	if explicit != "auto" {
		return explicit
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".parquet":
		return "parquet"
	case ".jsonl", ".ndjson":
		return "jsonl"
	case ".json":
		return "json"
	case ".csv":
		return "csv"
	default:
		return "jsonl" // sensible fallback
	}
}
