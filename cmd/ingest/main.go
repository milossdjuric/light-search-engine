// ingest is a multi-format dataset ingestion tool for search-eval-platform.
//
// It resolves a source (HuggingFace dataset or local file), reads records in
// the detected format (Parquet, JSONL, JSON, CSV), and streams them to the
// server's bulk index endpoint in configurable batches.
//
// Usage:
//
//	ingest -source <src> [flags]
//
// Sources:
//
//	hf://owner/repo              HuggingFace dataset (downloads parquet via datasets-server API)
//	gen://N                      Generate N synthetic documents with Zipf word distribution
//	/path/to/file.parquet        Local Parquet file
//	/path/to/file.jsonl          Local JSONL / NDJSON file
//	/path/to/file.json           Local JSON file (array or single object)
//	/path/to/file.csv            Local CSV file
//
// Examples:
//
//	ingest -source hf://BeIR/trec-covid
//	ingest -source hf://BeIR/scifact -config corpus -split corpus
//	ingest -source /data/corpus.jsonl -id-field id -text-fields title,body
//	ingest -source /data/articles.csv -id-field article_id -text-fields headline,body -meta-fields date,author
//	ingest -source gen://1000000
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Record is a document ready to be indexed.
type Record struct {
	ID       string            `json:"id"`
	Text     string            `json:"text"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Config holds all resolved ingest settings.
type Config struct {
	Source     string
	HFConfig   string
	HFSplit    string
	Format     string
	IDField    string
	TextFields []string
	MetaFields []string
	Server     string
	BatchSize  int
	Limit      int
	Token      string
	Workers    int
	Verbosity  int // 0=final only, 1=10s ticks, 2=per-batch
	// Gen source
	GenSeed int64
	// Parquet
	ParquetBatchSize int64 // rows per Arrow record batch (0 = default 256k)
}

// sourceFile is a resolved local file path plus its detected format.
type sourceFile struct {
	path   string
	format string
	temp   bool // delete after use (downloaded temp file)
}

// ── CLI flags ──────────────────────────────────────────────────────────────────

var (
	flagSource     = flag.String("source", "", "source: hf://owner/repo, gen://N, or local file path (required)")
	flagHFConfig   = flag.String("config", "corpus", "HuggingFace dataset config/subset name")
	flagHFSplit    = flag.String("split", "corpus", "HuggingFace dataset split name")
	flagFormat     = flag.String("format", "auto", "file format: auto|parquet|jsonl|json|csv")
	flagIDField    = flag.String("id-field", "_id", "field name for document ID")
	flagTextFields = flag.String("text-fields", "title,text", "comma-separated fields joined as document text")
	flagMetaFields = flag.String("meta-fields", "", "comma-separated fields to store as metadata")
	flagServer     = flag.String("server", "http://localhost:8080", "search server base URL")
	flagBatch      = flag.Int("batch", 10000, "documents per bulk POST request")
	flagLimit      = flag.Int("limit", 0, "max documents to ingest (0 = all)")
	flagToken      = flag.String("token", "", "HuggingFace API token (for private/gated datasets)")
	flagWorkers    = flag.Int("workers", 2, "parallel HTTP workers for bulk POSTs")
	flagVerbose     = flag.Bool("v", false, "10-second progress ticks (verbosity 1)")
	flagVeryVerbose = flag.Bool("vv", false, "per-batch stats + 10s ticks (verbosity 2)")
	flagGenSeed         = flag.Int64("gen-seed", 0, "random seed for gen:// sources (0 = time-based)")
	flagParquetBatch    = flag.Int64("parquet-batch", 256*1024, "Arrow rows per batch when reading Parquet (larger = fewer decompression setup calls)")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: ingest -source <src> [flags]\n\n")
		fmt.Fprintf(os.Stderr, "Sources:\n")
		fmt.Fprintf(os.Stderr, "  hf://owner/repo               HuggingFace dataset (parquet via datasets-server)\n")
		fmt.Fprintf(os.Stderr, "  gen://N                       Generate N synthetic documents with Zipf word distribution\n")
		fmt.Fprintf(os.Stderr, "  /path/to/file.parquet         Local Parquet file\n")
		fmt.Fprintf(os.Stderr, "  /path/to/file.jsonl           Local JSONL/NDJSON\n")
		fmt.Fprintf(os.Stderr, "  /path/to/file.json            Local JSON array or object\n")
		fmt.Fprintf(os.Stderr, "  /path/to/file.csv             Local CSV with header row\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	if *flagSource == "" {
		flag.Usage()
		os.Exit(1)
	}

	cfg := &Config{
		Source:     *flagSource,
		HFConfig:   *flagHFConfig,
		HFSplit:    *flagHFSplit,
		Format:     *flagFormat,
		IDField:    *flagIDField,
		TextFields: splitTrimmed(*flagTextFields),
		MetaFields: splitTrimmed(*flagMetaFields),
		Server:     strings.TrimRight(*flagServer, "/"),
		BatchSize:  *flagBatch,
		Limit:      *flagLimit,
		Token:      *flagToken,
		Workers:    *flagWorkers,
		Verbosity: func() int {
			if *flagVeryVerbose {
				return 2
			}
			if *flagVerbose {
				return 1
			}
			return 0
		}(),
		GenSeed:          *flagGenSeed,
		ParquetBatchSize: *flagParquetBatch,
	}

	if err := run(cfg); err != nil {
		log.Fatalf("ingest: %v", err)
	}
}

// run executes the full ingest pipeline.
func run(cfg *Config) error {
	totalStart := time.Now()

	// 1. Resolve source → local files (download if hf://).
	fetchStart := time.Now()
	files, err := resolveSource(cfg)
	if err != nil {
		return fmt.Errorf("resolve source: %w", err)
	}
	defer cleanupTempFiles(files)
	fetchElapsed := time.Since(fetchStart)
	log.Printf("fetch complete in %s", fetchElapsed.Round(time.Millisecond))

	ingestStart := time.Now()

	// 2. Channels.
	records := make(chan Record, cfg.BatchSize*4)
	batches := make(chan []Record, cfg.Workers*2)

	// 3. Reader goroutine: reads all source files, sends Records.
	var readErr error
	var wgRead sync.WaitGroup
	wgRead.Add(1)
	go func() {
		defer wgRead.Done()
		defer close(records)
		total := 0
		for _, sf := range files {
			remaining := 0
			if cfg.Limit > 0 {
				if total >= cfg.Limit {
					break
				}
				remaining = cfg.Limit - total
			}
			n, err := readSourceFile(sf, cfg, records, remaining)
			if err != nil {
				readErr = fmt.Errorf("read %s: %w", sf.path, err)
				return
			}
			total += n
		}
	}()

	// 4. Batcher goroutine: groups Records into fixed-size batches.
	var wgWork sync.WaitGroup
	wgWork.Add(1)
	go func() {
		defer wgWork.Done()
		defer close(batches)
		batch := make([]Record, 0, cfg.BatchSize)
		for rec := range records {
			batch = append(batch, rec)
			if len(batch) >= cfg.BatchSize {
				batches <- batch
				batch = make([]Record, 0, cfg.BatchSize)
			}
		}
		if len(batch) > 0 {
			batches <- batch
		}
	}()

	// 5. Worker goroutines: POST batches to /index/bulk.
	var indexed, failed, totalBytes int64
	for i := 0; i < cfg.Workers; i++ {
		wgWork.Add(1)
		go func() {
			defer wgWork.Done()
			cli := &http.Client{Timeout: 120 * time.Second}
			for batch := range batches {
				ok, fail, nb := postBatch(cli, cfg.Server, batch)
				atomic.AddInt64(&indexed, int64(ok))
				atomic.AddInt64(&failed, int64(fail))
				atomic.AddInt64(&totalBytes, int64(nb))
				if cfg.Verbosity >= 2 {
					secs := time.Since(ingestStart).Seconds()
					docs := atomic.LoadInt64(&indexed)
					mb := float64(atomic.LoadInt64(&totalBytes)) / (1 << 20)
					log.Printf("indexed %d | failed %d | %.0f docs/s | %.1f MB/s | elapsed %s",
						docs,
						atomic.LoadInt64(&failed),
						float64(docs)/secs,
						mb/secs,
						time.Since(ingestStart).Round(time.Second))
				}
			}
		}()
	}

	// 6. Progress ticker.
	progressDone := make(chan struct{})
	go func() {
		if cfg.Verbosity < 1 {
			<-progressDone
			return
		}
		tick := time.NewTicker(10 * time.Second)
		defer tick.Stop()
		for {
			select {
			case <-tick.C:
				secs := time.Since(ingestStart).Seconds()
				docs := atomic.LoadInt64(&indexed)
				mb := float64(atomic.LoadInt64(&totalBytes)) / (1 << 20)
				log.Printf("progress: %d indexed, %d failed | %.0f docs/s | %.1f MB/s | ingest %s",
					docs,
					atomic.LoadInt64(&failed),
					float64(docs)/secs,
					mb/secs,
					time.Since(ingestStart).Round(time.Second))
			case <-progressDone:
				return
			}
		}
	}()

	// 7. Wait for reader then workers, then stop progress ticker.
	wgRead.Wait()
	wgWork.Wait()
	close(progressDone)

	if readErr != nil {
		return readErr
	}

	ingestElapsed := time.Since(ingestStart)
	docs := atomic.LoadInt64(&indexed)
	mb := float64(atomic.LoadInt64(&totalBytes)) / (1 << 20)
	secs := ingestElapsed.Seconds()
	log.Printf("done: %d indexed, %d failed | %.0f docs/s | %.1f MB/s | %.1f MB total | fetch %s | ingest %s | total %s",
		docs,
		atomic.LoadInt64(&failed),
		float64(docs)/secs,
		mb/secs,
		mb,
		fetchElapsed.Round(time.Second),
		ingestElapsed.Round(time.Second),
		time.Since(totalStart).Round(time.Second))

	// Flush the server's in-memory buffer so the last partial chunk is persisted
	// to disk immediately — without this, docs below the flush threshold would
	// only be durably flushed on the next size-triggered flush or server shutdown.
	flushResp, err := http.Post(cfg.Server+"/index/flush", "application/json", nil)
	if err != nil {
		log.Printf("warning: flush after ingest failed: %v", err)
	} else {
		flushResp.Body.Close()
		log.Printf("post-ingest flush complete")
	}

	// GC: recycle the builder temp files in the server's buffer pool.
	// Temp files grow proportionally to flush size during ingest; after ingest
	// is done, releasing the OS disk blocks keeps idle disk usage near zero.
	gcResp, err := http.Post(cfg.Server+"/admin/gc", "application/json", nil)
	if err != nil {
		log.Printf("warning: post-ingest GC failed: %v", err)
	} else {
		gcResp.Body.Close()
		log.Printf("post-ingest GC complete")
	}

	return nil
}

// postBatch sends one batch to POST /index/bulk.
// Returns (indexed, failed, bytesSent) — bytesSent is the NDJSON payload size.
func postBatch(cli *http.Client, server string, batch []Record) (int, int, int) {
	var buf bytes.Buffer
	for _, rec := range batch {
		line, _ := json.Marshal(rec)
		buf.Write(line)
		buf.WriteByte('\n')
	}
	bytesSent := buf.Len()

	req, err := http.NewRequest("POST", server+"/index/bulk", &buf)
	if err != nil {
		return 0, len(batch), bytesSent
	}
	req.Header.Set("Content-Type", "application/x-ndjson")

	resp, err := cli.Do(req)
	if err != nil {
		log.Printf("post batch error: %v", err)
		return 0, len(batch), bytesSent
	}
	defer resp.Body.Close()

	var result struct {
		Indexed int `json:"indexed"`
		Failed  int `json:"failed"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, len(batch), bytesSent
	}
	return result.Indexed, result.Failed, bytesSent
}

// ── helpers ────────────────────────────────────────────────────────────────────

func splitTrimmed(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func cleanupTempFiles(files []sourceFile) {
	for _, f := range files {
		if f.temp {
			os.Remove(f.path)
		}
	}
}
