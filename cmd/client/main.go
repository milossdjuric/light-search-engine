// search-client is a command-line interface for interacting with a running
// search-eval-platform server.
//
// Usage:
//
//	search-client [flags] <command> [args...]
//
// Commands:
//
//	index   <id> <text>    Index a single document
//	delete  <id>           Delete a document by ID
//	search  <query>        Search for documents
//	bulk    [file]         Bulk-index NDJSON from a file (or stdin if no file)
//	health                 Show server health
//	segments               List all segment metadata
//	flush                  Flush all in-memory shard buffers to disk
//	merge                  Trigger an async segment merge
//
// Flags:
//
//	-server    http://host:port  (default: http://localhost:8080)
//	-top-k     N                 Results to return for search (default: 10)
//	-retriever bm25|tfidf        Retriever to use for search (default: bm25)
//	-pretty                      Pretty-print JSON output
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// ── CLI flags ─────────────────────────────────────────────────────────────────

var (
	serverURL = flag.String("server", "http://localhost:8080", "search server URL")
	topK      = flag.Int("top-k", 10, "number of results to return")
	retriever = flag.String("retriever", "bm25", "retriever type: bm25|tfidf")
	pretty    = flag.Bool("pretty", false, "pretty-print JSON output")
)

func main() {
	flag.Usage = usage
	flag.Parse()

	if flag.NArg() == 0 {
		usage()
		os.Exit(1)
	}

	cmd := flag.Arg(0)
	args := flag.Args()[1:]

	cli := &client{base: strings.TrimRight(*serverURL, "/")}

	var err error
	switch cmd {
	case "index":
		err = cli.cmdIndex(args)
	case "delete":
		err = cli.cmdDelete(args)
	case "search":
		err = cli.cmdSearch(args)
	case "bulk":
		err = cli.cmdBulk(args)
	case "health":
		err = cli.cmdHealth()
	case "segments":
		err = cli.cmdSegments()
	case "flush":
		err = cli.cmdFlush()
	case "merge":
		err = cli.cmdMerge()
	case "reset":
		err = cli.cmdReset()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", cmd)
		usage()
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `search-client — CLI for search-eval-platform

USAGE
  search-client [flags] <command> [args...]

COMMANDS
  index   <id> <text>    Index a single document
  delete  <id>           Delete a document by ID
  search  <query>        Search for documents
  bulk    [file]         Bulk-index NDJSON from file (stdin if no file given)
  health                 Show server health
  segments               List segment metadata
  flush                  Flush in-memory buffers to disk
  merge                  Trigger async segment merge
  reset                  Wipe all indexed data (segments, WAL, catalog)

FLAGS`)
	flag.PrintDefaults()
	fmt.Fprintln(os.Stderr, `
EXAMPLES
  search-client index doc1 "the quick brown fox"
  search-client search "quick fox" -top-k 5 -retriever bm25
  search-client delete doc1
  search-client bulk docs.ndjson
  echo '{"id":"d1","text":"hello world"}' | search-client bulk
  search-client health
  search-client segments
  search-client flush
  search-client merge`)
}

// ── HTTP client ───────────────────────────────────────────────────────────────

type client struct {
	base string
	http http.Client
}

func init() {
	http.DefaultClient.Timeout = 30 * time.Second
}

func (c *client) get(path string) ([]byte, int, error) {
	resp, err := http.Get(c.base + path)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	return body, resp.StatusCode, err
}

func (c *client) post(path string, body []byte, contentType string) ([]byte, int, error) {
	resp, err := http.Post(c.base+path, contentType, bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	return b, resp.StatusCode, err
}

func (c *client) delete(path string) ([]byte, int, error) {
	req, err := http.NewRequest(http.MethodDelete, c.base+path, nil)
	if err != nil {
		return nil, 0, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	return body, resp.StatusCode, err
}

// printJSON outputs raw JSON, optionally pretty-printed.
func printJSON(raw []byte) {
	if *pretty {
		var buf bytes.Buffer
		if err := json.Indent(&buf, raw, "", "  "); err == nil {
			fmt.Println(buf.String())
			return
		}
	}
	fmt.Println(string(raw))
}

// checkStatus prints the response body and returns an error if status >= 400.
func checkStatus(body []byte, status int, context string) error {
	if status >= 400 {
		return fmt.Errorf("%s: server returned %d: %s", context, status, body)
	}
	return nil
}

// ── Commands ──────────────────────────────────────────────────────────────────

// cmdIndex indexes a single document: index <id> <text...>
func (c *client) cmdIndex(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: index <id> <text>")
	}
	id := args[0]
	text := strings.Join(args[1:], " ")

	payload := map[string]string{"id": id, "text": text}
	body, _ := json.Marshal(payload)

	resp, status, err := c.post("/index", body, "application/json")
	if err != nil {
		return err
	}
	if err := checkStatus(resp, status, "index"); err != nil {
		return err
	}
	printJSON(resp)
	return nil
}

// cmdDelete deletes a document by ID.
func (c *client) cmdDelete(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: delete <id>")
	}
	resp, status, err := c.delete("/index/" + url.PathEscape(args[0]))
	if err != nil {
		return err
	}
	if err := checkStatus(resp, status, "delete"); err != nil {
		return err
	}
	printJSON(resp)
	return nil
}

// cmdSearch searches documents and prints results in a readable table.
func (c *client) cmdSearch(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: search <query>")
	}
	query := strings.Join(args, " ")

	params := url.Values{
		"q":         {query},
		"top_k":     {fmt.Sprintf("%d", *topK)},
		"retriever": {*retriever},
	}
	body, status, err := c.get("/search?" + params.Encode())
	if err != nil {
		return err
	}
	if err := checkStatus(body, status, "search"); err != nil {
		return err
	}

	// Parse and display results
	var resp struct {
		Results []struct {
			DocID    string            `json:"doc_id"`
			Score    float64           `json:"score"`
			Rank     int               `json:"rank"`
			Snippet  string            `json:"snippet"`
			Metadata map[string]string `json:"metadata"`
		} `json:"results"`
		Total  int   `json:"total"`
		TookMs int64 `json:"took_ms"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		printJSON(body)
		return nil
	}

	fmt.Printf("Found %d result(s) in %dms\n\n", resp.Total, resp.TookMs)
	for _, r := range resp.Results {
		fmt.Printf("  #%d  %-20s  score=%.4f\n", r.Rank, r.DocID, r.Score)
		if r.Snippet != "" {
			fmt.Printf("       %s\n", r.Snippet)
		}
		if len(r.Metadata) > 0 {
			for k, v := range r.Metadata {
				fmt.Printf("       [%s=%s]\n", k, v)
			}
		}
		fmt.Println()
	}
	return nil
}

// cmdBulk bulk-indexes documents from a NDJSON file or stdin.
// Each line: {"id":"...","text":"..."}
func (c *client) cmdBulk(args []string) error {
	var r io.Reader
	if len(args) == 0 {
		r = os.Stdin
		fmt.Fprintln(os.Stderr, "reading NDJSON from stdin (Ctrl+D to finish)...")
	} else {
		f, err := os.Open(args[0])
		if err != nil {
			return fmt.Errorf("open file: %w", err)
		}
		defer f.Close()
		r = f
	}

	// Stream directly to the server.
	pr, pw := io.Pipe()
	go func() {
		scanner := bufio.NewScanner(r)
		for scanner.Scan() {
			line := bytes.TrimSpace(scanner.Bytes())
			if len(line) == 0 {
				continue
			}
			pw.Write(line)
			pw.Write([]byte("\n"))
		}
		pw.Close()
	}()

	resp, err := http.Post(c.base+"/index/bulk", "application/x-ndjson", pr)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if err := checkStatus(body, resp.StatusCode, "bulk"); err != nil {
		return err
	}
	printJSON(body)
	return nil
}

// cmdHealth prints server health.
func (c *client) cmdHealth() error {
	body, status, err := c.get("/health")
	if err != nil {
		return err
	}
	if err := checkStatus(body, status, "health"); err != nil {
		return err
	}
	printJSON(body)
	return nil
}

// cmdSegments prints segment metadata.
func (c *client) cmdSegments() error {
	body, status, err := c.get("/segments")
	if err != nil {
		return err
	}
	if err := checkStatus(body, status, "segments"); err != nil {
		return err
	}

	var segs []struct {
		ShardID   string `json:"shard_id"`
		SegmentID string `json:"segment_id"`
		Level     int    `json:"level"`
		DocCount  int    `json:"doc_count"`
		FlushSeq  int64  `json:"flush_seq"`
		Path      string `json:"path"`
	}
	if err := json.Unmarshal(body, &segs); err != nil {
		printJSON(body)
		return nil
	}

	if len(segs) == 0 {
		fmt.Println("No segments (all data in memory buffers)")
		return nil
	}

	fmt.Printf("%-10s  %-24s  %-6s  %-9s  %-9s\n",
		"SHARD", "SEGMENT", "LEVEL", "DOCS", "FLUSH_SEQ")
	fmt.Println(strings.Repeat("-", 65))
	for _, s := range segs {
		fmt.Printf("%-10s  %-24s  %-6d  %-9d  %-9d\n",
			s.ShardID, s.SegmentID, s.Level, s.DocCount, s.FlushSeq)
	}
	fmt.Printf("\nTotal: %d segment(s)\n", len(segs))
	return nil
}

// cmdFlush flushes all in-memory buffers to disk.
func (c *client) cmdFlush() error {
	body, status, err := c.post("/index/flush", nil, "application/json")
	if err != nil {
		return err
	}
	if err := checkStatus(body, status, "flush"); err != nil {
		return err
	}
	printJSON(body)
	return nil
}

// cmdMerge triggers an async segment merge.
func (c *client) cmdMerge() error {
	body, status, err := c.post("/index/merge", nil, "application/json")
	if err != nil {
		return err
	}
	if err := checkStatus(body, status, "merge"); err != nil {
		return err
	}
	printJSON(body)
	return nil
}

// cmdReset wipes all indexed data on the server.
func (c *client) cmdReset() error {
	body, status, err := c.post("/admin/reset", nil, "application/json")
	if err != nil {
		return err
	}
	if err := checkStatus(body, status, "reset"); err != nil {
		return err
	}
	printJSON(body)
	return nil
}
