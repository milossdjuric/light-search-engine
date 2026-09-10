package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/apache/arrow/go/v17/arrow"
	"github.com/apache/arrow/go/v17/arrow/array"
	"github.com/apache/arrow/go/v17/arrow/memory"
	"github.com/apache/arrow/go/v17/parquet/file"
	"github.com/apache/arrow/go/v17/parquet/pqarrow"
)

// readSourceFile dispatches to the format-specific reader.
// maxRecords <= 0 means read everything.
// Returns the number of records sent to out.
func readSourceFile(sf sourceFile, cfg *Config, out chan<- Record, maxRecords int) (int, error) {
	switch sf.format {
	case "parquet":
		return readParquet(sf.path, cfg, out, maxRecords)
	case "jsonl", "ndjson":
		return readJSONL(sf.path, cfg, out, maxRecords)
	case "json":
		return readJSON(sf.path, cfg, out, maxRecords)
	case "csv":
		return readCSV(sf.path, cfg, out, maxRecords)
	case "gen":
		return readGen(sf.path, cfg, out, maxRecords)
	default:
		return 0, fmt.Errorf("unknown format %q", sf.format)
	}
}

// ── Parquet ────────────────────────────────────────────────────────────────────

func readParquet(path string, cfg *Config, out chan<- Record, maxRecords int) (int, error) {
	pf, err := file.OpenParquetFile(path, false)
	if err != nil {
		return 0, fmt.Errorf("open parquet: %w", err)
	}
	defer pf.Close()

	// BatchSize must be non-zero; zero → NextBatch(0) reads 0 rows → immediate EOF.
	// Larger batches amortize per-batch decompression setup over more rows.
	batchSize := cfg.ParquetBatchSize
	if batchSize <= 0 {
		batchSize = 256 * 1024
	}
	arrowReader, err := pqarrow.NewFileReader(pf, pqarrow.ArrowReadProperties{BatchSize: batchSize}, memory.DefaultAllocator)
	if err != nil {
		return 0, fmt.Errorf("parquet arrow reader: %w", err)
	}

	// Read Arrow schema from parquet metadata — no row-group I/O.
	schema, err := arrowReader.Schema()
	if err != nil {
		return 0, fmt.Errorf("parquet schema: %w", err)
	}

	idIdx := fieldIndex(schema, cfg.IDField)
	textIdxs := resolveIndices(schema, cfg.TextFields)
	metaIdxs := resolveIndices(schema, cfg.MetaFields)
	if idIdx < 0 && len(textIdxs) == 0 {
		return 0, fmt.Errorf("parquet schema has neither id field %q nor text fields %v; available: %v",
			cfg.IDField, cfg.TextFields, schemaFieldNames(schema))
	}

	// GetRecordReader: single setup for all row groups, lazy streaming.
	// nil args → reads all columns (column projection left as future work once
	// Arrow's projection semantics are verified against the installed version).
	rr, err := arrowReader.GetRecordReader(context.Background(), nil, nil)
	if err != nil {
		return 0, fmt.Errorf("parquet record reader: %w", err)
	}
	defer rr.Release()

	count := 0
	for rr.Next() {
		rec := rr.Record()
		nrows := int(rec.NumRows())
		for row := 0; row < nrows; row++ {
			if maxRecords > 0 && count >= maxRecords {
				return count, nil
			}

			docID := ""
			if idIdx >= 0 {
				docID = colStr(rec.Column(idIdx), row)
			}
			if docID == "" {
				docID = strconv.Itoa(count)
			}

			var parts []string
			for _, ti := range textIdxs {
				if v := colStr(rec.Column(ti), row); v != "" {
					parts = append(parts, v)
				}
			}
			text := strings.Join(parts, " ")
			if text == "" {
				continue
			}

			var meta map[string]string
			if len(metaIdxs) > 0 {
				meta = make(map[string]string, len(metaIdxs))
				for j, mi := range metaIdxs {
					meta[cfg.MetaFields[j]] = colStr(rec.Column(mi), row)
				}
			}

			out <- Record{ID: docID, Text: text, Metadata: meta}
			count++
		}
	}
	if err := rr.Err(); err != nil && err != io.EOF {
		return count, err
	}
	return count, nil
}

// ── JSONL / NDJSON ─────────────────────────────────────────────────────────────

func readJSONL(path string, cfg *Config, out chan<- Record, maxRecords int) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4<<20), 4<<20) // 4 MB per line

	count := 0
	for scanner.Scan() {
		if maxRecords > 0 && count >= maxRecords {
			break
		}
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		rec, err := mapToRecord(line, cfg)
		if err != nil {
			continue // skip malformed lines
		}
		out <- rec
		count++
	}
	return count, scanner.Err()
}

// ── JSON (array or single object) ─────────────────────────────────────────────

func readJSON(path string, cfg *Config, out chan<- Record, maxRecords int) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	dec := json.NewDecoder(f)

	// Peek at the first token to decide array vs object.
	tok, err := dec.Token()
	if err != nil {
		return 0, fmt.Errorf("read JSON: %w", err)
	}

	count := 0
	if delim, ok := tok.(json.Delim); ok && delim == '[' {
		// JSON array: stream one object at a time.
		for dec.More() {
			if maxRecords > 0 && count >= maxRecords {
				break
			}
			var raw json.RawMessage
			if err := dec.Decode(&raw); err != nil {
				continue
			}
			rec, err := mapToRecord(raw, cfg)
			if err != nil {
				continue
			}
			out <- rec
			count++
		}
	} else {
		// Single JSON object — re-read by seeking back to start.
		f.Seek(0, io.SeekStart)
		var raw json.RawMessage
		if err := json.NewDecoder(f).Decode(&raw); err != nil {
			return 0, err
		}
		rec, err := mapToRecord(raw, cfg)
		if err != nil {
			return 0, err
		}
		out <- rec
		count++
	}
	return count, nil
}

// ── CSV ────────────────────────────────────────────────────────────────────────

func readCSV(path string, cfg *Config, out chan<- Record, maxRecords int) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.LazyQuotes = true
	r.TrimLeadingSpace = true

	header, err := r.Read()
	if err != nil {
		return 0, fmt.Errorf("read CSV header: %w", err)
	}

	// Build column name → index map.
	colIdx := make(map[string]int, len(header))
	for i, h := range header {
		colIdx[strings.TrimSpace(h)] = i
	}

	idCol, hasID := colIdx[cfg.IDField]
	textCols := make([]int, 0, len(cfg.TextFields))
	for _, f := range cfg.TextFields {
		if i, ok := colIdx[f]; ok {
			textCols = append(textCols, i)
		}
	}
	metaCols := make([]int, 0, len(cfg.MetaFields))
	for _, f := range cfg.MetaFields {
		if i, ok := colIdx[f]; ok {
			metaCols = append(metaCols, i)
		}
	}

	count := 0
	for {
		if maxRecords > 0 && count >= maxRecords {
			break
		}
		row, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue // skip bad rows
		}

		docID := strconv.Itoa(count)
		if hasID && idCol < len(row) {
			if v := strings.TrimSpace(row[idCol]); v != "" {
				docID = v
			}
		}

		var parts []string
		for _, ci := range textCols {
			if ci < len(row) {
				if v := strings.TrimSpace(row[ci]); v != "" {
					parts = append(parts, v)
				}
			}
		}
		text := strings.Join(parts, " ")
		if text == "" {
			continue
		}

		var meta map[string]string
		if len(metaCols) > 0 {
			meta = make(map[string]string, len(metaCols))
			for j, ci := range metaCols {
				if ci < len(row) {
					meta[cfg.MetaFields[j]] = row[ci]
				}
			}
		}

		out <- Record{ID: docID, Text: text, Metadata: meta}
		count++
	}
	return count, nil
}

// ── shared helpers ─────────────────────────────────────────────────────────────

// mapToRecord parses a raw JSON object into a Record using cfg's field mapping.
func mapToRecord(raw []byte, cfg *Config) (Record, error) {
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return Record{}, err
	}

	docID := ""
	if v, ok := m[cfg.IDField]; ok {
		docID = fmt.Sprintf("%v", v)
	}
	if docID == "" {
		return Record{}, fmt.Errorf("missing id field %q", cfg.IDField)
	}

	var parts []string
	for _, f := range cfg.TextFields {
		if v, ok := m[f]; ok {
			s := strings.TrimSpace(fmt.Sprintf("%v", v))
			if s != "" && s != "<nil>" {
				parts = append(parts, s)
			}
		}
	}
	text := strings.Join(parts, " ")
	if text == "" {
		return Record{}, fmt.Errorf("all text fields empty for id %q", docID)
	}

	var meta map[string]string
	if len(cfg.MetaFields) > 0 {
		meta = make(map[string]string, len(cfg.MetaFields))
		for _, f := range cfg.MetaFields {
			if v, ok := m[f]; ok {
				meta[f] = fmt.Sprintf("%v", v)
			}
		}
	}
	return Record{ID: docID, Text: text, Metadata: meta}, nil
}

// colStr extracts a string value from an Arrow array at the given row index.
// Handles the types commonly found in Parquet files: strings, dictionary-encoded
// strings, and numeric types (used when IDs are integers).
func colStr(col arrow.Array, row int) string {
	if col == nil || col.IsNull(row) {
		return ""
	}
	switch c := col.(type) {
	case *array.String:
		return c.Value(row)
	case *array.LargeString:
		return c.Value(row)
	case *array.Dictionary:
		// Dictionary-encoded column: look up the value in the dictionary.
		return colStr(c.Dictionary(), dictIndex(c.Indices(), row))
	case *array.Int8:
		return strconv.Itoa(int(c.Value(row)))
	case *array.Int16:
		return strconv.Itoa(int(c.Value(row)))
	case *array.Int32:
		return strconv.Itoa(int(c.Value(row)))
	case *array.Int64:
		return strconv.FormatInt(c.Value(row), 10)
	case *array.Uint8:
		return strconv.FormatUint(uint64(c.Value(row)), 10)
	case *array.Uint16:
		return strconv.FormatUint(uint64(c.Value(row)), 10)
	case *array.Uint32:
		return strconv.FormatUint(uint64(c.Value(row)), 10)
	case *array.Uint64:
		return strconv.FormatUint(c.Value(row), 10)
	case *array.Float32:
		return strconv.FormatFloat(float64(c.Value(row)), 'f', -1, 32)
	case *array.Float64:
		return strconv.FormatFloat(c.Value(row), 'f', -1, 64)
	case *array.Boolean:
		if c.Value(row) {
			return "true"
		}
		return "false"
	default:
		return ""
	}
}

// dictIndex returns the integer index stored in an Arrow indices array at position row.
// Dictionary indices are typically Int8/Int16/Int32 depending on cardinality.
func dictIndex(indices arrow.Array, row int) int {
	switch idx := indices.(type) {
	case *array.Int8:
		return int(idx.Value(row))
	case *array.Int16:
		return int(idx.Value(row))
	case *array.Int32:
		return int(idx.Value(row))
	case *array.Int64:
		return int(idx.Value(row))
	case *array.Uint8:
		return int(idx.Value(row))
	case *array.Uint16:
		return int(idx.Value(row))
	case *array.Uint32:
		return int(idx.Value(row))
	case *array.Uint64:
		return int(idx.Value(row))
	default:
		return 0
	}
}

// fieldIndex returns the first column index for fieldName in schema, or -1.
func fieldIndex(schema *arrow.Schema, fieldName string) int {
	idxs := schema.FieldIndices(fieldName)
	if len(idxs) == 0 {
		return -1
	}
	return idxs[0]
}

// resolveIndices maps a list of field names to their column indices in schema.
// Fields not present in the schema are silently skipped.
func resolveIndices(schema *arrow.Schema, fields []string) []int {
	out := make([]int, 0, len(fields))
	for _, f := range fields {
		if i := fieldIndex(schema, f); i >= 0 {
			out = append(out, i)
		}
	}
	return out
}

// schemaFieldNames returns all field names in the schema (for error messages).
func schemaFieldNames(schema *arrow.Schema) []string {
	names := make([]string, schema.NumFields())
	for i := 0; i < schema.NumFields(); i++ {
		names[i] = schema.Field(i).Name
	}
	return names
}

// ── Synthetic generator ────────────────────────────────────────────────────────

// genVocab is the word pool used by readGen.  ~300 words across tech, science,
// and general-English domains gives a realistic Zipf-distributed vocabulary.
var genVocab = []string{
	// Common function words (will be very frequent under Zipf)
	"the", "of", "and", "to", "a", "in", "is", "it", "for", "that",
	"was", "with", "are", "as", "at", "this", "but", "from", "or", "which",
	"one", "have", "not", "by", "had", "an", "be", "on", "its", "also",
	// Technology
	"data", "system", "network", "algorithm", "model", "search", "index",
	"query", "vector", "embedding", "neural", "learning", "deep", "machine",
	"retrieval", "ranking", "score", "document", "token", "language",
	"distributed", "cluster", "shard", "segment", "storage", "cache",
	"latency", "throughput", "precision", "recall", "accuracy",
	"database", "server", "client", "request", "response", "api",
	"function", "method", "class", "object", "interface", "protocol",
	"compression", "encoding", "binary", "format", "schema", "field",
	"graph", "tree", "heap", "queue", "stack", "hash", "table", "map",
	"memory", "disk", "thread", "process", "concurrent", "parallel",
	"lock", "mutex", "atomic", "transaction", "commit", "rollback",
	"inverted", "posting", "skiplist", "merge", "flush", "block",
	"segment", "level", "tiered", "policy", "buffer", "write", "read",
	// Science
	"research", "study", "analysis", "experiment", "theory", "evidence",
	"result", "effect", "factor", "feature", "property", "attribute",
	"value", "measure", "metric", "evaluation", "performance", "quality",
	"signal", "noise", "frequency", "probability", "distribution",
	"sample", "population", "variable", "gradient", "loss",
	"optimization", "convergence", "iteration", "epoch", "batch",
	"training", "testing", "validation", "prediction", "classification",
	"regression", "clustering", "similarity", "distance", "correlation",
	"sparse", "dense", "matrix", "tensor", "dimension", "projection",
	// General English nouns
	"time", "year", "people", "way", "day", "life", "world", "part",
	"place", "case", "week", "company", "group", "number", "fact",
	"information", "point", "state", "area", "level", "problem",
	"question", "program", "project", "example", "message", "service",
	"user", "task", "term", "word", "text", "content", "type", "list",
	"set", "key", "node", "edge", "path", "link", "name", "id",
	"version", "status", "error", "warning", "log", "event",
	// Verbs / adjectives
	"provide", "support", "include", "create", "build", "use", "make",
	"based", "given", "large", "small", "high", "low", "fast", "slow",
	"new", "old", "first", "last", "next", "current", "final",
	"single", "multiple", "different", "similar", "specific", "general",
	"important", "significant", "effective", "efficient", "optimal",
	"local", "global", "remote", "internal", "external", "public",
	"open", "close", "start", "stop", "enable", "disable", "update",
}

// readGen produces N synthetic documents.
//
//	-source gen://1000000          one million random documents
//	-source gen://1000000 -gen-seed 42   deterministic (reproducible)
//
// Words are sampled with a Zipf distribution (s=1.07) over genVocab, giving a
// realistic long-tail word frequency distribution.  Document length is uniform
// [20, 150] words.  IDs are zero-padded sequentials: "doc-0000000001".
func readGen(uri string, cfg *Config, out chan<- Record, maxRecords int) (int, error) {
	nStr := strings.TrimPrefix(uri, "gen://")
	n, err := strconv.Atoi(nStr)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("invalid gen:// source %q: expected gen://N where N > 0", uri)
	}
	if maxRecords > 0 && maxRecords < n {
		n = maxRecords
	}

	seed := cfg.GenSeed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))
	zipf := rand.NewZipf(rng, 1.07, 1.0, uint64(len(genVocab)-1))

	for i := 0; i < n; i++ {
		docLen := 20 + rng.Intn(131) // [20, 150]
		words := make([]string, docLen)
		for j := range words {
			words[j] = genVocab[zipf.Uint64()]
		}
		out <- Record{
			ID:   fmt.Sprintf("doc-%010d", i),
			Text: strings.Join(words, " "),
		}
	}
	return n, nil
}
