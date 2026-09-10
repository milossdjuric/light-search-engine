package search

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	lz4 "github.com/pierrec/lz4/v4"
)

// WALEntry is one line in the NDJSON write-ahead log.
// Crc holds a CRC32 (IEEE) checksum computed over all fields except Crc itself.
// A zero value means the record predates checksum support and is accepted as-is.
type WALEntry struct {
	Seq      int64             `json:"seq"`
	Op       string            `json:"op"`
	DocID    string            `json:"doc_id"`
	Text     string            `json:"text,omitempty"`
	Fields   map[string]string `json:"fields,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
	TS       time.Time         `json:"ts"`
	Crc      uint32            `json:"crc,omitempty"`
}

const (
	walBinaryMagic      = "SWAL"
	walBinaryVersion    = byte(1) // uncompressed binary WAL
	walBinaryVersionLZ4 = byte(2) // LZ4-compressed binary WAL
	walOpIndex          = byte(1)
	walOpDelete         = byte(2)
)

// appendWAL assigns a sequence number, encodes entry as a binary WAL record, and
// appends it to the current WAL file. Thread-safe via walMu.
//
// Binary record layout: [payloadLen uint32 LE][crc32 uint32 LE][payload bytes]
// Payload: seq int64, tsNano int64, op byte, docIDLen uint16, docID,
//
//	(index only) textLen uint32, text, fieldsCount uint8, fields...,
//	metaCount uint8, metadata...
//
// nextSeq is protected by walMu alone (NOT by sm.mu). This is critical to avoid
// a lock-order inversion deadlock: flushLocked is called under sm.mu and acquires
// walMu (via rotateWALLocked), so appendWAL must never hold walMu while also
// acquiring sm.mu.
func (sm *SegmentManager) appendWAL(entry *WALEntry) error {
	sm.walMu.Lock()
	defer sm.walMu.Unlock()

	entry.Seq = sm.nextSeq
	sm.nextSeq++

	// Encode payload into scratch buffer (single pass — no double marshal).
	sm.walScratch = sm.walScratch[:0]
	sm.walScratch = walAppendEntry(sm.walScratch, entry)

	// Optionally compress the payload with LZ4 before writing.
	payload := sm.walScratch
	if sm.walCompress {
		need := lz4.CompressBlockBound(len(sm.walScratch))
		if cap(sm.walCompressBuf) < need {
			sm.walCompressBuf = make([]byte, need)
		}
		n, cerr := lz4.CompressBlock(sm.walScratch, sm.walCompressBuf[:need], nil)
		if cerr == nil && n > 0 {
			payload = sm.walCompressBuf[:n]
		}
		// If compression fails or expands, fall back to uncompressed.
	}

	// Write [payloadLen uint32 LE][crc32 uint32 LE][payload].
	if err := walWriteRecord(sm.walBuf, payload); err != nil {
		return fmt.Errorf("appendWAL: %w", err)
	}

	if sm.walDurability == "sync" {
		if err := sm.walBuf.Flush(); err != nil {
			return fmt.Errorf("appendWAL flush: %w", err)
		}
		return sm.walFile.Sync()
	}
	return nil
}

// rotateWALLocked seals the current WAL file, opens the next numbered file,
// and deletes the sealed file only after the new one is open (no crash window).
// All entries up to and including sealedSeq are guaranteed to be in a
// registered segment — safe to discard.
// Must be called with mu held (acquires walMu internally).
func (sm *SegmentManager) rotateWALLocked(_ int64) error {
	sm.walMu.Lock()
	defer sm.walMu.Unlock()

	// Flush any buffered bytes to the current WAL file before closing it.
	if err := sm.walBuf.Flush(); err != nil {
		return fmt.Errorf("rotateWAL flush: %w", err)
	}

	oldFile := sm.walFile
	oldPath := oldFile.Name()

	// Open the next WAL file BEFORE closing the old one — always have a valid
	// writable file, even if Remove() below fails.
	sm.walSeqNum++
	newPath := walFileName(sm.walDir, sm.shardID, sm.walSeqNum)
	f, err := os.OpenFile(newPath, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		sm.walSeqNum-- // roll back — old file is still valid
		return fmt.Errorf("rotateWAL open new file: %w", err)
	}
	sm.walFile = f
	sm.walBuf = bufio.NewWriterSize(f, 256*1024)
	// Write binary WAL file header to the new file.
	_, _ = sm.walBuf.WriteString(walBinaryMagic)
	if sm.walCompress {
		_ = sm.walBuf.WriteByte(walBinaryVersionLZ4)
	} else {
		_ = sm.walBuf.WriteByte(walBinaryVersion)
	}

	// Close and remove the sealed file. If Remove fails (e.g. permissions),
	// the file is harmless: on next startup it will be replayed and skipped
	// because all its entries have seq <= maxFlushSeq (already in a segment).
	oldFile.Close()
	if err := os.Remove(oldPath); err != nil {
		// Non-fatal — log and continue.  The orphan will be cleaned up on
		// the next startup by gcSealedWALFiles.
		_ = err
	}
	return nil
}

// runWALFlusher periodically flushes the WAL buffer to the OS in "async" durability mode.
// This bounds data loss on crash to at most one sync interval.
func (sm *SegmentManager) runWALFlusher() {
	defer sm.wg.Done()
	tick := time.NewTicker(time.Duration(sm.walSyncIntervalMs) * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-tick.C:
			sm.walMu.Lock()
			sm.walBuf.Flush() //nolint:errcheck
			sm.walMu.Unlock()
		case <-sm.stopCh:
			return
		}
	}
}

// runIdleFlusher flushes the in-memory buffer when no new documents have been
// indexed for idleFlushSecs seconds. This ensures the last chunk of a bulk
// ingest run is persisted to disk even if it never hits the size threshold.
func (sm *SegmentManager) runIdleFlusher() {
	defer sm.wg.Done()
	interval := time.Duration(sm.idleFlushSecs) * time.Second
	tick := time.NewTicker(interval / 2) // check twice per interval for responsiveness
	defer tick.Stop()
	for {
		select {
		case <-sm.stopCh:
			return
		case <-tick.C:
			last := sm.lastWriteTime.Load()
			if last == 0 {
				continue // no writes yet
			}
			if time.Since(time.Unix(0, last)) < interval {
				continue // still within idle window
			}
			// Buffer may be non-empty and stale — flush it.
			sm.mu.RLock()
			empty := sm.bufferDocs == 0
			sm.mu.RUnlock()
			if empty {
				continue
			}
			if err := sm.Flush(); err != nil {
				slog.Error("idle flush failed", "shard", sm.shardID, "err", err)
			} else {
				slog.Info("idle flush completed", "shard", sm.shardID)
				sm.lastWriteTime.Store(0) // reset so we don't re-flush empty buffer
			}
		}
	}
}

// replayWALLocked reads every WAL file for this shard in sequence-number order,
// applies entries with seq > afterSeq, verifies CRC32 on every record, and
// deletes files whose highest seq is <= afterSeq (fully covered by a segment).
// Must be called with mu held.
func (sm *SegmentManager) replayWALLocked(afterSeq int64) error {
	sm.walMu.Lock()
	defer sm.walMu.Unlock()

	// Flush current write buffer before scanning.
	if err := sm.walBuf.Flush(); err != nil {
		return fmt.Errorf("replayWAL flush: %w", err)
	}

	walPaths, err := listWALFiles(sm.walDir, sm.shardID)
	if err != nil {
		return fmt.Errorf("replayWAL list: %w", err)
	}

	var maxSeen int64

	for _, path := range walPaths {
		fileMaxSeq, replayed, gcErr := sm.replayOneWALFile(path, afterSeq)
		if gcErr != nil {
			return gcErr
		}
		if replayed > 0 {
			sm.bufferDirty = true
		}
		if fileMaxSeq > maxSeen {
			maxSeen = fileMaxSeq
		}

		// GC: if every entry in this file is covered by a segment, delete it —
		// unless it is the current active file (we still write to it).
		activeWALPath := walFileName(sm.walDir, sm.shardID, sm.walSeqNum)
		if fileMaxSeq > 0 && fileMaxSeq <= afterSeq && path != activeWALPath {
			os.Remove(path) //nolint:errcheck — best-effort GC
		}
	}

	if maxSeen >= sm.nextSeq {
		sm.nextSeq = maxSeen + 1
	}
	return nil
}

// replayOneWALFile reads a single WAL file, applying entries with seq > afterSeq.
// Detects binary (SWAL\x01 header) vs legacy JSON format automatically.
// Returns (maxSeqInFile, numReplayed, error).
func (sm *SegmentManager) replayOneWALFile(path string, afterSeq int64) (maxSeq int64, replayed int, err error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, 0, nil
		}
		return 0, 0, fmt.Errorf("replayWAL open %s: %w", path, err)
	}
	defer f.Close()

	// Peek at first 5 bytes to detect format.
	var peek [5]byte
	n, _ := io.ReadFull(f, peek[:])
	if n >= 5 && string(peek[:4]) == walBinaryMagic {
		switch peek[4] {
		case walBinaryVersion:
			// Uncompressed binary — file position is already past the 5-byte header.
			return sm.replayBinaryWALFile(bufio.NewReaderSize(f, 1<<20), path, afterSeq, false)
		case walBinaryVersionLZ4:
			// LZ4-compressed binary — each record payload is LZ4-compressed.
			return sm.replayBinaryWALFile(bufio.NewReaderSize(f, 1<<20), path, afterSeq, true)
		}
	}

	// Legacy JSON format — seek back to the beginning.
	if _, serr := f.Seek(0, io.SeekStart); serr != nil {
		return 0, 0, fmt.Errorf("replayWAL seek %s: %w", path, serr)
	}
	return sm.replayJSONWALFile(f, path, afterSeq)
}

// replayBinaryWALFile reads WAL records from a binary-format reader.
// The file header has already been consumed by the caller.
// If decompress is true, each record payload is LZ4-decompressed before decode.
func (sm *SegmentManager) replayBinaryWALFile(r io.Reader, _ string, afterSeq int64, decompress bool) (maxSeq int64, replayed int, err error) {
	var hdr [8]byte
	var decompBuf []byte // reused across records for LZ4 decompression
	for {
		if _, rerr := io.ReadFull(r, hdr[:]); rerr != nil {
			break // EOF or truncated tail — normal end of file
		}
		payloadLen := binary.LittleEndian.Uint32(hdr[0:4])
		storedCRC := binary.LittleEndian.Uint32(hdr[4:8])

		payload := make([]byte, payloadLen)
		if _, rerr := io.ReadFull(r, payload); rerr != nil {
			break // truncated record at EOF is OK (crash mid-write)
		}
		if crc32.ChecksumIEEE(payload) != storedCRC {
			continue // corrupted record — skip (matches JSON replay behaviour)
		}

		// Decompress if the file was written with LZ4 compression.
		if decompress {
			// LZ4 block decompression: we don't know the original size upfront,
			// so we start with 4× and grow until it fits.
			for size := len(payload) * 4; ; size *= 2 {
				if len(decompBuf) < size {
					decompBuf = make([]byte, size)
				}
				n, derr := lz4.UncompressBlock(payload, decompBuf)
				if derr == nil {
					payload = decompBuf[:n]
					break
				}
				if size > 64<<20 { // 64 MB ceiling — something is wrong
					payload = nil
					break
				}
			}
			if payload == nil {
				continue // decompression failed — skip corrupted record
			}
		}

		e, derr := walDecodeEntry(payload)
		if derr != nil {
			continue // malformed record — skip
		}
		if e.Seq > maxSeq {
			maxSeq = e.Seq
		}
		if e.Seq <= afterSeq {
			continue
		}

		switch e.Op {
		case opIndex:
			if _, wasTombstoned := sm.tombstones[e.DocID]; wasTombstoned {
				delete(sm.tombstones, e.DocID)
				_ = sm.meta.RemoveTombstone(sm.shardID, e.DocID)
			}
			if len(sm.bm25fFields) > 0 && len(e.Fields) > 0 {
				fieldTokens := make(map[string][]string, len(e.Fields))
				for fname, ftext := range e.Fields {
					fieldTokens[fname] = sm.tokenizer.Tokenize(ftext)
				}
				sm.buffer.AddFields(e.DocID, fieldTokens)
			} else {
				sm.buffer.Add(e.DocID, sm.tokenizer.Tokenize(e.Text))
			}
			if !sm.skipStoredFields {
				sm.bufferTexts[e.DocID] = e.Text
			}
			sm.bufferDocs++
			replayed++
		case opDelete:
			sm.tombstones[e.DocID] = struct{}{}
			replayed++
		}
	}
	return maxSeq, replayed, nil
}

// replayJSONWALFile reads WAL records from a legacy NDJSON-format reader.
func (sm *SegmentManager) replayJSONWALFile(r io.Reader, path string, afterSeq int64) (maxSeq int64, replayed int, err error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 4<<20), 4<<20)

	for scanner.Scan() {
		raw := scanner.Bytes()
		var e WALEntry
		if jerr := json.Unmarshal(raw, &e); jerr != nil {
			continue
		}

		// CRC verification: re-marshal with Crc=0 and compare.
		if e.Crc != 0 {
			stored := e.Crc
			e.Crc = 0
			check, merr := json.Marshal(e)
			if merr == nil && crc32.ChecksumIEEE(check) != stored {
				continue
			}
			e.Crc = stored
		}

		if e.Seq > maxSeq {
			maxSeq = e.Seq
		}
		if e.Seq <= afterSeq {
			continue
		}

		switch e.Op {
		case opIndex:
			if _, wasTombstoned := sm.tombstones[e.DocID]; wasTombstoned {
				delete(sm.tombstones, e.DocID)
				_ = sm.meta.RemoveTombstone(sm.shardID, e.DocID)
			}
			if len(sm.bm25fFields) > 0 && len(e.Fields) > 0 {
				fieldTokens := make(map[string][]string, len(e.Fields))
				for fname, ftext := range e.Fields {
					fieldTokens[fname] = sm.tokenizer.Tokenize(ftext)
				}
				sm.buffer.AddFields(e.DocID, fieldTokens)
			} else {
				sm.buffer.Add(e.DocID, sm.tokenizer.Tokenize(e.Text))
			}
			if !sm.skipStoredFields {
				sm.bufferTexts[e.DocID] = e.Text
			}
			sm.bufferDocs++
			replayed++
		case opDelete:
			sm.tombstones[e.DocID] = struct{}{}
			replayed++
		}
	}
	if serr := scanner.Err(); serr != nil {
		return maxSeq, replayed, fmt.Errorf("replayWAL scan %s: %w", path, serr)
	}
	return maxSeq, replayed, nil
}

// Binary WAL encoding / decoding
//
// Record layout: [payloadLen uint32 LE][crc32 uint32 LE][payload bytes]
//
// Payload layout (index op):
//
//	seq int64 LE | tsNano int64 LE | op byte(1)
//	docIDLen uint16 LE | docID bytes
//	textLen uint32 LE  | text bytes
//	fieldsCount uint8
//	  for each field: nameLen uint8 | name | valueLen uint32 LE | value
//	metaCount uint8
//	  for each meta:  keyLen uint8  | key  | valueLen uint16 LE | value
//
// Payload layout (delete op):
//
//	seq int64 LE | tsNano int64 LE | op byte(2)
//	docIDLen uint16 LE | docID bytes

// walWriteRecord writes [payloadLen][crc][payload] to w.
func walWriteRecord(w *bufio.Writer, payload []byte) error {
	var hdr [8]byte
	binary.LittleEndian.PutUint32(hdr[0:4], uint32(len(payload)))
	binary.LittleEndian.PutUint32(hdr[4:8], crc32.ChecksumIEEE(payload))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

// walAppendEntry serialises entry into b and returns the grown slice.
func walAppendEntry(b []byte, e *WALEntry) []byte {
	b = walAppendInt64(b, e.Seq)
	b = walAppendInt64(b, e.TS.UnixNano())
	if e.Op == opIndex {
		b = append(b, walOpIndex)
	} else {
		b = append(b, walOpDelete)
	}
	b = walAppendStr16(b, e.DocID)
	if e.Op == opIndex {
		b = walAppendStr32(b, e.Text)
		b = append(b, byte(len(e.Fields)))
		for k, v := range e.Fields {
			b = walAppendStr8(b, k)
			b = walAppendStr32(b, v)
		}
		b = append(b, byte(len(e.Metadata)))
		for k, v := range e.Metadata {
			b = walAppendStr8(b, k)
			b = walAppendStr16(b, v)
		}
	}
	return b
}

// walDecodeEntry decodes a binary payload into a WALEntry.
func walDecodeEntry(b []byte) (*WALEntry, error) {
	if len(b) < 17 { // 8+8+1 minimum
		return nil, fmt.Errorf("WAL binary record too short (%d bytes)", len(b))
	}
	seq := int64(binary.LittleEndian.Uint64(b[0:8]))
	tsNano := int64(binary.LittleEndian.Uint64(b[8:16]))
	opByte := b[16]
	b = b[17:]

	e := &WALEntry{Seq: seq, TS: time.Unix(0, tsNano).UTC()}

	var err error
	e.DocID, b, err = walReadStr16(b)
	if err != nil {
		return nil, fmt.Errorf("WAL binary decode docID: %w", err)
	}

	switch opByte {
	case walOpIndex:
		e.Op = opIndex
		e.Text, b, err = walReadStr32(b)
		if err != nil {
			return nil, fmt.Errorf("WAL binary decode text: %w", err)
		}
		if len(b) < 1 {
			return nil, fmt.Errorf("WAL binary decode: missing fields count")
		}
		fc := int(b[0])
		b = b[1:]
		if fc > 0 {
			e.Fields = make(map[string]string, fc)
			for i := 0; i < fc; i++ {
				var k, v string
				if k, b, err = walReadStr8(b); err != nil {
					return nil, err
				}
				if v, b, err = walReadStr32(b); err != nil {
					return nil, err
				}
				e.Fields[k] = v
			}
		}
		if len(b) < 1 {
			return nil, fmt.Errorf("WAL binary decode: missing meta count")
		}
		mc := int(b[0])
		b = b[1:]
		if mc > 0 {
			e.Metadata = make(map[string]string, mc)
			for i := 0; i < mc; i++ {
				var k, v string
				if k, b, err = walReadStr8(b); err != nil {
					return nil, err
				}
				if v, b, err = walReadStr16(b); err != nil {
					return nil, err
				}
				e.Metadata[k] = v
			}
		}
	case walOpDelete:
		e.Op = opDelete
	default:
		return nil, fmt.Errorf("WAL binary decode: unknown op byte %d", opByte)
	}
	return e, nil
}

func walAppendInt64(b []byte, v int64) []byte {
	var tmp [8]byte
	binary.LittleEndian.PutUint64(tmp[:], uint64(v))
	return append(b, tmp[:]...)
}

func walAppendStr8(b []byte, s string) []byte {
	return append(append(b, byte(len(s))), s...)
}

func walAppendStr16(b []byte, s string) []byte {
	var tmp [2]byte
	binary.LittleEndian.PutUint16(tmp[:], uint16(len(s)))
	return append(append(b, tmp[:]...), s...)
}

func walAppendStr32(b []byte, s string) []byte {
	var tmp [4]byte
	binary.LittleEndian.PutUint32(tmp[:], uint32(len(s)))
	return append(append(b, tmp[:]...), s...)
}

func walReadStr8(b []byte) (string, []byte, error) {
	if len(b) < 1 {
		return "", b, fmt.Errorf("WAL binary: short buffer (str8 len)")
	}
	n := int(b[0])
	b = b[1:]
	if len(b) < n {
		return "", b, fmt.Errorf("WAL binary: short buffer (str8 data, need %d have %d)", n, len(b))
	}
	return string(b[:n]), b[n:], nil
}

func walReadStr16(b []byte) (string, []byte, error) {
	if len(b) < 2 {
		return "", b, fmt.Errorf("WAL binary: short buffer (str16 len)")
	}
	n := int(binary.LittleEndian.Uint16(b[0:2]))
	b = b[2:]
	if len(b) < n {
		return "", b, fmt.Errorf("WAL binary: short buffer (str16 data, need %d have %d)", n, len(b))
	}
	return string(b[:n]), b[n:], nil
}

func walReadStr32(b []byte) (string, []byte, error) {
	if len(b) < 4 {
		return "", b, fmt.Errorf("WAL binary: short buffer (str32 len)")
	}
	n := int(binary.LittleEndian.Uint32(b[0:4]))
	b = b[4:]
	if len(b) < n {
		return "", b, fmt.Errorf("WAL binary: short buffer (str32 data, need %d have %d)", n, len(b))
	}
	return string(b[:n]), b[n:], nil
}

// walFileName returns the path for the Nth WAL file of a shard.
// Format: <walDir>/<shardID>_<NNNN>.wal  (zero-padded to 4 digits).
func walFileName(walDir, shardID string, seqNum int) string {
	return filepath.Join(walDir, fmt.Sprintf("%s_%04d.wal", shardID, seqNum))
}

// listWALFiles returns all WAL files for shardID in ascending sequence-number
// order.  Files must match the pattern "<shardID>_NNNN.wal".
func listWALFiles(walDir, shardID string) ([]string, error) {
	pattern := filepath.Join(walDir, shardID+"_*.wal")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}
	// Sort by the numeric suffix so replay happens in write order.
	sort.Slice(matches, func(i, j int) bool {
		return walSeqFromPath(matches[i]) < walSeqFromPath(matches[j])
	})
	return matches, nil
}

// findHighestWALSeqNum returns the highest sequence number among existing WAL
// files for shardID, or 0 if none exist.
func findHighestWALSeqNum(walDir, shardID string) (int, error) {
	paths, err := listWALFiles(walDir, shardID)
	if err != nil {
		return 0, err
	}
	max := 0
	for _, p := range paths {
		if n := walSeqFromPath(p); n > max {
			max = n
		}
	}
	return max, nil
}

// walSeqFromPath extracts the numeric suffix from a WAL filename.
// Returns 0 if the name does not match the expected pattern.
func walSeqFromPath(path string) int {
	base := filepath.Base(path)
	// base looks like "shard0_0003.wal"
	base = strings.TrimSuffix(base, ".wal")
	idx := strings.LastIndex(base, "_")
	if idx < 0 {
		return 0
	}
	n, err := strconv.Atoi(base[idx+1:])
	if err != nil {
		return 0
	}
	return n
}
