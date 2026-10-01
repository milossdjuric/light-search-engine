package replication

import (
	"encoding/json"
	"log/slog"
	"os"
	"time"
)

const (
	// maxApplyAttempts is how many times one WAL entry may fail to apply
	// (one attempt per reconnect) before it is dead-lettered and replication
	// moves past it. Without a limit a single poisoned entry would stall the
	// replica forever.
	maxApplyAttempts = 5

	// maxDeadLettersInMemory bounds DeadLetters(); the dead-letter file and
	// DeadLetterCount keep the complete history.
	maxDeadLettersInMemory = 100
)

// DeadLetter is a WAL entry the replica gave up applying. It carries the
// full entry so an operator can inspect and replay it by hand.
type DeadLetter struct {
	ShardID  string            `json:"shard_id"`
	Seq      uint64            `json:"seq"`
	Op       string            `json:"op"`
	DocID    string            `json:"doc_id"`
	Text     string            `json:"text,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
	Fields   map[string]string `json:"fields,omitempty"`
	Err      string            `json:"error"`
	Attempts int               `json:"attempts"`
	At       time.Time         `json:"at"`
}

// SetDeadLetterPath sets a file that each dead-lettered entry is appended to
// as one JSON line. Empty (the default) keeps dead letters in memory only.
// Call before Start.
func (r *ReplicaApplier) SetDeadLetterPath(path string) {
	r.deadLetterPath = path
}

// DeadLetters returns the most recent dead-lettered entries, oldest first.
func (r *ReplicaApplier) DeadLetters() []DeadLetter {
	r.dlMu.Lock()
	defer r.dlMu.Unlock()
	return append([]DeadLetter(nil), r.deadLetters...)
}

// DeadLetterCount returns the total number of entries dead-lettered since
// start. Non-zero means this replica has diverged from its primary.
func (r *ReplicaApplier) DeadLetterCount() uint64 {
	return r.deadLetterCount.Load()
}

// deadLetter records an entry that exhausted its apply attempts.
func (r *ReplicaApplier) deadLetter(entry *WALEntry, applyErr error, attempts int) {
	dl := DeadLetter{
		ShardID:  r.shardID,
		Seq:      entry.Seq,
		Op:       entry.Op,
		DocID:    entry.DocId,
		Text:     entry.Text,
		Metadata: entry.Metadata,
		Fields:   entry.Fields,
		Err:      applyErr.Error(),
		Attempts: attempts,
		At:       time.Now().UTC(),
	}
	slog.Error("replica: dead-lettering WAL entry after repeated apply failures; replica has diverged from primary",
		"shard", r.shardID, "seq", entry.Seq, "doc_id", entry.DocId, "attempts", attempts, "err", applyErr)

	r.dlMu.Lock()
	r.deadLetters = append(r.deadLetters, dl)
	if len(r.deadLetters) > maxDeadLettersInMemory {
		r.deadLetters = r.deadLetters[len(r.deadLetters)-maxDeadLettersInMemory:]
	}
	r.dlMu.Unlock()
	r.deadLetterCount.Add(1)

	if r.deadLetterPath != "" {
		if err := appendDeadLetter(r.deadLetterPath, dl); err != nil {
			slog.Error("replica: failed to persist dead letter", "path", r.deadLetterPath, "seq", entry.Seq, "err", err)
		}
	}
}

func appendDeadLetter(path string, dl DeadLetter) error {
	line, err := json.Marshal(dl)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(line, '\n')); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
