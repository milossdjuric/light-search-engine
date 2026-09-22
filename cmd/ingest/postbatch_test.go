package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestPostBatchTreatsErrorStatusAsFullBatchFailure verifies that a non-2xx
// response from /index/bulk is counted as a fully-failed batch, not
// silently decoded as {Indexed:0, Failed:0} — which previously looked
// identical to "nothing to do" in the run summary, hiding real data loss.
func TestPostBatchTreatsErrorStatusAsFullBatchFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"index error: boom"}`))
	}))
	defer srv.Close()

	batch := []Record{{ID: "d1", Text: "hello"}, {ID: "d2", Text: "world"}}
	indexed, failed, _ := postBatch(srv.Client(), srv.URL, batch)

	if indexed != 0 {
		t.Errorf("indexed = %d, want 0 (server returned an error, nothing was actually indexed)", indexed)
	}
	if failed != len(batch) {
		t.Errorf("failed = %d, want %d (an error response must count the whole batch as failed, not (0,0))", failed, len(batch))
	}
}
