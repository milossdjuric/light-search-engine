package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// TestClientSendsAPIKeyOnWrites verifies every write subcommand presents
// "Authorization: Bearer <key>" so the CLI works against a server with
// server.api_key set (previously it sent no header, so every write got 401).
func TestClientSendsAPIKeyOnWrites(t *testing.T) {
	var mu sync.Mutex
	got := map[string]string{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		got[r.Method+" "+r.URL.Path] = r.Header.Get("Authorization")
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	bulkFile := filepath.Join(t.TempDir(), "docs.ndjson")
	if err := os.WriteFile(bulkFile, []byte(`{"id":"d1","text":"hi"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	c := &client{base: srv.URL, apiKey: "secret"}
	steps := map[string]func() error{
		"POST /index":       func() error { return c.cmdIndex([]string{"d1", "hello"}) },
		"DELETE /index/d1":  func() error { return c.cmdDelete([]string{"d1"}) },
		"POST /index/bulk":  func() error { return c.cmdBulk([]string{bulkFile}) },
		"POST /index/flush": c.cmdFlush,
		"POST /index/merge": c.cmdMerge,
		"POST /admin/reset": c.cmdReset,
	}
	for name, fn := range steps {
		if err := fn(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if h := got[name]; h != "Bearer secret" {
			t.Errorf("%s: Authorization = %q, want %q", name, h, "Bearer secret")
		}
	}
}

// TestClientOmitsAuthHeaderWithoutKey keeps the no-auth default unchanged.
func TestClientOmitsAuthHeaderWithoutKey(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	c := &client{base: srv.URL}
	if err := c.cmdFlush(); err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Errorf("Authorization = %q, want empty when no API key is configured", got)
	}
}
