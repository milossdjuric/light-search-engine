package index_test

import (
	"math"
	"testing"

	"search-eval-platform/internal/retrieval/index"
)

func TestVarintRoundTrip(t *testing.T) {
	cases := []uint64{0, 1, 127, 128, 16383, 16384, math.MaxUint64}
	for _, v := range cases {
		buf := index.AppendVarint(nil, v)
		got, n := index.ReadVarint(buf, 0)
		if got != v {
			t.Errorf("value %d: decoded %d", v, got)
		}
		if n <= 0 {
			t.Errorf("value %d: bytesRead=%d", v, n)
		}
	}
}

func TestVarintByteCounts(t *testing.T) {
	// LEB128: 0–127 → 1 byte, 128–16383 → 2 bytes, 16384–2097151 → 3 bytes
	cases := []struct {
		v    uint64
		want int
	}{
		{0, 1}, {127, 1},
		{128, 2}, {16383, 2},
		{16384, 3},
		{math.MaxUint64, 10},
	}
	for _, c := range cases {
		buf := index.AppendVarint(nil, c.v)
		if len(buf) != c.want {
			t.Errorf("value %d: want %d bytes, got %d", c.v, c.want, len(buf))
		}
		_, n := index.ReadVarint(buf, 0)
		if n != c.want {
			t.Errorf("value %d: ReadVarint bytesRead=%d, want %d", c.v, n, c.want)
		}
	}
}

func TestVarintOffset(t *testing.T) {
	// Encode two values back-to-back and verify offset-based reading
	buf := index.AppendVarint(nil, 300)
	buf = index.AppendVarint(buf, 42)

	v1, n1 := index.ReadVarint(buf, 0)
	if v1 != 300 {
		t.Fatalf("first value: want 300, got %d", v1)
	}
	v2, n2 := index.ReadVarint(buf, n1)
	if v2 != 42 {
		t.Fatalf("second value: want 42, got %d", v2)
	}
	if n2 != 1 {
		t.Fatalf("second value bytes: want 1, got %d", n2)
	}
}

func TestDeltaListRoundTrip(t *testing.T) {
	ids := []uint64{1, 5, 10, 128, 300, 1000, 1001, 99999}
	enc := index.EncodeDeltaList(ids)
	dec := index.DecodeDeltaList(enc)

	if len(dec) != len(ids) {
		t.Fatalf("length mismatch: want %d, got %d", len(ids), len(dec))
	}
	for i, want := range ids {
		if dec[i] != want {
			t.Errorf("[%d]: want %d, got %d", i, want, dec[i])
		}
	}
}

func TestDeltaListEmpty(t *testing.T) {
	enc := index.EncodeDeltaList(nil)
	dec := index.DecodeDeltaList(enc)
	if len(dec) != 0 {
		t.Errorf("empty list: got %v", dec)
	}
}

func TestDeltaListSingle(t *testing.T) {
	ids := []uint64{42}
	dec := index.DecodeDeltaList(index.EncodeDeltaList(ids))
	if len(dec) != 1 || dec[0] != 42 {
		t.Errorf("single element: got %v", dec)
	}
}
