package codec_test

import (
	"math"
	"testing"

	"search-eval-platform/internal/codec"
)

func TestVarintRoundTrip(t *testing.T) {
	cases := []uint64{0, 1, 127, 128, 16383, 16384, math.MaxUint64}
	for _, v := range cases {
		buf := codec.AppendVarint(nil, v)
		got, n := codec.ReadVarint(buf, 0)
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
		buf := codec.AppendVarint(nil, c.v)
		if len(buf) != c.want {
			t.Errorf("value %d: want %d bytes, got %d", c.v, c.want, len(buf))
		}
		_, n := codec.ReadVarint(buf, 0)
		if n != c.want {
			t.Errorf("value %d: ReadVarint bytesRead=%d, want %d", c.v, n, c.want)
		}
	}
}

func TestVarintOffset(t *testing.T) {
	// Encode two values back-to-back and verify offset-based reading
	buf := codec.AppendVarint(nil, 300)
	buf = codec.AppendVarint(buf, 42)

	v1, n1 := codec.ReadVarint(buf, 0)
	if v1 != 300 {
		t.Fatalf("first value: want 300, got %d", v1)
	}
	v2, n2 := codec.ReadVarint(buf, n1)
	if v2 != 42 {
		t.Fatalf("second value: want 42, got %d", v2)
	}
	if n2 != 1 {
		t.Fatalf("second value bytes: want 1, got %d", n2)
	}
}
