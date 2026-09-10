package index_test

import (
	"testing"

	"search-eval-platform/internal/retrieval/index"
)

// TestPackFOR32RoundTrip covers the FOR32 encode/decode path for the three
// bit widths that exercise different code paths on amd64+AVX2:
//   - bits=8  (AVX2 pack8_avx2 / unpack8_avx2)
//   - bits=16 (AVX2 pack16_avx2 / unpack16_avx2)
//   - bits=24 (pure-Go packBits32 / unpackBits32)
func TestPackFOR32RoundTrip(t *testing.T) {
	cases := []struct {
		name   string
		maxVal uint32
	}{
		{"bits8", 200},
		{"bits16", 60000},
		{"bits24", 1 << 20},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			vals := make([]uint32, index.BlockSize)
			for i := range vals {
				vals[i] = uint32(i) % (tc.maxVal + 1)
			}
			// Ensure max value appears so bit width is accurate.
			vals[index.BlockSize-1] = tc.maxVal

			dst := make([]byte, index.PackFOR32BufSize)
			n := index.PackFOR32(vals, dst)
			if n <= 0 {
				t.Fatalf("PackFOR32 returned %d bytes", n)
			}

			out := make([]uint64, index.BlockSize)
			m := index.UnpackFOR32Into(dst[:n], out)
			if m != n {
				t.Errorf("UnpackFOR32Into consumed %d bytes, want %d", m, n)
			}
			for i, want := range vals {
				if got := uint32(out[i]); got != want {
					t.Errorf("[%d]: want %d, got %d", i, want, got)
				}
			}
		})
	}
}

// TestPackFOR32AllSame verifies that a block of identical values encodes correctly.
func TestPackFOR32AllSame(t *testing.T) {
	for _, val := range []uint32{0, 1, 127, 255, 256, 65535, 1 << 24} {
		vals := make([]uint32, index.BlockSize)
		for i := range vals {
			vals[i] = val
		}
		dst := make([]byte, index.PackFOR32BufSize)
		n := index.PackFOR32(vals, dst)
		out := make([]uint64, index.BlockSize)
		index.UnpackFOR32Into(dst[:n], out)
		for i := range out {
			if uint32(out[i]) != val {
				t.Errorf("val=%d [%d]: got %d", val, i, out[i])
			}
		}
	}
}

func TestPackUnpackRoundTrip(t *testing.T) {
	vals := make([]uint64, index.BlockSize)
	for i := range vals {
		vals[i] = uint64(i * 3)
	}

	dst := make([]byte, index.BlockPackBufSize)
	n := index.PackBlock(vals, dst)
	if n <= 0 {
		t.Fatal("PackBlock returned 0 bytes")
	}

	out := make([]uint64, index.BlockSize)
	m := index.UnpackBlock(dst[:n], out)
	if m != n {
		t.Errorf("UnpackBlock bytesRead=%d, want %d", m, n)
	}
	for i, want := range vals {
		if out[i] != want {
			t.Errorf("[%d]: want %d, got %d", i, want, out[i])
		}
	}
}

func TestPackBlockAllZeros(t *testing.T) {
	vals := make([]uint64, index.BlockSize)
	dst := make([]byte, index.BlockPackBufSize)
	n := index.PackBlock(vals, dst)

	if n <= 0 {
		t.Errorf("all-zeros block: PackBlock returned %d bytes, want > 0", n)
	}

	out := make([]uint64, index.BlockSize)
	index.UnpackBlock(dst[:n], out)
	for i, v := range out {
		if v != 0 {
			t.Errorf("[%d]: want 0, got %d", i, v)
		}
	}
}

func TestPackBlockSingleBit(t *testing.T) {
	vals := make([]uint64, index.BlockSize)
	vals[0] = 1 // max value = 1
	dst := make([]byte, index.BlockPackBufSize)
	n := index.PackBlock(vals, dst)

	out := make([]uint64, index.BlockSize)
	index.UnpackBlock(dst[:n], out)
	if out[0] != 1 {
		t.Errorf("[0]: want 1, got %d", out[0])
	}
	for i := 1; i < index.BlockSize; i++ {
		if out[i] != 0 {
			t.Errorf("[%d]: want 0, got %d", i, out[i])
		}
	}
}

func TestPackBlockLargeValues(t *testing.T) {
	vals := make([]uint64, index.BlockSize)
	for i := range vals {
		vals[i] = uint64(1) << 32 // 4GB values — 33-bit width
	}
	dst := make([]byte, index.BlockPackBufSize)
	n := index.PackBlock(vals, dst)

	out := make([]uint64, index.BlockSize)
	index.UnpackBlock(dst[:n], out)
	for i, want := range vals {
		if out[i] != want {
			t.Errorf("[%d]: want %d, got %d", i, want, out[i])
		}
	}
}
