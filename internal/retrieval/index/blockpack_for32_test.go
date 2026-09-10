package index

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestFOR32AllBitWidths(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for bits := uint8(1); bits <= 31; bits++ {
		bits := bits
		t.Run(fmt.Sprintf("bits=%d", bits), func(t *testing.T) {
			var maxVal uint32 = (1 << bits) - 1
			vals := make([]uint32, BlockSize)
			for i := range vals {
				vals[i] = r.Uint32() & maxVal
			}
			dst := make([]byte, PackFOR32BufSize)
			n := PackFOR32(vals, dst)
			out := make([]uint64, BlockSize)
			consumed := UnpackFOR32Into(dst[:n], out)
			if consumed != n {
				t.Fatalf("consumed %d want %d", consumed, n)
			}
			for i, v := range vals {
				if out[i] != uint64(v) {
					t.Fatalf("out[%d]=%d want %d", i, out[i], v)
				}
			}
		})
	}
}

func BenchmarkUnpackFOR32(b *testing.B) {
	r := rand.New(rand.NewSource(99))
	widths := []uint8{1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 24}
	for _, bits := range widths {
		bits := bits
		b.Run(fmt.Sprintf("bits=%d", bits), func(b *testing.B) {
			maxVal := uint32((1 << bits) - 1)
			vals := make([]uint32, BlockSize)
			for i := range vals {
				vals[i] = r.Uint32() & maxVal
			}
			dst := make([]byte, PackFOR32BufSize)
			n := PackFOR32(vals, dst)
			out := make([]uint64, BlockSize)
			b.SetBytes(int64(n))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				UnpackFOR32Into(dst[:n], out)
			}
		})
	}
}
