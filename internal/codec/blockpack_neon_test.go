//go:build cgo && arm64

package codec

import (
	"bytes"
	"math/rand"
	"testing"
)

// neonTestVectors returns 128-value blocks exercising every bit of a width:
// all zeros, all max, ascending, alternating extremes, and random values.
func neonTestVectors(bits uint8) [][]uint32 {
	max := uint32(1)<<bits - 1
	rng := rand.New(rand.NewSource(int64(bits)))
	var zeros, maxes, asc, alt, rnd [BlockSize]uint32
	for i := 0; i < BlockSize; i++ {
		maxes[i] = max
		asc[i] = uint32(i) & max
		if i%2 == 1 {
			alt[i] = max
		}
		rnd[i] = rng.Uint32() & max
	}
	return [][]uint32{zeros[:], maxes[:], asc[:], alt[:], rnd[:]}
}

// TestNEONPackMatchesGoReference checks the NEON pack kernels (bits 4, 8, 16)
// produce byte-for-byte the output of the pure-Go packBits32 — the on-disk
// v6 format. (Port of the former Rust crate's golden-vector tests, which were
// generated from this same Go implementation.)
func TestNEONPackMatchesGoReference(t *testing.T) {
	for _, bits := range []uint8{4, 8, 16} {
		n := (BlockSize*int(bits) + 7) / 8
		for vi, vals := range neonTestVectors(bits) {
			want := make([]byte, n)
			packBits32(vals, bits, want)
			got := make([]byte, n)
			packBitsNEON(vals, bits, got)
			if !bytes.Equal(got, want) {
				t.Errorf("bits=%d vector %d: NEON pack differs from Go reference", bits, vi)
			}
		}
	}
}

// TestNEONUnpackMatchesGoReference checks UnpackFOR32Into's NEON paths
// (bits 4, 8, 16, 24) decode exactly what the pure-Go unpackBits32 decodes.
func TestNEONUnpackMatchesGoReference(t *testing.T) {
	for _, bits := range []uint8{4, 8, 16, 24} {
		n := (BlockSize*int(bits) + 7) / 8
		for vi, vals := range neonTestVectors(bits) {
			src := make([]byte, 1+n+16) // header + payload + slack for 16-byte loads
			src[0] = bits
			packBits32(vals, bits, src[1:1+n])

			want := make([]uint64, BlockSize)
			unpackBits32(src[1:1+n], bits, want)
			got := make([]uint64, BlockSize)
			if used := UnpackFOR32Into(src, got); used != 1+n {
				t.Errorf("bits=%d vector %d: consumed %d bytes, want %d", bits, vi, used, 1+n)
			}
			for i := range want {
				if got[i] != want[i] || want[i] != uint64(vals[i]) {
					t.Errorf("bits=%d vector %d: value %d = %d, Go reference %d, original %d",
						bits, vi, i, got[i], want[i], vals[i])
					break
				}
			}
		}
	}
}
