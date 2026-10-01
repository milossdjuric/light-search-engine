// Package codec holds the posting-list encodings: LEB128 varints for block
// tails and the FOR-delta uint32 block codec for full 128-entry blocks, with
// native kernels (AVX2 C on amd64, NEON C++ on arm64) compiled by cgo and
// pure-Go fallbacks for !cgo and other architectures.
package codec

// BlockSize is the number of values in one compressed block.
// Must be 128 to align with skip list granularity.
const BlockSize = 128

// PackFOR32BufSize is the maximum byte size of one PackFOR32 output.
// Worst case: 1-byte header + BlockSize×32 bits = 1 + 512 = 513 bytes.
const PackFOR32BufSize = 1 + (BlockSize*32+7)/8 // = 513

// PackFOR32 compresses exactly BlockSize uint32 values into dst using bit-packing
// with a 1-byte bits-per-value header (Frame of Reference, zero-base assumed).
//
// Wire format:
//
//	[1] bits per value (0 = all values are zero)
//	[(BlockSize × bits + 7) / 8] packed bits (little-endian stream)
//
// dst must have capacity >= PackFOR32BufSize.
// Returns the number of bytes written.
func PackFOR32(vals []uint32, dst []byte) int {
	var maxVal uint32
	for _, v := range vals[:BlockSize] {
		if v > maxVal {
			maxVal = v
		}
	}
	bits := uint8(bits32Len(maxVal))
	dst[0] = bits
	if bits == 0 {
		return 1
	}
	packBitsImpl(vals[:BlockSize], bits, dst[1:])
	return 1 + (BlockSize*int(bits)+7)/8
}

// bits32Len returns the minimum number of bits needed to represent v,
// i.e. floor(log2(v))+1. Returns 0 for v==0.
func bits32Len(v uint32) int {
	n := 0
	for v > 0 {
		v >>= 1
		n++
	}
	return n
}

// packBitsImpl is the active bit-packing implementation. blockpack_cgo.go
// replaces this with an AVX2-accelerated version at init time when possible.
var packBitsImpl func(vals []uint32, bits uint8, dst []byte) = packBits32

// packBits32 encodes vals into dst using bits bits per value (little-endian
// bit stream). dst must be zeroed and long enough for the output.
func packBits32(vals []uint32, bits uint8, dst []byte) {
	bitsU := uint(bits)
	var bitBuf uint64
	var bitsFilled uint
	di := 0
	for _, v := range vals {
		bitBuf |= uint64(v) << bitsFilled
		bitsFilled += bitsU
		for bitsFilled >= 8 {
			dst[di] = byte(bitBuf)
			bitBuf >>= 8
			bitsFilled -= 8
			di++
		}
	}
	if bitsFilled > 0 {
		dst[di] = byte(bitBuf)
	}
}

// unpackBits32 decodes BlockSize uint32 values from src (a bit stream written
// by packBits32) into out as uint64. bits is the bits-per-value header byte.
// This is the pure-Go fallback used by blockpack_nocgo.go.
func unpackBits32(src []byte, bits uint8, out []uint64) {
	bitsU := uint(bits)
	mask := uint64((1 << bitsU) - 1)
	var bitBuf uint64
	var bitsFilled uint
	si := 0
	for i := range out[:BlockSize] {
		for bitsFilled < bitsU {
			bitBuf |= uint64(src[si]) << bitsFilled
			bitsFilled += 8
			si++
		}
		out[i] = bitBuf & mask
		bitBuf >>= bitsU
		bitsFilled -= bitsU
	}
}
