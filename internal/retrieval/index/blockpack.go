package index

import (
	"encoding/binary"

	"github.com/ronanh/intcomp"
)

// BlockSize is the number of values in one compressed block.
// Must be 128 to align with skip list granularity.
const BlockSize = 128

// BlockPackBufSize is the maximum byte size of one PackBlock output.
// Worst case: 4-byte header + 128 uint64s uncompressed = 4 + 128*8 = 1028 bytes.
// Add 64 bytes margin for intcomp header words.
const BlockPackBufSize = 4 + BlockSize*8 + 64

// blockPackBufSize is an unexported alias for internal use.
const blockPackBufSize = BlockPackBufSize

// compBlockMaxWords is the maximum number of uint64 words in a single intcomp-
// compressed block. Derived from BlockPackBufSize: (BlockPackBufSize-4)/8 = 136.
// Used to size the PostingIter.compScratch field so UnpackBlockWithScratch
// never falls back to a heap allocation.
const compBlockMaxWords = (BlockPackBufSize - 4) / 8 // = 136

// PackFOR32BufSize is the maximum byte size of one PackFOR32 output.
// Worst case: 1-byte header + BlockSize×32 bits = 1 + 512 = 513 bytes.
const PackFOR32BufSize = 1 + (BlockSize*32+7)/8 // = 513

// NOTE: this format is incompatible with .seg files written by the previous FOR-delta codec.
// Clear data/ before starting a server after upgrading.

// PackBlock compresses exactly BlockSize uint64 values into dst using
// delta binary packing (github.com/ronanh/intcomp).
//
// Wire format:
//
//	[4] number of compressed uint64 words (little-endian uint32)
//	[n*8] compressed words (each little-endian uint64)
//
// dst must have capacity >= blockPackBufSize.
// Returns the number of bytes written.
func PackBlock(vals []uint64, dst []byte) int {
	compressed := intcomp.CompressUint64(vals[:BlockSize], nil)
	n := len(compressed)
	binary.LittleEndian.PutUint32(dst[:4], uint32(n))
	for i, v := range compressed {
		binary.LittleEndian.PutUint64(dst[4+i*8:], v)
	}
	return 4 + n*8
}

// UnpackBlock decompresses exactly BlockSize uint64 values from src into out.
// src must start at the beginning of a block written by PackBlock.
// Returns total bytes consumed from src.
func UnpackBlock(src []byte, out []uint64) int {
	n := int(binary.LittleEndian.Uint32(src[:4]))
	compressed := make([]uint64, n)
	for i := range compressed {
		compressed[i] = binary.LittleEndian.Uint64(src[4+i*8:])
	}
	result := intcomp.UncompressUint64(compressed, out[:0])
	// copy in case intcomp allocated a new backing array
	copy(out[:BlockSize], result)
	return 4 + n*8
}

// UnpackBlockWithScratch is like UnpackBlock but uses the caller-provided
// scratch slice instead of allocating per call. scratch must have length >=
// compBlockMaxWords (136). If n exceeds len(scratch), it falls back to a
// heap allocation as a safety net (should not occur for valid blocks).
// Returns total bytes consumed from src.
func UnpackBlockWithScratch(src []byte, out []uint64, scratch []uint64) int {
	n := int(binary.LittleEndian.Uint32(src[:4]))
	var compressed []uint64
	if n <= len(scratch) {
		compressed = scratch[:n]
	} else {
		compressed = make([]uint64, n) // safety: should not happen for well-formed blocks
	}
	for i := range compressed {
		compressed[i] = binary.LittleEndian.Uint64(src[4+i*8:])
	}
	result := intcomp.UncompressUint64(compressed, out[:0])
	copy(out[:BlockSize], result)
	return 4 + n*8
}

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
