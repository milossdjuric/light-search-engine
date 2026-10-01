//go:build cgo && amd64

package codec

/*
#cgo CFLAGS: -O2 -mavx2
#include <stdint.h>

// Implemented in blockpack_avx2_amd64.c, which cgo compiles along with this
// package (no separate build step).
void unpack4_avx2(const uint8_t *src, uint32_t *out);
void unpack8_avx2(const uint8_t *src, uint32_t *out);
void unpack16_avx2(const uint16_t *src, uint32_t *out);
void unpack24_avx2(const uint8_t *src, uint32_t *out);
void unpack1_c(const uint8_t *src, uint32_t *out);
void unpack2_c(const uint8_t *src, uint32_t *out);
void unpack3_c(const uint8_t *src, uint32_t *out);
void unpack5_c(const uint8_t *src, uint32_t *out);
void unpack6_c(const uint8_t *src, uint32_t *out);
void unpack7_c(const uint8_t *src, uint32_t *out);
void unpack9_c(const uint8_t *src, uint32_t *out);
void unpack10_c(const uint8_t *src, uint32_t *out);
void unpack11_c(const uint8_t *src, uint32_t *out);
void unpack12_c(const uint8_t *src, uint32_t *out);
void unpack13_c(const uint8_t *src, uint32_t *out);
void unpack14_c(const uint8_t *src, uint32_t *out);
void unpack15_c(const uint8_t *src, uint32_t *out);
void pack4_avx2(const uint32_t *src, uint8_t *dst);
void pack8_avx2(const uint32_t *src, uint8_t *dst);
void pack16_avx2(const uint32_t *src, uint16_t *dst);
*/
import "C"
import (
	"unsafe"

	cpuid "github.com/klauspost/cpuid/v2"
)

// avx2OK is set once at package init; true when the CPU supports AVX2.
var avx2OK = cpuid.CPU.Supports(cpuid.AVX2)

func init() {
	if avx2OK {
		packBitsImpl = packBitsAVX2
	}
}

// packBitsAVX2 uses AVX2 SIMD to pack BlockSize uint32 delta values into dst.
// bits=4, 8, 16 use vectorized paths; other widths fall back to the pure-Go
// bit-shift loop (same as the non-SIMD path).
func packBitsAVX2(vals []uint32, bits uint8, dst []byte) {
	switch bits {
	case 4:
		C.pack4_avx2((*C.uint32_t)(unsafe.Pointer(&vals[0])),
			(*C.uint8_t)(unsafe.Pointer(&dst[0])))
	case 8:
		C.pack8_avx2((*C.uint32_t)(unsafe.Pointer(&vals[0])),
			(*C.uint8_t)(unsafe.Pointer(&dst[0])))
	case 16:
		C.pack16_avx2((*C.uint32_t)(unsafe.Pointer(&vals[0])),
			(*C.uint16_t)(unsafe.Pointer(&dst[0])))
	default:
		packBits32(vals, bits, dst)
	}
}

// UnpackFOR32Into decodes exactly BlockSize uint32 values from src (written by
// PackFOR32) into out as uint64. Returns the number of bytes consumed from src.
// On amd64 with AVX2, bits=8 and bits=16 use SIMD zero-extension; other bit
// widths and the fallback path use pure Go bit-unpacking.
func UnpackFOR32Into(src []byte, out []uint64) int {
	bits := src[0]
	if bits == 0 {
		for i := range out[:BlockSize] {
			out[i] = 0
		}
		return 1
	}

	nBytes := (BlockSize*int(bits) + 7) / 8

	var tmp [BlockSize]uint32
	p8 := (*C.uint8_t)(unsafe.Pointer(&src[1]))
	dispatched := true

	if avx2OK {
		switch bits {
		case 4:
			C.unpack4_avx2(p8, (*C.uint32_t)(unsafe.Pointer(&tmp[0])))
		case 8:
			C.unpack8_avx2(p8, (*C.uint32_t)(unsafe.Pointer(&tmp[0])))
		case 16:
			C.unpack16_avx2((*C.uint16_t)(unsafe.Pointer(&src[1])), (*C.uint32_t)(unsafe.Pointer(&tmp[0])))
		case 24:
			C.unpack24_avx2(p8, (*C.uint32_t)(unsafe.Pointer(&tmp[0])))
		default:
			dispatched = false
		}
	} else {
		dispatched = false
	}

	if !dispatched {
		out32 := (*C.uint32_t)(unsafe.Pointer(&tmp[0]))
		switch bits {
		case 1:
			C.unpack1_c(p8, out32)
		case 2:
			C.unpack2_c(p8, out32)
		case 3:
			C.unpack3_c(p8, out32)
		case 5:
			C.unpack5_c(p8, out32)
		case 6:
			C.unpack6_c(p8, out32)
		case 7:
			C.unpack7_c(p8, out32)
		case 9:
			C.unpack9_c(p8, out32)
		case 10:
			C.unpack10_c(p8, out32)
		case 11:
			C.unpack11_c(p8, out32)
		case 12:
			C.unpack12_c(p8, out32)
		case 13:
			C.unpack13_c(p8, out32)
		case 14:
			C.unpack14_c(p8, out32)
		case 15:
			C.unpack15_c(p8, out32)
		default:
			unpackBits32(src[1:], bits, out)
			return 1 + nBytes
		}
	}

	for i, v := range tmp {
		out[i] = uint64(v)
	}
	return 1 + nBytes
}
