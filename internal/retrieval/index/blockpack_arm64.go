//go:build cgo && arm64

package index

/*
#cgo LDFLAGS: -L${SRCDIR}/../../../native/blockpack_neon/target/aarch64-unknown-linux-gnu/release -lblockpack_neon
#include <stdint.h>

// Prototypes for the Rust NEON kernels in native/blockpack_neon (built via
// `cargo build --release --target aarch64-unknown-linux-gnu`; see that
// crate's Cargo.toml — crate-type = ["staticlib"]). Hand-written rather than
// cbindgen-generated since the surface is small and fixed.
void unpack4_neon(const uint8_t *src, uint32_t *out);
void unpack8_neon(const uint8_t *src, uint32_t *out);
void unpack16_neon(const uint16_t *src, uint32_t *out);
void unpack24_neon(const uint8_t *src, uint32_t *out);
void pack4_neon(const uint32_t *src, uint8_t *dst);
void pack8_neon(const uint32_t *src, uint8_t *dst);
void pack16_neon(const uint32_t *src, uint16_t *dst);
*/
import "C"
import "unsafe"

// NEON (Advanced SIMD) is part of the mandatory AArch64 baseline — unlike
// AVX2 on x86-64, there is no optional CPU feature to detect at runtime, so
// unlike blockpack_cgo.go's avx2OK there is no gate here.
func init() {
	packBitsImpl = packBitsNEON
}

// packBitsNEON mirrors packBitsAVX2's dispatch shape (blockpack_cgo.go):
// bits 4, 8, 16 use vectorized NEON paths; other widths fall back to the
// pure-Go bit-shift loop.
func packBitsNEON(vals []uint32, bits uint8, dst []byte) {
	switch bits {
	case 4:
		C.pack4_neon((*C.uint32_t)(unsafe.Pointer(&vals[0])),
			(*C.uint8_t)(unsafe.Pointer(&dst[0])))
	case 8:
		C.pack8_neon((*C.uint32_t)(unsafe.Pointer(&vals[0])),
			(*C.uint8_t)(unsafe.Pointer(&dst[0])))
	case 16:
		C.pack16_neon((*C.uint32_t)(unsafe.Pointer(&vals[0])),
			(*C.uint16_t)(unsafe.Pointer(&dst[0])))
	default:
		packBits32(vals, bits, dst)
	}
}

// UnpackFOR32Into decodes exactly BlockSize uint32 values from src (written
// by PackFOR32) into out as uint64. Returns the number of bytes consumed
// from src. Mirrors blockpack_cgo.go's UnpackFOR32Into (the AVX2 path):
// bits=4, 8, 16, 24 use NEON; other bit widths fall back to the pure-Go
// bit-unpacking used by the non-cgo build.
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

	switch bits {
	case 4:
		C.unpack4_neon(p8, (*C.uint32_t)(unsafe.Pointer(&tmp[0])))
	case 8:
		C.unpack8_neon(p8, (*C.uint32_t)(unsafe.Pointer(&tmp[0])))
	case 16:
		C.unpack16_neon((*C.uint16_t)(unsafe.Pointer(&src[1])), (*C.uint32_t)(unsafe.Pointer(&tmp[0])))
	case 24:
		C.unpack24_neon(p8, (*C.uint32_t)(unsafe.Pointer(&tmp[0])))
	default:
		unpackBits32(src[1:], bits, out)
		return 1 + nBytes
	}

	for i, v := range tmp {
		out[i] = uint64(v)
	}
	return 1 + nBytes
}
