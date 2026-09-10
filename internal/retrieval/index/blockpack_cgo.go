//go:build cgo && amd64

package index

/*
#cgo CFLAGS: -O2 -mavx2
#include <stdint.h>
#include <immintrin.h>

// unpack4_avx2 unpacks 128 packed 4-bit values (64 bytes) to 128 uint32.
// Each input byte holds two values: low nibble = even index, high nibble = odd index.
// Uses SSSE3 pshufb to duplicate each byte, then blends low/high nibbles.
static void unpack4_avx2(const uint8_t *src, uint32_t *out) {
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    // Select odd byte positions from hi (high bit set = take from second operand).
    const __m128i odd_sel = _mm_set_epi8(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0);
    int i;
    for (i = 0; i < 64; i += 8) {
        // Load 8 bytes = 16 packed nibbles.
        __m128i raw = _mm_loadl_epi64((const __m128i *)(src + i));
        // Duplicate each byte: [b0,b1,...,b7] → [b0,b0,b1,b1,...,b7,b7].
        const __m128i shuf = _mm_set_epi8(7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0);
        __m128i v = _mm_shuffle_epi8(raw, shuf);
        // lo: keep low nibble of each byte; hi: keep high nibble shifted to low position.
        __m128i lo = _mm_and_si128(v, lo_mask);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), lo_mask);
        // Even byte positions: value at even index (lo nibble).
        // Odd byte positions: value at odd index (hi nibble).
        __m128i nibbles = _mm_blendv_epi8(lo, hi, odd_sel);
        // Zero-extend 8 bytes at a time to uint32.
        int base = i * 2;
        _mm256_storeu_si256((__m256i *)(out + base),
            _mm256_cvtepu8_epi32(nibbles));
        _mm256_storeu_si256((__m256i *)(out + base + 8),
            _mm256_cvtepu8_epi32(_mm_srli_si128(nibbles, 8)));
    }
}

// unpack8_avx2 zero-extends 128 packed uint8 values to uint32.
static void unpack8_avx2(const uint8_t *src, uint32_t *out) {
    int i;
    for (i = 0; i < 128; i += 8) {
        __m256i v = _mm256_cvtepu8_epi32(
            _mm_loadl_epi64((const __m128i *)(src + i)));
        _mm256_storeu_si256((__m256i *)(out + i), v);
    }
}

// unpack16_avx2 zero-extends 128 packed uint16 values to uint32.
static void unpack16_avx2(const uint16_t *src, uint32_t *out) {
    int i;
    for (i = 0; i < 128; i += 8) {
        __m256i v = _mm256_cvtepu16_epi32(
            _mm_loadu_si128((const __m128i *)(src + i)));
        _mm256_storeu_si256((__m256i *)(out + i), v);
    }
}

// unpack24_avx2 unpacks 128 packed 3-byte (24-bit) values to 128 uint32.
static void unpack24_avx2(const uint8_t *src, uint32_t *out) {
    const __m128i shuf = _mm_set_epi8(-1,11,10,9, -1,8,7,6, -1,5,4,3, -1,2,1,0);
    int i;
    for (i = 0; i < 124; i += 4) {
        __m128i raw = _mm_loadu_si128((const __m128i *)(src + i * 3));
        __m128i v   = _mm_shuffle_epi8(raw, shuf);
        _mm_storeu_si128((__m128i *)(out + i), v);
    }
    for (; i < 128; i++) {
        out[i] = (uint32_t)src[i*3] | ((uint32_t)src[i*3+1] << 8) | ((uint32_t)src[i*3+2] << 16);
    }
}

// ── Specialized unpackers for bits 1–3, 5–7 (8 values per b-byte group, fits in uint64) ──

static void unpack1_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        uint8_t b = src[g];
        for (int i = 0; i < 8; i++) out[g*8+i] = (b >> i) & 1;
    }
}
static void unpack2_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        uint32_t w = (uint32_t)src[g*2] | ((uint32_t)src[g*2+1] << 8);
        for (int i = 0; i < 8; i++) out[g*8+i] = (w >> (i*2)) & 3;
    }
}
static void unpack3_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*3;
        uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
        for (int i = 0; i < 8; i++) out[g*8+i] = (w >> (i*3)) & 7;
    }
}
static void unpack5_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*5;
        uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) |
                     ((uint64_t)p[3]<<24) | ((uint64_t)p[4]<<32);
        for (int i = 0; i < 8; i++) out[g*8+i] = (w >> (i*5)) & 31;
    }
}
static void unpack6_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*6;
        uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) |
                     ((uint64_t)p[3]<<24) | ((uint64_t)p[4]<<32) | ((uint64_t)p[5]<<40);
        for (int i = 0; i < 8; i++) out[g*8+i] = (w >> (i*6)) & 63;
    }
}
static void unpack7_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*7;
        uint64_t w = (uint64_t)p[0] | ((uint64_t)p[1]<<8) | ((uint64_t)p[2]<<16) |
                     ((uint64_t)p[3]<<24) | ((uint64_t)p[4]<<32) | ((uint64_t)p[5]<<40) |
                     ((uint64_t)p[6]<<48);
        for (int i = 0; i < 8; i++) out[g*8+i] = (w >> (i*7)) & 127;
    }
}

// ── Specialized unpackers for bits 9–15 (8 values per b-byte group, lo+hi uint64) ──

static void unpack9_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*9;
        uint64_t lo = (uint64_t)p[0]|(uint64_t)p[1]<<8|(uint64_t)p[2]<<16|(uint64_t)p[3]<<24|
                      (uint64_t)p[4]<<32|(uint64_t)p[5]<<40|(uint64_t)p[6]<<48|(uint64_t)p[7]<<56;
        uint64_t hi = (uint64_t)p[8];
        out[g*8+0]=(uint32_t)(lo & 0x1FF); out[g*8+1]=(uint32_t)((lo>>9) & 0x1FF);
        out[g*8+2]=(uint32_t)((lo>>18) & 0x1FF); out[g*8+3]=(uint32_t)((lo>>27) & 0x1FF);
        out[g*8+4]=(uint32_t)((lo>>36) & 0x1FF); out[g*8+5]=(uint32_t)((lo>>45) & 0x1FF);
        out[g*8+6]=(uint32_t)((lo>>54) & 0x1FF);
        out[g*8+7]=(uint32_t)(((lo>>63)|(hi<<1)) & 0x1FF);
    }
}
static void unpack10_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*10;
        uint64_t lo = (uint64_t)p[0]|(uint64_t)p[1]<<8|(uint64_t)p[2]<<16|(uint64_t)p[3]<<24|
                      (uint64_t)p[4]<<32|(uint64_t)p[5]<<40|(uint64_t)p[6]<<48|(uint64_t)p[7]<<56;
        uint64_t hi = (uint64_t)p[8]|(uint64_t)p[9]<<8;
        out[g*8+0]=(uint32_t)(lo & 0x3FF); out[g*8+1]=(uint32_t)((lo>>10) & 0x3FF);
        out[g*8+2]=(uint32_t)((lo>>20) & 0x3FF); out[g*8+3]=(uint32_t)((lo>>30) & 0x3FF);
        out[g*8+4]=(uint32_t)((lo>>40) & 0x3FF); out[g*8+5]=(uint32_t)((lo>>50) & 0x3FF);
        out[g*8+6]=(uint32_t)(((lo>>60)|(hi<<4)) & 0x3FF);
        out[g*8+7]=(uint32_t)((hi>>6) & 0x3FF);
    }
}
static void unpack11_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*11;
        uint64_t lo = (uint64_t)p[0]|(uint64_t)p[1]<<8|(uint64_t)p[2]<<16|(uint64_t)p[3]<<24|
                      (uint64_t)p[4]<<32|(uint64_t)p[5]<<40|(uint64_t)p[6]<<48|(uint64_t)p[7]<<56;
        uint64_t hi = (uint64_t)p[8]|(uint64_t)p[9]<<8|(uint64_t)p[10]<<16;
        out[g*8+0]=(uint32_t)(lo & 0x7FF); out[g*8+1]=(uint32_t)((lo>>11) & 0x7FF);
        out[g*8+2]=(uint32_t)((lo>>22) & 0x7FF); out[g*8+3]=(uint32_t)((lo>>33) & 0x7FF);
        out[g*8+4]=(uint32_t)((lo>>44) & 0x7FF);
        out[g*8+5]=(uint32_t)(((lo>>55)|(hi<<9)) & 0x7FF);
        out[g*8+6]=(uint32_t)((hi>>2) & 0x7FF); out[g*8+7]=(uint32_t)((hi>>13) & 0x7FF);
    }
}
static void unpack12_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*12;
        uint64_t lo = (uint64_t)p[0]|(uint64_t)p[1]<<8|(uint64_t)p[2]<<16|(uint64_t)p[3]<<24|
                      (uint64_t)p[4]<<32|(uint64_t)p[5]<<40|(uint64_t)p[6]<<48|(uint64_t)p[7]<<56;
        uint64_t hi = (uint64_t)p[8]|(uint64_t)p[9]<<8|(uint64_t)p[10]<<16|(uint64_t)p[11]<<24;
        out[g*8+0]=(uint32_t)(lo & 0xFFF); out[g*8+1]=(uint32_t)((lo>>12) & 0xFFF);
        out[g*8+2]=(uint32_t)((lo>>24) & 0xFFF); out[g*8+3]=(uint32_t)((lo>>36) & 0xFFF);
        out[g*8+4]=(uint32_t)((lo>>48) & 0xFFF);
        out[g*8+5]=(uint32_t)(((lo>>60)|(hi<<4)) & 0xFFF);
        out[g*8+6]=(uint32_t)((hi>>8) & 0xFFF); out[g*8+7]=(uint32_t)((hi>>20) & 0xFFF);
    }
}
static void unpack13_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*13;
        uint64_t lo = (uint64_t)p[0]|(uint64_t)p[1]<<8|(uint64_t)p[2]<<16|(uint64_t)p[3]<<24|
                      (uint64_t)p[4]<<32|(uint64_t)p[5]<<40|(uint64_t)p[6]<<48|(uint64_t)p[7]<<56;
        uint64_t hi = (uint64_t)p[8]|(uint64_t)p[9]<<8|(uint64_t)p[10]<<16|
                      (uint64_t)p[11]<<24|(uint64_t)p[12]<<32;
        out[g*8+0]=(uint32_t)(lo & 0x1FFF); out[g*8+1]=(uint32_t)((lo>>13) & 0x1FFF);
        out[g*8+2]=(uint32_t)((lo>>26) & 0x1FFF); out[g*8+3]=(uint32_t)((lo>>39) & 0x1FFF);
        out[g*8+4]=(uint32_t)(((lo>>52)|(hi<<12)) & 0x1FFF);
        out[g*8+5]=(uint32_t)((hi>>1) & 0x1FFF); out[g*8+6]=(uint32_t)((hi>>14) & 0x1FFF);
        out[g*8+7]=(uint32_t)((hi>>27) & 0x1FFF);
    }
}
static void unpack14_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*14;
        uint64_t lo = (uint64_t)p[0]|(uint64_t)p[1]<<8|(uint64_t)p[2]<<16|(uint64_t)p[3]<<24|
                      (uint64_t)p[4]<<32|(uint64_t)p[5]<<40|(uint64_t)p[6]<<48|(uint64_t)p[7]<<56;
        uint64_t hi = (uint64_t)p[8]|(uint64_t)p[9]<<8|(uint64_t)p[10]<<16|
                      (uint64_t)p[11]<<24|(uint64_t)p[12]<<32|(uint64_t)p[13]<<40;
        out[g*8+0]=(uint32_t)(lo & 0x3FFF); out[g*8+1]=(uint32_t)((lo>>14) & 0x3FFF);
        out[g*8+2]=(uint32_t)((lo>>28) & 0x3FFF); out[g*8+3]=(uint32_t)((lo>>42) & 0x3FFF);
        out[g*8+4]=(uint32_t)(((lo>>56)|(hi<<8)) & 0x3FFF);
        out[g*8+5]=(uint32_t)((hi>>6) & 0x3FFF); out[g*8+6]=(uint32_t)((hi>>20) & 0x3FFF);
        out[g*8+7]=(uint32_t)((hi>>34) & 0x3FFF);
    }
}
static void unpack15_c(const uint8_t *src, uint32_t *out) {
    for (int g = 0; g < 16; g++) {
        const uint8_t *p = src + g*15;
        uint64_t lo = (uint64_t)p[0]|(uint64_t)p[1]<<8|(uint64_t)p[2]<<16|(uint64_t)p[3]<<24|
                      (uint64_t)p[4]<<32|(uint64_t)p[5]<<40|(uint64_t)p[6]<<48|(uint64_t)p[7]<<56;
        uint64_t hi = (uint64_t)p[8]|(uint64_t)p[9]<<8|(uint64_t)p[10]<<16|
                      (uint64_t)p[11]<<24|(uint64_t)p[12]<<32|(uint64_t)p[13]<<40|(uint64_t)p[14]<<48;
        out[g*8+0]=(uint32_t)(lo & 0x7FFF); out[g*8+1]=(uint32_t)((lo>>15) & 0x7FFF);
        out[g*8+2]=(uint32_t)((lo>>30) & 0x7FFF); out[g*8+3]=(uint32_t)((lo>>45) & 0x7FFF);
        out[g*8+4]=(uint32_t)(((lo>>60)|(hi<<4)) & 0x7FFF);
        out[g*8+5]=(uint32_t)((hi>>11) & 0x7FFF); out[g*8+6]=(uint32_t)((hi>>26) & 0x7FFF);
        out[g*8+7]=(uint32_t)((hi>>41) & 0x7FFF);
    }
}

// pack4_avx2 packs 128 uint32 values (each < 16) into 64 nibble-packed bytes.
// Pairs are packed as: dst[k] = src[2k] | (src[2k+1] << 4).
// Processes 32 values per iteration using packus narrowing + nibble OR.
static void pack4_avx2(const uint32_t *src, uint8_t *dst) {
    int i;
    for (i = 0; i < 128; i += 32) {
        __m256i a  = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i b  = _mm256_loadu_si256((const __m256i *)(src + i + 8));
        __m256i c  = _mm256_loadu_si256((const __m256i *)(src + i + 16));
        __m256i d  = _mm256_loadu_si256((const __m256i *)(src + i + 24));
        // Narrow 32→16 (values ≤ 15, saturation is a no-op).
        __m256i ab = _mm256_packus_epi32(a, b);
        ab = _mm256_permute4x64_epi64(ab, 0xD8);
        __m256i cd = _mm256_packus_epi32(c, d);
        cd = _mm256_permute4x64_epi64(cd, 0xD8);
        // Narrow 16→8; p = [v0,v1,...,v31] as bytes.
        __m256i p  = _mm256_packus_epi16(ab, cd);
        p = _mm256_permute4x64_epi64(p, 0xD8);
        // Pack nibble pairs: combined[k] = p[2k] | (p[2k+1] << 4) as uint16.
        const __m256i mask_ff = _mm256_set1_epi16(0x00FF);
        __m256i lo = _mm256_and_si256(p, mask_ff);       // even values as uint16
        __m256i hi = _mm256_srli_epi16(p, 8);            // odd values as uint16
        __m256i combined = _mm256_or_si256(lo, _mm256_slli_epi16(hi, 4));
        // Narrow 16→8 (combined values ≤ 0xFF); store 16 bytes.
        __m256i result = _mm256_packus_epi16(combined, _mm256_setzero_si256());
        result = _mm256_permute4x64_epi64(result, 0xD8);
        _mm_storeu_si128((__m128i *)(dst + i / 2), _mm256_castsi256_si128(result));
    }
}

// pack8_avx2 narrows 128 uint32 values (each fits in uint8) to 128 packed bytes.
// Processes 32 values per iteration using two stages:
//   stage 1: _mm256_packus_epi32  (32-bit → 16-bit, two regs at a time)
//   stage 2: _mm256_packus_epi16  (16-bit → 8-bit, two regs at a time)
// _mm256_packus interleaves across 128-bit lanes, so a permute4x64 (0xD8)
// restores sequential order after each stage.
static void pack8_avx2(const uint32_t *src, uint8_t *dst) {
    int i;
    for (i = 0; i < 128; i += 32) {
        __m256i a  = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i b  = _mm256_loadu_si256((const __m256i *)(src + i + 8));
        __m256i c  = _mm256_loadu_si256((const __m256i *)(src + i + 16));
        __m256i d  = _mm256_loadu_si256((const __m256i *)(src + i + 24));
        // 32→16: pack a+b → [a0..a7, b0..b7] as uint16
        __m256i ab = _mm256_packus_epi32(a, b);
        ab = _mm256_permute4x64_epi64(ab, 0xD8);
        // 32→16: pack c+d → [c0..c7, d0..d7] as uint16
        __m256i cd = _mm256_packus_epi32(c, d);
        cd = _mm256_permute4x64_epi64(cd, 0xD8);
        // 16→8: pack ab+cd → [a0..a7, c0..c7, b0..b7, d0..d7] as uint8 (lanes interleaved)
        __m256i p  = _mm256_packus_epi16(ab, cd);
        // Restore sequential order: [a, b, c, d]
        p = _mm256_permute4x64_epi64(p, 0xD8);
        _mm256_storeu_si256((__m256i *)(dst + i), p);
    }
}

// pack16_avx2 narrows 128 uint32 values (each fits in uint16) to 256 packed bytes.
// Processes 16 values per iteration using _mm256_packus_epi32 + permute.
static void pack16_avx2(const uint32_t *src, uint16_t *dst) {
    int i;
    for (i = 0; i < 128; i += 16) {
        __m256i a = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(src + i + 8));
        // Pack a+b → [a0..a7, b0..b7] as uint16 (with lane-interleave fix)
        __m256i p = _mm256_packus_epi32(a, b);
        p = _mm256_permute4x64_epi64(p, 0xD8);
        _mm256_storeu_si256((__m256i *)(dst + i), p);
    }
}
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
