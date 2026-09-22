//! ARM NEON (AArch64 "Advanced SIMD") pack/unpack kernels for the FOR-delta
//! v6 block codec, mirroring the coverage of the AVX2 path in
//! internal/retrieval/index/blockpack_cgo.go: unpack for bits ∈ {4, 8, 16,
//! 24}, pack for bits ∈ {4, 8, 16}. Every function operates on exactly
//! `BLOCK_SIZE` (128) values, matching the fixed block size the Go side
//! always calls with.
//!
//! Exposed as `extern "C"` functions with the same argument shapes as their
//! AVX2/C counterparts so the Go side's cgo glue is a straight swap.
//!
//! Wire format is NOT implementation-defined: it must match
//! `packBits32`/`unpackBits32` in internal/retrieval/index/blockpack.go
//! exactly (little-endian bit stream, values packed low-to-high in
//! insertion order) — the same format the pure-Go and AVX2/C paths already
//! produce and consume. See tests/vectors.rs, which is generated from that
//! Go implementation directly, for the compatibility tests.

#[cfg(not(target_arch = "aarch64"))]
compile_error!("blockpack_neon only supports aarch64 (NEON is part of the mandatory AArch64 baseline, unlike AVX2 on x86-64)");

pub const BLOCK_SIZE: usize = 128;

mod neon {
    use std::arch::aarch64::*;

    /// Zero-extends 128 packed u8 values to u32. 8 lanes per iteration.
    ///
    /// # Safety
    /// `src` must be valid for 128 reads, `out` valid for 128 `u32` writes.
    pub unsafe fn unpack8(src: *const u8, out: *mut u32) {
        let mut i = 0usize;
        while i < 128 {
            let bytes = vld1_u8(src.add(i));
            let widened16 = vmovl_u8(bytes);
            let lo32 = vmovl_u16(vget_low_u16(widened16));
            let hi32 = vmovl_high_u16(widened16);
            vst1q_u32(out.add(i), lo32);
            vst1q_u32(out.add(i + 4), hi32);
            i += 8;
        }
    }

    /// Zero-extends 128 packed u16 values (little-endian) to u32.
    ///
    /// # Safety
    /// `src` must be valid for 128 `u16` reads, `out` valid for 128 `u32` writes.
    pub unsafe fn unpack16(src: *const u16, out: *mut u32) {
        let mut i = 0usize;
        while i < 128 {
            let half = vld1q_u16(src.add(i));
            let lo32 = vmovl_u16(vget_low_u16(half));
            let hi32 = vmovl_high_u16(half);
            vst1q_u32(out.add(i), lo32);
            vst1q_u32(out.add(i + 4), hi32);
            i += 8;
        }
    }

    /// Unpacks 128 packed 4-bit (nibble) values to u32. Each source byte
    /// holds two values: low nibble = even index, high nibble = odd index
    /// (matches packBits32's little-endian bit-stream convention).
    ///
    /// # Safety
    /// `src` must be valid for 64 reads, `out` valid for 128 `u32` writes.
    pub unsafe fn unpack4(src: *const u8, out: *mut u32) {
        let mut i = 0usize; // byte offset into src
        let mut o = 0usize; // value offset into out
        while i < 64 {
            let v = vld1_u8(src.add(i));
            let lo = vand_u8(v, vdup_n_u8(0x0F));
            let hi = vshr_n_u8::<4>(v);
            // Interleave to [lo0,hi0,lo1,hi1,...,lo7,hi7] = values 0..15
            // for these 8 source bytes.
            let inter_lo = vzip1_u8(lo, hi); // values 0..7
            let inter_hi = vzip2_u8(lo, hi); // values 8..15

            let w1 = vmovl_u8(inter_lo);
            vst1q_u32(out.add(o), vmovl_u16(vget_low_u16(w1)));
            vst1q_u32(out.add(o + 4), vmovl_high_u16(w1));

            let w2 = vmovl_u8(inter_hi);
            vst1q_u32(out.add(o + 8), vmovl_u16(vget_low_u16(w2)));
            vst1q_u32(out.add(o + 12), vmovl_high_u16(w2));

            i += 8;
            o += 16;
        }
    }

    /// Unpacks 128 packed 3-byte (24-bit) little-endian values to u32.
    /// Vectorizes the first 124 values via a table lookup that spreads each
    /// 3-byte group to a 4-byte (one zero byte inserted) group; the last 4
    /// values fall back to a scalar loop, matching unpack24_avx2's split.
    ///
    /// # Safety
    /// `src` must be valid for 384 reads (128*3), `out` valid for 128 `u32`
    /// writes. The vectorized path reads up to 4 bytes past the 12 bytes it
    /// needs per iteration (16-byte table loads) but never past byte 384 of
    /// `src` — the same over-read shape already used by unpack24_avx2's
    /// `_mm_loadu_si128` in the C AVX2 path.
    pub unsafe fn unpack24(src: *const u8, out: *mut u32) {
        let idx: [u8; 16] = [0, 1, 2, 255, 3, 4, 5, 255, 6, 7, 8, 255, 9, 10, 11, 255];
        let table = vld1q_u8(idx.as_ptr());

        let mut i = 0usize; // value index
        while i < 124 {
            let raw = vld1q_u8(src.add(i * 3));
            let v = vqtbl1q_u8(raw, table);
            vst1q_u32(out.add(i), vreinterpretq_u32_u8(v));
            i += 4;
        }
        while i < 128 {
            let o = i * 3;
            let b0 = *src.add(o) as u32;
            let b1 = *src.add(o + 1) as u32;
            let b2 = *src.add(o + 2) as u32;
            *out.add(i) = b0 | (b1 << 8) | (b2 << 16);
            i += 1;
        }
    }

    /// Narrows 128 u32 values (each fitting in u8) to 128 packed bytes.
    ///
    /// # Safety
    /// `src` must be valid for 128 `u32` reads, `dst` valid for 128 writes.
    pub unsafe fn pack8(src: *const u32, dst: *mut u8) {
        let mut i = 0usize;
        while i < 128 {
            let a = vld1q_u32(src.add(i));
            let b = vld1q_u32(src.add(i + 4));
            let a16 = vqmovn_u32(a);
            let b16 = vqmovn_u32(b);
            let ab16 = vcombine_u16(a16, b16);
            let ab8 = vqmovn_u16(ab16);
            vst1_u8(dst.add(i), ab8);
            i += 8;
        }
    }

    /// Narrows 128 u32 values (each fitting in u16) to 128 packed u16s
    /// (little-endian).
    ///
    /// # Safety
    /// `src` must be valid for 128 `u32` reads, `dst` valid for 128 `u16` writes.
    pub unsafe fn pack16(src: *const u32, dst: *mut u16) {
        let mut i = 0usize;
        while i < 128 {
            let a = vld1q_u32(src.add(i));
            let b = vld1q_u32(src.add(i + 4));
            let a16 = vqmovn_u32(a);
            let b16 = vqmovn_u32(b);
            let ab16 = vcombine_u16(a16, b16);
            vst1q_u16(dst.add(i), ab16);
            i += 8;
        }
    }

    /// Packs 128 u32 values (each < 16) into 64 nibble-packed bytes:
    /// dst[k] = src[2k] | (src[2k+1] << 4).
    ///
    /// # Safety
    /// `src` must be valid for 128 `u32` reads, `dst` valid for 64 writes.
    pub unsafe fn pack4(src: *const u32, dst: *mut u8) {
        let mut i = 0usize; // value index
        let mut o = 0usize; // dst byte index
        while i < 128 {
            let a = vld1q_u32(src.add(i));
            let b = vld1q_u32(src.add(i + 4));
            let a16 = vqmovn_u32(a);
            let b16 = vqmovn_u32(b);
            let ab16 = vcombine_u16(a16, b16);
            let v8 = vqmovn_u16(ab16); // 8 nibble values v0..v7

            let even = vuzp1_u8(v8, v8); // [v0,v2,v4,v6, ...]
            let odd = vuzp2_u8(v8, v8); // [v1,v3,v5,v7, ...]
            let combined = vorr_u8(even, vshl_n_u8::<4>(odd));
            let combined_u32 = vreinterpret_u32_u8(combined);
            vst1_lane_u32::<0>(dst.add(o) as *mut u32, combined_u32);

            i += 8;
            o += 4;
        }
    }
}

/// # Safety
/// `src` valid for 64 reads, `out` valid for 128 `u32` writes.
#[no_mangle]
pub unsafe extern "C" fn unpack4_neon(src: *const u8, out: *mut u32) {
    neon::unpack4(src, out)
}

/// # Safety
/// `src` valid for 128 reads, `out` valid for 128 `u32` writes.
#[no_mangle]
pub unsafe extern "C" fn unpack8_neon(src: *const u8, out: *mut u32) {
    neon::unpack8(src, out)
}

/// # Safety
/// `src` valid for 128 `u16` reads, `out` valid for 128 `u32` writes.
#[no_mangle]
pub unsafe extern "C" fn unpack16_neon(src: *const u16, out: *mut u32) {
    neon::unpack16(src, out)
}

/// # Safety
/// `src` valid for 384 reads, `out` valid for 128 `u32` writes.
#[no_mangle]
pub unsafe extern "C" fn unpack24_neon(src: *const u8, out: *mut u32) {
    neon::unpack24(src, out)
}

/// # Safety
/// `src` valid for 128 `u32` reads, `dst` valid for 64 writes.
#[no_mangle]
pub unsafe extern "C" fn pack4_neon(src: *const u32, dst: *mut u8) {
    neon::pack4(src, dst)
}

/// # Safety
/// `src` valid for 128 `u32` reads, `dst` valid for 128 writes.
#[no_mangle]
pub unsafe extern "C" fn pack8_neon(src: *const u32, dst: *mut u8) {
    neon::pack8(src, dst)
}

/// # Safety
/// `src` valid for 128 `u32` reads, `dst` valid for 128 `u16` writes.
#[no_mangle]
pub unsafe extern "C" fn pack16_neon(src: *const u32, dst: *mut u16) {
    neon::pack16(src, dst)
}

#[cfg(test)]
mod tests {
    use super::neon;

    include!("../tests/vectors.rs");

    fn run_unpack4(cases: &[Vector]) {
        for c in cases {
            let packed_bytes = (128 * 4 + 7) / 8;
            let mut src = c.packed.to_vec();
            src.resize(packed_bytes, 0);
            let mut out = [0u32; 128];
            unsafe { neon::unpack4(src.as_ptr(), out.as_mut_ptr()) };
            assert_eq!(out, c.vals, "unpack4 mismatch");
        }
    }

    fn run_pack4(cases: &[Vector]) {
        for c in cases {
            let mut dst = vec![0u8; (128 * 4 + 7) / 8];
            unsafe { neon::pack4(c.vals.as_ptr(), dst.as_mut_ptr()) };
            assert_eq!(dst, c.packed, "pack4 mismatch");
        }
    }

    fn run_unpack8(cases: &[Vector]) {
        for c in cases {
            let mut src = c.packed.to_vec();
            src.resize(128, 0);
            let mut out = [0u32; 128];
            unsafe { neon::unpack8(src.as_ptr(), out.as_mut_ptr()) };
            assert_eq!(out, c.vals, "unpack8 mismatch");
        }
    }

    fn run_pack8(cases: &[Vector]) {
        for c in cases {
            let mut dst = vec![0u8; 128];
            unsafe { neon::pack8(c.vals.as_ptr(), dst.as_mut_ptr()) };
            assert_eq!(dst, c.packed, "pack8 mismatch");
        }
    }

    fn run_unpack16(cases: &[Vector]) {
        for c in cases {
            let mut src16 = [0u16; 128];
            for (i, chunk) in c.packed.chunks(2).enumerate() {
                src16[i] = chunk[0] as u16 | ((chunk[1] as u16) << 8);
            }
            let mut out = [0u32; 128];
            unsafe { neon::unpack16(src16.as_ptr(), out.as_mut_ptr()) };
            assert_eq!(out, c.vals, "unpack16 mismatch");
        }
    }

    fn run_pack16(cases: &[Vector]) {
        for c in cases {
            let mut dst = vec![0u16; 128];
            unsafe { neon::pack16(c.vals.as_ptr(), dst.as_mut_ptr()) };
            let mut dst_bytes = vec![0u8; 256];
            for (i, v) in dst.iter().enumerate() {
                dst_bytes[i * 2] = (*v & 0xFF) as u8;
                dst_bytes[i * 2 + 1] = (*v >> 8) as u8;
            }
            assert_eq!(dst_bytes, c.packed, "pack16 mismatch");
        }
    }

    fn run_unpack24(cases: &[Vector]) {
        for c in cases {
            let mut src = c.packed.to_vec();
            src.resize(384 + 16, 0); // padding for the over-read tail load
            let mut out = [0u32; 128];
            unsafe { neon::unpack24(src.as_ptr(), out.as_mut_ptr()) };
            assert_eq!(out, c.vals, "unpack24 mismatch");
        }
    }

    #[test]
    fn test_unpack4_matches_go_golden_vectors() {
        run_unpack4(BITS4);
    }

    #[test]
    fn test_pack4_matches_go_golden_vectors() {
        run_pack4(BITS4);
    }

    #[test]
    fn test_unpack8_matches_go_golden_vectors() {
        run_unpack8(BITS8);
    }

    #[test]
    fn test_pack8_matches_go_golden_vectors() {
        run_pack8(BITS8);
    }

    #[test]
    fn test_unpack16_matches_go_golden_vectors() {
        run_unpack16(BITS16);
    }

    #[test]
    fn test_pack16_matches_go_golden_vectors() {
        run_pack16(BITS16);
    }

    #[test]
    fn test_unpack24_matches_go_golden_vectors() {
        run_unpack24(BITS24);
    }
}
