//! Rust port of github.com/ronanh/intcomp's UncompressDeltaVarByteUint64.
//!
//! This is NOT a SIMD kernel like blockpack_neon: the format is a
//! variable-byte (LEB128-style) delta codec with a data-dependent
//! continuation bit per byte, so byte boundaries aren't known ahead of
//! decoding and the loop can't be vectorized the way fixed-width bit-packing
//! can. The win here is from removing Go's bounds checks, GC interaction and
//! function-call overhead on a hot scalar loop, not from parallelism.
//!
//! Wire format (must match intcomp's CompressDeltaVarByteUint64 exactly):
//! payload bytes are packed 8-per-u64 word, MSB-first within each word (the
//! byte at bit position 56 is the first byte of the word). Each value is
//! encoded as a little-endian-style base-128 varint of its delta from the
//! previous value: 7 payload bits per byte, high bit set means "more bytes
//! follow". The caller is responsible for stripping the header word (which
//! encodes how many words to consume and how many values to produce) before
//! calling this function — see decodeDeltaVarByteRust in
//! internal/retrieval/index/deltavarbyte_cgo.go.

/// Decodes delta-varbyte-encoded payload words into `out_len` u64 values.
///
/// # Safety
/// `in_ptr` must be valid for `in_words` reads, `out_ptr` valid for
/// `out_len` writes. Both must be non-overlapping. `in_ptr`/`in_words` must
/// be a well-formed encoding produced by intcomp's CompressDeltaVarByteUint64
/// (payload only, header word already stripped by the caller) — malformed
/// input may produce fewer than `out_len` decoded values (the remainder of
/// `out` is left untouched) but will not write out of bounds or panic.
#[no_mangle]
pub unsafe extern "C" fn decode_delta_varbyte_u64(
    in_ptr: *const u64,
    in_words: usize,
    out_ptr: *mut u64,
    out_len: usize,
) {
    if in_words == 0 || out_len == 0 {
        return;
    }
    let input = std::slice::from_raw_parts(in_ptr, in_words);
    let output = std::slice::from_raw_parts_mut(out_ptr, out_len);

    let mut inpos: usize = 0;
    let mut outpos: usize = 0;
    let mut shift_in: u32 = 56;
    let mut init_offset: u64 = 0;
    let mut delta: u64 = 0;
    let mut shift_out: u32 = 0;

    while inpos < in_words && outpos < out_len {
        let c = input[inpos] >> shift_in;

        if shift_in < 8 {
            shift_in = 56;
            inpos += 1;
        } else {
            shift_in -= 8;
        }

        delta = delta.wrapping_add((c & 0x7F).wrapping_shl(shift_out));
        shift_out += 7;

        if c & 0x80 == 0 {
            shift_out = 0;
            let val = delta.wrapping_add(init_offset);
            output[outpos] = val;
            init_offset = val;
            outpos += 1;
            delta = 0;
        }
    }
}
