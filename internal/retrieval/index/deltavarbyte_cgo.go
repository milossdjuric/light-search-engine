//go:build cgo && amd64

package index

/*
#cgo LDFLAGS: -L${SRCDIR}/../../../native/deltavarbyte/target/x86_64-unknown-linux-gnu/release -ldeltavarbyte
#include <stdint.h>
#include <stddef.h>

// Prototype for the Rust decoder in native/deltavarbyte (built via
// `cargo build --release --target x86_64-unknown-linux-gnu`; see that
// crate's Cargo.toml — crate-type = ["staticlib"]). Hand-written rather
// than cbindgen-generated since the surface is a single function.
void decode_delta_varbyte_u64(const uint64_t *in_ptr, size_t in_words,
                               uint64_t *out_ptr, size_t out_len);
*/
import "C"
import "unsafe"

// decodeDeltaVarByteRust is a Rust-backed port of
// github.com/ronanh/intcomp's UncompressDeltaVarByteUint64, matching its
// exact signature and on-disk format (see native/deltavarbyte/src/lib.rs for
// the format documentation). Built for amd64+cgo only; other platforms use
// the intcomp-backed fallback in deltavarbyte_nocgo.go. It decodes one block
// from in (compressed with intcomp.CompressDeltaVarByteUint64) and appends
// the result to out, resizing out if necessary. Returns the unconsumed
// remainder of in and the updated out slice.
func decodeDeltaVarByteRust(in, out []uint64) ([]uint64, []uint64) {
	if len(in) == 0 {
		return in, out
	}

	outlen := int(int32(in[0]))
	inlen := int(in[0] >> 32)
	resin := in[inlen:]
	payload := in[1:inlen]

	if cap(out)-len(out) < outlen {
		tmpout := make([]uint64, len(out), len(out)+outlen)
		copy(tmpout, out)
		out = tmpout
	}
	start := len(out)
	out = out[:start+outlen]

	if outlen > 0 {
		var inPtr *C.uint64_t
		if len(payload) > 0 {
			inPtr = (*C.uint64_t)(unsafe.Pointer(&payload[0]))
		}
		C.decode_delta_varbyte_u64(
			inPtr, C.size_t(len(payload)),
			(*C.uint64_t)(unsafe.Pointer(&out[start])), C.size_t(outlen),
		)
	}

	return resin, out
}
