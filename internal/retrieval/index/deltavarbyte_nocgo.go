//go:build !cgo || !amd64

package index

import "github.com/ronanh/intcomp"

// decodeDeltaVarByteRust falls back to the pure-Go intcomp implementation on
// platforms without the amd64+cgo Rust build (see deltavarbyte_cgo.go). Same
// signature and behavior as the Rust-backed version, just without the ~1.9x
// decode speedup measured by BenchmarkDeltaVarByteDecode on amd64.
func decodeDeltaVarByteRust(in, out []uint64) ([]uint64, []uint64) {
	return intcomp.UncompressDeltaVarByteUint64(in, out)
}
