package index

import (
	"reflect"
	"testing"

	"github.com/ronanh/intcomp"
)

// deltaVarByteTestCases exercises the shapes decodeDeltaVarByteRust must
// handle: empty input, a single value, small sequential deltas (1-byte
// varints), and large deltas (multi-byte varints) — matching the varint
// continuation-bit format intcomp.CompressDeltaVarByteUint64 produces.
func deltaVarByteTestCases() map[string][]uint64 {
	return map[string][]uint64{
		"empty":  {},
		"single": {42},
		"small_sequential": func() []uint64 {
			vals := make([]uint64, 200)
			for i := range vals {
				vals[i] = uint64(i)
			}
			return vals
		}(),
		"large_deltas": func() []uint64 {
			vals := make([]uint64, 50)
			var cur uint64
			for i := range vals {
				cur += 1 << 40 // forces multi-byte varint deltas
				vals[i] = cur
			}
			return vals
		}(),
		"mixed": func() []uint64 {
			vals := make([]uint64, 300)
			var cur uint64
			for i := range vals {
				if i%10 == 0 {
					cur += 1 << 30
				} else {
					cur += uint64(i % 5)
				}
				vals[i] = cur
			}
			return vals
		}(),
	}
}

// TestDecodeDeltaVarByteRust_MatchesIntcompReference cross-checks the Rust
// decoder against the real ronanh/intcomp encoder+decoder: it encodes with
// the actual library, decodes with both the library's own decoder and ours,
// and requires identical output. This is the format-compatibility oracle —
// UncompressDeltaVarByteUint64's on-disk format is owned by intcomp, not us.
func TestDecodeDeltaVarByteRust_MatchesIntcompReference(t *testing.T) {
	for name, vals := range deltaVarByteTestCases() {
		t.Run(name, func(t *testing.T) {
			encoded := intcomp.CompressDeltaVarByteUint64(vals, nil)

			wantRemaining, wantOut := intcomp.UncompressDeltaVarByteUint64(encoded, nil)
			gotRemaining, gotOut := decodeDeltaVarByteRust(encoded, nil)

			if !reflect.DeepEqual(gotOut, wantOut) {
				t.Errorf("decodeDeltaVarByteRust(%s) output = %v, want %v", name, gotOut, wantOut)
			}
			if !reflect.DeepEqual(gotRemaining, wantRemaining) {
				t.Errorf("decodeDeltaVarByteRust(%s) remaining = %v, want %v", name, gotRemaining, wantRemaining)
			}
		})
	}
}

// TestDecodeDeltaVarByteRust_AppendsToExistingOut verifies the out slice is
// appended to (not overwritten), matching intcomp.UncompressDeltaVarByteUint64's
// documented behavior ("append the result to out").
func TestDecodeDeltaVarByteRust_AppendsToExistingOut(t *testing.T) {
	vals := []uint64{10, 20, 30}
	encoded := intcomp.CompressDeltaVarByteUint64(vals, nil)

	prefix := []uint64{999}
	_, got := decodeDeltaVarByteRust(encoded, prefix)

	want := []uint64{999, 10, 20, 30}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("decodeDeltaVarByteRust with existing out = %v, want %v", got, want)
	}
}

// BenchmarkDeltaVarByteDecode compares the Rust FFI decoder against the pure-Go
// intcomp.UncompressDeltaVarByteUint64 on identical input, isolating exactly
// the decode-time win (or lack thereof) from switching implementations.
func BenchmarkDeltaVarByteDecode(b *testing.B) {
	vals := make([]uint64, 2000)
	var cur uint64
	for i := range vals {
		cur += uint64(i%997) + 1 // realistic skewed small-to-medium deltas
		vals[i] = cur
	}
	encoded := intcomp.CompressDeltaVarByteUint64(vals, nil)

	b.Run("Go/intcomp", func(b *testing.B) {
		out := make([]uint64, 0, len(vals))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = intcomp.UncompressDeltaVarByteUint64(encoded, out[:0])
		}
	})

	b.Run("Rust", func(b *testing.B) {
		out := make([]uint64, 0, len(vals))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = decodeDeltaVarByteRust(encoded, out[:0])
		}
	})
}

// TestUnpackBlock_MatchesIntcompReference checks that PackBlock/UnpackBlock
// (and the scratch variant), which route delta-varbyte blocks through
// decodeDeltaVarByteRust, produce exactly what intcomp.UncompressUint64
// produces for the same PackBlock output.
func TestUnpackBlock_MatchesIntcompReference(t *testing.T) {
	blocks := map[string][]uint64{
		"zeros":      make([]uint64, BlockSize),
		"small":      make([]uint64, BlockSize),
		"multi_byte": make([]uint64, BlockSize),
		"max_u32":    make([]uint64, BlockSize),
	}
	for i := 0; i < BlockSize; i++ {
		blocks["small"][i] = uint64(i%37) + 1
		blocks["multi_byte"][i] = uint64(i) * 70000
		blocks["max_u32"][i] = 1<<32 - 1
	}

	packBuf := make([]byte, blockPackBufSize)
	for name, vals := range blocks {
		n := PackBlock(vals, packBuf)

		want := intcomp.UncompressUint64(intcomp.CompressUint64(vals, nil), nil)

		got := make([]uint64, BlockSize)
		if consumed := UnpackBlock(packBuf[:n], got); consumed != n {
			t.Errorf("%s: UnpackBlock consumed %d bytes, want %d", name, consumed, n)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: UnpackBlock = %v, want %v", name, got, want)
		}

		gotScratch := make([]uint64, BlockSize)
		scratch := make([]uint64, compBlockMaxWords)
		if consumed := UnpackBlockWithScratch(packBuf[:n], gotScratch, scratch); consumed != n {
			t.Errorf("%s: UnpackBlockWithScratch consumed %d bytes, want %d", name, consumed, n)
		}
		if !reflect.DeepEqual(gotScratch, want) {
			t.Errorf("%s: UnpackBlockWithScratch = %v, want %v", name, gotScratch, want)
		}
	}
}
