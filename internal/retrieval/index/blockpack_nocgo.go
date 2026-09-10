//go:build !cgo || !amd64

package index

// UnpackFOR32Into decodes exactly BlockSize uint32 values from src (written by
// PackFOR32) into out as uint64. Returns the number of bytes consumed from src.
// Specialized paths for common bit widths avoid the generic shift loop.
func UnpackFOR32Into(src []byte, out []uint64) int {
	bits := src[0]
	if bits == 0 {
		for i := range out[:BlockSize] {
			out[i] = 0
		}
		return 1
	}

	nBytes := (BlockSize*int(bits) + 7) / 8
	data := src[1:]

	switch bits {
	case 1: // 128 values × 1 bit = 16 bytes; 8 values per byte
		for g := 0; g < 16; g++ {
			b := data[g]
			out[g*8+0] = uint64(b & 1)
			out[g*8+1] = uint64((b >> 1) & 1)
			out[g*8+2] = uint64((b >> 2) & 1)
			out[g*8+3] = uint64((b >> 3) & 1)
			out[g*8+4] = uint64((b >> 4) & 1)
			out[g*8+5] = uint64((b >> 5) & 1)
			out[g*8+6] = uint64((b >> 6) & 1)
			out[g*8+7] = uint64(b >> 7)
		}
	case 2: // 32 bytes; 8 values per 2 bytes
		for g := 0; g < 16; g++ {
			o := g * 2
			w := uint32(data[o]) | uint32(data[o+1])<<8
			out[g*8+0] = uint64(w & 3)
			out[g*8+1] = uint64((w >> 2) & 3)
			out[g*8+2] = uint64((w >> 4) & 3)
			out[g*8+3] = uint64((w >> 6) & 3)
			out[g*8+4] = uint64((w >> 8) & 3)
			out[g*8+5] = uint64((w >> 10) & 3)
			out[g*8+6] = uint64((w >> 12) & 3)
			out[g*8+7] = uint64((w >> 14) & 3)
		}
	case 3: // 48 bytes; 8 values per 3 bytes
		for g := 0; g < 16; g++ {
			o := g * 3
			w := uint32(data[o]) | uint32(data[o+1])<<8 | uint32(data[o+2])<<16
			out[g*8+0] = uint64(w & 7)
			out[g*8+1] = uint64((w >> 3) & 7)
			out[g*8+2] = uint64((w >> 6) & 7)
			out[g*8+3] = uint64((w >> 9) & 7)
			out[g*8+4] = uint64((w >> 12) & 7)
			out[g*8+5] = uint64((w >> 15) & 7)
			out[g*8+6] = uint64((w >> 18) & 7)
			out[g*8+7] = uint64((w >> 21) & 7)
		}
	case 4: // 64 bytes; two nibbles per byte
		b := data[:nBytes]
		for i := 0; i < BlockSize/2; i++ {
			out[i*2] = uint64(b[i] & 0x0F)
			out[i*2+1] = uint64(b[i] >> 4)
		}
	case 5: // 80 bytes; 8 values per 5 bytes (fits in uint64)
		for g := 0; g < 16; g++ {
			o := g * 5
			w := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32
			out[g*8+0] = w & 31
			out[g*8+1] = (w >> 5) & 31
			out[g*8+2] = (w >> 10) & 31
			out[g*8+3] = (w >> 15) & 31
			out[g*8+4] = (w >> 20) & 31
			out[g*8+5] = (w >> 25) & 31
			out[g*8+6] = (w >> 30) & 31
			out[g*8+7] = (w >> 35) & 31
		}
	case 6: // 96 bytes; 8 values per 6 bytes
		for g := 0; g < 16; g++ {
			o := g * 6
			w := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40
			out[g*8+0] = w & 63
			out[g*8+1] = (w >> 6) & 63
			out[g*8+2] = (w >> 12) & 63
			out[g*8+3] = (w >> 18) & 63
			out[g*8+4] = (w >> 24) & 63
			out[g*8+5] = (w >> 30) & 63
			out[g*8+6] = (w >> 36) & 63
			out[g*8+7] = (w >> 42) & 63
		}
	case 7: // 112 bytes; 8 values per 7 bytes
		for g := 0; g < 16; g++ {
			o := g * 7
			w := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40 |
				uint64(data[o+6])<<48
			out[g*8+0] = w & 127
			out[g*8+1] = (w >> 7) & 127
			out[g*8+2] = (w >> 14) & 127
			out[g*8+3] = (w >> 21) & 127
			out[g*8+4] = (w >> 28) & 127
			out[g*8+5] = (w >> 35) & 127
			out[g*8+6] = (w >> 42) & 127
			out[g*8+7] = (w >> 49) & 127
		}
	case 8: // 128 bytes; one value per byte
		for i := range out[:BlockSize] {
			out[i] = uint64(data[i])
		}
	case 9: // 144 bytes; 8 values per 9 bytes; v7 spans byte 63-64 (lo/hi boundary)
		for g := 0; g < 16; g++ {
			o := g * 9
			lo := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40 |
				uint64(data[o+6])<<48 | uint64(data[o+7])<<56
			hi := uint64(data[o+8])
			out[g*8+0] = lo & 0x1FF
			out[g*8+1] = (lo >> 9) & 0x1FF
			out[g*8+2] = (lo >> 18) & 0x1FF
			out[g*8+3] = (lo >> 27) & 0x1FF
			out[g*8+4] = (lo >> 36) & 0x1FF
			out[g*8+5] = (lo >> 45) & 0x1FF
			out[g*8+6] = (lo >> 54) & 0x1FF
			out[g*8+7] = ((lo >> 63) | (hi << 1)) & 0x1FF
		}
	case 10: // 160 bytes; v6 spans lo/hi
		for g := 0; g < 16; g++ {
			o := g * 10
			lo := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40 |
				uint64(data[o+6])<<48 | uint64(data[o+7])<<56
			hi := uint64(data[o+8]) | uint64(data[o+9])<<8
			out[g*8+0] = lo & 0x3FF
			out[g*8+1] = (lo >> 10) & 0x3FF
			out[g*8+2] = (lo >> 20) & 0x3FF
			out[g*8+3] = (lo >> 30) & 0x3FF
			out[g*8+4] = (lo >> 40) & 0x3FF
			out[g*8+5] = (lo >> 50) & 0x3FF
			out[g*8+6] = ((lo >> 60) | (hi << 4)) & 0x3FF
			out[g*8+7] = (hi >> 6) & 0x3FF
		}
	case 11: // 176 bytes; v5 spans lo/hi
		for g := 0; g < 16; g++ {
			o := g * 11
			lo := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40 |
				uint64(data[o+6])<<48 | uint64(data[o+7])<<56
			hi := uint64(data[o+8]) | uint64(data[o+9])<<8 | uint64(data[o+10])<<16
			out[g*8+0] = lo & 0x7FF
			out[g*8+1] = (lo >> 11) & 0x7FF
			out[g*8+2] = (lo >> 22) & 0x7FF
			out[g*8+3] = (lo >> 33) & 0x7FF
			out[g*8+4] = (lo >> 44) & 0x7FF
			out[g*8+5] = ((lo >> 55) | (hi << 9)) & 0x7FF
			out[g*8+6] = (hi >> 2) & 0x7FF
			out[g*8+7] = (hi >> 13) & 0x7FF
		}
	case 12: // 192 bytes; v5 spans lo/hi
		for g := 0; g < 16; g++ {
			o := g * 12
			lo := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40 |
				uint64(data[o+6])<<48 | uint64(data[o+7])<<56
			hi := uint64(data[o+8]) | uint64(data[o+9])<<8 |
				uint64(data[o+10])<<16 | uint64(data[o+11])<<24
			out[g*8+0] = lo & 0xFFF
			out[g*8+1] = (lo >> 12) & 0xFFF
			out[g*8+2] = (lo >> 24) & 0xFFF
			out[g*8+3] = (lo >> 36) & 0xFFF
			out[g*8+4] = (lo >> 48) & 0xFFF
			out[g*8+5] = ((lo >> 60) | (hi << 4)) & 0xFFF
			out[g*8+6] = (hi >> 8) & 0xFFF
			out[g*8+7] = (hi >> 20) & 0xFFF
		}
	case 13: // 208 bytes; v4 spans lo/hi
		for g := 0; g < 16; g++ {
			o := g * 13
			lo := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40 |
				uint64(data[o+6])<<48 | uint64(data[o+7])<<56
			hi := uint64(data[o+8]) | uint64(data[o+9])<<8 | uint64(data[o+10])<<16 |
				uint64(data[o+11])<<24 | uint64(data[o+12])<<32
			out[g*8+0] = lo & 0x1FFF
			out[g*8+1] = (lo >> 13) & 0x1FFF
			out[g*8+2] = (lo >> 26) & 0x1FFF
			out[g*8+3] = (lo >> 39) & 0x1FFF
			out[g*8+4] = ((lo >> 52) | (hi << 12)) & 0x1FFF
			out[g*8+5] = (hi >> 1) & 0x1FFF
			out[g*8+6] = (hi >> 14) & 0x1FFF
			out[g*8+7] = (hi >> 27) & 0x1FFF
		}
	case 14: // 224 bytes; v4 spans lo/hi
		for g := 0; g < 16; g++ {
			o := g * 14
			lo := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40 |
				uint64(data[o+6])<<48 | uint64(data[o+7])<<56
			hi := uint64(data[o+8]) | uint64(data[o+9])<<8 | uint64(data[o+10])<<16 |
				uint64(data[o+11])<<24 | uint64(data[o+12])<<32 | uint64(data[o+13])<<40
			out[g*8+0] = lo & 0x3FFF
			out[g*8+1] = (lo >> 14) & 0x3FFF
			out[g*8+2] = (lo >> 28) & 0x3FFF
			out[g*8+3] = (lo >> 42) & 0x3FFF
			out[g*8+4] = ((lo >> 56) | (hi << 8)) & 0x3FFF
			out[g*8+5] = (hi >> 6) & 0x3FFF
			out[g*8+6] = (hi >> 20) & 0x3FFF
			out[g*8+7] = (hi >> 34) & 0x3FFF
		}
	case 15: // 240 bytes; v4 spans lo/hi
		for g := 0; g < 16; g++ {
			o := g * 15
			lo := uint64(data[o]) | uint64(data[o+1])<<8 | uint64(data[o+2])<<16 |
				uint64(data[o+3])<<24 | uint64(data[o+4])<<32 | uint64(data[o+5])<<40 |
				uint64(data[o+6])<<48 | uint64(data[o+7])<<56
			hi := uint64(data[o+8]) | uint64(data[o+9])<<8 | uint64(data[o+10])<<16 |
				uint64(data[o+11])<<24 | uint64(data[o+12])<<32 | uint64(data[o+13])<<40 |
				uint64(data[o+14])<<48
			out[g*8+0] = lo & 0x7FFF
			out[g*8+1] = (lo >> 15) & 0x7FFF
			out[g*8+2] = (lo >> 30) & 0x7FFF
			out[g*8+3] = (lo >> 45) & 0x7FFF
			out[g*8+4] = ((lo >> 60) | (hi << 4)) & 0x7FFF
			out[g*8+5] = (hi >> 11) & 0x7FFF
			out[g*8+6] = (hi >> 26) & 0x7FFF
			out[g*8+7] = (hi >> 41) & 0x7FFF
		}
	case 16: // 256 bytes; two bytes per value, little-endian
		b := data[:nBytes]
		for i := range out[:BlockSize] {
			out[i] = uint64(b[i*2]) | uint64(b[i*2+1])<<8
		}
	case 24: // 384 bytes; three bytes per value, little-endian
		b := data[:nBytes]
		for i := range out[:BlockSize] {
			out[i] = uint64(b[i*3]) | uint64(b[i*3+1])<<8 | uint64(b[i*3+2])<<16
		}
	default:
		unpackBits32(data, bits, out)
	}
	return 1 + nBytes
}
