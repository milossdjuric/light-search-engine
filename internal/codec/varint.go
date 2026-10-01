package codec

// AppendVarint encodes x as LEB128 and appends the bytes to buf.
func AppendVarint(buf []byte, x uint64) []byte {
	for x >= 0x80 {
		buf = append(buf, byte(x)|0x80)
		x >>= 7
	}
	return append(buf, byte(x))
}

// ReadVarint decodes a LEB128 value from data starting at offset.
// Returns the decoded value and the number of bytes consumed.
func ReadVarint(data []byte, offset int) (value uint64, bytesRead int) {
	var x uint64
	var shift uint
	for {
		b := data[offset+bytesRead]
		bytesRead++
		x |= uint64(b&0x7F) << shift
		if b&0x80 == 0 {
			break
		}
		shift += 7
	}
	return x, bytesRead
}
