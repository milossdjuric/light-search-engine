package index

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

// EncodeDeltaList encodes a sorted slice of uint64 IDs as delta-compressed varints.
// The first value is stored as-is; subsequent values store the difference from the previous.
func EncodeDeltaList(ids []uint64) []byte {
	buf := make([]byte, 0, len(ids)*2)
	var prev uint64
	for _, id := range ids {
		buf = AppendVarint(buf, id-prev)
		prev = id
	}
	return buf
}

// DecodeDeltaList decodes a delta-compressed varint list back to the original IDs.
func DecodeDeltaList(data []byte) []uint64 {
	var ids []uint64
	var prev uint64
	offset := 0
	for offset < len(data) {
		delta, n := ReadVarint(data, offset)
		offset += n
		prev += delta
		ids = append(ids, prev)
	}
	return ids
}
