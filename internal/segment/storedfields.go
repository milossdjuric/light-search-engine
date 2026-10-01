package segment

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
)

// storedFieldEntry is one entry in the .seg.fld index section.
type storedFieldEntry struct {
	docID      string
	dataOffset uint32 // byte offset within the data section
	textLen    uint32
}

// fldMagic is the 4-byte magic for .seg.fld sidecar files.
var fldMagic = [4]byte{'S', 'F', 'L', 'D'}

const fldVersion = byte(1)

// GetText returns the original text for docID from the stored fields sidecar.
// Returns ("", false) if no sidecar exists or the doc is not found.
func (s *Segment) GetText(docID string) (string, bool) {
	if len(s.fldIndex) == 0 {
		return "", false
	}
	i := sort.Search(len(s.fldIndex), func(i int) bool {
		return s.fldIndex[i].docID >= docID
	})
	if i >= len(s.fldIndex) || s.fldIndex[i].docID != docID {
		return "", false
	}
	entry := s.fldIndex[i]
	if entry.textLen == 0 {
		return "", true
	}

	f, err := os.Open(s.fldPath)
	if err != nil {
		return "", false
	}
	defer f.Close()

	buf := make([]byte, entry.textLen)
	if _, err := f.ReadAt(buf, s.fldDataStart+int64(entry.dataOffset)); err != nil {
		return "", false
	}
	return string(buf), true
}

// WriteStoredFields writes a .seg.fld sidecar file mapping docID → original text.
// texts is a map from docID to document text.
// The index section is sorted by docID to allow binary search.
func WriteStoredFields(path string, texts map[string]string) error {
	if len(texts) == 0 {
		return nil
	}

	// Sort docIDs for binary-searchable index.
	docIDs := make([]string, 0, len(texts))
	for id := range texts {
		docIDs = append(docIDs, id)
	}
	sort.Strings(docIDs)

	// Compute per-entry data offsets.
	type entry struct {
		docID      string
		dataOffset uint32
		textLen    uint32
	}
	entries := make([]entry, len(docIDs))
	var offset uint32
	for i, id := range docIDs {
		t := texts[id]
		entries[i] = entry{id, offset, uint32(len(t))}
		offset += uint32(len(t))
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("WriteStoredFields create %s: %w", path, err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1<<20)

	// Header: magic + version + numEntries.
	if _, err := w.Write(fldMagic[:]); err != nil {
		return err
	}
	if err := w.WriteByte(fldVersion); err != nil {
		return err
	}
	var tmp [4]byte
	binary.LittleEndian.PutUint32(tmp[:], uint32(len(entries)))
	if _, err := w.Write(tmp[:]); err != nil {
		return err
	}

	// Index section.
	for _, e := range entries {
		if len(e.docID) > 255 {
			return fmt.Errorf("WriteStoredFields: docID too long (%d > 255): %s", len(e.docID), e.docID)
		}
		if err := w.WriteByte(byte(len(e.docID))); err != nil {
			return err
		}
		if _, err := w.WriteString(e.docID); err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(tmp[:], e.dataOffset)
		if _, err := w.Write(tmp[:]); err != nil {
			return err
		}
		binary.LittleEndian.PutUint32(tmp[:], e.textLen)
		if _, err := w.Write(tmp[:]); err != nil {
			return err
		}
	}

	// Data section: concatenated texts.
	for _, id := range docIDs {
		if _, err := w.WriteString(texts[id]); err != nil {
			return err
		}
	}

	return w.Flush()
}

// LoadStoredFields reads the index section of a .seg.fld sidecar into memory.
// The data section is read on demand in GetText via ReadAt.
// Returns nil error if path does not exist (sidecar is optional).
func LoadStoredFields(seg *Segment, path string) error {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil // sidecar absent — older segment or stored fields disabled
	}
	if err != nil {
		return fmt.Errorf("LoadStoredFields open %s: %w", path, err)
	}
	defer f.Close()

	// Read header: magic(4) + version(1) + numEntries(4) = 9 bytes.
	var hdr [9]byte
	if _, err := io.ReadFull(f, hdr[:]); err != nil {
		return fmt.Errorf("LoadStoredFields read header %s: %w", path, err)
	}
	if hdr[0] != fldMagic[0] || hdr[1] != fldMagic[1] || hdr[2] != fldMagic[2] || hdr[3] != fldMagic[3] {
		return fmt.Errorf("LoadStoredFields %s: bad magic", path)
	}
	// hdr[4] is version — ignore for forward compat
	n := int(binary.LittleEndian.Uint32(hdr[5:9]))

	// Read index section.
	fldIdx := make([]storedFieldEntry, n)
	for i := 0; i < n; i++ {
		// docIDLen uint8
		lenBuf := make([]byte, 1)
		if _, err := io.ReadFull(f, lenBuf); err != nil {
			return fmt.Errorf("LoadStoredFields read docIDLen %s[%d]: %w", path, i, err)
		}
		docIDLen := int(lenBuf[0])
		docIDBuf := make([]byte, docIDLen)
		if _, err := io.ReadFull(f, docIDBuf); err != nil {
			return fmt.Errorf("LoadStoredFields read docID %s[%d]: %w", path, i, err)
		}
		var offLen [8]byte
		if _, err := io.ReadFull(f, offLen[:]); err != nil {
			return fmt.Errorf("LoadStoredFields read offLen %s[%d]: %w", path, i, err)
		}
		fldIdx[i] = storedFieldEntry{
			docID:      string(docIDBuf),
			dataOffset: binary.LittleEndian.Uint32(offLen[0:4]),
			textLen:    binary.LittleEndian.Uint32(offLen[4:8]),
		}
	}

	// Data section starts at current file position.
	dataStart, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("LoadStoredFields seek %s: %w", path, err)
	}

	seg.fldPath = path
	seg.fldIndex = fldIdx
	seg.fldDataStart = dataStart
	return nil
}
