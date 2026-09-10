package index

import (
	"bufio"
	"os"
	"strings"
)

// SynonymMap holds query-time synonym expansions.
// Each token maps to all other tokens in its synonym group.
// Applied at query time only — never at index time.
type SynonymMap struct {
	expand map[string][]string
}

// LoadSynonyms reads a synonyms file and builds a SynonymMap.
// File format: one synonym group per line, terms separated by commas.
// Example: "car,automobile,vehicle,auto"
// Empty lines and lines starting with '#' are ignored.
// Returns nil (not an error) when path is empty.
func LoadSynonyms(path string) (*SynonymMap, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	expand := make(map[string][]string)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, ",")
		var terms []string
		for _, p := range parts {
			t := strings.TrimSpace(strings.ToLower(p))
			if t != "" {
				terms = append(terms, t)
			}
		}
		if len(terms) < 2 {
			continue // single-term group — no expansion needed
		}
		// Map each term → all other terms in the group.
		for i, term := range terms {
			others := make([]string, 0, len(terms)-1)
			for j, other := range terms {
				if i != j {
					others = append(others, other)
				}
			}
			// Merge with any existing mapping for this term.
			existing := expand[term]
		outer:
			for _, o := range others {
				for _, e := range existing {
					if e == o {
						continue outer // already present
					}
				}
				existing = append(existing, o)
			}
			expand[term] = existing
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return &SynonymMap{expand: expand}, nil
}

// Expand returns the input token followed by all its synonyms.
// If the token has no synonyms, the returned slice contains only the token.
// The returned slice is always a new allocation.
func (s *SynonymMap) Expand(token string) []string {
	if s == nil || len(s.expand) == 0 {
		return []string{token}
	}
	syns, ok := s.expand[token]
	if !ok || len(syns) == 0 {
		return []string{token}
	}
	result := make([]string, 1, 1+len(syns))
	result[0] = token
	result = append(result, syns...)
	return result
}
