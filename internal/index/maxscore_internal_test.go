package index

import "testing"

// TestFindPivotIsRankSafe checks the MaxScore pivot: the head entries[:p]
// may only be skipped while their combined upper bound stays below θ, since
// a document matching only head terms could otherwise still enter the
// top-K. (A previous version picked the largest i whose *tail* sum reached
// θ instead, silently dropping such documents.)
func TestFindPivotIsRankSafe(t *testing.T) {
	// Four terms, ub-ascending: 1, 2, 3, 10 → suffixUB = [16, 15, 13, 10].
	// Head sums: p=1 → 1, p=2 → 3, p=3 → 6, p=4 → 16.
	suffixUB := []float64{16, 15, 13, 10}

	tests := []struct {
		name  string
		theta float64
		want  int
	}{
		{"theta zero: every term essential", 0, 0},
		{"theta 0.5: even the lowest term alone could beat it", 0.5, 0},
		{"theta 2: head {1} sums to 1 < 2", 2, 1},
		{"theta 5: head {1,2} sums to 3 < 5, adding 3 reaches 6", 5, 2},
		{"theta 6: head {1,2,3} sums to exactly 6, not below", 6, 2},
		{"theta 10: head {1,2,3} sums to 6 < 10", 10, 3},
		{"theta 16: all terms together only tie it", 16, 3},
		{"theta 17: nothing can beat it", 17, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := findPivot(suffixUB, tt.theta); got != tt.want {
				t.Errorf("findPivot(%v, %v) = %d, want %d", suffixUB, tt.theta, got, tt.want)
			}
		})
	}
}
