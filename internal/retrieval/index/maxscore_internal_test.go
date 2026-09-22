package index

import "testing"

// TestFindPivotSkipsLowUBTerms verifies that findPivot returns the LARGEST
// index i with suffixUB[i] >= θ, not just the first one it finds. Since
// entries are sorted ub-ascending, suffixUB is non-increasing, so
// suffixUB[0] (the sum of every term's UB) is always >= any smaller suffix
// sum — meaning "first i satisfying the condition" is always 0, which would
// make every term "essential" and disable block-max pruning entirely. The
// pivot must advance as far as the cumulative UB from the high-impact tail
// still clears θ, so the low-UB head terms can be treated as optional.
func TestFindPivotSkipsLowUBTerms(t *testing.T) {
	// Four terms, ub-ascending: 1, 2, 3, 10.
	// suffixUB = [16, 15, 13, 10].
	suffixUB := []float64{16, 15, 13, 10}

	tests := []struct {
		name  string
		theta float64
		want  int
	}{
		{"theta zero: every term mandatory", 0, 0},
		{"theta clearable by last term alone: only it is essential", 10, 3},
		{"theta needs last two terms combined", 12, 2},
		{"theta needs last three terms combined", 14, 1},
		{"theta needs all terms", 16, 0},
		{"theta unreachable by any combination: no pivot", 17, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := findPivot(suffixUB, tt.theta); got != tt.want {
				t.Errorf("findPivot(%v, %v) = %d, want %d", suffixUB, tt.theta, got, tt.want)
			}
		})
	}
}
