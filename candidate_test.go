package fasthnsw

import "testing"

func TestExactCandidates(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{3, 0},
		{0, 2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := exactCandidates(flat, dim, MetricL2, 2)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}

	want := [][]candidate{
		{{id: 1, distance: 1}, {id: 3, distance: 4}},
		{{id: 0, distance: 1}, {id: 2, distance: 4}},
		{{id: 1, distance: 4}, {id: 0, distance: 9}},
		{{id: 0, distance: 4}, {id: 1, distance: 5}},
	}
	assertCandidates(t, got, want)
}

func TestExactCandidatesExcludesSelfAndTruncatesK(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0},
		{1},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := exactCandidates(flat, dim, MetricL2, 10)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("len(got) = %d, want 3", len(got))
	}
	for sourceID, candidates := range got {
		if len(candidates) != 2 {
			t.Fatalf("len(got[%d]) = %d, want 2", sourceID, len(candidates))
		}
		for _, candidate := range candidates {
			if candidate.id == sourceID {
				t.Fatalf("got[%d] contains self candidate: %+v", sourceID, candidates)
			}
		}
	}
}

func TestExactCandidatesTieBreaksByID(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0},
		{-1},
		{1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := exactCandidates(flat, dim, MetricL2, 2)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}
	want := [][]candidate{
		{{id: 1, distance: 1}, {id: 2, distance: 1}},
		{{id: 0, distance: 1}, {id: 2, distance: 4}},
		{{id: 0, distance: 1}, {id: 1, distance: 4}},
	}
	assertCandidates(t, got, want)
}

func TestExactCandidatesEmptyAndSingleVector(t *testing.T) {
	got, err := exactCandidates(nil, 2, MetricL2, 3)
	if err != nil {
		t.Fatalf("exactCandidates empty returned error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("len(empty candidates) = %d, want 0", len(got))
	}

	flat, dim, err := flattenVectors([][]float32{{1, 2}}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}
	got, err = exactCandidates(flat, dim, MetricL2, 3)
	if err != nil {
		t.Fatalf("exactCandidates single returned error: %v", err)
	}
	if len(got) != 1 || len(got[0]) != 0 {
		t.Fatalf("single-vector candidates = %v, want one empty list", got)
	}
}

func TestExactCandidatesRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name    string
		vectors []float32
		dim     int
		k       int
	}{
		{name: "bad dim", vectors: []float32{1, 2}, dim: 0, k: 1},
		{name: "unaligned storage", vectors: []float32{1, 2, 3}, dim: 2, k: 1},
		{name: "bad k", vectors: []float32{1, 2}, dim: 2, k: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := exactCandidates(tt.vectors, tt.dim, MetricL2, tt.k); err == nil {
				t.Fatal("exactCandidates returned nil error")
			}
		})
	}
}

func TestExactCandidatesCosineUsesNormalizedVectors(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{10, 0},
		{1, 0},
		{0, 1},
	}, 0, MetricCosine)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := exactCandidates(flat, dim, MetricCosine, 2)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}
	want := [][]candidate{
		{{id: 1, distance: 0}, {id: 2, distance: 1}},
		{{id: 0, distance: 0}, {id: 2, distance: 1}},
		{{id: 0, distance: 1}, {id: 1, distance: 1}},
	}
	assertCandidates(t, got, want)
}

func assertCandidates(t *testing.T, got, want [][]candidate) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(candidates) = %d, want %d: %v", len(got), len(want), got)
	}
	for sourceID := range want {
		if len(got[sourceID]) != len(want[sourceID]) {
			t.Fatalf("len(candidates[%d]) = %d, want %d: %v", sourceID, len(got[sourceID]), len(want[sourceID]), got[sourceID])
		}
		for i := range want[sourceID] {
			if got[sourceID][i].id != want[sourceID][i].id {
				t.Fatalf("candidates[%d][%d].id = %d, want %d", sourceID, i, got[sourceID][i].id, want[sourceID][i].id)
			}
			if !almostEqual(got[sourceID][i].distance, want[sourceID][i].distance) {
				t.Fatalf("candidates[%d][%d].distance = %v, want %v", sourceID, i, got[sourceID][i].distance, want[sourceID][i].distance)
			}
		}
	}
}
