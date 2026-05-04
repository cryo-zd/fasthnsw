package core

import "testing"

func TestExactTopKL2(t *testing.T) {
	vectors := []float32{
		0, 0,
		2, 0,
		1, 0,
	}

	got, err := exactTopK(vectors, 2, MetricL2, []float32{0.9, 0}, 2)
	if err != nil {
		t.Fatalf("exactTopK returned error: %v", err)
	}

	want := []Result{
		{ID: 2, Distance: 0.010000004},
		{ID: 0, Distance: 0.80999994},
	}
	assertResults(t, got, want)
}

func TestExactTopKTieBreaksByID(t *testing.T) {
	vectors := []float32{
		1, 0,
		-1, 0,
	}

	got, err := exactTopK(vectors, 2, MetricL2, []float32{0, 0}, 2)
	if err != nil {
		t.Fatalf("exactTopK returned error: %v", err)
	}

	want := []Result{
		{ID: 0, Distance: 1},
		{ID: 1, Distance: 1},
	}
	assertResults(t, got, want)
}

func TestExactTopKCosineNormalizesQuery(t *testing.T) {
	vectors, dim, err := flattenVectors([][]float32{{1, 0}, {0, 1}}, 0, MetricCosine)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := exactTopK(vectors, dim, MetricCosine, []float32{10, 0}, 2)
	if err != nil {
		t.Fatalf("exactTopK returned error: %v", err)
	}

	want := []Result{
		{ID: 0, Distance: 0},
		{ID: 1, Distance: 1},
	}
	assertResults(t, got, want)
}

func TestExactTopKRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name    string
		vectors []float32
		dim     int
		query   []float32
		k       int
	}{
		{name: "bad dim", vectors: []float32{1, 2}, dim: 0, query: []float32{1}, k: 1},
		{name: "unaligned", vectors: []float32{1, 2, 3}, dim: 2, query: []float32{1, 2}, k: 1},
		{name: "bad query dim", vectors: []float32{1, 2}, dim: 2, query: []float32{1}, k: 1},
		{name: "bad k", vectors: []float32{1, 2}, dim: 2, query: []float32{1, 2}, k: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := exactTopK(tt.vectors, tt.dim, MetricL2, tt.query, tt.k); err == nil {
				t.Fatal("exactTopK returned nil error")
			}
		})
	}
}

func assertResults(t *testing.T, got, want []Result) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(results) = %d, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ID != want[i].ID {
			t.Fatalf("result %d ID = %d, want %d", i, got[i].ID, want[i].ID)
		}
		if !almostEqual(got[i].Distance, want[i].Distance) {
			t.Fatalf("result %d distance = %v, want %v", i, got[i].Distance, want[i].Distance)
		}
	}
}
