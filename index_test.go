package fasthnsw

import (
	"bytes"
	"errors"
	"testing"
)

func TestBuildStoresVectorsAndConstructsGraph(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	if idx.dim != 2 {
		t.Fatalf("idx.dim = %d, want 2", idx.dim)
	}
	if idx.count != 2 {
		t.Fatalf("idx.count = %d, want 2", idx.count)
	}
	if len(idx.vectors) != 4 {
		t.Fatalf("len(idx.vectors) = %d, want 4", len(idx.vectors))
	}
	if idx.vectors[0] != 1 || idx.vectors[3] != 4 {
		t.Fatalf("stored vectors = %v, want flattened input", idx.vectors)
	}
	if !idx.graphReady {
		t.Fatal("idx.graphReady = false, want true")
	}
	if idx.entryPoint < 0 {
		t.Fatalf("idx.entryPoint = %d, want non-negative", idx.entryPoint)
	}
	if len(idx.layers) == 0 {
		t.Fatal("idx.layers is empty")
	}
	if len(idx.levels) != idx.count {
		t.Fatalf("len(idx.levels) = %d, want %d", len(idx.levels), idx.count)
	}
}

func TestBuildInfersDimension(t *testing.T) {
	idx, err := New(Config{})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{1, 2, 3}, {4, 5, 6}})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	if idx.dim != 3 {
		t.Fatalf("idx.dim = %d, want 3", idx.dim)
	}
	if idx.cfg.Dim != 3 {
		t.Fatalf("idx.cfg.Dim = %d, want 3", idx.cfg.Dim)
	}
}

func TestBuildStoresNormalizedCosineVectors(t *testing.T) {
	idx, err := New(Config{Metric: MetricCosine})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{3, 4}})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	if !almostEqual(idx.vectors[0], 0.6) || !almostEqual(idx.vectors[1], 0.8) {
		t.Fatalf("stored cosine vector = %v, want normalized [0.6 0.8]", idx.vectors)
	}
}

func TestBuildRejectsInvalidVectors(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tests := []struct {
		name    string
		vectors [][]float32
	}{
		{name: "empty", vectors: nil},
		{name: "zero dim", vectors: [][]float32{{}}},
		{name: "mismatch", vectors: [][]float32{{1, 2}, {3}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := idx.Build(tt.vectors)
			if err == nil {
				t.Fatal("Build returned nil error")
			}
		})
	}
}

func TestBuildRejectsCosineZeroVector(t *testing.T) {
	idx, err := New(Config{Metric: MetricCosine, Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{1, 0}, {0, 0}})
	if err == nil {
		t.Fatal("Build returned nil error")
	}
}

func TestSearchValidatesInputsBeforeGraphCheck(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tests := []struct {
		name     string
		query    []float32
		k        int
		efSearch int
	}{
		{name: "empty query", query: nil, k: 1, efSearch: 2},
		{name: "wrong dimension", query: []float32{1}, k: 1, efSearch: 2},
		{name: "bad k", query: []float32{1, 2}, k: 0, efSearch: 2},
		{name: "bad efSearch", query: []float32{1, 2}, k: 1, efSearch: 0},
		{name: "efSearch less than k", query: []float32{1, 2}, k: 2, efSearch: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := idx.Search(tt.query, tt.k, tt.efSearch)
			if err == nil {
				t.Fatal("Search returned nil error")
			}
		})
	}
}

func TestSearchUnbuiltIndexReturnsNotBuilt(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = idx.Search([]float32{1, 2}, 1, 1)
	if !errors.Is(err, ErrIndexNotBuilt) {
		t.Fatalf("Search error = %v, want ErrIndexNotBuilt", err)
	}
}

func TestSearchTrustsReadyGraphInHotPath(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	err = idx.Build([][]float32{{0, 0}, {1, 0}})
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}

	idx.layers = [][][]int{{{1}}}
	idx.entryPoint = 0
	idx.maxLayer = 0
	idx.graphReady = true

	got, err := idx.Search([]float32{0, 0}, 1, 1)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	assertResults(t, got, []Result{{ID: 0, Distance: 0}})
}

func TestSaveAndLoadReturnNotImplemented(t *testing.T) {
	idx, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if err := idx.Save(bytes.NewBuffer(nil)); !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Save error = %v, want ErrNotImplemented", err)
	}
	if _, err := Load(bytes.NewBuffer(nil)); !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Load error = %v, want ErrNotImplemented", err)
	}
}

func TestSaveAndLoadRejectNilIO(t *testing.T) {
	idx, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if err := idx.Save(nil); err == nil || errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Save(nil) error = %v, want validation error", err)
	}
	if _, err := Load(nil); err == nil || errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Load(nil) error = %v, want validation error", err)
	}
}
