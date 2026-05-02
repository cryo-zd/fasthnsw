package fasthnsw

import (
	"errors"
	"testing"
)

func TestSearchCompleteGraphMatchesExactTopK(t *testing.T) {
	vectors := [][]float32{
		{0, 0},
		{1, 0},
		{2, 0},
		{3, 0},
	}
	idx := newTestIndexWithGraph(t, MetricL2, vectors, [][][]int{completeLayer(len(vectors))}, 3, 0)

	got, err := idx.Search([]float32{0.1, 0}, 2, 4)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	want, err := exactTopK(idx.vectors, idx.dim, idx.cfg.Metric, []float32{0.1, 0}, 2)
	if err != nil {
		t.Fatalf("exactTopK returned error: %v", err)
	}
	assertResults(t, got, want)
}

func TestSearchTieBreaksByID(t *testing.T) {
	vectors := [][]float32{
		{1, 0},
		{-1, 0},
	}
	idx := newTestIndexWithGraph(t, MetricL2, vectors, [][][]int{completeLayer(len(vectors))}, 1, 0)

	got, err := idx.Search([]float32{0, 0}, 2, 2)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}

	want := []Result{
		{ID: 0, Distance: 1},
		{ID: 1, Distance: 1},
	}
	assertResults(t, got, want)
}

func TestSearchUsesUpperLayerGreedyDescent(t *testing.T) {
	vectors := [][]float32{
		{0, 0},
		{10, 0},
		{1, 0},
	}
	layers := [][][]int{
		{
			{2},
			nil,
			{0},
		},
		{
			nil,
			{2},
			nil,
		},
	}
	idx := newTestIndexWithGraph(t, MetricL2, vectors, layers, 1, 1)

	got, err := idx.Search([]float32{0, 0}, 1, 2)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	want := []Result{{ID: 0, Distance: 0}}
	assertResults(t, got, want)
}

func TestSearchRejectsCosineZeroQueryBeforeGraphCheck(t *testing.T) {
	idx, err := New(Config{Metric: MetricCosine, Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = idx.Search([]float32{0, 0}, 1, 1)
	if err == nil {
		t.Fatal("Search returned nil error")
	}
	if errors.Is(err, ErrIndexNotBuilt) {
		t.Fatalf("Search returned graph error before cosine query validation: %v", err)
	}
}

func TestValidateSearchableGraphRejectsMalformedLayers(t *testing.T) {
	idx := &Index{
		cfg:        DefaultConfig(),
		vectors:    []float32{0, 0, 1, 0},
		dim:        2,
		count:      2,
		layers:     [][][]int{{{1}}},
		entryPoint: 0,
		maxLayer:   0,
		graphReady: true,
	}

	err := idx.validateSearchableGraph()
	if err == nil {
		t.Fatal("validateSearchableGraph returned nil error")
	}
}

func TestValidateSearchableGraphRejectsOutOfRangeNeighbor(t *testing.T) {
	idx := &Index{
		cfg:        DefaultConfig(),
		vectors:    []float32{0, 0, 1, 0},
		dim:        2,
		count:      2,
		layers:     [][][]int{{{2}, {0}}},
		entryPoint: 0,
		maxLayer:   0,
		graphReady: true,
	}

	err := idx.validateSearchableGraph()
	if err == nil {
		t.Fatal("validateSearchableGraph returned nil error")
	}
}

func newTestIndexWithGraph(t *testing.T, metric Metric, vectors [][]float32, layers [][][]int, entryPoint int, maxLayer int) *Index {
	t.Helper()

	idx, err := New(Config{Metric: metric, Dim: len(vectors[0])})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	err = idx.Build(vectors)
	if !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Build error = %v, want ErrNotImplemented", err)
	}

	idx.layers = layers
	idx.entryPoint = entryPoint
	idx.maxLayer = maxLayer
	idx.graphReady = true

	if err := idx.validateSearchableGraph(); err != nil {
		t.Fatalf("test graph is invalid: %v", err)
	}
	return idx
}

func completeLayer(count int) [][]int {
	layer := make([][]int, count)
	for id := 0; id < count; id++ {
		for neighborID := 0; neighborID < count; neighborID++ {
			if neighborID != id {
				layer[id] = append(layer[id], neighborID)
			}
		}
	}
	return layer
}
