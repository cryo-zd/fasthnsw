package fasthnsw

import "testing"

func TestRNGPruneKeepsCandidatesWhenNoDominance(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{0, 1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := rngPrune(0, []candidate{{id: 1}, {id: 2}}, 4, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	want := []candidate{{id: 1, distance: 1}, {id: 2, distance: 1}}
	assertCandidateList(t, got, want)
}

func TestRNGPruneRemovesDominatedCollinearCandidate(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{2, 0},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := rngPrune(0, []candidate{{id: 2}, {id: 1}}, 4, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	want := []candidate{{id: 1, distance: 1}}
	assertCandidateList(t, got, want)
}

func TestRNGPruneHonorsMaxDegree(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0},
		{1},
		{-1},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := rngPrune(0, []candidate{{id: 3}, {id: 2}, {id: 1}}, 1, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	want := []candidate{{id: 1, distance: 1}}
	assertCandidateList(t, got, want)
}

func TestPruneNormalizesSelfDuplicatesAndStaleDistances(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0},
		{1},
		{2},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := rngPrune(0, []candidate{
		{id: 0, distance: -1},
		{id: 2, distance: 0},
		{id: 1, distance: 99},
		{id: 1, distance: 100},
	}, 4, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	want := []candidate{{id: 1, distance: 1}}
	assertCandidateList(t, got, want)
}

func TestAlphaPruneCanBeLessAggressiveThanRNG(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0, 0},
		{1, 0},
		{1, 1},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}
	candidates := []candidate{{id: 1}, {id: 2}}

	rng, err := rngPrune(0, candidates, 4, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("rngPrune returned error: %v", err)
	}
	alpha60, err := alphaPrune(0, candidates, 4, 60, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("alphaPrune(60) returned error: %v", err)
	}
	alpha120, err := alphaPrune(0, candidates, 4, 120, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("alphaPrune(120) returned error: %v", err)
	}

	assertCandidateList(t, rng, []candidate{{id: 1, distance: 1}})
	assertCandidateList(t, alpha60, rng)
	assertCandidateList(t, alpha120, []candidate{{id: 1, distance: 1}, {id: 2, distance: 2}})
}

func TestPruneRejectsInvalidInput(t *testing.T) {
	tests := []struct {
		name     string
		sourceID int
		vectors  []float32
		dim      int
		maxDeg   int
		alpha    float64
		metric   Metric
	}{
		{name: "bad max degree", sourceID: 0, vectors: []float32{0, 1}, dim: 1, maxDeg: 0, alpha: 60, metric: MetricL2},
		{name: "bad metric", sourceID: 0, vectors: []float32{0, 1}, dim: 1, maxDeg: 1, alpha: 60, metric: Metric(99)},
		{name: "bad dim", sourceID: 0, vectors: []float32{0, 1}, dim: 0, maxDeg: 1, alpha: 60, metric: MetricL2},
		{name: "unaligned storage", sourceID: 0, vectors: []float32{0, 1, 2}, dim: 2, maxDeg: 1, alpha: 60, metric: MetricL2},
		{name: "bad source", sourceID: 2, vectors: []float32{0, 1}, dim: 1, maxDeg: 1, alpha: 60, metric: MetricL2},
		{name: "bad alpha low", sourceID: 0, vectors: []float32{0, 1}, dim: 1, maxDeg: 1, alpha: 59, metric: MetricL2},
		{name: "bad alpha high", sourceID: 0, vectors: []float32{0, 1}, dim: 1, maxDeg: 1, alpha: 181, metric: MetricL2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := alphaPrune(tt.sourceID, []candidate{{id: 1}}, tt.maxDeg, tt.alpha, tt.vectors, tt.dim, tt.metric); err == nil {
				t.Fatal("alphaPrune returned nil error")
			}
		})
	}
}

func TestPruneRejectsOutOfRangeCandidate(t *testing.T) {
	_, err := rngPrune(0, []candidate{{id: 2}}, 1, []float32{0, 1}, 1, MetricL2)
	if err == nil {
		t.Fatal("rngPrune returned nil error")
	}
}

func TestAngleUWVGreaterThanAlphaHandlesDegenerateGeometry(t *testing.T) {
	flat, dim, err := flattenVectors([][]float32{
		{0, 0},
		{0, 0},
		{1, 0},
	}, 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	if angleUWVGreaterThanAlpha(0, 1, 2, flat, dim, -0.5) {
		t.Fatal("degenerate angle unexpectedly satisfied alpha condition")
	}
}

func assertCandidateList(t *testing.T, got, want []candidate) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("len(candidates) = %d, want %d: got %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].id != want[i].id {
			t.Fatalf("candidate %d id = %d, want %d", i, got[i].id, want[i].id)
		}
		if !almostEqual(got[i].distance, want[i].distance) {
			t.Fatalf("candidate %d distance = %v, want %v", i, got[i].distance, want[i].distance)
		}
	}
}
