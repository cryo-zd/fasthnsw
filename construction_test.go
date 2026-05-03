package fasthnsw

import (
	"reflect"
	"testing"
)

func TestBuildSearchesTinyCompleteLayer(t *testing.T) {
	vectors := [][]float32{
		{0, 0},
		{1, 0},
		{2, 0},
		{3, 0},
	}
	idx, err := New(Config{Dim: 2, M: 4, K0: 4, EfConstruction: 4, Seed: 11})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build returned error: %v", err)
	}

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

func TestBuildSearchesSingleVector(t *testing.T) {
	idx := mustBuildIndex(t, Config{Dim: 2}, [][]float32{{3, 4}})

	got, err := idx.Search([]float32{3, 4}, 1, 1)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	assertResults(t, got, []Result{{ID: 0, Distance: 0}})
}

func TestBuildSearchesCosineIndex(t *testing.T) {
	vectors := [][]float32{
		{1, 0},
		{0, 1},
		{1, 1},
	}
	idx, err := New(Config{Metric: MetricCosine, Dim: 2, M: 4, K0: 4, EfConstruction: 4})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build returned error: %v", err)
	}

	got, err := idx.Search([]float32{1, 0}, 2, 3)
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	want, err := exactTopK(idx.vectors, idx.dim, idx.cfg.Metric, []float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("exactTopK returned error: %v", err)
	}
	assertResults(t, got, want)
}

func TestBuildUsesDoubleDegreeOnBaseLayer(t *testing.T) {
	idx := mustBuildIndex(t, Config{Dim: 1, M: 2, K0: 4, EfConstruction: 4, Seed: 3}, lineVectors(4))

	if len(idx.layers) == 0 {
		t.Fatal("idx.layers is empty")
	}
	for sourceID, neighbors := range idx.layers[0] {
		if len(neighbors) <= idx.cfg.M {
			t.Fatalf("base layer node %d degree = %d, want greater than M=%d in complete tiny layer", sourceID, len(neighbors), idx.cfg.M)
		}
		if len(neighbors) > baseLayerMaxDegree(idx.cfg) {
			t.Fatalf("base layer node %d degree = %d, want <= %d", sourceID, len(neighbors), baseLayerMaxDegree(idx.cfg))
		}
	}
}

func TestBuildHNSWLayerCompletesWhenDegreeEqualsBound(t *testing.T) {
	flat, dim, err := flattenVectors(lineVectors(3), 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}

	got, err := buildHNSWLayer(1, []int{0, 1, 2}, 3, flat, dim, Config{Metric: MetricL2}, 2)
	if err != nil {
		t.Fatalf("buildHNSWLayer returned error: %v", err)
	}
	for sourceID, neighbors := range got {
		if len(neighbors) != 2 {
			t.Fatalf("len(adjacency[%d]) = %d, want complete degree 2: %v", sourceID, len(neighbors), neighbors)
		}
	}
}

func TestFastHNSWOptKCNAConfigDisablesConnectivityRepair(t *testing.T) {
	cfg := fastHNSWOptKCNAConfig(Config{Alpha: 90}, 12, 24, 6)

	if cfg.CandidateK != 12 {
		t.Fatalf("CandidateK = %d, want 12", cfg.CandidateK)
	}
	if cfg.SearchEf != 24 {
		t.Fatalf("SearchEf = %d, want 24", cfg.SearchEf)
	}
	if cfg.MaxDegree != 6 {
		t.Fatalf("MaxDegree = %d, want 6", cfg.MaxDegree)
	}
	if cfg.AlphaDegrees != 90 {
		t.Fatalf("AlphaDegrees = %v, want 90", cfg.AlphaDegrees)
	}
	if cfg.ConnectComponents {
		t.Fatal("ConnectComponents = true, want false for FastHNSW/HNSW construction")
	}
}

func TestBuildIsDeterministicWithFixedSeed(t *testing.T) {
	vectors := deterministicVectors(48, 3)
	cfg := Config{Dim: 3, M: 4, K0: 8, EfConstruction: 8, Iterations: 1, Seed: 17}

	left := mustBuildIndex(t, cfg, vectors)
	right := mustBuildIndex(t, cfg, vectors)

	if !reflect.DeepEqual(left.levels, right.levels) {
		t.Fatalf("levels differ under fixed seed:\nleft=%v\nright=%v", left.levels, right.levels)
	}
	if left.entryPoint != right.entryPoint {
		t.Fatalf("entryPoint = %d and %d, want equal", left.entryPoint, right.entryPoint)
	}
	if left.maxLayer != right.maxLayer {
		t.Fatalf("maxLayer = %d and %d, want equal", left.maxLayer, right.maxLayer)
	}
	if !reflect.DeepEqual(left.layers, right.layers) {
		t.Fatal("layers differ under fixed seed")
	}
}

func TestAssignLevelsUsesSeed(t *testing.T) {
	left := assignLevels(512, 8, 1)
	right := assignLevels(512, 8, 2)
	if reflect.DeepEqual(left, right) {
		t.Fatal("assignLevels produced identical levels for different seeds")
	}
}

func TestRefineCandidatesStopsWhenQualityRequirementIsMet(t *testing.T) {
	flat, dim, err := flattenVectors(lineVectors(32), 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}
	candidates, err := exactCandidates(flat, dim, MetricL2, 4)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}

	got, stats, err := refineCandidatesUntilRecall(candidates, optKCNAConfig{
		CandidateK:        4,
		SearchEf:          6,
		MaxDegree:         4,
		AlphaDegrees:      90,
		ConnectComponents: true,
	}, candidateQualityConfig{
		TargetRecall: 0.98,
		Controls:     16,
		Seed:         12,
		K:            4,
	}, flat, dim, MetricL2, 5)
	if err != nil {
		t.Fatalf("refineCandidatesUntilRecall returned error: %v", err)
	}
	if stats.Iterations != 0 {
		t.Fatalf("iterations = %d, want 0 when initial quality meets requirement", stats.Iterations)
	}
	if stats.InitialRecall < 0.98 || stats.FinalRecall < 0.98 {
		t.Fatalf("recall stats = %+v, want both >= 0.98", stats)
	}
	assertCandidates(t, got, candidates)
}

func TestRefineCandidatesUsesIterationsAsMaximumCap(t *testing.T) {
	flat, dim, err := flattenVectors(lineVectors(48), 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}
	candidates := make([][]candidate, len(flat)/dim)

	_, stats, err := refineCandidatesUntilRecall(candidates, optKCNAConfig{
		CandidateK:        8,
		SearchEf:          8,
		MaxDegree:         4,
		AlphaDegrees:      90,
		ConnectComponents: true,
	}, candidateQualityConfig{
		TargetRecall: 1,
		Controls:     24,
		Seed:         19,
		K:            8,
	}, flat, dim, MetricL2, 1)
	if err != nil {
		t.Fatalf("refineCandidatesUntilRecall returned error: %v", err)
	}
	if stats.Iterations != 1 {
		t.Fatalf("iterations = %d, want exactly the max cap 1", stats.Iterations)
	}
}

func TestAcquireCandidatesByGraphSearchDropsSelfAndKeepsNearest(t *testing.T) {
	flat, dim, err := flattenVectors(lineVectors(5), 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}
	adjacency := completeLayer(5)

	got := acquireCandidatesByGraphSearch(adjacency, flat, dim, MetricL2, 2, 5)
	assertCandidateList(t, got[2], []candidate{
		{id: 1, distance: 1},
		{id: 3, distance: 1},
	})
}

func TestEstimateCandidateRecallUsesFixedControls(t *testing.T) {
	flat, dim, err := flattenVectors(lineVectors(20), 0, MetricL2)
	if err != nil {
		t.Fatalf("flattenVectors returned error: %v", err)
	}
	candidates, err := exactCandidates(flat, dim, MetricL2, 4)
	if err != nil {
		t.Fatalf("exactCandidates returned error: %v", err)
	}
	cfg := candidateQualityConfig{
		TargetRecall: 0.98,
		ControlIDs:   []int{1, 3, 5, 7},
		Seed:         1,
		K:            4,
	}

	left, err := estimateCandidateRecall(candidates, cfg, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("estimateCandidateRecall returned error: %v", err)
	}
	cfg.Seed = 99
	right, err := estimateCandidateRecall(candidates, cfg, flat, dim, MetricL2)
	if err != nil {
		t.Fatalf("estimateCandidateRecall second run returned error: %v", err)
	}
	if left != right {
		t.Fatalf("fixed-control recall = %v and %v, want equal", left, right)
	}
}

func TestBuildLayerInvariants(t *testing.T) {
	vectors := deterministicVectors(40, 2)
	cfg := Config{Dim: 2, M: 4, K0: 8, EfConstruction: 8, Iterations: 1, Seed: 23}
	idx := mustBuildIndex(t, cfg, vectors)

	if !idx.graphReady {
		t.Fatal("idx.graphReady = false, want true")
	}
	if len(idx.layers) != idx.maxLayer+1 {
		t.Fatalf("len(layers) = %d, want %d", len(idx.layers), idx.maxLayer+1)
	}
	if idx.entryPoint < 0 || idx.entryPoint >= idx.count {
		t.Fatalf("entryPoint = %d out of range", idx.entryPoint)
	}
	if idx.levels[idx.entryPoint] != idx.maxLayer {
		t.Fatalf("entryPoint level = %d, want maxLayer %d", idx.levels[idx.entryPoint], idx.maxLayer)
	}

	for layerID, layer := range idx.layers {
		if len(layer) != idx.count {
			t.Fatalf("len(layers[%d]) = %d, want %d", layerID, len(layer), idx.count)
		}
		maxDegree := idx.cfg.M
		if layerID == 0 {
			maxDegree = baseLayerMaxDegree(idx.cfg)
		}
		for sourceID, neighbors := range layer {
			if idx.levels[sourceID] < layerID && len(neighbors) != 0 {
				t.Fatalf("layer %d node %d has neighbors below assigned level: %v", layerID, sourceID, neighbors)
			}
			if len(neighbors) > maxDegree {
				t.Fatalf("layer %d node %d degree = %d, want <= %d: %v", layerID, sourceID, len(neighbors), maxDegree, neighbors)
			}
			seen := make(map[int]bool, len(neighbors))
			for _, neighborID := range neighbors {
				if neighborID == sourceID {
					t.Fatalf("layer %d node %d contains self edge: %v", layerID, sourceID, neighbors)
				}
				if neighborID < 0 || neighborID >= idx.count {
					t.Fatalf("layer %d node %d neighbor %d out of range", layerID, sourceID, neighborID)
				}
				if idx.levels[neighborID] < layerID {
					t.Fatalf("layer %d node %d links to node %d below layer", layerID, sourceID, neighborID)
				}
				if seen[neighborID] {
					t.Fatalf("layer %d node %d has duplicate neighbor %d: %v", layerID, sourceID, neighborID, neighbors)
				}
				seen[neighborID] = true
			}
		}
	}
}

func mustBuildIndex(t *testing.T, cfg Config, vectors [][]float32) *Index {
	t.Helper()

	idx, err := New(cfg)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		t.Fatalf("Build returned error: %v", err)
	}
	return idx
}

func deterministicVectors(count int, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for id := 0; id < count; id++ {
		vector := make([]float32, dim)
		for d := 0; d < dim; d++ {
			vector[d] = float32((id+1)*(d+3)%17) + float32(d)/10
		}
		vectors[id] = vector
	}
	return vectors
}
