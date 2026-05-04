package core

import (
	"testing"

	"github.com/cryo-zd/fasthnsw/internal/synth"
)

func TestRecallAtK(t *testing.T) {
	got := []Result{{ID: 1}, {ID: 2}, {ID: 3}}
	want := []Result{{ID: 2}, {ID: 4}, {ID: 1}}

	recall := recallAtK(got, want, 3)
	if recall != float64(2)/3 {
		t.Fatalf("recallAtK = %v, want %v", recall, float64(2)/3)
	}
}

func TestGeneratedDatasetsAreDeterministic(t *testing.T) {
	if !synth.SameVectors(synth.UniformVectors(16, 3), synth.UniformVectors(16, 3)) {
		t.Fatal("uniformVectors is not deterministic")
	}
	if !synth.SameVectors(synth.ClusteredVectors(24, 4, 3), synth.ClusteredVectors(24, 4, 3)) {
		t.Fatal("clusteredVectors is not deterministic")
	}
	if !synth.SameVectors(synth.ClusteredQueries(12, 4, 3), synth.ClusteredQueries(12, 4, 3)) {
		t.Fatal("clusteredQueries is not deterministic")
	}
}

func TestBuildSearchRecallClusteredData(t *testing.T) {
	vectors := synth.ClusteredVectors(180, 6, 6)
	queries := synth.ClusteredQueries(36, 6, 6)
	idx := mustBuildIndex(t, recallTestConfig(6), vectors)

	recall := averageRecallAtK(t, idx, queries, 10, 48)
	if recall < 0.85 {
		t.Fatalf("clustered Recall@10 = %.3f, want >= 0.85", recall)
	}
}

func TestBuildSearchRecallUniformData(t *testing.T) {
	vectors := synth.UniformVectors(160, 5)
	queries := synth.UniformQueries(32, 5)
	idx := mustBuildIndex(t, recallTestConfig(5), vectors)

	recall := averageRecallAtK(t, idx, queries, 10, 64)
	if recall < 0.70 {
		t.Fatalf("uniform Recall@10 = %.3f, want >= 0.70", recall)
	}
}

func TestBuildSearchResultsDeterministicWithFixedSeed(t *testing.T) {
	vectors := synth.ClusteredVectors(120, 4, 4)
	queries := synth.ClusteredQueries(8, 4, 4)
	cfg := recallTestConfig(4)

	left := mustBuildIndex(t, cfg, vectors)
	right := mustBuildIndex(t, cfg, vectors)

	for _, query := range queries {
		leftResults, err := left.Search(query, 10, 48)
		if err != nil {
			t.Fatalf("left Search returned error: %v", err)
		}
		rightResults, err := right.Search(query, 10, 48)
		if err != nil {
			t.Fatalf("right Search returned error: %v", err)
		}
		assertResults(t, leftResults, rightResults)
	}
}

func recallTestConfig(dim int) Config {
	return Config{
		Dim:               dim,
		M:                 12,
		K0:                32,
		CandidateK:        32,
		ConstructionL:     64,
		Alpha:             67,
		Iterations:        3,
		Seed:              41,
		CandidateRecall:   0.90,
		CandidateControls: 64,
	}
}

func averageRecallAtK(t *testing.T, idx *Index, queries [][]float32, k int, efSearch int) float64 {
	t.Helper()

	var total float64
	for _, query := range queries {
		got, err := idx.Search(query, k, efSearch)
		if err != nil {
			t.Fatalf("Search returned error: %v", err)
		}
		want, err := exactTopK(idx.vectors, idx.dim, idx.cfg.Metric, query, k)
		if err != nil {
			t.Fatalf("exactTopK returned error: %v", err)
		}
		total += recallAtK(got, want, k)
	}
	return total / float64(len(queries))
}

func recallAtK(got []Result, want []Result, k int) float64 {
	wantIDs := make(map[int]bool, k)
	for i := 0; i < k && i < len(want); i++ {
		wantIDs[want[i].ID] = true
	}

	var hits int
	limit := k
	if limit > len(got) {
		limit = len(got)
	}
	for i := 0; i < limit; i++ {
		if wantIDs[got[i].ID] {
			hits++
		}
	}
	if k == 0 {
		return 1
	}
	return float64(hits) / float64(k)
}
