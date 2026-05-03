package fasthnsw

import (
	"fmt"
	"testing"
)

func BenchmarkCandidateAcquisition(b *testing.B) {
	vectors, dim, err := flattenVectors(uniformVectors(256, 8), 0, MetricL2)
	if err != nil {
		b.Fatalf("flattenVectors returned error: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := approximateKNNGCandidates(vectors, dim, MetricL2, 16, int64(i+1), 4); err != nil {
			b.Fatalf("approximateKNNGCandidates returned error: %v", err)
		}
	}
}

func BenchmarkLayerConstruction(b *testing.B) {
	vectors, dim, err := flattenVectors(uniformVectors(256, 8), 0, MetricL2)
	if err != nil {
		b.Fatalf("flattenVectors returned error: %v", err)
	}
	nodes := make([]int, 256)
	for i := range nodes {
		nodes[i] = i
	}
	cfg := Config{
		Metric:            MetricL2,
		M:                 8,
		K0:                16,
		CandidateK:        16,
		ConstructionL:     32,
		Alpha:             67,
		Iterations:        2,
		Seed:              99,
		CandidateRecall:   0.90,
		CandidateControls: 64,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := buildHNSWLayer(0, nodes, len(nodes), vectors, dim, cfg, baseLayerMaxDegree(cfg)); err != nil {
			b.Fatalf("buildHNSWLayer returned error: %v", err)
		}
	}
}

func BenchmarkBuild(b *testing.B) {
	vectors := uniformVectors(256, 8)
	cfg := Config{Dim: 8, M: 8, K0: 16, CandidateK: 16, ConstructionL: 32, Iterations: 2, Seed: 99, CandidateRecall: 0.90, CandidateControls: 64}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := New(cfg)
		if err != nil {
			b.Fatalf("New returned error: %v", err)
		}
		if err := idx.Build(vectors); err != nil {
			b.Fatalf("Build returned error: %v", err)
		}
	}
}

func BenchmarkSearch(b *testing.B) {
	vectors := uniformVectors(256, 8)
	idx, err := New(Config{Dim: 8, M: 8, K0: 16, CandidateK: 16, ConstructionL: 32, Iterations: 2, Seed: 99, CandidateRecall: 0.90, CandidateControls: 64})
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		b.Fatalf("Build returned error: %v", err)
	}
	query := uniformQueries(1, 8)[0]

	for _, efSearch := range []int{16, 32, 64} {
		b.Run(fmt.Sprintf("ef%d", efSearch), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := idx.Search(query, 10, efSearch); err != nil {
					b.Fatalf("Search returned error: %v", err)
				}
			}
		})
	}
}
