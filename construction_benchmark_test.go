package fasthnsw

import "testing"

func BenchmarkBuild(b *testing.B) {
	vectors := deterministicVectors(256, 8)
	cfg := Config{Dim: 8, M: 8, K0: 16, EfConstruction: 16, Iterations: 1, Seed: 99}

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
	vectors := deterministicVectors(256, 8)
	idx, err := New(Config{Dim: 8, M: 8, K0: 16, EfConstruction: 16, Iterations: 1, Seed: 99})
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}
	if err := idx.Build(vectors); err != nil {
		b.Fatalf("Build returned error: %v", err)
	}
	query := vectors[0]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := idx.Search(query, 10, 32); err != nil {
			b.Fatalf("Search returned error: %v", err)
		}
	}
}
