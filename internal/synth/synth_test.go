package synth

import "testing"

func TestGeneratedDatasetsAreDeterministic(t *testing.T) {
	if !SameVectors(UniformVectors(16, 3), UniformVectors(16, 3)) {
		t.Fatal("UniformVectors is not deterministic")
	}
	if !SameVectors(UniformQueries(16, 3), UniformQueries(16, 3)) {
		t.Fatal("UniformQueries is not deterministic")
	}
	if !SameVectors(ClusteredVectors(24, 4, 3), ClusteredVectors(24, 4, 3)) {
		t.Fatal("ClusteredVectors is not deterministic")
	}
	if !SameVectors(ClusteredQueries(12, 4, 3), ClusteredQueries(12, 4, 3)) {
		t.Fatal("ClusteredQueries is not deterministic")
	}
}
