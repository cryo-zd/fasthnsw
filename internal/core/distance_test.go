package core

import "testing"

func TestSquaredL2(t *testing.T) {
	got := squaredL2([]float32{1, 2, 3}, []float32{4, 2, -1})
	const want float32 = 25
	if got != want {
		t.Fatalf("squaredL2 = %v, want %v", got, want)
	}
}

func TestNormalizeInto(t *testing.T) {
	dst := make([]float32, 2)
	if err := normalizeInto(dst, []float32{3, 4}); err != nil {
		t.Fatalf("normalizeInto returned error: %v", err)
	}

	if !almostEqual(dst[0], 0.6) || !almostEqual(dst[1], 0.8) {
		t.Fatalf("normalized vector = %v, want [0.6 0.8]", dst)
	}
}

func TestNormalizeIntoRejectsZeroVector(t *testing.T) {
	err := normalizeInto(make([]float32, 2), []float32{0, 0})
	if err == nil {
		t.Fatal("normalizeInto returned nil error")
	}
}

func TestCosineDistanceNormalized(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	if got := cosineDistanceNormalized(a, b); got != 1 {
		t.Fatalf("orthogonal cosine distance = %v, want 1", got)
	}

	if got := cosineDistanceNormalized(a, a); got != 0 {
		t.Fatalf("identical cosine distance = %v, want 0", got)
	}
}
