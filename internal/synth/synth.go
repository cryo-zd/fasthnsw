package synth

// UniformVectors returns deterministic pseudo-uniform vectors.
func UniformVectors(count int, dim int) [][]float32 {
	vectors := make([][]float32, count)
	for id := 0; id < count; id++ {
		vector := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v := (id+1)*(d+5)*37 + (id%11)*17 + d*13
			vector[d] = float32(v%997) / 997
		}
		vectors[id] = vector
	}
	return vectors
}

// UniformQueries returns deterministic pseudo-uniform query vectors.
func UniformQueries(count int, dim int) [][]float32 {
	queries := make([][]float32, count)
	for id := 0; id < count; id++ {
		query := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v := (id+3)*(d+7)*29 + id*19 + d*23
			query[d] = float32(v%991) / 991
		}
		queries[id] = query
	}
	return queries
}

// ClusteredVectors returns deterministic clustered vectors for recall tests.
func ClusteredVectors(count int, dim int, clusters int) [][]float32 {
	vectors := make([][]float32, count)
	for id := 0; id < count; id++ {
		cluster := id % clusters
		vector := make([]float32, dim)
		for d := 0; d < dim; d++ {
			center := float32((cluster+1)*(d+2)) / float32(clusters+dim)
			offsetRaw := ((id+1)*(d+3) + cluster*7) % 17
			offset := (float32(offsetRaw) - 8) * 0.0025
			vector[d] = center + offset
		}
		vectors[id] = vector
	}
	return vectors
}

// ClusteredQueries returns deterministic clustered query vectors.
func ClusteredQueries(count int, dim int, clusters int) [][]float32 {
	queries := make([][]float32, count)
	for id := 0; id < count; id++ {
		cluster := id % clusters
		query := make([]float32, dim)
		for d := 0; d < dim; d++ {
			center := float32((cluster+1)*(d+2)) / float32(clusters+dim)
			offsetRaw := ((id+5)*(d+1) + cluster*3) % 13
			offset := (float32(offsetRaw) - 6) * 0.0015
			query[d] = center + offset
		}
		queries[id] = query
	}
	return queries
}

// SameVectors reports whether two vector collections are exactly identical.
func SameVectors(left [][]float32, right [][]float32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if len(left[i]) != len(right[i]) {
			return false
		}
		for d := range left[i] {
			if left[i][d] != right[i][d] {
				return false
			}
		}
	}
	return true
}
