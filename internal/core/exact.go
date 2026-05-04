package core

import (
	"container/heap"
	"fmt"
)

// exactTopK scans every stored vector and returns the exact nearest neighbors.
// It is an internal correctness oracle for tests, recall checks, and future
// benchmarks; the public Search API should use graph search once implemented.
func exactTopK(vectors []float32, dim int, metric Metric, query []float32, k int) ([]Result, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("fasthnsw: vector dimension must be positive")
	}
	if len(vectors)%dim != 0 {
		return nil, fmt.Errorf("fasthnsw: flat vector storage is not aligned to dimension")
	}
	if len(query) != dim {
		return nil, fmt.Errorf("fasthnsw: query vector has dimension %d, want %d", len(query), dim)
	}
	if k <= 0 {
		return nil, fmt.Errorf("fasthnsw: k must be positive")
	}

	query, err := prepareQuery(metric, query)
	if err != nil {
		return nil, err
	}

	count := len(vectors) / dim
	if k > count {
		k = count
	}

	results := make(resultMaxHeap, 0, k)
	heap.Init(&results)
	for id := 0; id < count; id++ {
		result := Result{
			ID:       id,
			Distance: distance(metric, vectorAt(vectors, dim, id), query),
		}
		if results.Len() < k || betterResult(result, results.worst()) {
			heap.Push(&results, result)
			if results.Len() > k {
				heap.Pop(&results)
			}
		}
	}
	return results.sorted(), nil
}
