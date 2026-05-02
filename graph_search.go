package fasthnsw

import "container/heap"

// graphSearchLayer implements the HNSW SEARCH-LAYER routine over one adjacency
// table. The query must already be in the representation expected by the
// metric; public Search normalizes cosine queries before calling this helper,
// while construction code searches with stored vectors that are already
// normalized when the metric is cosine.
func graphSearchLayer(adjacency [][]int, vectors []float32, dim int, metric Metric, entries []int, query []float32, ef int) []Result {
	count := len(adjacency)
	visited := make([]bool, count)
	candidates := make(resultMinHeap, 0, ef)
	results := make(resultMaxHeap, 0, ef)
	heap.Init(&candidates)
	heap.Init(&results)

	for _, entry := range entries {
		if entry < 0 || entry >= count || visited[entry] {
			continue
		}
		visited[entry] = true
		result := resultForNode(vectors, dim, metric, entry, query)
		heap.Push(&candidates, result)
		heap.Push(&results, result)
	}

	for candidates.Len() > 0 {
		current := heap.Pop(&candidates).(Result)
		if results.Len() > 0 && current.Distance > results.worst().Distance {
			break
		}

		for _, neighborID := range adjacency[current.ID] {
			if neighborID < 0 || neighborID >= count || visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			candidate := resultForNode(vectors, dim, metric, neighborID, query)
			if results.Len() < ef || betterResult(candidate, results.worst()) {
				heap.Push(&candidates, candidate)
				heap.Push(&results, candidate)
				if results.Len() > ef {
					heap.Pop(&results)
				}
			}
		}
	}

	return results.sorted()
}

// resultForNode computes a query result for one stored vector id.
func resultForNode(vectors []float32, dim int, metric Metric, id int, query []float32) Result {
	return Result{
		ID:       id,
		Distance: distance(metric, vectorAt(vectors, dim, id), query),
	}
}
