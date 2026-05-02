package fasthnsw

import "container/heap"

// search runs the standard HNSW query procedure with a query already prepared
// for the index metric. Each upper layer calls SEARCH-LAYER with ef=1, then
// layer 0 calls SEARCH-LAYER with the caller-provided efSearch and returns the
// best k results.
func (idx *Index) search(query []float32, k int, efSearch int) ([]Result, error) {
	entries := []int{idx.entryPoint}
	for layer := idx.maxLayer; layer > 0; layer-- {
		results := idx.searchLayer(layer, entries, query, 1)
		if len(results) == 0 {
			return nil, ErrIndexNotBuilt
		}
		entries = []int{results[0].ID}
	}

	results := idx.searchLayer(0, entries, query, efSearch)
	if k > len(results) {
		k = len(results)
	}
	out := make([]Result, k)
	copy(out, results[:k])
	return out, nil
}

// searchLayer implements the HNSW SEARCH-LAYER routine.
//
// The candidate set C is a nearest-first min-heap because the algorithm must
// expand the closest unexpanded candidate next. The result set W is a
// farthest-first max-heap because the algorithm repeatedly compares against,
// and evicts, the current worst retained result. Using a max-heap for C would
// expand the farthest candidate first and would not match HNSW search.
func (idx *Index) searchLayer(layer int, entries []int, query []float32, ef int) []Result {
	visited := make([]bool, idx.count)
	candidates := make(resultMinHeap, 0, ef)
	results := make(resultMaxHeap, 0, ef)
	heap.Init(&candidates)
	heap.Init(&results)

	for _, entry := range entries {
		if entry < 0 || entry >= idx.count || visited[entry] {
			continue
		}
		visited[entry] = true
		result := idx.resultForNode(entry, query)
		heap.Push(&candidates, result)
		heap.Push(&results, result)
	}

	for candidates.Len() > 0 {
		current := heap.Pop(&candidates).(Result)
		if results.Len() > 0 && current.Distance > results.worst().Distance {
			break
		}

		for _, neighborID := range idx.layers[layer][current.ID] {
			if visited[neighborID] {
				continue
			}
			visited[neighborID] = true

			candidate := idx.resultForNode(neighborID, query)
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
func (idx *Index) resultForNode(id int, query []float32) Result {
	return Result{
		ID:       id,
		Distance: distance(idx.cfg.Metric, vectorAt(idx.vectors, idx.dim, id), query),
	}
}
