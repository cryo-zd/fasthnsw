package core

// graphSearchScratch owns temporary state for one or more SEARCH-LAYER calls.
// It is query-local or construction-loop-local scratch, not shared index state,
// so callers can reduce allocations without introducing synchronization or
// cross-query reuse.
type graphSearchScratch struct {
	visited    []uint32
	visitMark  uint32
	candidates resultMinHeap
	results    resultMaxHeap
}

func (scratch *graphSearchScratch) reset(count int, ef int) {
	scratch.reserve(count, ef)
	scratch.visitMark++
	if scratch.visitMark == 0 {
		allVisited := scratch.visited[:cap(scratch.visited)]
		clear(allVisited)
		scratch.visitMark = 1
	}
}

func (scratch *graphSearchScratch) reserve(count int, ef int) {
	if cap(scratch.visited) < count {
		scratch.visited = make([]uint32, count)
		scratch.visitMark = 0
	} else {
		scratch.visited = scratch.visited[:count]
	}

	if cap(scratch.candidates) < ef {
		scratch.candidates = make(resultMinHeap, 0, ef)
	} else {
		scratch.candidates = scratch.candidates[:0]
	}
	if cap(scratch.results) < ef {
		scratch.results = make(resultMaxHeap, 0, ef)
	} else {
		scratch.results = scratch.results[:0]
	}
}

func (scratch *graphSearchScratch) isVisited(id int) bool {
	return scratch.visited[id] == scratch.visitMark
}

func (scratch *graphSearchScratch) markVisited(id int) {
	scratch.visited[id] = scratch.visitMark
}

func (scratch *graphSearchScratch) sortedResults(limit int) []Result {
	if limit < 0 || limit > scratch.results.Len() {
		limit = scratch.results.Len()
	}
	results := scratch.results.sorted()
	return results[:limit]
}

// graphSearchLayer implements the HNSW SEARCH-LAYER routine over one adjacency
// table. The query must already be in the representation expected by the
// metric; public Search normalizes cosine queries before calling this helper,
// while construction code searches with stored vectors that are already
// normalized when the metric is cosine.
func graphSearchLayer(adjacency [][]int, vectors []float32, dim int, metric Metric, entry int, query []float32, ef int, scratch *graphSearchScratch) []Result {
	if scratch == nil {
		scratch = &graphSearchScratch{}
	}
	if !graphSearchLayerInto(adjacency, vectors, dim, metric, entry, query, ef, scratch) {
		return nil
	}
	return scratch.sortedResults(scratch.results.Len())
}

func graphSearchLayerTopK(adjacency [][]int, vectors []float32, dim int, metric Metric, entry int, query []float32, ef int, k int, scratch *graphSearchScratch) []Result {
	if scratch == nil {
		scratch = &graphSearchScratch{}
	}
	if !graphSearchLayerInto(adjacency, vectors, dim, metric, entry, query, ef, scratch) {
		return nil
	}
	return scratch.sortedResults(k)
}

func graphSearchLayerInto(adjacency [][]int, vectors []float32, dim int, metric Metric, entry int, query []float32, ef int, scratch *graphSearchScratch) bool {
	count := len(adjacency)
	if ef <= 0 || entry < 0 || entry >= count {
		return false
	}
	if scratch == nil {
		scratch = &graphSearchScratch{}
	}
	scratch.reset(count, ef)

	scratch.markVisited(entry)
	result := resultForNode(vectors, dim, metric, entry, query)
	scratch.candidates.push(result)
	scratch.results.push(result)

	for scratch.candidates.Len() > 0 {
		current := scratch.candidates.pop()
		if scratch.results.Len() > 0 && current.Distance > scratch.results.worst().Distance {
			break
		}

		for _, neighborID := range adjacency[current.ID] {
			if neighborID < 0 || neighborID >= count || scratch.isVisited(neighborID) {
				continue
			}
			scratch.markVisited(neighborID)

			candidate := resultForNode(vectors, dim, metric, neighborID, query)
			if scratch.results.Len() < ef {
				scratch.candidates.push(candidate)
				scratch.results.push(candidate)
				continue
			}
			if betterResult(candidate, scratch.results.worst()) {
				scratch.candidates.push(candidate)
				scratch.results.replaceWorst(candidate)
			}
		}
	}

	return scratch.results.Len() > 0
}

// resultForNode computes a query result for one stored vector id.
func resultForNode(vectors []float32, dim int, metric Metric, id int, query []float32) Result {
	return Result{
		ID:       id,
		Distance: distance(metric, vectorAt(vectors, dim, id), query),
	}
}
