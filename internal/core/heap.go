package core

import "sort"

// resultMinHeap orders Result values nearest-first.
// HNSW uses this ordering for the candidate set C so SEARCH-LAYER expands the
// closest unexpanded candidate before farther ones.
type resultMinHeap []Result

func (h resultMinHeap) Len() int { return len(h) }

func (h resultMinHeap) Less(i, j int) bool {
	return betterResult(h[i], h[j])
}

func (h resultMinHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *resultMinHeap) Push(x any) {
	*h = append(*h, x.(Result))
}

func (h *resultMinHeap) Pop() any {
	old := *h
	n := len(old)
	result := old[n-1]
	*h = old[:n-1]
	return result
}

// resultMaxHeap orders Result values farthest-first.
// HNSW uses this ordering for the result set W so SEARCH-LAYER can compare
// against, and evict, the current worst retained result in logarithmic time.
type resultMaxHeap []Result

func (h resultMaxHeap) Len() int { return len(h) }

func (h resultMaxHeap) Less(i, j int) bool {
	return worseResult(h[i], h[j])
}

func (h resultMaxHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *resultMaxHeap) Push(x any) {
	*h = append(*h, x.(Result))
}

func (h *resultMaxHeap) Pop() any {
	old := *h
	n := len(old)
	result := old[n-1]
	*h = old[:n-1]
	return result
}

// worst returns the farthest retained result, which is the root of W.
func (h resultMaxHeap) worst() Result {
	return h[0]
}

// sorted returns results in public order: nearest distance first, then id.
func (h resultMaxHeap) sorted() []Result {
	results := make([]Result, len(h))
	copy(results, h)
	sort.Slice(results, func(i, j int) bool {
		return betterResult(results[i], results[j])
	})
	return results
}
