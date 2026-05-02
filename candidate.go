package fasthnsw

import (
	"container/heap"
	"fmt"
	"sort"
)

// candidate is one construction-time neighbor candidate for a source node.
// It is intentionally separate from Result because construction code will add
// pruning and graph-building metadata around candidates while public Search
// results should remain a stable user-facing type.
type candidate struct {
	id       int
	distance float32
}

// exactCandidates computes exact per-node k-nearest candidate lists.
//
// This O(n^2) builder is the correctness baseline for pruning, IterNSG, recall
// checks, and future approximate candidate acquisition. It excludes each source
// node itself, returns candidates sorted by distance then id, and operates on
// the same flat vector storage used by the index.
func exactCandidates(vectors []float32, dim int, metric Metric, k int) ([][]candidate, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("fasthnsw: vector dimension must be positive")
	}
	if len(vectors)%dim != 0 {
		return nil, fmt.Errorf("fasthnsw: flat vector storage is not aligned to dimension")
	}
	if k <= 0 {
		return nil, fmt.Errorf("fasthnsw: k must be positive")
	}

	count := len(vectors) / dim
	out := make([][]candidate, count)
	if count <= 1 {
		return out, nil
	}

	limit := k
	if limit > count-1 {
		limit = count - 1
	}

	for sourceID := 0; sourceID < count; sourceID++ {
		out[sourceID] = exactCandidatesForNode(vectors, dim, metric, sourceID, limit)
	}
	return out, nil
}

// exactCandidatesForNode scans all other vectors and keeps the nearest limit
// candidates for one source node.
func exactCandidatesForNode(vectors []float32, dim int, metric Metric, sourceID int, limit int) []candidate {
	source := vectorAt(vectors, dim, sourceID)
	best := make(candidateMaxHeap, 0, limit)
	heap.Init(&best)

	count := len(vectors) / dim
	for id := 0; id < count; id++ {
		if id == sourceID {
			continue
		}
		next := candidate{
			id:       id,
			distance: distance(metric, source, vectorAt(vectors, dim, id)),
		}
		if best.Len() < limit || betterCandidate(next, best.worst()) {
			heap.Push(&best, next)
			if best.Len() > limit {
				heap.Pop(&best)
			}
		}
	}

	return best.sorted()
}

// betterCandidate reports whether a should sort before b in candidate order.
// Candidate order is deterministic: distance ascending, then smaller id.
func betterCandidate(a, b candidate) bool {
	if a.distance != b.distance {
		return a.distance < b.distance
	}
	return a.id < b.id
}

// worseCandidate reports whether a should sort after b in candidate order.
func worseCandidate(a, b candidate) bool {
	if a.distance != b.distance {
		return a.distance > b.distance
	}
	return a.id > b.id
}

// candidateMaxHeap keeps the current worst retained candidate at the root so
// exact candidate generation can maintain a bounded top-k set in O(log k).
type candidateMaxHeap []candidate

func (h candidateMaxHeap) Len() int { return len(h) }

func (h candidateMaxHeap) Less(i, j int) bool {
	return worseCandidate(h[i], h[j])
}

func (h candidateMaxHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *candidateMaxHeap) Push(x any) {
	*h = append(*h, x.(candidate))
}

func (h *candidateMaxHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// worst returns the farthest retained candidate, which is the heap root.
func (h candidateMaxHeap) worst() candidate {
	return h[0]
}

// sorted returns candidates in construction order: nearest distance first,
// then id. It returns a copy so callers can mutate their list without changing
// heap internals.
func (h candidateMaxHeap) sorted() []candidate {
	items := make([]candidate, len(h))
	copy(items, h)
	sort.Slice(items, func(i, j int) bool {
		return betterCandidate(items[i], items[j])
	})
	return items
}
