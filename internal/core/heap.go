package core

import "slices"

// resultMinHeap orders Result values nearest-first.
// HNSW uses this ordering for the candidate set C so SEARCH-LAYER expands the
// closest unexpanded candidate before farther ones.
type resultMinHeap []Result

func (h resultMinHeap) Len() int { return len(h) }

func (h resultMinHeap) less(i, j int) bool {
	return betterResult(h[i], h[j])
}

func (h resultMinHeap) swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *resultMinHeap) init() {
	for i := len(*h)/2 - 1; i >= 0; i-- {
		h.down(i)
	}
}

func (h *resultMinHeap) push(result Result) {
	*h = append(*h, result)
	h.up(len(*h) - 1)
}

func (h *resultMinHeap) pop() Result {
	old := *h
	result := old[0]
	last := len(old) - 1
	if last > 0 {
		old[0] = old[last]
		*h = old[:last]
		h.down(0)
	} else {
		*h = old[:0]
	}
	return result
}

func (h resultMinHeap) up(child int) {
	for child > 0 {
		parent := (child - 1) / 2
		if !h.less(child, parent) {
			return
		}
		h.swap(parent, child)
		child = parent
	}
}

func (h resultMinHeap) down(parent int) {
	n := len(h)
	for {
		left := 2*parent + 1
		if left >= n {
			return
		}
		best := left
		right := left + 1
		if right < n && h.less(right, left) {
			best = right
		}
		if !h.less(best, parent) {
			return
		}
		h.swap(parent, best)
		parent = best
	}
}

// resultMaxHeap orders Result values farthest-first.
// HNSW uses this ordering for the result set W so SEARCH-LAYER can compare
// against, and evict, the current worst retained result in logarithmic time.
type resultMaxHeap []Result

func (h resultMaxHeap) Len() int { return len(h) }

func (h resultMaxHeap) less(i, j int) bool {
	return worseResult(h[i], h[j])
}

func (h resultMaxHeap) swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *resultMaxHeap) init() {
	for i := len(*h)/2 - 1; i >= 0; i-- {
		h.down(i)
	}
}

func (h *resultMaxHeap) push(result Result) {
	*h = append(*h, result)
	h.up(len(*h) - 1)
}

// replaceWorst replaces the root of W with a better result and restores the
// heap invariant without growing the heap beyond its configured ef bound.
func (h resultMaxHeap) replaceWorst(result Result) {
	h[0] = result
	h.down(0)
}

func (h *resultMaxHeap) pop() Result {
	old := *h
	result := old[0]
	last := len(old) - 1
	if last > 0 {
		old[0] = old[last]
		*h = old[:last]
		h.down(0)
	} else {
		*h = old[:0]
	}
	return result
}

// worst returns the farthest retained result, which is the root of W.
func (h resultMaxHeap) worst() Result {
	return h[0]
}

func (h resultMaxHeap) up(child int) {
	for child > 0 {
		parent := (child - 1) / 2
		if !h.less(child, parent) {
			return
		}
		h.swap(parent, child)
		child = parent
	}
}

func (h resultMaxHeap) down(parent int) {
	n := len(h)
	for {
		left := 2*parent + 1
		if left >= n {
			return
		}
		best := left
		right := left + 1
		if right < n && h.less(right, left) {
			best = right
		}
		if !h.less(best, parent) {
			return
		}
		h.swap(parent, best)
		parent = best
	}
}

// sorted returns results in public order: nearest distance first, then id. It
// sorts the heap storage in place; callers should not use the heap after
// calling sorted.
func (h resultMaxHeap) sorted() []Result {
	slices.SortFunc(h, func(a, b Result) int {
		return compareDistanceID(a.Distance, a.ID, b.Distance, b.ID)
	})
	return h
}
