package fasthnsw

import (
	"container/heap"
	"fmt"
	"math/rand"
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
// checks, and tiny-layer fallbacks. It excludes each source node itself,
// returns candidates sorted by distance then id, and operates on the same flat
// vector storage used by the index.
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

// approximateKNNGCandidates builds the initial KNNG used by FastHNSW
// construction.
//
// The paper uses an approximate KNNG as the first construction phase rather
// than exact all-pairs neighbors. This pure Go implementation follows the same
// NN-Descent style idea without copying the official artifact: initialize each
// node with deterministic random neighbors, repeatedly exchange candidates
// through forward and reverse neighborhoods, then keep the nearest k candidates.
// Exact candidates are used only when k already covers the whole tiny layer.
func approximateKNNGCandidates(vectors []float32, dim int, metric Metric, k int, seed int64, iterations int) ([][]candidate, error) {
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
	if count <= limit+1 {
		return exactCandidates(vectors, dim, metric, limit)
	}

	for sourceID := 0; sourceID < count; sourceID++ {
		initialIDs := initialApproxNeighborIDs(count, sourceID, limit, seed)
		out[sourceID] = candidatesFromIDSet(sourceID, initialIDs, limit, vectors, dim, metric)
	}

	rounds := iterations
	if rounds < 1 {
		rounds = 1
	}
	for round := 0; round < rounds; round++ {
		reverse := reverseCandidateIDs(out, count)
		next := make([][]candidate, count)
		for sourceID := 0; sourceID < count; sourceID++ {
			ids := make(map[int]struct{}, limit*(limit+2))
			addCandidateIDs(ids, sourceID, out[sourceID])
			for _, reverseID := range reverse[sourceID] {
				if reverseID != sourceID {
					ids[reverseID] = struct{}{}
				}
			}
			for _, current := range out[sourceID] {
				addCandidateIDs(ids, sourceID, out[current.id])
			}
			for _, reverseID := range reverse[sourceID] {
				addCandidateIDs(ids, sourceID, out[reverseID])
			}
			next[sourceID] = candidatesFromIDSet(sourceID, ids, limit, vectors, dim, metric)
		}
		out = next
	}
	return out, nil
}

func initialApproxNeighborIDs(count int, sourceID int, limit int, seed int64) map[int]struct{} {
	ids := make(map[int]struct{}, limit)

	// Adjacent ids are cheap deterministic seeds and are useful when input ids
	// already carry locality, while random ids keep the initializer from
	// degenerating on shuffled datasets.
	localBudget := limit / 2
	for offset := 1; len(ids) < localBudget && offset < count; offset++ {
		ids[(sourceID+offset)%count] = struct{}{}
		if len(ids) >= localBudget {
			break
		}
		ids[(sourceID-offset+count)%count] = struct{}{}
	}

	rng := rand.New(rand.NewSource(mixSeed(seed, int64(sourceID), int64(count))))
	for len(ids) < limit {
		id := rng.Intn(count)
		if id == sourceID {
			continue
		}
		ids[id] = struct{}{}
	}
	return ids
}

func reverseCandidateIDs(candidates [][]candidate, count int) [][]int {
	reverse := make([][]int, count)
	for sourceID, sourceCandidates := range candidates {
		for _, candidate := range sourceCandidates {
			reverse[candidate.id] = append(reverse[candidate.id], sourceID)
		}
	}
	return reverse
}

func addCandidateIDs(ids map[int]struct{}, sourceID int, candidates []candidate) {
	for _, candidate := range candidates {
		if candidate.id == sourceID {
			continue
		}
		ids[candidate.id] = struct{}{}
	}
}

func candidatesFromIDSet(sourceID int, ids map[int]struct{}, limit int, vectors []float32, dim int, metric Metric) []candidate {
	source := vectorAt(vectors, dim, sourceID)
	best := make(candidateMaxHeap, 0, limit)
	heap.Init(&best)
	count := len(vectors) / dim
	for id := range ids {
		if id == sourceID || id < 0 || id >= count {
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

// mixSeed derives deterministic independent random streams from one user seed.
func mixSeed(seed int64, values ...int64) int64 {
	x := uint64(seed) + 0x9e3779b97f4a7c15
	for _, value := range values {
		x ^= uint64(value) + 0x9e3779b97f4a7c15 + (x << 6) + (x >> 2)
	}
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return int64(x)
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
