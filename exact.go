package fasthnsw

import "fmt"

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

	results := makeTopK(k)
	for id := 0; id < count; id++ {
		results.insert(Result{
			ID:       id,
			Distance: distance(metric, vectorAt(vectors, dim, id), query),
		})
	}
	return results.results(), nil
}

// topK keeps a bounded, sorted result list using deterministic distance then
// id ordering. The simple insertion strategy is intentional here because the
// exact baseline is primarily for tests and small recall oracles.
type topK struct {
	k     int
	items []Result
}

// makeTopK initializes a bounded result list.
func makeTopK(k int) topK {
	return topK{k: k, items: make([]Result, 0, k)}
}

// insert adds result when it belongs in the current top-k set.
func (t *topK) insert(result Result) {
	if t.k <= 0 {
		return
	}

	pos := len(t.items)
	for i, existing := range t.items {
		if betterResult(result, existing) {
			pos = i
			break
		}
	}

	t.items = append(t.items, Result{})
	copy(t.items[pos+1:], t.items[pos:])
	t.items[pos] = result
	if len(t.items) > t.k {
		t.items = t.items[:t.k]
	}
}

// results returns a copy so callers cannot mutate topK internals.
func (t topK) results() []Result {
	out := make([]Result, len(t.items))
	copy(out, t.items)
	return out
}

// betterResult reports whether a should sort before b.
func betterResult(a, b Result) bool {
	if a.Distance != b.Distance {
		return a.Distance < b.Distance
	}
	return a.ID < b.ID
}
