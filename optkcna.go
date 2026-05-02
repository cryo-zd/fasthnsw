package fasthnsw

import (
	"fmt"
	"sort"
)

// optKCNAConfig controls one OptKCNA candidate-refresh step.
type optKCNAConfig struct {
	CandidateK        int
	SearchEf          int
	MaxDegree         int
	AlphaDegrees      float64
	ConnectComponents bool
}

// optKCNA refreshes k-CNA candidates by searching an intermediate alpha-PG.
//
// The current candidate lists are first converted to an alpha-pruned graph.
// When requested, weak components in that intermediate graph are connected
// deterministically. Each node then searches the graph using its own vector as
// the query, drops itself from the results, and keeps up to CandidateK nearest
// candidates for the next construction iteration.
func optKCNA(candidates [][]candidate, cfg optKCNAConfig, vectors []float32, dim int, metric Metric) ([][]candidate, error) {
	count, err := validateOptKCNAInput(candidates, cfg, vectors, dim, metric)
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return [][]candidate{}, nil
	}

	alphaGraph, err := buildPrunedLayer(candidates, cfg.MaxDegree, alphaPruneMode(cfg.AlphaDegrees), vectors, dim, metric)
	if err != nil {
		return nil, err
	}
	if cfg.ConnectComponents {
		alphaGraph, err = connectWeakComponentsInPlace(alphaGraph, vectors, dim, metric)
		if err != nil {
			return nil, err
		}
	}

	refreshed := make([][]candidate, count)
	for sourceID := 0; sourceID < count; sourceID++ {
		results := graphSearchLayer(alphaGraph, vectors, dim, metric, []int{sourceID}, vectorAt(vectors, dim, sourceID), cfg.SearchEf)
		refreshed[sourceID] = candidatesFromResults(sourceID, results, cfg.CandidateK)
	}
	return refreshed, nil
}

func validateOptKCNAInput(candidates [][]candidate, cfg optKCNAConfig, vectors []float32, dim int, metric Metric) (int, error) {
	if cfg.CandidateK <= 0 {
		return 0, fmt.Errorf("fasthnsw: CandidateK must be positive")
	}
	if cfg.SearchEf < cfg.CandidateK {
		return 0, fmt.Errorf("fasthnsw: SearchEf must be greater than or equal to CandidateK")
	}
	if cfg.MaxDegree <= 0 {
		return 0, fmt.Errorf("fasthnsw: MaxDegree must be positive")
	}
	if cfg.AlphaDegrees < minAlpha || cfg.AlphaDegrees > maxAlphaDegrees {
		return 0, fmt.Errorf("fasthnsw: AlphaDegrees must be in [%.0f, %.0f]", float64(minAlpha), float64(maxAlphaDegrees))
	}
	if !validMetric(metric) {
		return 0, fmt.Errorf("fasthnsw: unsupported metric %d", metric)
	}
	if dim <= 0 {
		return 0, fmt.Errorf("fasthnsw: vector dimension must be positive")
	}
	if len(vectors)%dim != 0 {
		return 0, fmt.Errorf("fasthnsw: flat vector storage is not aligned to dimension")
	}

	count := len(vectors) / dim
	if len(candidates) != count {
		return 0, fmt.Errorf("fasthnsw: candidate list count %d, want %d", len(candidates), count)
	}
	return count, nil
}

func candidatesFromResults(sourceID int, results []Result, limit int) []candidate {
	out := make([]candidate, 0, limit)
	for _, result := range results {
		if result.ID == sourceID {
			continue
		}
		out = append(out, candidate{id: result.ID, distance: result.Distance})
		if len(out) >= limit {
			break
		}
	}
	return out
}

// connectWeakComponentsInPlace makes an intermediate alpha graph weakly
// connected by mutating adjacency.
//
// This is intentionally in-place because OptKCNA owns the temporary alpha graph
// produced for the current refresh step. Keeping a pre-repair copy is not
// required by the algorithm and would add avoidable edge-slice allocations in
// an IterNSG hot path.
//
// The main component is the largest weak component, with ties resolved by its
// smallest node id. Each other component is connected to the growing main
// component by the nearest cross-component pair, with ties resolved by source
// id then target id.
func connectWeakComponentsInPlace(adjacency [][]int, vectors []float32, dim int, metric Metric) ([][]int, error) {
	if err := validateAdjacencyForConstruction(adjacency, vectors, dim, metric); err != nil {
		return nil, err
	}

	components := weakComponents(adjacency)
	if len(components) <= 1 {
		return normalizeAdjacencyForConstruction(adjacency, vectors, dim, metric), nil
	}

	mainIndex := mainComponentIndex(components)
	mainSet := make(map[int]bool, len(components[mainIndex]))
	for _, nodeID := range components[mainIndex] {
		mainSet[nodeID] = true
	}

	for componentIndex, component := range components {
		if componentIndex == mainIndex {
			continue
		}
		edge := nearestComponentBridge(component, mainSet, vectors, dim, metric)
		adjacency[edge.source] = append(adjacency[edge.source], edge.target)
		adjacency[edge.target] = append(adjacency[edge.target], edge.source)
		for _, nodeID := range component {
			mainSet[nodeID] = true
		}
	}

	return normalizeAdjacencyForConstruction(adjacency, vectors, dim, metric), nil
}

func validateAdjacencyForConstruction(adjacency [][]int, vectors []float32, dim int, metric Metric) error {
	if !validMetric(metric) {
		return fmt.Errorf("fasthnsw: unsupported metric %d", metric)
	}
	if dim <= 0 {
		return fmt.Errorf("fasthnsw: vector dimension must be positive")
	}
	if len(vectors)%dim != 0 {
		return fmt.Errorf("fasthnsw: flat vector storage is not aligned to dimension")
	}
	count := len(vectors) / dim
	if len(adjacency) != count {
		return fmt.Errorf("fasthnsw: adjacency count %d, want %d", len(adjacency), count)
	}
	for sourceID, neighbors := range adjacency {
		for _, neighborID := range neighbors {
			if neighborID < 0 || neighborID >= count {
				return fmt.Errorf("fasthnsw: adjacency[%d] has neighbor %d out of range [0,%d)", sourceID, neighborID, count)
			}
		}
	}
	return nil
}

func weakComponents(adjacency [][]int) [][]int {
	undirected := make([][]int, len(adjacency))
	for sourceID, neighbors := range adjacency {
		for _, neighborID := range neighbors {
			undirected[sourceID] = append(undirected[sourceID], neighborID)
			undirected[neighborID] = append(undirected[neighborID], sourceID)
		}
	}

	visited := make([]bool, len(adjacency))
	var components [][]int
	for start := 0; start < len(adjacency); start++ {
		if visited[start] {
			continue
		}

		queue := []int{start}
		visited[start] = true
		component := make([]int, 0)
		for len(queue) > 0 {
			nodeID := queue[0]
			queue = queue[1:]
			component = append(component, nodeID)
			for _, neighborID := range undirected[nodeID] {
				if visited[neighborID] {
					continue
				}
				visited[neighborID] = true
				queue = append(queue, neighborID)
			}
		}
		sort.Ints(component)
		components = append(components, component)
	}
	return components
}

func mainComponentIndex(components [][]int) int {
	mainIndex := 0
	for i := 1; i < len(components); i++ {
		if len(components[i]) > len(components[mainIndex]) {
			mainIndex = i
			continue
		}
		if len(components[i]) == len(components[mainIndex]) && components[i][0] < components[mainIndex][0] {
			mainIndex = i
		}
	}
	return mainIndex
}

type componentBridge struct {
	source   int
	target   int
	distance float32
}

func nearestComponentBridge(component []int, mainSet map[int]bool, vectors []float32, dim int, metric Metric) componentBridge {
	best := componentBridge{source: -1, target: -1}
	for _, sourceID := range component {
		for targetID := range mainSet {
			next := componentBridge{
				source:   sourceID,
				target:   targetID,
				distance: distance(metric, vectorAt(vectors, dim, sourceID), vectorAt(vectors, dim, targetID)),
			}
			if best.source == -1 || betterBridge(next, best) {
				best = next
			}
		}
	}
	return best
}

func betterBridge(a, b componentBridge) bool {
	if a.distance != b.distance {
		return a.distance < b.distance
	}
	if a.source != b.source {
		return a.source < b.source
	}
	return a.target < b.target
}

func normalizeAdjacencyForConstruction(adjacency [][]int, vectors []float32, dim int, metric Metric) [][]int {
	out := make([][]int, len(adjacency))
	for sourceID, neighbors := range adjacency {
		seen := make(map[int]bool, len(neighbors))
		candidates := make([]candidate, 0, len(neighbors))
		for _, neighborID := range neighbors {
			if neighborID == sourceID || seen[neighborID] {
				continue
			}
			seen[neighborID] = true
			candidates = append(candidates, candidate{
				id:       neighborID,
				distance: distance(metric, vectorAt(vectors, dim, sourceID), vectorAt(vectors, dim, neighborID)),
			})
		}
		sort.Slice(candidates, func(i, j int) bool {
			return betterCandidate(candidates[i], candidates[j])
		})
		out[sourceID] = candidateIDs(candidates)
	}
	return out
}
