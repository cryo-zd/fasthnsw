package fasthnsw

import "fmt"

type pruneKind int

const (
	pruneKindRNG pruneKind = iota
	pruneKindAlpha
)

// pruneMode selects the pruning rule used while assembling construction
// graphs. RNG mode is used for final HNSW layers; alpha mode is used for
// intermediate graphs in the paper's refinement-before-search construction.
type pruneMode struct {
	kind         pruneKind
	alphaDegrees float64
}

// rngPruneMode returns the final-layer RNG pruning mode.
func rngPruneMode() pruneMode {
	return pruneMode{kind: pruneKindRNG}
}

// alphaPruneMode returns the intermediate alpha-pruning mode.
func alphaPruneMode(alphaDegrees float64) pruneMode {
	return pruneMode{kind: pruneKindAlpha, alphaDegrees: alphaDegrees}
}

// buildPrunedLayer converts per-node candidate lists into one adjacency layer.
//
// The assembly mirrors the graph-construction flow used by NSG/HNSW-style
// builders: prune each node's forward candidates, add retained edges in reverse
// direction, then prune again so reverse-edge repair does not violate the
// degree bound. The output shape matches Index.layers[layer]: adjacency[nodeID]
// is a deterministic list of neighbor ids ordered by candidate distance then id.
func buildPrunedLayer(candidates [][]candidate, maxDegree int, mode pruneMode, vectors []float32, dim int, metric Metric) ([][]int, error) {
	count, err := validatePrunedLayerInput(candidates, maxDegree, mode, vectors, dim, metric)
	if err != nil {
		return nil, err
	}

	adjacency := make([][]int, count)
	if count == 0 {
		return adjacency, nil
	}

	forward := make([][]candidate, count)
	for sourceID := 0; sourceID < count; sourceID++ {
		pruned, err := applyPruneMode(sourceID, candidates[sourceID], maxDegree, mode, vectors, dim, metric)
		if err != nil {
			return nil, err
		}
		forward[sourceID] = pruned
	}

	merged := make([][]candidate, count)
	for sourceID, pruned := range forward {
		merged[sourceID] = append(merged[sourceID], pruned...)
		for _, neighbor := range pruned {
			merged[neighbor.id] = append(merged[neighbor.id], candidate{id: sourceID})
		}
	}

	for sourceID := 0; sourceID < count; sourceID++ {
		pruned, err := applyPruneMode(sourceID, merged[sourceID], maxDegree, mode, vectors, dim, metric)
		if err != nil {
			return nil, err
		}
		adjacency[sourceID] = candidateIDs(pruned)
	}
	return adjacency, nil
}

func validatePrunedLayerInput(candidates [][]candidate, maxDegree int, mode pruneMode, vectors []float32, dim int, metric Metric) (int, error) {
	if maxDegree <= 0 {
		return 0, fmt.Errorf("fasthnsw: maxDegree must be positive")
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
	if err := validatePruneMode(mode); err != nil {
		return 0, err
	}

	count := len(vectors) / dim
	if len(candidates) != count {
		return 0, fmt.Errorf("fasthnsw: candidate list count %d, want %d", len(candidates), count)
	}
	return count, nil
}

func validatePruneMode(mode pruneMode) error {
	switch mode.kind {
	case pruneKindRNG:
		return nil
	case pruneKindAlpha:
		if mode.alphaDegrees < minAlpha || mode.alphaDegrees > maxAlphaDegrees {
			return fmt.Errorf("fasthnsw: alpha must be in [%.0f, %.0f]", float64(minAlpha), float64(maxAlphaDegrees))
		}
		return nil
	default:
		return fmt.Errorf("fasthnsw: unsupported prune mode %d", mode.kind)
	}
}

func applyPruneMode(sourceID int, candidates []candidate, maxDegree int, mode pruneMode, vectors []float32, dim int, metric Metric) ([]candidate, error) {
	switch mode.kind {
	case pruneKindRNG:
		return rngPrune(sourceID, candidates, maxDegree, vectors, dim, metric)
	case pruneKindAlpha:
		return alphaPrune(sourceID, candidates, maxDegree, mode.alphaDegrees, vectors, dim, metric)
	default:
		return nil, fmt.Errorf("fasthnsw: unsupported prune mode %d", mode.kind)
	}
}

func candidateIDs(candidates []candidate) []int {
	ids := make([]int, len(candidates))
	for i, candidate := range candidates {
		ids[i] = candidate.id
	}
	return ids
}
