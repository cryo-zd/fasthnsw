package fasthnsw

import "fmt"

// resetGraph clears searchable graph metadata after vectors are replaced.
func (idx *Index) resetGraph() {
	idx.layers = nil
	idx.entryPoint = -1
	idx.maxLayer = -1
	idx.graphReady = false
}

// validateSearchableGraph checks the internal graph shape before search.
// The construction code added in later phases should produce this same layout:
// one adjacency table per layer, each table indexed by global node id.
func (idx *Index) validateSearchableGraph() error {
	if !idx.graphReady {
		return ErrIndexNotBuilt
	}
	if idx.count <= 0 || idx.dim <= 0 || len(idx.vectors) != idx.count*idx.dim {
		return fmt.Errorf("fasthnsw: index has invalid vector storage")
	}
	if idx.entryPoint < 0 || idx.entryPoint >= idx.count {
		return fmt.Errorf("fasthnsw: entry point %d out of range [0,%d)", idx.entryPoint, idx.count)
	}
	if idx.maxLayer < 0 {
		return fmt.Errorf("fasthnsw: max layer must be non-negative")
	}
	if len(idx.layers) != idx.maxLayer+1 {
		return fmt.Errorf("fasthnsw: graph has %d layers, want %d", len(idx.layers), idx.maxLayer+1)
	}

	for layer, adjacency := range idx.layers {
		if len(adjacency) != idx.count {
			return fmt.Errorf("fasthnsw: layer %d has %d node lists, want %d", layer, len(adjacency), idx.count)
		}
		for nodeID, neighbors := range adjacency {
			for _, neighborID := range neighbors {
				if neighborID < 0 || neighborID >= idx.count {
					return fmt.Errorf("fasthnsw: layer %d node %d has neighbor %d out of range [0,%d)", layer, nodeID, neighborID, idx.count)
				}
			}
		}
	}

	return nil
}
