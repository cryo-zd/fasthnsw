package core

import (
	"fmt"
	"math/rand"
)

type levelSampler struct {
	rng   *rand.Rand
	count int
}

func newLevelSampler(seed int64) *levelSampler {
	return &levelSampler{rng: rand.New(rand.NewSource(seed))}
}

func (s *levelSampler) advance(count int, maxDegree int) {
	for s.count < count {
		s.next(maxDegree)
	}
}

func (s *levelSampler) next(maxDegree int) int {
	s.count++
	return sampleLevel(s.rng, maxDegree)
}

// AddVector appends one vector to the index and inserts it with the standard
// HNSW insertion algorithm.
//
// Batch Build remains the FastHNSW construction path. AddVector is intended for
// incremental updates after Build, or for building a small index one vector at a
// time. It returns the assigned internal id, which is the previous vector
// count. Index mutation is not internally synchronized; callers must not run
// AddVector concurrently with Search, Save, or Build.
func (idx *Index) AddVector(vector []float32) (int, error) {
	if idx == nil {
		return -1, fmt.Errorf("fasthnsw: nil index")
	}
	if err := idx.validateAddVectorInput(vector); err != nil {
		return -1, err
	}
	if idx.count > 0 {
		if err := idx.validateIncrementalGraphState(); err != nil {
			return -1, err
		}
	}

	oldCount := idx.count
	dim := idx.dim
	if dim == 0 {
		dim = idx.cfg.Dim
	}
	if dim == 0 {
		dim = len(vector)
	}

	stored, err := prepareStoredVector(vector, dim, idx.cfg.Metric)
	if err != nil {
		return -1, err
	}
	level := idx.nextIncrementalLevel()

	if oldCount == 0 {
		idx.vectors = append(idx.vectors[:0], stored...)
		idx.dim = dim
		idx.count = 1
		idx.cfg.Dim = dim
		idx.levels = []int{level}
		idx.layers = make([][][]int, level+1)
		for layer := range idx.layers {
			idx.layers[layer] = make([][]int, 1)
		}
		idx.entryPoint = 0
		idx.maxLayer = level
		idx.graphReady = true
		return 0, nil
	}

	idx.vectors = append(idx.vectors, stored...)
	idx.count = oldCount + 1
	idx.levels = append(idx.levels, level)
	idx.expandLayersForNewNode(level)
	if err := idx.insertStandardHNSWNode(oldCount, level, effectiveStandardEfConstruction(idx.cfg), &graphSearchScratch{}); err != nil {
		idx.graphReady = false
		idx.levelSampler = nil
		return -1, err
	}
	idx.graphReady = true
	return oldCount, nil
}

func (idx *Index) validateAddVectorInput(vector []float32) error {
	dim := idx.dim
	if dim == 0 {
		dim = idx.cfg.Dim
	}
	return validateVector(vector, dim, "vector")
}

func (idx *Index) validateIncrementalGraphState() error {
	if !idx.graphReady {
		return ErrIndexNotBuilt
	}
	if idx.dim <= 0 {
		return fmt.Errorf("fasthnsw: index dimension must be positive")
	}
	if len(idx.vectors) != idx.count*idx.dim {
		return fmt.Errorf("fasthnsw: flat vector length %d does not match count*dim %d", len(idx.vectors), idx.count*idx.dim)
	}
	if len(idx.levels) != idx.count {
		return fmt.Errorf("fasthnsw: level count %d does not match vector count %d", len(idx.levels), idx.count)
	}
	if idx.entryPoint < 0 || idx.entryPoint >= idx.count {
		return fmt.Errorf("fasthnsw: entry point %d out of range", idx.entryPoint)
	}
	if idx.maxLayer < 0 || len(idx.layers) != idx.maxLayer+1 {
		return fmt.Errorf("fasthnsw: layer count %d does not match max layer %d", len(idx.layers), idx.maxLayer)
	}
	for layer, adjacency := range idx.layers {
		if len(adjacency) != idx.count {
			return fmt.Errorf("fasthnsw: layer %d adjacency length %d does not match vector count %d", layer, len(adjacency), idx.count)
		}
	}
	return nil
}

func prepareStoredVector(vector []float32, dim int, metric Metric) ([]float32, error) {
	stored := make([]float32, dim)
	switch metric {
	case MetricL2:
		copy(stored, vector)
		return stored, nil
	case MetricCosine:
		if err := normalizeInto(stored, vector); err != nil {
			return nil, fmt.Errorf("fasthnsw: vector: %w", err)
		}
		return stored, nil
	default:
		return nil, fmt.Errorf("fasthnsw: unsupported metric %d", metric)
	}
}

func (idx *Index) nextIncrementalLevel() int {
	if idx.levelSampler == nil || idx.levelSampler.count != idx.count {
		idx.levelSampler = newLevelSampler(idx.cfg.Seed)
		idx.levelSampler.advance(idx.count, idx.cfg.M)
	}
	return idx.levelSampler.next(idx.cfg.M)
}

func (idx *Index) expandLayersForNewNode(level int) {
	newCount := idx.count
	for layer := range idx.layers {
		idx.layers[layer] = append(idx.layers[layer], nil)
	}
	for layer := len(idx.layers); layer <= level; layer++ {
		idx.layers = append(idx.layers, make([][]int, newCount))
	}
}
