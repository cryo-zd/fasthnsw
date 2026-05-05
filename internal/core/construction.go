package core

import (
	"math"
	"math/rand"
	"sort"

	"github.com/cryo-zd/fasthnsw/internal/eval"
)

const minApproxKNNGIterations = 4

type candidateQualityConfig struct {
	TargetRecall float64
	Controls     int
	ControlIDs   []int
	Seed         int64
	K            int
	Workers      int
}

type candidateRefinementStats struct {
	InitialRecall float64
	FinalRecall   float64
	Iterations    int
}

// buildSearchableGraph constructs all HNSW layers for the vectors already
// owned by idx. It follows the FastHNSW construction shape: assign every
// node's maximum layer first, then build each layer over the complete set of
// nodes present in that layer.
func (idx *Index) buildSearchableGraph() error {
	levels := assignLevels(idx.count, idx.cfg.M, idx.cfg.Seed)
	maxLayer := maxAssignedLayer(levels)
	entryPoint := selectEntryPoint(levels, maxLayer, idx.cfg.Seed)

	layers := make([][][]int, maxLayer+1)
	for layer := maxLayer; layer >= 0; layer-- {
		nodes := nodesAtOrAboveLayer(levels, layer)
		maxDegree := idx.cfg.M
		if layer == 0 {
			maxDegree = baseLayerMaxDegree(idx.cfg)
		}
		adjacency, err := buildHNSWLayer(layer, nodes, idx.count, idx.vectors, idx.dim, idx.cfg, maxDegree)
		if err != nil {
			return err
		}
		layers[layer] = adjacency
	}

	idx.levels = levels
	idx.layers = layers
	idx.entryPoint = entryPoint
	idx.maxLayer = maxLayer
	idx.graphReady = true
	return nil
}

// assignLevels samples each node's maximum HNSW layer using the standard
// exponential distribution. The configured seed is the only source of
// randomness, which keeps builds reproducible.
func assignLevels(count int, maxDegree int, seed int64) []int {
	levels := make([]int, count)
	if count == 0 || maxDegree <= 1 {
		// M=1 is a valid but sparse configuration. The logarithmic sampler is
		// undefined there, so construction falls back to a single-layer graph.
		return levels
	}

	rng := rand.New(rand.NewSource(seed))
	normalizer := math.Log(float64(maxDegree))
	for id := 0; id < count; id++ {
		u := rng.Float64()
		if u == 0 {
			u = math.SmallestNonzeroFloat64
		}
		levels[id] = int(-math.Log(u) / normalizer)
	}
	return levels
}

// maxAssignedLayer returns the highest sampled layer in a level table.
func maxAssignedLayer(levels []int) int {
	maxLayer := 0
	for _, level := range levels {
		if level > maxLayer {
			maxLayer = level
		}
	}
	return maxLayer
}

// selectEntryPoint picks a deterministic seeded-random node from the top layer.
func selectEntryPoint(levels []int, layer int, seed int64) int {
	top := nodesAtExactLayer(levels, layer)
	if len(top) == 0 {
		return -1
	}
	rng := rand.New(rand.NewSource(mixSeed(seed, int64(layer), int64(len(top)))))
	return top[rng.Intn(len(top))]
}

func nodesAtExactLayer(levels []int, layer int) []int {
	nodes := make([]int, 0)
	for id, level := range levels {
		if level == layer {
			nodes = append(nodes, id)
		}
	}
	return nodes
}

// nodesAtOrAboveLayer returns D_i = {u | level(u) >= i} in global id order.
func nodesAtOrAboveLayer(levels []int, layer int) []int {
	nodes := make([]int, 0, len(levels))
	for id, level := range levels {
		if level >= layer {
			nodes = append(nodes, id)
		}
	}
	return nodes
}

// buildHNSWLayer returns one global adjacency table for a layer node set.
// Small layers are made complete because their degree is already within the
// layer's HNSW bound: M for upper layers and 2*M for layer 0. Larger layers use
// IterNSG and final RNG pruning.
func buildHNSWLayer(layer int, nodes []int, totalCount int, vectors []float32, dim int, cfg Config, maxDegree int) ([][]int, error) {
	if len(nodes)-1 <= maxDegree {
		return buildCompleteLayer(nodes, totalCount, vectors, dim, cfg.Metric), nil
	}
	return buildIterNSGLayer(layer, nodes, totalCount, vectors, dim, cfg, maxDegree)
}

// buildCompleteLayer connects every node in a tiny layer to every other layer
// node. Neighbor order still follows distance then id so search behavior stays
// deterministic even before pruning is needed.
func buildCompleteLayer(nodes []int, totalCount int, vectors []float32, dim int, metric Metric) [][]int {
	adjacency := make([][]int, totalCount)
	for _, sourceID := range nodes {
		source := vectorAt(vectors, dim, sourceID)
		neighbors := make([]candidate, 0, len(nodes)-1)
		for _, neighborID := range nodes {
			if neighborID == sourceID {
				continue
			}
			neighbors = append(neighbors, candidate{
				id:       neighborID,
				distance: distance(metric, source, vectorAt(vectors, dim, neighborID)),
			})
		}
		sort.Slice(neighbors, func(i, j int) bool {
			return betterCandidate(neighbors[i], neighbors[j])
		})
		adjacency[sourceID] = candidateIDs(neighbors)
	}
	return adjacency
}

// buildIterNSGLayer builds one non-trivial HNSW layer. Candidate ids are local
// while IterNSG runs, then the final RNG-pruned adjacency is mapped back to the
// index's global node ids.
func buildIterNSGLayer(layer int, nodes []int, totalCount int, vectors []float32, dim int, cfg Config, maxDegree int) ([][]int, error) {
	localVectors := extractLayerVectors(nodes, vectors, dim)
	initialK := cfg.K0
	if initialK > len(nodes)-1 {
		initialK = len(nodes) - 1
	}
	candidateK := cfg.CandidateK
	if candidateK > len(nodes)-1 {
		candidateK = len(nodes) - 1
	}

	initialKNNG, err := approximateKNNGCandidates(
		localVectors,
		dim,
		cfg.Metric,
		initialK,
		mixSeed(cfg.Seed, int64(layer), int64(len(nodes))),
		approxKNNGIterations(cfg),
		cfg.Workers,
	)
	if err != nil {
		return nil, err
	}

	searchEf := cfg.ConstructionL
	if searchEf > len(nodes)-1 {
		searchEf = len(nodes) - 1
	}
	initialGraph := candidatesToAdjacency(initialKNNG)
	candidates := acquireCandidatesByGraphSearch(initialGraph, localVectors, dim, cfg.Metric, candidateK, searchEf, cfg.Workers)

	refreshCfg := fastHNSWOptKCNAConfig(cfg, candidateK, searchEf, maxDegree)
	controlSeed := mixSeed(cfg.Seed, int64(layer), int64(candidateK))
	qualityCfg := candidateQualityConfig{
		TargetRecall: cfg.CandidateRecall,
		Controls:     cfg.CandidateControls,
		ControlIDs:   selectCandidateControls(len(nodes), cfg.CandidateControls, controlSeed),
		Seed:         controlSeed,
		K:            candidateK,
		Workers:      cfg.Workers,
	}
	candidates, _, err = refineCandidatesUntilRecall(candidates, refreshCfg, qualityCfg, localVectors, dim, cfg.Metric, cfg.Iterations)
	if err != nil {
		return nil, err
	}

	localAdjacency, err := buildFinalHNSWLayer(candidates, maxDegree, localVectors, dim, cfg.Metric, cfg.Workers)
	if err != nil {
		return nil, err
	}
	return mapLayerAdjacencyToGlobal(localAdjacency, nodes, totalCount), nil
}

func buildFinalHNSWLayer(candidates [][]candidate, maxDegree int, vectors []float32, dim int, metric Metric, workers int) ([][]int, error) {
	return buildPrunedLayer(candidates, maxDegree, rngPruneMode(), vectors, dim, metric, workers)
}

func fastHNSWOptKCNAConfig(cfg Config, candidateK int, searchEf int, maxDegree int) optKCNAConfig {
	return optKCNAConfig{
		CandidateK:        candidateK,
		SearchEf:          searchEf,
		MaxDegree:         maxDegree,
		AlphaDegrees:      cfg.Alpha,
		ConnectComponents: false,
		Workers:           cfg.Workers,
	}
}

func baseLayerMaxDegree(cfg Config) int {
	return 2 * cfg.M
}

// approxKNNGIterations keeps the KNNG initializer independent from IterNSG's
// refinement loop count while still deriving a deterministic effort level from
// the public config.
func approxKNNGIterations(cfg Config) int {
	rounds := cfg.Iterations * 2
	if rounds < minApproxKNNGIterations {
		return minApproxKNNGIterations
	}
	return rounds
}

func candidatesToAdjacency(candidates [][]candidate) [][]int {
	adjacency := make([][]int, len(candidates))
	for sourceID, sourceCandidates := range candidates {
		adjacency[sourceID] = candidateIDs(sourceCandidates)
	}
	return adjacency
}

// acquireCandidatesByGraphSearch implements the Algorithm 6 transition from an
// initial KNNG to k-CNA candidate sets: each node searches the current graph
// using its own vector as the query, then keeps the nearest k non-self results.
func acquireCandidatesByGraphSearch(adjacency [][]int, vectors []float32, dim int, metric Metric, candidateK int, searchEf int, workers int) [][]candidate {
	out := make([][]candidate, len(adjacency))
	workerCount := effectiveWorkerCount(workers, len(adjacency))
	scratches := makeGraphSearchScratches(workerCount, len(adjacency), searchEf)
	_ = parallelForNodes(len(adjacency), workers, func(workerID int, sourceID int) error {
		results := graphSearchLayer(adjacency, vectors, dim, metric, sourceID, vectorAt(vectors, dim, sourceID), searchEf, &scratches[workerID])
		out[sourceID] = candidatesFromResults(sourceID, results, candidateK)
		return nil
	})
	return out
}

// refineCandidatesUntilRecall follows Algorithm 6's stopping rule. Candidate
// quality is estimated before each OptKCNA round and after each refresh; the
// public Iterations config is only a maximum round cap, matching the official
// artifact's recall-threshold plus loop-cap structure.
func refineCandidatesUntilRecall(candidates [][]candidate, refreshCfg optKCNAConfig, qualityCfg candidateQualityConfig, vectors []float32, dim int, metric Metric, maxIterations int) ([][]candidate, candidateRefinementStats, error) {
	recall, err := estimateCandidateRecall(candidates, qualityCfg, vectors, dim, metric)
	if err != nil {
		return nil, candidateRefinementStats{}, err
	}
	stats := candidateRefinementStats{InitialRecall: recall, FinalRecall: recall}
	for stats.Iterations < maxIterations && stats.FinalRecall < qualityCfg.TargetRecall {
		candidates, err = optKCNA(candidates, refreshCfg, vectors, dim, metric)
		if err != nil {
			return nil, candidateRefinementStats{}, err
		}
		stats.Iterations++
		stats.FinalRecall, err = estimateCandidateRecall(candidates, qualityCfg, vectors, dim, metric)
		if err != nil {
			return nil, candidateRefinementStats{}, err
		}
	}
	return candidates, stats, nil
}

// estimateCandidateRecall is the paper's r-hat estimator for k-CNA quality. It
// compares candidate lists on a deterministic control subset against exact
// neighbors for those controls, avoiding an all-node exact evaluation during
// normal construction.
func estimateCandidateRecall(candidates [][]candidate, cfg candidateQualityConfig, vectors []float32, dim int, metric Metric) (float64, error) {
	if cfg.K <= 0 {
		return 0, nil
	}
	count := len(vectors) / dim
	if count == 0 {
		return 1, nil
	}
	controls := cfg.ControlIDs
	if controls == nil {
		controls = selectCandidateControls(count, cfg.Controls, cfg.Seed)
	}
	controlHits := make([]int, len(controls))
	controlTotals := make([]int, len(controls))
	_ = parallelForNodes(len(controls), cfg.Workers, func(_ int, controlIndex int) error {
		sourceID := controls[controlIndex]
		exact := exactCandidatesForNode(vectors, dim, metric, sourceID, minInt(cfg.K, count-1))
		sourceHits, sourceTotal := eval.CountHitsAtK(candidateIDs(candidates[sourceID]), candidateIDs(exact), cfg.K)
		controlHits[controlIndex] = sourceHits
		controlTotals[controlIndex] = sourceTotal
		return nil
	})

	var hits int
	var total int
	for i := range controls {
		hits += controlHits[i]
		total += controlTotals[i]
	}
	if total == 0 {
		return 1, nil
	}
	return float64(hits) / float64(total), nil
}

func selectCandidateControls(count int, maxControls int, seed int64) []int {
	if count <= 0 || maxControls == 0 {
		return nil
	}
	if maxControls >= count {
		controls := make([]int, count)
		for i := range controls {
			controls[i] = i
		}
		return controls
	}
	rng := rand.New(rand.NewSource(seed))
	perm := rng.Perm(count)
	controls := append([]int(nil), perm[:maxControls]...)
	sort.Ints(controls)
	return controls
}

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

// extractLayerVectors copies a layer's global vectors into a dense local block
// so construction helpers can use compact local ids without sparse lookups.
func extractLayerVectors(nodes []int, vectors []float32, dim int) []float32 {
	out := make([]float32, len(nodes)*dim)
	for localID, globalID := range nodes {
		copy(out[localID*dim:(localID+1)*dim], vectorAt(vectors, dim, globalID))
	}
	return out
}

// mapLayerAdjacencyToGlobal expands local IterNSG adjacency back to the full
// index shape expected by Search: adjacency[globalNodeID] -> global neighbors.
func mapLayerAdjacencyToGlobal(localAdjacency [][]int, nodes []int, totalCount int) [][]int {
	globalAdjacency := make([][]int, totalCount)
	for localID, neighbors := range localAdjacency {
		globalID := nodes[localID]
		globalNeighbors := make([]int, len(neighbors))
		for i, localNeighborID := range neighbors {
			globalNeighbors[i] = nodes[localNeighborID]
		}
		globalAdjacency[globalID] = globalNeighbors
	}
	return globalAdjacency
}
