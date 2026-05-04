package benchdata

import (
	"fmt"
	"math"
	"os"
	"time"

	"github.com/cryo-zd/fasthnsw"
)

// Dataset is an in-memory benchmark dataset. It is intentionally internal so
// public users do not inherit benchmark-loader compatibility constraints.
type Dataset struct {
	Name        string
	Metric      fasthnsw.Metric
	Base        [][]float32
	Queries     [][]float32
	GroundTruth [][]int
}

// Dim returns the dataset vector dimension.
func (d Dataset) Dim() int {
	if len(d.Base) == 0 {
		return 0
	}
	return len(d.Base[0])
}

// Validate checks that the dataset can be used for Recall@k validation.
func (d Dataset) Validate(k int) error {
	if d.Name == "" {
		return fmt.Errorf("dataset name must not be empty")
	}
	if len(d.Base) == 0 {
		return fmt.Errorf("dataset base vectors must not be empty")
	}
	if len(d.Queries) == 0 {
		return fmt.Errorf("dataset queries must not be empty")
	}
	dim := len(d.Base[0])
	if dim == 0 {
		return fmt.Errorf("dataset dimension must be positive")
	}
	for i, vector := range d.Base {
		if len(vector) != dim {
			return fmt.Errorf("base vector %d has dimension %d, want %d", i, len(vector), dim)
		}
	}
	for i, query := range d.Queries {
		if len(query) != dim {
			return fmt.Errorf("query vector %d has dimension %d, want %d", i, len(query), dim)
		}
	}
	if len(d.GroundTruth) != len(d.Queries) {
		return fmt.Errorf("ground truth count %d does not match query count %d", len(d.GroundTruth), len(d.Queries))
	}
	for i, neighbors := range d.GroundTruth {
		if len(neighbors) < k {
			return fmt.Errorf("ground truth row %d has %d neighbors, want at least %d", i, len(neighbors), k)
		}
		for rank := 0; rank < k; rank++ {
			if neighbors[rank] < 0 || neighbors[rank] >= len(d.Base) {
				return fmt.Errorf("ground truth row %d rank %d has id %d outside loaded base size %d", i, rank, neighbors[rank], len(d.Base))
			}
		}
	}
	return nil
}

// ValidationResult is the machine-readable summary produced by a validation
// run over one dataset and one build/search configuration.
type ValidationResult struct {
	Dataset     Dataset
	K           int
	EfSearch    int
	BuildTime   time.Duration
	QueryTime   time.Duration
	Recall      float64
	IndexBytes  int64
	SearchCount int
}

// QPS returns single-query throughput over the measured search loop.
func (r ValidationResult) QPS() float64 {
	if r.QueryTime <= 0 {
		return 0
	}
	return float64(r.SearchCount) / r.QueryTime.Seconds()
}

// RunValidation builds an index, searches every query, and compares results
// against the dataset ground truth. If saveIndexPath is non-empty, the built
// index is persisted and the resulting file size is reported.
func RunValidation(dataset Dataset, cfg fasthnsw.Config, k int, efSearch int, saveIndexPath string) (ValidationResult, error) {
	if k <= 0 {
		return ValidationResult{}, fmt.Errorf("k must be positive")
	}
	if efSearch < k {
		return ValidationResult{}, fmt.Errorf("ef must be greater than or equal to k")
	}
	if err := dataset.Validate(k); err != nil {
		return ValidationResult{}, err
	}

	cfg.Metric = dataset.Metric
	if cfg.Dim == 0 {
		cfg.Dim = dataset.Dim()
	}
	idx, err := fasthnsw.New(cfg)
	if err != nil {
		return ValidationResult{}, err
	}

	buildStart := time.Now()
	if err := idx.Build(dataset.Base); err != nil {
		return ValidationResult{}, err
	}
	buildTime := time.Since(buildStart)

	var indexBytes int64
	if saveIndexPath != "" {
		file, err := os.Create(saveIndexPath)
		if err != nil {
			return ValidationResult{}, fmt.Errorf("create saved index: %w", err)
		}
		if err := idx.Save(file); err != nil {
			if closeErr := file.Close(); closeErr != nil {
				return ValidationResult{}, fmt.Errorf("save index: %w; close index: %w", err, closeErr)
			}
			return ValidationResult{}, fmt.Errorf("save index: %w", err)
		}
		if err := file.Close(); err != nil {
			return ValidationResult{}, fmt.Errorf("close saved index: %w", err)
		}
		info, err := os.Stat(saveIndexPath)
		if err != nil {
			return ValidationResult{}, fmt.Errorf("stat saved index: %w", err)
		}
		indexBytes = info.Size()
	}

	searchStart := time.Now()
	var recallTotal float64
	for queryID, query := range dataset.Queries {
		results, err := idx.Search(query, k, efSearch)
		if err != nil {
			return ValidationResult{}, fmt.Errorf("query %d: %w", queryID, err)
		}
		recallTotal += RecallAtK(results, dataset.GroundTruth[queryID], k)
	}
	queryTime := time.Since(searchStart)

	return ValidationResult{
		Dataset:     dataset,
		K:           k,
		EfSearch:    efSearch,
		BuildTime:   buildTime,
		QueryTime:   queryTime,
		Recall:      recallTotal / float64(len(dataset.Queries)),
		IndexBytes:  indexBytes,
		SearchCount: len(dataset.Queries),
	}, nil
}

// RecallAtK returns the fraction of the true top-k neighbor ids found in got.
func RecallAtK(got []fasthnsw.Result, truth []int, k int) float64 {
	if k == 0 {
		return 1
	}
	truthIDs := make(map[int]bool, k)
	for i := 0; i < k && i < len(truth); i++ {
		truthIDs[truth[i]] = true
	}

	var hits int
	limit := k
	if limit > len(got) {
		limit = len(got)
	}
	for i := 0; i < limit; i++ {
		if truthIDs[got[i].ID] {
			hits++
		}
	}
	return float64(hits) / float64(k)
}

// ExactGroundTruth computes deterministic exact nearest-neighbor ids for
// generated validation datasets that do not ship precomputed ground truth.
func ExactGroundTruth(base [][]float32, queries [][]float32, metric fasthnsw.Metric, k int) ([][]int, error) {
	if k <= 0 {
		return nil, fmt.Errorf("k must be positive")
	}
	if len(base) == 0 || len(queries) == 0 {
		return nil, fmt.Errorf("base and query vectors must not be empty")
	}
	if k > len(base) {
		k = len(base)
	}
	preparedBase, err := prepareVectors(base, metric)
	if err != nil {
		return nil, err
	}
	preparedQueries, err := prepareVectors(queries, metric)
	if err != nil {
		return nil, err
	}

	truth := make([][]int, len(preparedQueries))
	for queryID, query := range preparedQueries {
		candidates := make([]neighbor, 0, len(preparedBase))
		for id, vector := range preparedBase {
			candidates = append(candidates, neighbor{
				id:       id,
				distance: distance(metric, vector, query),
			})
		}
		sortNeighbors(candidates)
		ids := make([]int, k)
		for i := 0; i < k; i++ {
			ids[i] = candidates[i].id
		}
		truth[queryID] = ids
	}
	return truth, nil
}

type neighbor struct {
	id       int
	distance float32
}

func sortNeighbors(neighbors []neighbor) {
	for i := 1; i < len(neighbors); i++ {
		current := neighbors[i]
		j := i - 1
		for j >= 0 && worseNeighbor(neighbors[j], current) {
			neighbors[j+1] = neighbors[j]
			j--
		}
		neighbors[j+1] = current
	}
}

func worseNeighbor(left neighbor, right neighbor) bool {
	if left.distance != right.distance {
		return left.distance > right.distance
	}
	return left.id > right.id
}

func prepareVectors(vectors [][]float32, metric fasthnsw.Metric) ([][]float32, error) {
	if metric != fasthnsw.MetricCosine {
		return vectors, nil
	}
	out := make([][]float32, len(vectors))
	for i, vector := range vectors {
		normalized, err := normalize(vector)
		if err != nil {
			return nil, fmt.Errorf("vector %d: %w", i, err)
		}
		out[i] = normalized
	}
	return out, nil
}

func normalize(vector []float32) ([]float32, error) {
	var normSquared float64
	for _, value := range vector {
		v := float64(value)
		normSquared += v * v
	}
	if normSquared == 0 {
		return nil, fmt.Errorf("cosine vectors must not be zero vectors")
	}
	inv := float32(1 / math.Sqrt(normSquared))
	out := make([]float32, len(vector))
	for i, value := range vector {
		out[i] = value * inv
	}
	return out, nil
}

func distance(metric fasthnsw.Metric, left []float32, right []float32) float32 {
	switch metric {
	case fasthnsw.MetricL2:
		var sum float32
		for i := range left {
			d := left[i] - right[i]
			sum += d * d
		}
		return sum
	case fasthnsw.MetricCosine:
		var dot float32
		for i := range left {
			dot += left[i] * right[i]
		}
		return 1 - dot
	default:
		panic("unsupported metric reached benchmark distance")
	}
}
