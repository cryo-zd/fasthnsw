package fasthnsw

import (
	"fmt"
	"runtime"
)

// Metric identifies the distance function used by an index.
type Metric int

const (
	// MetricL2 uses squared Euclidean distance.
	MetricL2 Metric = iota
	// MetricCosine uses cosine distance.
	MetricCosine
)

const (
	defaultM                 = 32
	defaultEfConstruction    = 200
	defaultK0                = 200
	defaultAlpha             = 67
	defaultIterations        = 2
	defaultSeed              = int64(1)
	defaultCandidateRecall   = 0.98
	defaultCandidateControls = 100
	minAlpha                 = 60
)

// Config controls index construction and search behavior.
//
// A zero value for tunable construction fields uses the package defaults.
// Dim may be zero to infer dimensionality from the dataset during Build.
type Config struct {
	Metric         Metric
	Dim            int
	M              int
	EfConstruction int
	K0             int
	Alpha          float64
	Iterations     int
	Seed           int64
	Workers        int
	// CandidateRecall is the IterNSG candidate-quality requirement. A zero
	// value uses the package default.
	CandidateRecall float64
	// CandidateControls is the number of deterministic control nodes used by
	// the IterNSG candidate-quality estimator. A zero value uses the package
	// default.
	CandidateControls int
}

// DefaultConfig returns deterministic defaults suitable for a first index.
func DefaultConfig() Config {
	return Config{
		Metric:            MetricL2,
		M:                 defaultM,
		EfConstruction:    defaultEfConstruction,
		K0:                defaultK0,
		Alpha:             defaultAlpha,
		Iterations:        defaultIterations,
		Seed:              defaultSeed,
		Workers:           runtime.GOMAXPROCS(0),
		CandidateRecall:   defaultCandidateRecall,
		CandidateControls: defaultCandidateControls,
	}
}

func normalizeConfig(cfg Config) (Config, error) {
	if !validMetric(cfg.Metric) {
		return Config{}, fmt.Errorf("fasthnsw: unsupported metric %d", cfg.Metric)
	}
	if cfg.Dim < 0 {
		return Config{}, fmt.Errorf("fasthnsw: dimension must be non-negative")
	}

	defaults := DefaultConfig()
	if cfg.M == 0 {
		cfg.M = defaults.M
	}
	if cfg.EfConstruction == 0 {
		cfg.EfConstruction = defaults.EfConstruction
	}
	if cfg.K0 == 0 {
		cfg.K0 = defaults.K0
	}
	if cfg.Alpha == 0 {
		cfg.Alpha = defaults.Alpha
	}
	if cfg.Iterations == 0 {
		cfg.Iterations = defaults.Iterations
	}
	if cfg.Seed == 0 {
		cfg.Seed = defaults.Seed
	}
	if cfg.Workers == 0 {
		cfg.Workers = defaults.Workers
	}
	if cfg.CandidateRecall == 0 {
		cfg.CandidateRecall = defaults.CandidateRecall
	}
	if cfg.CandidateControls == 0 {
		cfg.CandidateControls = defaults.CandidateControls
	}

	if cfg.M < 0 {
		return Config{}, fmt.Errorf("fasthnsw: M must be positive")
	}
	if cfg.EfConstruction < 0 {
		return Config{}, fmt.Errorf("fasthnsw: EfConstruction must be positive")
	}
	if cfg.K0 < 0 {
		return Config{}, fmt.Errorf("fasthnsw: K0 must be positive")
	}
	if cfg.Alpha < minAlpha {
		return Config{}, fmt.Errorf("fasthnsw: Alpha must be at least %.0f", float64(minAlpha))
	}
	if cfg.Alpha > maxAlphaDegrees {
		return Config{}, fmt.Errorf("fasthnsw: Alpha must be at most %.0f", float64(maxAlphaDegrees))
	}
	if cfg.Iterations < 0 {
		return Config{}, fmt.Errorf("fasthnsw: Iterations must be positive")
	}
	if cfg.Workers < 0 {
		return Config{}, fmt.Errorf("fasthnsw: Workers must be positive")
	}
	if cfg.CandidateRecall <= 0 || cfg.CandidateRecall > 1 {
		return Config{}, fmt.Errorf("fasthnsw: CandidateRecall must be in (0,1]")
	}
	if cfg.CandidateControls < 0 {
		return Config{}, fmt.Errorf("fasthnsw: CandidateControls must be positive")
	}

	return cfg, nil
}

func validMetric(metric Metric) bool {
	return metric == MetricL2 || metric == MetricCosine
}
