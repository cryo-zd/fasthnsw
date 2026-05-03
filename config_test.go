package fasthnsw

import "testing"

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Metric != MetricL2 {
		t.Fatalf("default metric = %v, want %v", cfg.Metric, MetricL2)
	}
	if cfg.M != defaultM {
		t.Fatalf("default M = %d, want %d", cfg.M, defaultM)
	}
	if cfg.EfConstruction != defaultEfConstruction {
		t.Fatalf("default EfConstruction = %d, want %d", cfg.EfConstruction, defaultEfConstruction)
	}
	if cfg.K0 != defaultK0 {
		t.Fatalf("default K0 = %d, want %d", cfg.K0, defaultK0)
	}
	if cfg.Alpha != defaultAlpha {
		t.Fatalf("default Alpha = %v, want %v", cfg.Alpha, defaultAlpha)
	}
	if cfg.Iterations != defaultIterations {
		t.Fatalf("default Iterations = %d, want %d", cfg.Iterations, defaultIterations)
	}
	if cfg.Seed != defaultSeed {
		t.Fatalf("default Seed = %d, want %d", cfg.Seed, defaultSeed)
	}
	if cfg.Workers <= 0 {
		t.Fatalf("default Workers = %d, want positive", cfg.Workers)
	}
	if cfg.CandidateRecall != defaultCandidateRecall {
		t.Fatalf("default CandidateRecall = %v, want %v", cfg.CandidateRecall, defaultCandidateRecall)
	}
	if cfg.CandidateControls != defaultCandidateControls {
		t.Fatalf("default CandidateControls = %d, want %d", cfg.CandidateControls, defaultCandidateControls)
	}
}

func TestNewAcceptsDefaultConfig(t *testing.T) {
	idx, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New(DefaultConfig()) returned error: %v", err)
	}
	if idx == nil {
		t.Fatal("New(DefaultConfig()) returned nil index")
	}
}

func TestNewNormalizesZeroConfig(t *testing.T) {
	idx, err := New(Config{})
	if err != nil {
		t.Fatalf("New(Config{}) returned error: %v", err)
	}
	if idx.cfg.M != defaultM {
		t.Fatalf("normalized M = %d, want %d", idx.cfg.M, defaultM)
	}
	if idx.cfg.Workers <= 0 {
		t.Fatalf("normalized Workers = %d, want positive", idx.cfg.Workers)
	}
}

func TestNewRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
	}{
		{name: "metric", cfg: Config{Metric: Metric(99)}},
		{name: "dim", cfg: Config{Dim: -1}},
		{name: "m", cfg: Config{M: -1}},
		{name: "ef construction", cfg: Config{EfConstruction: -1}},
		{name: "k0", cfg: Config{K0: -1}},
		{name: "alpha", cfg: Config{Alpha: 59}},
		{name: "alpha high", cfg: Config{Alpha: 181}},
		{name: "iterations", cfg: Config{Iterations: -1}},
		{name: "workers", cfg: Config{Workers: -1}},
		{name: "candidate recall low", cfg: Config{CandidateRecall: -1}},
		{name: "candidate recall high", cfg: Config{CandidateRecall: 1.01}},
		{name: "candidate controls", cfg: Config{CandidateControls: -1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := New(tt.cfg); err == nil {
				t.Fatal("New returned nil error")
			}
		})
	}
}
