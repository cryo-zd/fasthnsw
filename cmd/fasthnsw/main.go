package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/cryo-zd/fasthnsw"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout io.Writer, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stderr)
		return 2
	}

	var err error
	switch args[0] {
	case "build":
		err = runBuild(args[1:], stdout)
	case "query":
		err = runQuery(args[1:], stdout)
	case "help", "-help", "--help":
		printUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fasthnsw: unknown subcommand %q\n", args[0])
		printUsage(stderr)
		return 2
	}
	if err != nil {
		fmt.Fprintf(stderr, "fasthnsw: %v\n", err)
		return 1
	}
	return 0
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fasthnsw <build|query> [flags]")
	fmt.Fprintln(w, "  build -input vectors.txt -output index.fhnsw [config flags]")
	fmt.Fprintln(w, "  query -index index.fhnsw -queries queries.txt -k 10 -ef 64")
}

func runBuild(args []string, stdout io.Writer) error {
	cfgFlags := defaultConfigFlags()
	fs := newFlagSet("build")
	inputPath := fs.String("input", "", "input text vector file")
	outputPath := fs.String("output", "", "output index file")
	addConfigFlags(fs, &cfgFlags)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("build received unexpected positional arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *inputPath == "" || *outputPath == "" {
		return errors.New("build requires -input and -output")
	}

	cfg, err := cfgFlags.config()
	if err != nil {
		return err
	}
	vectors, err := readTextVectors(*inputPath)
	if err != nil {
		return err
	}
	idx, err := fasthnsw.New(cfg)
	if err != nil {
		return err
	}
	if err := idx.Build(vectors); err != nil {
		return err
	}

	file, err := os.Create(*outputPath)
	if err != nil {
		return fmt.Errorf("create index file: %w", err)
	}
	if err := idx.Save(file); err != nil {
		if closeErr := file.Close(); closeErr != nil {
			return errors.Join(err, fmt.Errorf("close index file: %w", closeErr))
		}
		return err
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close index file: %w", err)
	}

	fmt.Fprintf(stdout, "built index vectors=%d dim=%d output=%s\n", len(vectors), len(vectors[0]), *outputPath)
	return nil
}

func runQuery(args []string, stdout io.Writer) error {
	fs := newFlagSet("query")
	indexPath := fs.String("index", "", "input index file")
	queriesPath := fs.String("queries", "", "input text query vector file")
	k := fs.Int("k", 10, "number of nearest neighbors")
	efSearch := fs.Int("ef", 64, "HNSW search width")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("query received unexpected positional arguments: %s", strings.Join(fs.Args(), " "))
	}
	if *indexPath == "" || *queriesPath == "" {
		return errors.New("query requires -index and -queries")
	}

	file, err := os.Open(*indexPath)
	if err != nil {
		return fmt.Errorf("open index file: %w", err)
	}
	defer file.Close()
	idx, err := fasthnsw.Load(file)
	if err != nil {
		return err
	}

	queries, err := readTextVectors(*queriesPath)
	if err != nil {
		return err
	}
	for queryID, query := range queries {
		results, err := idx.Search(query, *k, *efSearch)
		if err != nil {
			return fmt.Errorf("query %d: %w", queryID, err)
		}
		for rank, result := range results {
			fmt.Fprintf(stdout, "query=%d rank=%d id=%d distance=%.6f\n", queryID, rank, result.ID, result.Distance)
		}
	}
	return nil
}

func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	return fs
}

type configFlags struct {
	metric            string
	dim               int
	m                 int
	k0                int
	candidateK        int
	constructionL     int
	alpha             float64
	iterations        int
	seed              int64
	candidateRecall   float64
	candidateControls int
}

func defaultConfigFlags() configFlags {
	cfg := fasthnsw.DefaultConfig()
	return configFlags{
		metric:            "l2",
		dim:               0,
		m:                 cfg.M,
		k0:                cfg.K0,
		candidateK:        cfg.CandidateK,
		constructionL:     cfg.ConstructionL,
		alpha:             cfg.Alpha,
		iterations:        cfg.Iterations,
		seed:              cfg.Seed,
		candidateRecall:   cfg.CandidateRecall,
		candidateControls: cfg.CandidateControls,
	}
}

func addConfigFlags(fs *flag.FlagSet, cfg *configFlags) {
	fs.StringVar(&cfg.metric, "metric", cfg.metric, "distance metric: l2 or cosine")
	fs.IntVar(&cfg.dim, "dim", cfg.dim, "vector dimension; zero infers from input")
	fs.IntVar(&cfg.m, "m", cfg.m, "HNSW max degree parameter")
	fs.IntVar(&cfg.k0, "k0", cfg.k0, "initial approximate KNNG size")
	fs.IntVar(&cfg.candidateK, "candidate-k", cfg.candidateK, "retained k-CNA candidate count")
	fs.IntVar(&cfg.constructionL, "construction-l", cfg.constructionL, "construction graph-search width")
	fs.Float64Var(&cfg.alpha, "alpha", cfg.alpha, "alpha-pruning threshold in degrees")
	fs.IntVar(&cfg.iterations, "iterations", cfg.iterations, "maximum IterNSG refinement iterations")
	fs.Int64Var(&cfg.seed, "seed", cfg.seed, "deterministic construction seed")
	fs.Float64Var(&cfg.candidateRecall, "candidate-recall", cfg.candidateRecall, "IterNSG candidate recall requirement")
	fs.IntVar(&cfg.candidateControls, "candidate-controls", cfg.candidateControls, "deterministic candidate-quality estimator sample size")
}

func (cfg configFlags) config() (fasthnsw.Config, error) {
	metric, err := parseMetric(cfg.metric)
	if err != nil {
		return fasthnsw.Config{}, err
	}
	return fasthnsw.Config{
		Metric:            metric,
		Dim:               cfg.dim,
		M:                 cfg.m,
		K0:                cfg.k0,
		CandidateK:        cfg.candidateK,
		ConstructionL:     cfg.constructionL,
		Alpha:             cfg.alpha,
		Iterations:        cfg.iterations,
		Seed:              cfg.seed,
		CandidateRecall:   cfg.candidateRecall,
		CandidateControls: cfg.candidateControls,
	}, nil
}

func parseMetric(value string) (fasthnsw.Metric, error) {
	switch strings.ToLower(value) {
	case "l2":
		return fasthnsw.MetricL2, nil
	case "cosine":
		return fasthnsw.MetricCosine, nil
	default:
		return 0, fmt.Errorf("unsupported metric %q", value)
	}
}

func readTextVectors(path string) ([][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open vector file: %w", err)
	}
	defer file.Close()
	vectors, err := parseTextVectors(file)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return vectors, nil
}

func parseTextVectors(r io.Reader) ([][]float32, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1024), 1024*1024)

	var vectors [][]float32
	dim := 0
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if dim == 0 {
			dim = len(fields)
		}
		if len(fields) != dim {
			return nil, fmt.Errorf("line %d has dimension %d, want %d", lineNumber, len(fields), dim)
		}
		vector := make([]float32, dim)
		for i, field := range fields {
			value, err := strconv.ParseFloat(field, 32)
			if err != nil {
				return nil, fmt.Errorf("line %d field %d: %w", lineNumber, i+1, err)
			}
			vector[i] = float32(value)
		}
		vectors = append(vectors, vector)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(vectors) == 0 {
		return nil, errors.New("vector file has no vectors")
	}
	return vectors, nil
}
