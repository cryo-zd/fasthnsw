# Repository Guidelines

## Project Structure & Module Organization

FastHNSW is a Go module at `github.com/cryo-zd/fasthnsw`. Public API files live at the repository root, including `fasthnsw.go`, `doc.go`, and root-level tests. Core ANN implementation details are under `internal/core`, with focused tests beside the code. Benchmark and validation dataset loaders are in `internal/benchdata`, recall helpers are in `internal/eval`, and synthetic data generation is in `internal/synth`. The optional smoke and validation CLI is in `cmd/fasthnsw`; usage examples are in `examples/basic` and `examples/persistence`.

## Build, Test, and Development Commands

- `go test ./...`: run the full unit test suite.
- `go test -bench=. -benchmem ./...`: run benchmarks for distance, construction, build, and search paths.
- `go run ./cmd/fasthnsw validate -dataset clustered -vectors 1000 -query-count 100 -dim 32 -k 10 -ef 64`: run a synthetic validation smoke test.
- `go run ./cmd/fasthnsw build ...` and `go run ./cmd/fasthnsw query ...`: build and query an index using whitespace-separated text vectors.
- `go vet ./...`: run standard Go static checks before larger changes.

## Coding Style & Naming Conventions

Use standard Go formatting: run `gofmt` on edited `.go` files. Keep package names short and lowercase, matching existing directories such as `core`, `benchdata`, and `synth`. Exported identifiers require useful doc comments when they are part of the public root package. Prefer table-driven tests for config, search, and validation behavior, and keep helpers in package-local `*_test.go` files.

## Testing Guidelines

Tests use Go’s built-in `testing` package and live next to the code as `*_test.go`. Name tests by behavior, for example `TestSearchRejectsBadK` or `TestConfigDefaults`. Add tests for error paths and deterministic behavior when changing construction, search, persistence, or CLI parsing. For performance-sensitive changes, include a benchmark run and call out any notable regression or improvement.

## Commit & Pull Request Guidelines

Recent history uses concise conventional prefixes such as `feat:`, `docs:`, `test:`, and `benchmark:`. Keep commits focused and imperative, for example `test: cover cosine zero vector validation`. Pull requests should include a short problem statement, a summary of changes, validation commands run, and linked issues when applicable. Include CLI output snippets for behavior changes; screenshots are usually unnecessary for this repository.

## Security & Configuration Tips

Do not commit large benchmark datasets, generated indexes, CPU profiles, or memory profiles. Keep external dataset paths local and document required ANN-Benchmarks HDF5 datasets (`train`, `test`, and `neighbors`) when sharing validation steps.
