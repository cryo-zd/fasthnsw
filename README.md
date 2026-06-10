# FastHNSW

FastHNSW is a Go approximate nearest neighbor solution implementing an initial FastHNSW construction path inspired by "Revisiting the Index Construction of Proximity Graph-Based Approximate Nearest Neighbor Search"(PVLDB 18(6), 2025).

## Example

```go
package main

import "github.com/cryo-zd/fasthnsw"

func main() {
	cfg := fasthnsw.DefaultConfig()
	cfg.Dim = 3
	cfg.M = 4
	cfg.K0 = 4
	cfg.CandidateK = 4
	cfg.ConstructionL = 4

	idx, err := fasthnsw.New(cfg)
	if err != nil {
		panic(err)
	}

	vectors := [][]float32{
		{0.2, 0, 0},
		{0, 0.5, 0},
		{0, 0, 0.4},
		{0.3, 0.2, 0},
	}
	if err := idx.Build(vectors); err != nil {
		panic(err)
	}

	newID, err := idx.AddVector([]float32{0.4, 0.2, 0.8})
	if err != nil {
		panic(err)
	}
	_ = newID

	results, err := idx.Search([]float32{0.3, 0, 0}, 2, 4)
	if err != nil {
		panic(err)
	}
}
```

## Configuration

Start from `DefaultConfig()` and override only the fields needed for the dataset. `Dim` may be left as zero to infer dimensionality during `Build`.

- `Metric` selects squared L2 or cosine distance. Cosine builds normalize stored vectors and reject zero vectors.
- `M` controls final graph degree. Layer 0 uses `2*M`; upper layers use `M`.
- `K0`, `CandidateK`, and `ConstructionL` control IterNSG initialization, retained k-CNA size, and construction search width.
- `Alpha` controls intermediate alpha-pruning. `60` is RNG-equivalent; larger values prune less aggressively.
- `CandidateRecall`, `CandidateControls`, and `Iterations` control the IterNSG stopping rule. `Iterations` is a maximum cap.
- `Seed` controls deterministic randomized construction choices.
- `Workers` controls node-local construction parallelism. Use `1` for a sequential build.

## References
- [Revisiting the Index Construction of Proximity Graph-Based Approximate Nearest Neighbor Search](https://dl.acm.org/doi/10.14778/3725688.3725709)

## License

MIT.
