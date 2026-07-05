# FastHNSW

&emsp;&emsp;FastHNSW is a Go approximate nearest neighbor solution implementing an initial FastHNSW construction path inspired by "Revisiting the Index Construction of Proximity Graph-Based Approximate Nearest Neighbor Search"(PVLDB 18(6), 2025).

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

&emsp;&emsp;Start from `DefaultConfig()` and override only the fields needed for the dataset. `Dim` may be left as zero to infer dimensionality during `Build`.

- `Metric` selects squared L2 or cosine distance. Cosine builds normalize stored vectors and reject zero vectors.
- `M` controls final graph degree. Layer 0 uses `2*M`; upper layers use `M`.
- `K0`, `CandidateK`, and `ConstructionL` control IterNSG initialization, retained k-CNA size, and construction search width.
- `Alpha` controls intermediate alpha-pruning. `60` is RNG-equivalent; larger values prune less aggressively.
- `CandidateRecall`, `CandidateControls`, and `Iterations` control the IterNSG stopping rule. `Iterations` is a maximum cap.
- `Seed` controls deterministic randomized construction choices.
- `Workers` controls node-local construction parallelism. Use `1` for a sequential build.

## References
- [Revisiting the Index Construction of Proximity Graph-Based Approximate Nearest Neighbor Search](https://dl.acm.org/doi/10.14778/3725688.3725709)

&emsp;&emsp;Standard HNSW construction inserts vectors one by one. Each insertion searches the graph built so far, selects neighbors, links the new node, and then moves to the next vector. That procedure is robust, but it creates a long dependency chain: node `i+1` cannot be inserted until node `i` has already modified the graph.

```text
Standard HNSW construction

input order
   |
   v
insert v0 -> insert v1 -> insert v2 -> ... -> insert vn
    |           |            |                 |
    v           v            v                 v
 graph0      graph1       graph2            graphn

Each insert searches the current partial graph and mutates it before the next insert can start.
```

&emsp;&emsp;FastHNSW keeps the HNSW search model, but changes how the graph is built. It assigns every node's maximum layer before creating edges, then builds each layer over the complete node set that belongs to that layer:

```text
Preassigned levels

node id:   0   1   2   3   4   5   6   7
level:     0   2   0   1   0   3   1   0

layer 3:                   5
layer 2:       1           5
layer 1:       1       3   5   6
layer 0:   0   1   2   3   4   5   6   7
```

&emsp;&emsp;For one layer, the construction pipeline is:

```text
D_i = all nodes with level >= i

        approximate KNNG initialization
                    |
                    v
       k-CNA candidate acquisition by graph search
                    |
                    v
        alpha-pruned intermediate graph
                    |
                    v
       beam-search refresh of candidate sets
                    |
                    v
     repeat until candidate recall target or max cap
                    |
                    v
         final RNG pruning for the HNSW layer
```

&emsp;&emsp;The key distinction is that the alpha-pruned graph is only a construction-time tool used to refresh candidates. The final index layer is an RNG-pruned HNSW layer, and queries still use ordinary HNSW semantics: greedy descent in upper layers and beam search in layer 0.

&emsp;&emsp;This design attacks build time without changing the query algorithm. Instead of relying on a serial insertion order to discover good neighbors, FastHNSW uses layer-wide candidate acquisition and refinement. Node-local work inside a layer can be scheduled deterministically across workers, and the final pruning step keeps the graph degree bounded like HNSW. With suitable construction parameters, the resulting graph is intended to match the recall of a standard HNSW baseline while reducing construction cost, especially on larger datasets where the serial insertion bottleneck dominates.

## License

MIT.
