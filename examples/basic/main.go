package main

import (
	"fmt"

	"github.com/cryo-zd/fasthnsw"
)

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
	for _, result := range results {
		fmt.Printf("id=%d distance=%.6f\n", result.ID, result.Distance)
	}
}
