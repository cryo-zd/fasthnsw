package fasthnsw_test

import (
	"bytes"
	"fmt"

	"github.com/cryo-zd/fasthnsw"
)

func ExampleIndex_Save() {
	cfg := fasthnsw.DefaultConfig()
	cfg.Dim = 2
	cfg.M = 4
	cfg.K0 = 4
	cfg.CandidateK = 4
	cfg.ConstructionL = 4

	idx, err := fasthnsw.New(cfg)
	if err != nil {
		panic(err)
	}
	if err := idx.Build([][]float32{
		{0, 0},
		{1, 0},
		{0, 1},
	}); err != nil {
		panic(err)
	}

	var buf bytes.Buffer
	if err := idx.Save(&buf); err != nil {
		panic(err)
	}

	loaded, err := fasthnsw.Load(&buf)
	if err != nil {
		panic(err)
	}
	results, err := loaded.Search([]float32{0, 0}, 1, 4)
	if err != nil {
		panic(err)
	}
	fmt.Println(results[0].ID)

	// Output: 0
}
