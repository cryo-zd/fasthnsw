package fasthnsw

import (
	"bytes"
	"errors"
	"testing"
)

func TestBuildValidatesVectorsBeforeNotImplemented(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	err = idx.Build([][]float32{{1, 2}, {3, 4}})
	if !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Build error = %v, want ErrNotImplemented", err)
	}
}

func TestBuildRejectsInvalidVectors(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tests := []struct {
		name    string
		vectors [][]float32
	}{
		{name: "empty", vectors: nil},
		{name: "zero dim", vectors: [][]float32{{}}},
		{name: "mismatch", vectors: [][]float32{{1, 2}, {3}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := idx.Build(tt.vectors)
			if err == nil {
				t.Fatal("Build returned nil error")
			}
			if errors.Is(err, ErrNotImplemented) {
				t.Fatalf("Build returned ErrNotImplemented before validation: %v", err)
			}
		})
	}
}

func TestSearchValidatesInputsBeforeNotImplemented(t *testing.T) {
	idx, err := New(Config{Dim: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tests := []struct {
		name     string
		query    []float32
		k        int
		efSearch int
		wantStub bool
	}{
		{name: "valid", query: []float32{1, 2}, k: 1, efSearch: 2, wantStub: true},
		{name: "empty query", query: nil, k: 1, efSearch: 2},
		{name: "wrong dimension", query: []float32{1}, k: 1, efSearch: 2},
		{name: "bad k", query: []float32{1, 2}, k: 0, efSearch: 2},
		{name: "bad efSearch", query: []float32{1, 2}, k: 1, efSearch: 0},
		{name: "efSearch less than k", query: []float32{1, 2}, k: 2, efSearch: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := idx.Search(tt.query, tt.k, tt.efSearch)
			if tt.wantStub {
				if !errors.Is(err, ErrNotImplemented) {
					t.Fatalf("Search error = %v, want ErrNotImplemented", err)
				}
				return
			}
			if err == nil {
				t.Fatal("Search returned nil error")
			}
			if errors.Is(err, ErrNotImplemented) {
				t.Fatalf("Search returned ErrNotImplemented before validation: %v", err)
			}
		})
	}
}

func TestSaveAndLoadReturnNotImplemented(t *testing.T) {
	idx, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if err := idx.Save(bytes.NewBuffer(nil)); !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Save error = %v, want ErrNotImplemented", err)
	}
	if _, err := Load(bytes.NewBuffer(nil)); !errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Load error = %v, want ErrNotImplemented", err)
	}
}

func TestSaveAndLoadRejectNilIO(t *testing.T) {
	idx, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if err := idx.Save(nil); err == nil || errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Save(nil) error = %v, want validation error", err)
	}
	if _, err := Load(nil); err == nil || errors.Is(err, ErrNotImplemented) {
		t.Fatalf("Load(nil) error = %v, want validation error", err)
	}
}
