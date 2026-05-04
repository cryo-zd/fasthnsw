package core

import (
	"container/heap"
	"testing"
)

func TestResultMinHeapPopsNearestFirst(t *testing.T) {
	candidates := resultMinHeap{
		{ID: 2, Distance: 0.5},
		{ID: 1, Distance: 0.5},
		{ID: 3, Distance: 0.1},
	}
	heap.Init(&candidates)

	want := []Result{
		{ID: 3, Distance: 0.1},
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 0.5},
	}
	for i, expected := range want {
		got := heap.Pop(&candidates).(Result)
		if got != expected {
			t.Fatalf("pop %d = %+v, want %+v", i, got, expected)
		}
	}
}

func TestResultMaxHeapPopsWorstFirst(t *testing.T) {
	results := resultMaxHeap{
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 0.5},
		{ID: 3, Distance: 0.1},
	}
	heap.Init(&results)

	got := heap.Pop(&results).(Result)
	want := Result{ID: 2, Distance: 0.5}
	if got != want {
		t.Fatalf("first pop = %+v, want worst result %+v", got, want)
	}
}

func TestResultMaxHeapSortedReturnsPublicOrder(t *testing.T) {
	results := resultMaxHeap{
		{ID: 2, Distance: 0.5},
		{ID: 1, Distance: 0.5},
		{ID: 3, Distance: 0.1},
	}
	heap.Init(&results)

	got := results.sorted()
	want := []Result{
		{ID: 3, Distance: 0.1},
		{ID: 1, Distance: 0.5},
		{ID: 2, Distance: 0.5},
	}
	assertResults(t, got, want)
}
