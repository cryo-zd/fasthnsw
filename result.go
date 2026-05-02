package fasthnsw

// Result is one nearest-neighbor result returned by Search.
type Result struct {
	ID       int
	Distance float32
}

// betterResult reports whether a should sort before b in public result order.
// Distances sort ascending, and equal distances break ties by smaller id.
func betterResult(a, b Result) bool {
	if a.Distance != b.Distance {
		return a.Distance < b.Distance
	}
	return a.ID < b.ID
}

// worseResult reports whether a should sort after b in public result order.
// It is used by max-heaps that keep the farthest retained result at the root.
func worseResult(a, b Result) bool {
	if a.Distance != b.Distance {
		return a.Distance > b.Distance
	}
	return a.ID > b.ID
}
