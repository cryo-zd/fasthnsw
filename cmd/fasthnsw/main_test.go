package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildAndQueryCommands(t *testing.T) {
	dir := t.TempDir()
	vectorsPath := writeTestFile(t, dir, "vectors.txt", "0 0\n1 0\n0 1\n")
	queriesPath := writeTestFile(t, dir, "queries.txt", "0 0\n")
	indexPath := filepath.Join(dir, "index.fhnsw")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"build",
		"-input", vectorsPath,
		"-output", indexPath,
		"-dim", "2",
		"-m", "2",
		"-k0", "2",
		"-candidate-k", "2",
		"-construction-l", "2",
		"-iterations", "1",
		"-candidate-recall", "0.9",
		"-candidate-controls", "2",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("build exit code = %d, stderr = %s", code, stderr.String())
	}
	if _, err := os.Stat(indexPath); err != nil {
		t.Fatalf("stat built index: %v", err)
	}
	if !strings.Contains(stdout.String(), "built index vectors=3 dim=2") {
		t.Fatalf("build stdout = %q, want summary", stdout.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = run([]string{
		"query",
		"-index", indexPath,
		"-queries", queriesPath,
		"-k", "1",
		"-ef", "2",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("query exit code = %d, stderr = %s", code, stderr.String())
	}
	got := stdout.String()
	for _, want := range []string{"query=0", "rank=0", "id=0", "distance=0.000000"} {
		if !strings.Contains(got, want) {
			t.Fatalf("query stdout = %q, want %q", got, want)
		}
	}
}

func TestRunRejectsUnknownSubcommand(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{"missing"}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("run returned success for unknown subcommand")
	}
	if !strings.Contains(stderr.String(), "unknown subcommand") {
		t.Fatalf("stderr = %q, want unknown subcommand", stderr.String())
	}
}

func TestBuildRejectsMissingRequiredFlags(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{"build"}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("build returned success without required flags")
	}
	if !strings.Contains(stderr.String(), "requires -input and -output") {
		t.Fatalf("stderr = %q, want required flag error", stderr.String())
	}
}

func TestBuildRejectsInconsistentVectorDimensions(t *testing.T) {
	dir := t.TempDir()
	vectorsPath := writeTestFile(t, dir, "vectors.txt", "0 0\n1\n")
	indexPath := filepath.Join(dir, "index.fhnsw")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"build",
		"-input", vectorsPath,
		"-output", indexPath,
	}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("build returned success for inconsistent vectors")
	}
	if !strings.Contains(stderr.String(), "dimension") {
		t.Fatalf("stderr = %q, want dimension error", stderr.String())
	}
}

func TestBuildRejectsInvalidMetric(t *testing.T) {
	dir := t.TempDir()
	vectorsPath := writeTestFile(t, dir, "vectors.txt", "0 0\n1 0\n")
	indexPath := filepath.Join(dir, "index.fhnsw")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"build",
		"-input", vectorsPath,
		"-output", indexPath,
		"-metric", "dot",
	}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("build returned success for invalid metric")
	}
	if !strings.Contains(stderr.String(), "unsupported metric") {
		t.Fatalf("stderr = %q, want metric error", stderr.String())
	}
}

func TestQueryRejectsWrongDimension(t *testing.T) {
	dir := t.TempDir()
	vectorsPath := writeTestFile(t, dir, "vectors.txt", "0 0\n1 0\n0 1\n")
	queriesPath := writeTestFile(t, dir, "queries.txt", "0 0 0\n")
	indexPath := filepath.Join(dir, "index.fhnsw")

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	code := run([]string{
		"build",
		"-input", vectorsPath,
		"-output", indexPath,
		"-dim", "2",
		"-m", "2",
		"-k0", "2",
		"-candidate-k", "2",
		"-construction-l", "2",
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("build exit code = %d, stderr = %s", code, stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	code = run([]string{
		"query",
		"-index", indexPath,
		"-queries", queriesPath,
		"-k", "1",
		"-ef", "2",
	}, &stdout, &stderr)
	if code == 0 {
		t.Fatal("query returned success for wrong query dimension")
	}
	if !strings.Contains(stderr.String(), "query vector has dimension") {
		t.Fatalf("stderr = %q, want query dimension error", stderr.String())
	}
}

func writeTestFile(t *testing.T, dir string, name string, content string) string {
	t.Helper()

	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write test file: %v", err)
	}
	return path
}
