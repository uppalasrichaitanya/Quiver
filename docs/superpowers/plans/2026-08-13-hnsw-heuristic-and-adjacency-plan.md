# HNSW Heuristic and Adjacency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Quiver HNSW recall with diversified neighbor selection, then reduce graph memory overhead with packed 32-bit adjacency storage while preserving correctness and durability behavior.

**Architecture:** First replace the direct nearest-neighbor truncation in `HnswIndex` with the HNSW diversity rule at both forward selection and reverse overflow pruning. Commit and benchmark that algorithmic change independently. Then replace each node's nested `Vec<Vec<usize>>` adjacency with fixed-capacity per-level blocks in one packed `Vec<u32>` arena; keep the same selection algorithm and public API so the second benchmark isolates layout effects.

**Tech Stack:** Rust 2024, `quiver-core`, existing HNSW unit tests, Criterion/SIFT1M benchmark binaries.

---

## File structure

- Modify: `quiver-core/src/index/hnsw.rs` — selection helper, graph links, tests, then packed adjacency representation.
- Modify: `benchmarks/README.md` — append separately dated recall and layout benchmark results.
- Modify: `PROGRESS.md` — update Phase 3 status only after both measured steps.
- Create: `benchmarks/results/<date-host>/raw/*.json` — benchmark output from existing `sift_benchmark` binary.

### Task 1: Add a deterministic diversified-selection test

**Files:**
- Modify: `quiver-core/src/index/hnsw.rs` inside `#[cfg(test)]`

- [ ] **Step 1: Write the failing test**

Add a candidate set where candidate `1` is nearest to query `q`, candidate `2` is near candidate `1`, and candidate `3` is farther from `q` but far from `1`. Assert that selecting two candidates retains `1` and `3`, not `1` and `2`.

```rust
#[test]
fn diversified_selection_keeps_a_farther_bridge_candidate() {
    let candidates = vec![Candidate { node_idx: 1, distance: 1.0 }, Candidate { node_idx: 2, distance: 1.1 }, Candidate { node_idx: 3, distance: 2.0 }];
    let selected = select_neighbors_heuristic(&[0.0], &candidates, 2, |node_idx| match node_idx {
        1 => vec![1.0], 2 => vec![1.1], 3 => vec![2.0], _ => unreachable!(),
    }, Metric::L2);
    assert_eq!(selected, vec![1, 3]);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p quiver-core diversified_selection_keeps_a_farther_bridge_candidate`

Expected: compilation failure because `select_neighbors_heuristic` does not exist.

- [ ] **Step 3: Implement the minimal heuristic helper**

Sort candidates by query distance. For each candidate `c`, accept it only if `distance(c, s) >= distance(query, c)` for every selected neighbor `s`; otherwise put it in a discarded list. Fill unused capacity from discarded candidates in query-distance order. Return node IDs only.

- [ ] **Step 4: Re-run the focused test**

Run: `cargo test -p quiver-core diversified_selection_keeps_a_farther_bridge_candidate`

Expected: PASS.

### Task 2: Apply diversified selection on insert and reverse pruning

**Files:**
- Modify: `quiver-core/src/index/hnsw.rs:449-487`
- Modify: `quiver-core/src/index/hnsw.rs:584-614`

- [ ] **Step 1: Write failing graph-invariant tests**

Add tests that all neighbor lists remain within `m_max0` at level zero and `m` above level zero after random insertion, and that reopening/rebuilding remains searchable.

- [ ] **Step 2: Run the new tests and verify the nearest-only implementation cannot satisfy the diversity assertion**

Run: `cargo test -p quiver-core hnsw::tests`

Expected: the new diversity-specific test fails before call sites use the heuristic.

- [ ] **Step 3: Replace both call sites**

Use the helper to select forward links from `search_layer` candidates. On reverse overflow, construct candidates from the existing neighbor list plus the new node, then select relative to the overflowing node's stored vector. Keep level-specific capacities unchanged.

- [ ] **Step 4: Run HNSW tests**

Run: `cargo test -p quiver-core hnsw::tests`

Expected: PASS.

- [ ] **Step 5: Commit the algorithmic change**

```powershell
git add quiver-core/src/index/hnsw.rs
git commit -m "feat: diversify HNSW neighbor selection"
```

### Task 3: Measure the heuristic independently

**Files:**
- Create: `benchmarks/results/<date-host>/raw/quiver-m32-efc200-diverse.json`
- Modify: `benchmarks/README.md`

- [ ] **Step 1: Run the existing SIFT1M benchmark configuration**

Run `quiver-core/src/bin/sift_benchmark.rs` with exactly `M=32`, `efConstruction=200`, seed `42`, one thread, and the existing SIFT1M base/query/ground-truth inputs.

- [ ] **Step 2: Compare only like-for-like values**

Compare the new raw JSON against `benchmarks/results/2026-07-28-i7-12650h/raw/quiver-m32-efc200.json`: Recall@10/100 by efSearch, build time, QPS, p50/p99, and RSS.

- [ ] **Step 3: Append the result with environment disclosure**

Document whether this is the same host and dataset. If SIFT1M data is unavailable, record that the algorithm is unit-tested but macro-benchmark comparison is pending; do not invent numbers.

### Task 4: Add a packed adjacency abstraction

**Files:**
- Modify: `quiver-core/src/index/hnsw.rs`

- [ ] **Step 1: Write failing adjacency tests**

Write tests for a `PackedAdjacency` type that allocates fixed-capacity per-node level lists, returns slices for read traversal, replaces one list without affecting another, and stores node IDs as `u32`.

- [ ] **Step 2: Run focused tests to verify they fail**

Run: `cargo test -p quiver-core packed_adjacency`

Expected: compilation failure because the type does not exist.

- [ ] **Step 3: Implement a compact arena**

Define `PackedAdjacency { links: Vec<u32>, levels: Vec<LevelBlock> }`, where every `LevelBlock` records an offset, length, and fixed capacity in one shared `Vec<u32>`. Allocate capacity `m_max0` for level zero and `m` above it when each node is created. Replace a list in its existing block, so pruning does not retain obsolete links. Convert `u32` IDs to `usize` only at lookup boundaries, with checked conversion.

- [ ] **Step 4: Run packed adjacency tests**

Run: `cargo test -p quiver-core packed_adjacency`

Expected: PASS.

### Task 5: Migrate HNSW traversal and mutation to packed adjacency

**Files:**
- Modify: `quiver-core/src/index/hnsw.rs`

- [ ] **Step 1: Write failing regression tests before migration**

Add tests covering bidirectional connection insertion, pruning after capacity overflow, deletion/compaction, and reopen graph rebuild using the public HNSW API.

- [ ] **Step 2: Run and confirm baseline tests pass**

Run: `cargo test -p quiver-core hnsw::tests`

Expected: PASS before internal representation changes.

- [ ] **Step 3: Migrate all list access**

Replace every `node.neighbors[level]` read, assignment, push, clone, and iteration with `HnswIndex` helpers that read/replace a list in the packed arena. Preserve `m`, `m_max0`, tombstones, deterministic RNG, WAL ordering, compaction, and reopen behavior.

- [ ] **Step 4: Run all HNSW tests**

Run: `cargo test -p quiver-core hnsw::tests`

Expected: PASS.

- [ ] **Step 5: Commit the layout change**

```powershell
git add quiver-core/src/index/hnsw.rs
git commit -m "refactor: pack HNSW adjacency links"
```

### Task 6: Verify and measure the final layout independently

**Files:**
- Modify: `benchmarks/README.md`
- Modify: `PROGRESS.md`

- [ ] **Step 1: Run workspace verification**

Run:

```powershell
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: all pass with no warnings.

- [ ] **Step 2: Repeat the M=32 / efConstruction=200 SIFT1M run**

Use the same host, input files, seed, and one-thread setting from Task 3. Save raw JSON separately and compare it with both baseline and heuristic-only data.

- [ ] **Step 3: Document only measured conclusions**

Append a table separating baseline, diversified-only, and diversified-plus-packed-layout results. Update `PROGRESS.md` to state what was measured, or explicitly state macro results are pending when dataset execution is unavailable.

- [ ] **Step 4: Commit docs and raw benchmark outputs**

```powershell
git add benchmarks PROGRESS.md
git commit -m "docs: record HNSW Phase 3 measurements"
```
