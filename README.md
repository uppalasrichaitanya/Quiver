# Quiver

[![CI](https://github.com/uppalasrichaitanya/Quiver/actions/workflows/ci.yml/badge.svg)](https://github.com/uppalasrichaitanya/Quiver/actions/workflows/ci.yml)

Quiver is a portfolio-grade, single-node vector database built from scratch in Rust. Its purpose is to demonstrate systems-engineering workâ€”binary formats, mmap storage, write-ahead logging, crash recovery, graph indexing, SIMD, fuzzing, and honest measurementâ€”in a repository that can be inspected and tested.

Quiver is **not production software** and is not intended to compete with Qdrant, Pinecone, FAISS, Milvus, or hnswlib. It currently lacks the operational hardening, feature breadth, and benchmark evidence required for those comparisons. The goal is correctness rigor and a transparent account of the remaining gap, not a claim that a small from-scratch implementation wins.

## Current status

Implemented in `quiver-core`:

- Versioned mmap vector storage with a 64-byte header; legacy v1 and explicit-ID v2 files are readable.
- CRC32-checksummed write-ahead log with idempotent insert replay, durable deletion, and incomplete-tail truncation.
- Crash-safe live-only compaction using durable replacement files, a swap journal, and recovery from interruption during the swap.
- Bounds-checked file parsing for headers and records, plus a dedicated libFuzzer target.
- Scalar and hand-written x86_64 AVX2/FMA kernels for L2, dot product, and cosine distance, with runtime dispatch and scalar fallback.
- Exact brute-force search and multi-layer HNSW insert/search/delete.
- Batch-built SQ8 flat search with per-dimension calibration, asymmetric distance evaluation, and one-byte vector components.
- Per-vector key-value metadata and filtered search: a `Metadata` model (`bool`/`int`/`float`/`str`), an `Eq`/`And` `Filter` predicate, durable metadata (WAL op + CRC32 snapshot sidecar, format version 3), and a filter-aware `search_filtered` that traverses the HNSW graph with the filter live during the beam search, using non-matching nodes as waypoints.
- Tests comparing HNSW recall with brute-force ground truth and real subprocess-kill tests for ordinary WAL recovery and compaction recovery.

The workspace includes an Axum server with insert, search, batch-search, and delete endpoints, plus a PyO3 local `Index` API for the same core operations. Insert accepts optional `metadata`, and search/batch-search accept an optional `filter`. The server accepts concurrent HTTP connections around one `RwLock`-protected HNSW index (parallel reads, exclusive writes); mutations remain serialized by the core API's single-writer model.

## Known limitations

- HNSW graph topology is persisted as a snapshot on `flush`/`compact` and loaded on `open`, but vectors inserted after the last snapshot still require a rebuild on reopen; a hard kill leaves the snapshot stale (the server flushes on graceful shutdown).
- The HNSW API is single-threaded. Mutation safety comes from Rust's `&mut self`; the server wraps the index in an `RwLock` (parallel reads, exclusive writes).
- Filtered search is a single-pass filter-aware traversal, but cost still grows as selectivity shrinks: at SIFT1M/M=32/efc200/ef=100 the 50% case runs ~2124 QPS while the 1% case runs ~195 QPS (an ~11x gap, down from ~25x under the earlier naive post-filtering). Recall stays >= 0.9837 at 1%/10%/50% selectivity. Only `Eq` and `And` filters are implemented; `Or`/`In`/range are deferred, and metadata is immutable after insert.
- SQ8 is currently an in-memory, batch-built index; online recalibration and persistence are not implemented. IVF-PQ remains planned.
- Benchmark evidence, including reproducible SIFT1M comparisons and scalar/SIMD Criterion results, is documented in `benchmarks/`. A sampling flamegraph remains unavailable after the documented `samply` install attempt.
- The crates have not yet been released to crates.io or PyPI.

## Build and test

Install a current stable Rust toolchain, then run:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

The default workspace test run includes hard-kill subprocess recovery tests. On Unix, `Child::kill` sends SIGKILL; on Windows it uses `TerminateProcess`.

To execute the file parser fuzz target on Linux:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz --locked
cd fuzz
cargo +nightly fuzz run file_format -- -max_total_time=60
```

CI runs that target for 60 seconds on a Linux runner on every push and pull request.

## Benchmarks

Run the currently available Criterion microbenchmarks with:

```bash
cargo bench -p quiver-core
```

These cover distance kernels and local brute-force/HNSW operations. They are development harnesses, not the final comparative benchmark suite, and no performance claim should be inferred from their presence. Phase 6 will profile the scalar path first and then commit reproducible, same-hardware comparisons with FAISS and hnswlib.

A reproducible single-threaded SIFT1M comparison is available in
[`benchmarks/`](benchmarks/README.md). It includes raw JSON for the complete
Quiver/FAISS/hnswlib configuration sweep and documents the measured
quality/cost trade-offs.

## Semantic-search demo

Run semantic search over a small real-text corpus through the HTTP API:

```powershell
$env:QUIVER_DIMENSION=384
cargo run -p quiver-server
# In another terminal:
python examples/semantic_search.py
```

On Windows GNU, use the 64-bit MSYS2 MinGW64 tools first in `PATH`:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;" + ($env:PATH -replace "C:\\MinGW\\bin;?", "")
```

Axum was temporarily removed in an earlier commit because `C:\MinGW\bin\dlltool.exe` was selected and failed to create 64-bit import libraries with `Invalid bfd target`. That was a PATH/toolchain conflict, not an Axum or Tokio limitation. Axum is restored, and the server now opens an existing index on restart or creates one when the configured data path does not exist.

The server exposes `POST /vectors` (optional `metadata`), `POST /search` and
`POST /search/batch` (both with an optional `filter`), `DELETE /vectors/{id}`,
and `POST /shutdown`. A local native Python API is also available through
[`quiver-py`](quiver-py/README.md).

![Quiver semantic-search terminal demo](examples/semantic-search-demo.gif)

## Workspace layout

```text
quiver-core/    mmap storage, WAL, distance kernels, brute-force, HNSW
quiver-server/  Axum HTTP insert/search/delete API
quiver-py/      PyO3 local Index API
fuzz/           dedicated file-format libFuzzer package and seed corpus
```

Quiver is dual-licensed under MIT or Apache-2.0.
