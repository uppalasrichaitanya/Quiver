# Quiver - Project Progress and Context Transfer

> **Last updated:** 2026-08-13
> **Purpose:** Current implementation status and next-step context.

## Architecture

Quiver is a portfolio-grade, single-node, embeddable vector search engine in Rust.

```text
quiver-core/    mmap storage, WAL, recovery, distances, brute-force, HNSW, SQ8
quiver-server/  Axum REST API with restart-safe open-or-create lifecycle
quiver-py/      PyO3 local Index API
fuzz/           file-format libFuzzer target and seed corpus
benchmarks/     reproducible Criterion and SIFT1M comparisons
```

## Completed

- Versioned 64-byte vector storage with legacy v1 and explicit-ID v2 support.
- CRC32 WAL, idempotent replay, durable deletion, crash-safe compaction, and hard-kill recovery tests.
- Bounds-checked parsing plus a dedicated file-format fuzz target.
- Scalar and AVX2/FMA L2, dot-product, and cosine kernels with runtime fallback.
- Exact brute-force search and multi-layer HNSW insert/search/delete.
- Batch-built SQ8 flat search with per-dimension calibration and asymmetric distance evaluation.
- Reproducible SIFT1M comparisons against FAISS and hnswlib, including recall, latency, QPS, and RSS.
- Axum HTTP API: health, insert, search, and delete. The server opens an existing index or creates one when paths are absent.
- PyO3 `Index` API and semantic-search demo.

## Verification Status

The latest workspace run has **97 tests passing** with zero failures: 96 workspace/unit tests plus the server-process restart persistence integration test. CI also runs clippy, rustfmt, and 60 seconds of file-format fuzzing.

## Known Limitations

- HNSW graph topology is rebuilt on reopen rather than persisted.
- HNSW mutation is single-writer through `&mut self`; the server wraps it in a mutex.
- HNSW uses nested adjacency allocations and first-M pruning, leaving recall and RSS gaps versus mature libraries.
- SQ8 is in-memory and batch-built; online recalibration and persistence are not implemented.
- Metadata/filtering and IVF-PQ are not implemented.
- Crates.io and PyPI releases are not published.

## Windows GNU Build Decision

Axum was removed in commit `8b86302` after the demo picked up `C:\MinGW\bin\dlltool.exe`, which cannot create 64-bit import libraries for the `x86_64-pc-windows-gnu` target (`Invalid bfd target`). This is a toolchain PATH conflict, not a platform limitation. The supported fix is to use the MSYS2 MinGW64 binutils first and remove the conflicting 32-bit MinGW directory:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;" + ($env:PATH -replace "C:\\MinGW\\bin;?", "")
```

## Next Phases

1. Complete project hygiene and API lifecycle (this phase).
2. HNSW diversified neighbor selection and packed 32-bit fixed-capacity adjacency are implemented and covered by unit/regression tests. Same-host SIFT1M reruns are pending because the dataset files are not present in this workspace.
3. Add metadata and filtered search, starting with post-filtering and selectivity benchmarks.
4. Persist SQ8 and evaluate quantized HNSW.
5. Implement IVF-PQ only after the HNSW and filtering layers are measured and stable.
6. Add packaging, release automation, and generated benchmark charts.
