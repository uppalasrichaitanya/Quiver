# Quiver - Project Progress and Context Transfer

> **Last updated:** 2026-08-23
> **Purpose:** Current implementation status and next-step context.
> **Git:** `main` is clean and pushed to `origin` (github.com/uppalasrichaitanya/Quiver) through the metadata + filtered-search milestone (2026-08-23).

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
- HNSW search/build hot-path optimizations (2026-08-22): borrowed-slice `neighbors()`, a thread-local generation-counted visited pool, neighbor prefetch, cached once-per-process SIMD dispatch, four-accumulator AVX2 distance kernels, and group-commit batch WAL inserts.
- HNSW graph-topology snapshot persistence (2026-08-23): `flush`/`compact` write a CRC32-protected `<data>.graph` snapshot; `open` loads it and skips the rebuild when it validates against the store and config, falling back to rebuild on any mismatch or corruption. The HTTP server flushes on graceful shutdown (Ctrl+C or `POST /shutdown`) so restarts reopen from the snapshot.
- Server concurrency and batching (2026-08-22): `HnswIndex` is shared behind an `RwLock` (parallel reads), and a `POST /search/batch` endpoint runs multiple queries under one read lock.
- Metadata and filtered search (2026-08-23, plan week-14 milestone): a `Metadata` key-value model (`bool`/`int`/`float`/`str` scalars) and `Filter` predicate (`Eq`/`And`) in `quiver-core`. Metadata is durable — a new `InsertMeta` WAL op, a CRC32-protected `<data>.meta` snapshot sidecar that participates in flush and crash-safe compaction, and a data-file format-version bump to 3 (v3 records are byte-identical to v2; the version signals metadata may exist). `HnswIndex::insert_with_metadata` / `insert_batch_with_metadata` attach metadata at insert; `search_filtered` post-filters HNSW candidates with adaptive over-fetch (beam width expands until `k` matches are collected or the graph is exhausted) so recall holds as selectivity shrinks. Exposed through the HTTP server (`metadata` on insert, `filter` on search/batch) and `quiver-py`. Filtered SIFT1M benchmarks at 1%/10%/50% selectivity are recorded under `benchmarks/results/2026-08-23-i7-12650h/raw`.

## Verification Status

The latest run has **162 `quiver-core` tests** and **9 `quiver-server` integration tests** passing with zero failures (the server suite covers restart persistence, batch search, graceful-shutdown snapshot persistence, and metadata/filtered search incl. survival across a graceful restart). CI also runs clippy, rustfmt, and 60 seconds of file-format fuzzing. The Python bindings were verified with a maturin-built extension smoke test (metadata insert + Eq/And/bool/float filters + malformed-filter rejection).

## Known Limitations

- HNSW graph topology is persisted as a snapshot on `flush`/`compact` and loaded on `open`, but vectors inserted after the last snapshot still require a rebuild on reopen (the snapshot is only as fresh as the last flush). The HTTP server flushes on graceful shutdown (Ctrl+C or `POST /shutdown`), but a hard kill still leaves the snapshot stale.
- HNSW mutation is single-writer through `&mut self`; the server wraps it in an `RwLock` (parallel reads, exclusive writes).
- HNSW build is slower than in-memory competitors because Quiver fsyncs a CRC32 WAL during the build, while FAISS/hnswlib build in memory and serialize afterward. Group-commit batch inserts (2026-08-22) cut build time ~2-2.7x, but the durability-during-build cost remains the gap. Search speed and recall are no longer the gap: at M=32/efc200/ef=100 Quiver now measures 2680 QPS / p50 0.38 ms / Recall@10 0.9961, on par with or above FAISS and hnswlib.
- SQ8 is in-memory and batch-built; online recalibration and persistence are not implemented.
- Filtered search is naive post-filtering: candidates are over-fetched from the HNSW graph and then filtered, so cost scales roughly with the inverse of filter selectivity. At SIFT1M/M=32/efc200/ef=100 the 50% case runs ~2156 QPS (p50 0.46 ms) but the 1% case drops to ~84 QPS (p50 ~9.9 ms) even though adaptive over-fetch keeps recall at 0.9995. A filter-aware graph traversal (or per-filter secondary structure) is the future fix. The predicate vocabulary is also intentionally naive — only `Eq` and `And`; `Or`, `In`, and range predicates are deferred. Metadata is immutable after insert (no update/delete-metadata API).
- IVF-PQ is not implemented.
- Crates.io and PyPI releases are not published.

## Windows GNU Build Decision

Axum was removed in commit `8b86302` after the demo picked up `C:\MinGW\bin\dlltool.exe`, which cannot create 64-bit import libraries for the `x86_64-pc-windows-gnu` target (`Invalid bfd target`). This is a toolchain PATH conflict, not a platform limitation. The supported fix is to use the MSYS2 MinGW64 binutils first and remove the conflicting 32-bit MinGW directory:

```powershell
$env:PATH = "C:\msys64\mingw64\bin;" + ($env:PATH -replace "C:\\MinGW\\bin;?", "")
```

## Next Phases

1. Complete project hygiene and API lifecycle (done).
2. HNSW diversified neighbor selection and packed 32-bit fixed-capacity adjacency are implemented and covered by unit/regression tests. The same-host SIFT1M rerun is **complete** (2026-08-22): recall at M=32/efc200/ef=100 rose 0.9595 -> 0.9961 (now above FAISS 0.9922 and hnswlib 0.9920), peak RSS fell 3.4x, build is ~1.0-1.5x slower, and search QPS regressed to ~0.77x of the July baseline. Raw JSON in `benchmarks/results/2026-08-22-i7-12650h/raw`.
3. Closed the search-speed gap (done, 2026-08-22): `neighbors()` returns a borrowed slice, SIMD dispatch resolves once per process, the visited set is a generation-counted pool, neighbor vectors are prefetched, AVX2 kernels use four accumulators, and inserts batch into one WAL fsync per group. Re-measured on the same SIFT1M harness: at M=32/efc200/ef=100 search is now 2680 QPS / p50 0.38 ms (3.4x the July baseline, on par with FAISS/hnswlib) and build is 2.06-2.68x faster, with bit-identical recall. Raw JSON in `benchmarks/results/2026-08-22b-i7-12650h/raw` and `.../2026-08-22c-i7-12650h/raw` (the latter a cool-down rerun of the two efc=200 configs).
4. Add metadata and filtered search (done, 2026-08-23; plan week-14 milestone, the last missing Full-tier feature). Implemented per the `HANDOFF-metadata.md` design: out-of-band `Metadata` + `Filter` (`Eq`/`And`), durable via an `InsertMeta` WAL op and a CRC32 `<data>.meta` snapshot sidecar under format version 3, naive post-filtering `search_filtered` with adaptive over-fetch, surfaced through the server and `quiver-py`. Filtered Recall@10 at 1%/10%/50% selectivity is >= 0.992 everywhere vs brute-force filtered ground truth (raw JSON in `benchmarks/results/2026-08-23-i7-12650h/raw`). The remaining gap is the inverse-selectivity cost of post-filtering, documented under Known Limitations.
5. Persist SQ8 and evaluate quantized HNSW.
6. Implement IVF-PQ only after the HNSW and filtering layers are measured and stable.
7. Add packaging, release automation, and generated benchmark charts.
