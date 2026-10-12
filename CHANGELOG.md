# Changelog

All notable changes to Quiver are documented in this file.

## [0.2.0] - 2026-10-12

Initial public release to crates.io. Includes all bug fixes from the audit, format v4 header CRC, auto-flush, WAL-size compaction, and improved error handling.

## [0.1.0] - 2026-08-24

### Added

- **Header CRC (format v4):** Data-file header now carries a CRC32 checksum at bytes 60–64. A torn or corrupt header is a loud open-time error instead of silently shrinking `vector_count`.
- **Auto-flush:** Server can periodically flush the HNSW graph snapshot to bound staleness after a hard kill (`QUIVER_FLUSH_INTERVAL_SECS`, default 300 seconds).
- **WAL-size bound:** `HnswConfig::max_wal_bytes` triggers compaction when the WAL grows beyond a threshold, preventing unbounded growth from metadata mutations and deletes. Checked after each delete, update, or clear operation.
- **Input validation:** New `QuiverError::InvalidInput` variant for client mistakes (non-finite vectors, f32 overflows, invalid k/ef_search); these now return HTTP 400 instead of 500.

### Fixed

- **Server crash from huge k:** A search request with an enormous `k` (e.g., 10^15) no longer causes `BinaryHeap::with_capacity` to request a multi-petabyte allocation and crash the process. Core clamps `k` to the candidate count; server rejects values outside 1–10,000.
- **Legacy v1 store corruption:** Inserting metadata into a v1-format store no longer relabels the header version while keeping v1-layout records, which would corrupt the file. Metadata on v1 stores is now rejected with a clear error until `compact()` upgrades the layout.
- **Unbounded WAL growth from metadata:** The WAL checkpoint now rewrites `InsertMeta` entries (which carried full vector payloads) as metadata-only `UpdateMeta` entries, preventing metadata-heavy workloads from growing the WAL indefinitely.
- **Unnecessary full compaction:** The WAL-size trigger now calls `flush()` first to drop checkpointed insert payloads; only if the retained delete/metadata entries still exceed the bound does it trigger a full compaction and rebuild.

### Changed

- **Format version 3 → 4:** New stores are format v4 (with header CRC). Older v1–v3 stores remain readable but unprotected until compacted. There is no automatic upgrade.
- **Async-safe auto-flush:** Server auto-flush runs on Tokio's blocking pool to avoid stalling async worker threads during fsync.
- **as_chunks for parsing:** Replaced `chunks_exact(4)` with `as_chunks::<4>()` in benchmark and test code for compatibility with Rust 1.98+ clippy lints.

### Performance

- **Filtered search (2026-10-12 rerun):** [Pending benchmark results]
  - Previous (2026-08-23b) at M=32/efC=200: 195 QPS at 1% selectivity, 2124 QPS at 50%, recall ≥ 0.9837 throughout.
  - Selective `Eq` filters (≤ 32,768 matches) now take an exact-scan fast path with recall 1.0.

### Infrastructure

- **CI:** GitHub Actions runs tests, clippy, formatting, and 60-second file-format fuzzing on every push. All passing.
- **Benchmarks:** Reproducible SIFT1M harnesses with raw JSON output committed alongside results. Includes FAISS and hnswlib comparisons on the same host and harness.

## [0.0.1] - 2026-08-24

Initial portfolio-grade release: HNSW and SQ8 indexes, filtered search, metadata, WAL recovery, crash-safe compaction, HTTP server, Python bindings, and comprehensive tests.
