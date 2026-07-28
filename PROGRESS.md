# Quiver — Project Progress & Context Transfer

> **Last updated:** 2026-07-28
> **Purpose:** Context transfer document for continuing development on Linux

---

## 🏗️ Architecture Overview

Quiver is a **portfolio-grade, single-node, embeddable vector search engine** written in Rust, published as `quiver-db`.

```
quiver/
├── Cargo.toml              # Workspace root (3 crates)
├── quiver-core/            # Core engine library
│   ├── src/
│   │   ├── lib.rs          # Public API re-exports
│   │   ├── error.rs        # QuiverError enum (thiserror)
│   │   ├── distance.rs     # L2, dot, cosine — scalar + AVX2 SIMD
│   │   ├── index/
│   │   │   ├── mod.rs      # SearchResult type
│   │   │   ├── brute_force.rs  # Flat index (baseline)
│   │   │   └── hnsw.rs     # HNSW graph index
│   │   └── storage/
│   │       ├── mod.rs
│   │       ├── header.rs   # 64-byte binary file header (QVDB magic)
│   │       ├── wal.rs      # CRC32-checksummed WAL
│   │       └── vecstore.rs # mmap vector storage + crash recovery
│   └── benches/
│       ├── distance_benchmarks.rs  # Scalar vs SIMD comparison
│       └── index_benchmarks.rs     # HNSW/BF insert/search/recall
├── quiver-server/          # axum REST API (scaffolded, not implemented)
└── quiver-py/              # PyO3/maturin bindings (scaffolded, not implemented)
```

## ✅ Completed Phases

### Phase 1: Project Initialization
- Rust workspace with 3 crates (quiver-core, quiver-server, quiver-py)
- Dual MIT/Apache-2.0 licensing
- Centralized workspace dependencies in root `Cargo.toml`

### Phase 2: Storage Engine
- **FileHeader** — 64-byte binary header with magic `QVDB`, format version, dimension, metric, vector count
- **WAL** — Length-prefixed, CRC32-checksummed entries; supports Insert + Delete ops; truncates corrupt tails
- **VectorStore** — mmap-backed vector storage with amortized 2x file growth
- **Crash Recovery** — WAL replay on open with idempotent insert recovery
- **Durable Deletion** — delete records are fsynced before tombstones become visible and are restored on reopen
- **Crash-safe compaction** — rewrites live vectors to a durable replacement, journals the data/WAL swap, and resets the WAL only after installation
- **Stable IDs after compaction** — format v2 stores an explicit u64 vector ID per record; format v1 remains readable

### Phase 3: Distance Metrics
- **Scalar:** L2 squared, dot product, cosine similarity
- **AVX2 SIMD (Phase 6):** All three metrics with `_mm256_*` + FMA intrinsics
- **Runtime dispatch:** `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`
- **Tail handling:** Scalar fallback for dimensions not divisible by 8
- Criterion benchmarks at dims [128, 256, 512, 768, 1024, 1536]

### Phase 4: Brute-Force Index
- Linear scan + top-K binary heap
- Wraps `VectorStore` for persistence
- All metrics (L2, dot product, cosine)
- 100% recall (exact search, by definition)

### Phase 5: HNSW Index
- Multi-layer graph with exponential-decay layer assignment
- Greedy + beam search (`ef_search` parameter)
- Configurable `HnswConfig`: M, m_max0, ef_construction, ml, tombstone_ratio
- Durable tombstone deletion restored from the WAL on reopen
- Automatic live-only storage + graph compaction above the configured tombstone threshold
- `parking_lot::RwLock` concurrency wrapper (v1: coarse-grained)
- Graph rebuild from VectorStore on reopen
- **Recall@10 > 95%** against brute-force ground truth (1000 vectors, 50 queries)

### Phase 6: SIMD Distance Kernels (PARTIALLY COMPLETE)
- ✅ AVX2+FMA kernels for L2, dot product, cosine
- ✅ Runtime feature detection + scalar fallback
- ✅ Correctness validated (75 tests pass, including durable deletion, kill-mid-compaction recovery, legacy-format migration, and non-aligned SIMD dimensions)
- ✅ Criterion benchmark harness (scalar vs SIMD, index-level)
- ⬜ Run benchmarks and capture before/after numbers
- ⬜ Document SIMD speedup results

## 📊 Test Status

```
test result: ok. 75 passed; 0 failed; 0 ignored
```

All tests pass as of the last run:
- 18 distance tests (including 3 SIMD-specific: simd_vs_scalar, non_aligned_dims, simd_available)
- 10 brute-force index tests
- 12 HNSW index tests
- 7 header tests
- 7 vecstore tests
- 7 WAL tests
- 3 doc-tests (0 — none written yet)

## 🔧 Build Environment

### Windows (current)
```powershell
# CRITICAL: Must use MSYS2 MinGW64 toolchain, NOT the 32-bit MinGW
$env:PATH = "C:\msys64\mingw64\bin;" + ($env:PATH -replace "C:\\MinGW\\bin;?","")

# Build
cargo build --package quiver-core

# Test
cargo test --package quiver-core

# Bench (distance)
cargo bench --package quiver-core --bench distance_benchmarks

# Bench (index)
cargo bench --package quiver-core --bench index_benchmarks
```

**Known issue on Windows:** `C:\MinGW\bin` has a 32-bit `dlltool.exe` and `as.exe` that conflict with the 64-bit Rust target. Always prepend `C:\msys64\mingw64\bin` to PATH.

### Linux (target)
```bash
# Should work out of the box with:
rustup default stable
cargo test --package quiver-core
cargo bench --package quiver-core
```

No special toolchain setup needed on Linux. AVX2+FMA will be auto-detected at runtime.

## 📋 Remaining Phases

### Phase 6 (finish): Benchmark Results
- Run `cargo bench` and capture scalar vs SIMD numbers
- Expected 3-6x speedup on distance functions at dim≥128

### Phase 7: Scalar Quantization (SQ8)
- Per-dimension min/max tracking, float32→uint8, asymmetric distance

### Phase 8: IVF-PQ Index
- K-means clustering + product quantization + optional reranking

### Phase 9: Metadata & Filtering
- Key-value metadata storage, post-filtering, selectivity benchmarks

### Phase 10: API Layer & Python Bindings
- axum REST API (insert, batch, search, delete, /metrics)
- PyO3/maturin bindings → publish to PyPI as `quiver-db`

### Phase 11: Benchmarking & Documentation
- SIFT1M benchmarks, comparison vs FAISS/hnswlib
- README, quickstart demo, blog posts

### Phase 12: Stretch Goals
- Filter-aware traversal, lock-free reads, DiskANN, hybrid search

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `parking_lot::RwLock` | Faster than std RwLock, writer-preferring; v1 is coarse-grained |
| mmap + WAL | Durability without full DB overhead; WAL cleared on flush |
| FMA intrinsics | Single rounding = more accurate than scalar (slight float diff is expected) |
| Relative tolerance in SIMD tests | FMA accumulation error grows with magnitude; absolute tolerance fails at dim≥255 |
| `compute_distance` returns "lower = better" | Uniform semantics: L2 natural, dot/cosine negated. Enables single min-heap |
| No sqrt in L2 | Monotonic transform — ranking preserved without it |

## 📁 Key Files to Read First

1. [distance.rs](quiver-core/src/distance.rs) — SIMD kernels + dispatch logic
2. [hnsw.rs](quiver-core/src/index/hnsw.rs) — Graph index implementation
3. [vecstore.rs](quiver-core/src/storage/vecstore.rs) — mmap storage + WAL integration
4. [Cargo.toml](Cargo.toml) — Workspace dependencies
