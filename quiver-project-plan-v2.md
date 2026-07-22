# Quiver — Vector Database From Scratch, Project Plan (v2)

**Package name note:** `quiver` is taken on both crates.io and PyPI (unrelated projects). Publish as `quiver-db` on both registries; use "Quiver" as the project/brand name everywhere else (repo, README, resume, talking about it). **Re-verify `quiver-db` is still unclaimed right before you actually publish (week 14), not just now** — package namespaces get squatted over a 12+ week window.

## Positioning (read this first, even before the cut line)

This is a **portfolio project aimed at landing an infra/systems/ML-platform engineering role**, not a startup and not a production service. Say this explicitly in the README rather than leaving it implied — it's a strength, not a hedge: it tells a reviewer you understand the difference between a learning artifact and a product, and it preempts the "why did you reinvent this, Qdrant/Pinecone/FAISS already exist" question by answering it before anyone asks.

The corollary: the ROI on this project is **career capital**, not commercial viability. As a company, this has no realistic path — Pinecone, Weaviate, Qdrant, Milvus, Chroma, LanceDB, Turbopuffer, and pgvector already cover this space with years of head start and real funding, and even the "embedded, SQLite-for-vectors" niche you'd naturally retreat to is already contested (LanceDB, sqlite-vec, usearch). That's fine — it's not what this project is for. Just make the decision on purpose so scope doesn't quietly drift toward "make it a real service" mid-project (at which point the non-goals below stop being non-goals).

## Goal
Build a single-node, embeddable vector search engine implementing HNSW and IVF-PQ from scratch, with real benchmarks (recall@k, QPS, p50/p99 latency, memory footprint) measured against a public standard dataset (SIFT1M) and a real embedding dataset, and compared honestly against **both FAISS and hnswlib**, run locally on the *same hardware* — not published numbers. Every benchmark result discloses CPU model, core count, RAM, and thread count for all engines compared, on both sides.

*Why hnswlib in addition to FAISS:* FAISS is a large, general research library with overhead unrelated to HNSW quality specifically. hnswlib is a minimal, well-regarded pure-HNSW C++ implementation. Comparing against it isolates "is my HNSW well-implemented" from "is FAISS's overall architecture fast" — a more precise read than FAISS alone.

*Expectation to set explicitly in the README, before anyone else points it out:* after 12–14 weeks of solo work, you will very likely be several times slower than both baselines. That's expected — they have years of tuning. The value is in the honest gap measurement, the profiling that explains the gap, and what you'd do next — not in "winning."

## MVP cut line (unchanged — this is the best risk-management idea in the plan)

This project has a natural stopping point at every milestone. Treat weeks 1–6 as **non-negotiable core** — a correct HNSW index with honest benchmarks against brute-force ground truth is a complete, demoable project on its own. Everything after that is a valuable but sequential add-on. If time runs out, stop after any milestone and you still have something real to show.

**When you do have to cut, cut in this order: IVF-PQ first, then filtering, then anything else — before you cut the demo or the writeup.** A demo and a clear writeup have higher signal-per-hour for a hiring audience than a second index type does.

| Tier | Scope | If you only get this far |
|---|---|---|
| **Core (must ship)** | Storage + CI + distance metrics + brute-force + HNSW + benchmark harness + profiling writeup + basic crash-recovery test | A working, correct, benchmarked ANN index with verified crash safety — legitimate on its own |
| **Extended** | SQ8 + IVF-PQ (validated against a Python reference) + memory comparison | The "memory efficiency" story |
| **Full** | Metadata filtering (naive, tested across selectivities) + REST API + graceful shutdown + Python bindings + quickstart demo | A usable library/service you can actually show someone |
| **Stretch** | Filter-aware traversal, concurrency, DiskANN, Raft clustering, content trail (blog posts) if not already woven in | Only if everything above is solid |

## Non-goals (explicit, so it reads as a boundary, not a gap)

Auth, TLS, multi-tenancy, horizontal scaling, and production hardening are explicitly out of scope for this project. Quiver is a portfolio-grade single-node engine, not a production service — say so directly in the README rather than leaving it implied.

## Tech stack decision

| Choice | Pick | Why |
|---|---|---|
| Core language | **Rust** | Memory safety without GC pauses, strong SIMD story, and it's what the serious vector-DB ecosystem (Qdrant, LanceDB) has converged on |
| SIMD approach | **Hand-written `std::arch` intrinsics** (AVX2 on x86, NEON on ARM) behind `#[target_feature]` + `is_x86_feature_detected!` runtime dispatch, written **after** the profiling pass confirms distance computation is the hot path — not before | `std::simd` (portable_simd) remains nightly-only with no near-term stabilization signal as of early 2026, and `packed_simd` has been abandoned for years. Hand-written intrinsics are stable, and as of Rust 1.87 most intrinsics no longer require `unsafe` to call (the `safe_unaligned_simd` crate covers the rest), so the historical "unsafe overhead" argument against hand-rolled intrinsics is weaker than it used to be. "I profiled, found the hot path, then wrote AVX2 kernels with runtime dispatch and a before/after flamegraph" is a stronger and more honest technical story than writing SIMD first and profiling something else after. (`pulp` or `wide` are solid stable-Rust fallbacks if portability matters more than squeezing out max performance.) |
| Bindings | PyO3 via `maturin` | Makes the library usable as `pip install`, and lets you run FAISS/hnswlib and your engine through the *same* Python benchmark harness for a fair comparison |
| Storage | Custom memory-mapped file format, versioned header | No external DB dependency; version byte in the header from day one so format changes later (e.g., metadata support) don't break old indexes |
| Concurrency model (v1) | Single-writer, multi-reader via a global `RwLock` | Decide this now, not in week 14 — it shapes your node representation. Use `parking_lot::RwLock` rather than `std::sync::RwLock` for better write-fairness under read-heavy load. Lock-free/sharded reads are a stretch goal, not a v1 requirement |
| API async runtime | `axum` on `tokio` (or `tonic` if gRPC) | Committing now avoids a mismatch between the `RwLock`-based engine state and async handlers later |
| Benchmarking | `criterion` for microbenchmarks, custom harness for macro recall/QPS runs, both logging hardware metadata alongside results | Kept separate deliberately — see Benchmarking plan |
| Observability | `tracing` for structured logs/spans in the API layer; explicit graceful shutdown (flush WAL + mmap on SIGTERM before exit) | Cheap, and "handles shutdown correctly" is exactly the kind of detail that separates a demo from something that reads as production-minded |
| Packaging | Dual MIT/Apache-2.0 license, published as `quiver-db` via `maturin` to PyPI and `cargo publish` to crates.io | Standard Rust ecosystem convention |
| CI | GitHub Actions, **stood up in week 1**, not bolted on at the end | The entire point of CI in a solo project is catching regressions while you build, not certifying the finished thing |

## Architecture — components, in build order

### 1. Storage engine (foundation)
- Custom binary file format: **versioned header** (format version, dimension, count, metric type) + fixed-size vector records
- Memory-mapped access (`mmap`)
- Write-ahead log (WAL), scoped deliberately for v1: length-prefixed, checksummed entries; recovery replays until the first checksum failure and truncates there; periodic full mmap flush + fsync on a timer. This is *not* ARIES-style recovery — don't try to build that, it's not what the project needs to prove
- **A basic crash-recovery test lands in week 2, not the week 15 buffer**: kill the process mid-write (`SIGKILL` during insert) and assert the file reopens cleanly with no corruption. This is the storage engine's central correctness claim — verify it early, not last
- **A `cargo fuzz` target on the file-format parser specifically**, separate from the insert/delete fuzz tests below. Parsing a binary mmap'd file with corrupted length/offset fields is exactly the class of bug (integer overflow, out-of-bounds slice) that survives "it's Rust so it's safe" assumptions
- No dependency on the index — build and test standalone first

### 2. Distance metrics module
- Cosine similarity, L2, dot product — **scalar implementations first**
- If vectors are normalized to unit length at insert time (reasonable for embedding models), cosine reduces to a dot product — one fast kernel covers two of the three metrics. Call this out explicitly in the writeup as a deliberate optimization
- Correctness tests: scalar implementation validated against a naive reference on random vectors, floating-point tolerance
- **SIMD kernels are added later (see component 5), after profiling — not here.** This is a deliberate change from the original sequencing: writing SIMD before profiling and then claiming "profiling justified it" is backwards

### 3. Brute-force baseline index
- Build **before** HNSW — this is your ground truth for recall measurement and your fallback for small datasets
- Linear scan + top-K heap selection

### 4. HNSW index (primary ANN structure)
- Multi-layer graph, exponential-decay layer assignment
- Insert: greedy search from top layer to entry point, connect to M nearest neighbors per layer down to layer 0
- Search: greedy descent, then beam search (`ef_search`) at layer 0
- Tunable parameters to sweep and benchmark: `M`, `ef_construction`, `ef_search` — show the recall/speed tradeoff curve
- Delete handling: tombstone approach with a **concrete compaction trigger** — track tombstone ratio, trigger full rebuild-from-live-vectors above a threshold (e.g., 20%)
- **Graph memory layout — decide and benchmark explicitly.** Neighbor-list cache misses during beam search are often the real bottleneck at scale, not the distance math. Store each node's neighbor IDs contiguously to enable prefetching, and treat this as its own benchmarkable decision alongside the SIMD work, not an afterthought

### 5. Profiling pass (do this before writing SIMD kernels — sequencing fixed from v1)
- Use `perf` + flamegraphs (`cargo flamegraph`) to find the actual hot path on the **scalar** implementation, before any SIMD or layout optimization exists
- Capture a before/after flamegraph for each major optimization (SIMD kernels, memory layout) — this is what turns "I wrote SIMD because that's where performance comes from" into "I profiled it, found X was the bottleneck, fixed it, here's the proof," a materially stronger interview story that also protects you from optimizing the wrong thing
- **Now write the SIMD distance kernels** (moved from component 2), with runtime feature detection, against the evidence this pass produced

### 6. Scalar quantization (SQ8)
- Per-dimension min/max, quantize each float to a byte
- ~4x memory reduction with a much simpler implementation than PQ
- Gives you a working "compressed index" milestone independent of whether full PQ lands on schedule

### 7. IVF-PQ index (memory-efficient alternative)
- **Before writing the Rust implementation, write a ~50-line numpy/scikit-learn reference of the same k-means + PQ pipeline** on a small synthetic dataset. When Rust recall is off, diff against the reference instead of debugging algorithm-correctness and systems-code-correctness simultaneously — this is the single biggest schedule risk in the original plan and the cheapest way to de-risk it
- **IVF**: k-means (k-means++ init) clusters the dataset into N coarse clusters at build time; query searches only the nearest few clusters
- **PQ**: split each vector into sub-vectors, quantize each independently against a small codebook
- Benchmark RAM usage of IVF-PQ vs. raw HNSW at comparable recall
- Rerank step: after compressed-distance candidates, optionally rerank top-K against full-precision vectors

### 8. Metadata + filtered search layer
- Attach key-value metadata at insert time
- Naive approach first: over-fetch candidates, filter after
- **Test filtered-recall across a range of filter selectivities (1%, 10%, 50%)**, not just one — this is what makes the naive-approach limitation visible and sets up the stretch-goal narrative honestly
- Filter-aware traversal moved to stretch goals (see below)

### 9. Query API layer
- `axum`-based REST server on `tokio`: insert, delete, search (with optional filter), batch insert
- **Graceful shutdown**: flush WAL and mmap on SIGTERM before the process exits — otherwise every restart is an unplanned crash-recovery test
- `tracing`-based structured logs and spans
- `/metrics` endpoint (query latency histogram, index size, cache stats) — cheap once the API layer exists, signals "built like a service"
- Python client via PyO3/maturin bindings, published as `quiver-db`
- **A quickstart**: a short notebook or script showing semantic search over a small real corpus (your own blog, a Wikipedia slice) hit through the REST API — plus a 30-second terminal recording or GIF for the README. This is the single biggest missing piece in the original plan: benchmark charts prove engineering rigor to people who already know what to look for; a working demo is what everyone else — recruiters, hiring managers, engineers skimming — actually looks at

### 10. CI (moved to week 1 — see Tech stack table)
- GitHub Actions: correctness + fuzz tests (including the file-format fuzz target) on every push, from day one
- Commit benchmark JSON output over time and plot it, so the README can show a "performance over the life of the project" chart, not just a final snapshot — cheap addition, disproportionate signal value

### 11. Content trail (new — not in v1)
- 2–4 short technical posts written **as you go**, not after the fact: e.g. "Building HNSW from scratch," "What profiling taught me before I wrote a line of SIMD," "3x slower than FAISS and hnswlib: here's why." A repo with strong engineering and zero visibility gets found by nobody; the same repo with a couple of well-written posts gets found by search and gives you something concrete to talk about in an interview that isn't "go read my code"
- This has a strong claim to being higher-ROI than IVF-PQ if the timeline is tight — see the cut-line note above

## Milestone plan

Timeline assumes roughly full-time effort. If this is nights-and-weekends work, expect **20–24 weeks** rather than 12–15 — treat the table below as sequencing, not a calendar.

| Weeks | Milestone | Tier |
|---|---|---|
| 1 | Storage engine (versioned format) + **CI pipeline stood up** + scalar distance metrics + brute-force baseline | Core |
| 2 | Correctness tests solid + **basic crash-recovery test (kill mid-write)** + **file-format fuzz target** | Core |
| 3–6 | HNSW insert + search (validated against brute-force), parameter sweep, benchmark harness (recall@k, QPS, p50/p99), **profiling pass with flamegraphs on the scalar implementation** | Core |
| 6–7 | **SIMD distance kernels**, written against profiling evidence, before/after flamegraph comparison; graph memory layout decision + benchmark | Core |
| 8 | SQ8 quantization + benchmarks | Extended |
| 9 | **Python reference implementation of k-means + PQ**, validated on synthetic data | Extended |
| 10–12 | IVF-PQ in Rust (k-means, PQ codebook training, quantized search), validated against the Python reference | Extended |
| 13 | IVF-PQ vs HNSW vs SQ8 comparison benchmarks (recall, memory, speed), **hardware/thread count disclosed** | Extended |
| 14 | Metadata filtering (naive) + filtered-recall benchmarks across multiple selectivities | Full |
| 15 | REST API (axum) + graceful shutdown + `tracing` + `/metrics` + Python bindings via maturin | Full |
| 16 | Quickstart demo (notebook + recording) + documentation + README with real benchmark charts (FAISS **and** hnswlib, disclosed hardware) | Full |
| 17+ (buffer) | Deletes edge cases, deeper crash-recovery fuzzing, content trail posts, any stretch goals if time allows | Buffer / Stretch |

## Benchmarking plan

- **Datasets**:
  - SIFT1M (1M vectors, 128-d, ships with precomputed ground truth) for the classic recall/QPS story
  - A second, smaller benchmark on a real embedding dataset (e.g., a subset of a Cohere/OpenAI Wikipedia-embeddings set, cosine-native) — closer to the actual target use case (RAG-style embedding search) than raw SIFT descriptors
- **Two benchmark layers, kept separate**:
  - `criterion` microbenchmarks: distance function throughput, single insert/search latency — statistically sound regression detection during development
  - Macro harness: end-to-end recall@k / QPS / p50-p99 against SIFT1M and the embedding dataset
- **Metrics**: Recall@10/@100 vs. `ef_search`/`nprobe` sweeps; QPS at each recall level; p50/p99 latency; RSS memory for HNSW vs SQ8 vs IVF-PQ at comparable recall; index build time
- **Comparison point**: run **both FAISS and hnswlib** locally, on the same machine, same dataset, same benchmark harness — not published numbers. Disclose CPU model, core count, RAM, and thread count for every engine in every result table. This is the single highest-leverage credibility fix in the whole plan, and the hardware disclosure is what makes it actually verifiable rather than just claimed
- **Set expectations in the README explicitly**: expect to land several times slower than both baselines after 12–17 weeks of solo work — that's normal, and the honest gap analysis is more valuable than a suspiciously close result

## Testing strategy
- Unit tests per module (storage, distance metrics, HNSW insert/search, PQ quantization)
- Correctness: recall against brute-force baseline must exceed a threshold (e.g., >95% recall@10) at reasonable parameters
- **Two fuzz surfaces**: random insert/delete sequences (index-logic correctness) **and** the binary file-format parser (memory-safety on malformed/corrupted files) — these catch different bug classes and both belong in CI
- Basic crash-recovery test in week 2; deeper crash-injection fuzzing in the buffer weeks
- Concurrency tests once the `RwLock` model is in place: readers shouldn't block on writes
- CI runs correctness + fuzz tests on every push, from week 1

## Cost
**$0.** SIFT1M and the embedding dataset download, all development, all benchmarking — CPU-only, your own machine.

## Deliverables
1. `quiver-db` Rust crate + Python bindings, published to crates.io / PyPI via maturin, dual MIT/Apache-2.0 licensed
2. README with real benchmark charts (recall/QPS curves, memory comparison, before/after flamegraphs), including local comparisons against **both FAISS and hnswlib**, with hardware disclosed, and an explicit statement of positioning (portfolio-grade, not production) and expected-gap framing
3. A **quickstart demo**: notebook or script + short recording showing real semantic search through the API
4. Write-up explaining the HNSW vs SQ8 vs IVF-PQ tradeoffs you actually measured, and what profiling revealed
5. 2–4 short technical blog posts, written during the project, not after
6. GitHub repo with clean commit history (storage → baseline → HNSW → SQ8 → IVF-PQ → filtering → API), CI badge (running since week 1), benchmark-history chart

## Stretch goals (only if the Full tier is solid and you have time left)
- **Filter-aware traversal** — skip non-matching nodes during graph walk rather than post-filtering; note the failure mode to design around: naive skip-during-traversal can strand greedy search in a region with no matching neighbors when the filter predicate is selective, tanking recall. Worth reading how ACORN-style approaches handle this before implementing
- Lock-free / sharded concurrent reads, replacing the v1 `RwLock`
- Raft-based clustering for multi-node distribution
- On-disk DiskANN-style index for datasets larger than RAM
- Dense + BM25 hybrid search fusion
