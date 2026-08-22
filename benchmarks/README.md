# Reproducible SIFT1M benchmark suite

This directory contains a single-threaded SIFT1M comparison of Quiver HNSW,
FAISS `IndexHNSWFlat`, and hnswlib. Raw JSON under
[`results/2026-07-28-i7-12650h/raw`](results/2026-07-28-i7-12650h/raw) is the
source of truth for the July HNSW runs and for the FAISS/hnswlib reference
numbers. SQ8 and brute-force memory results are under
[`results/2026-07-29-i7-12650h/raw`](results/2026-07-29-i7-12650h/raw).

A same-host Quiver rerun on **2026-08-22** — after the diversified
neighbor-selection and packed `u32` adjacency commits — lives under
[`results/2026-08-22-i7-12650h/raw`](results/2026-08-22-i7-12650h/raw). In the
tables below, Quiver rows carry both the 2026-07-28 baseline and the
2026-08-22 rerun; FAISS and hnswlib rows are the unchanged 2026-07-28
reference. Regenerate the before/after delta with
`python benchmarks/compare_runs.py benchmarks/results/2026-07-28-i7-12650h/raw benchmarks/results/2026-08-22-i7-12650h/raw`.

## Hardware and methodology

| Host | CPU | Cores / logical CPUs | RAM | Dataset | Threads (build / query) |
| --- | --- | ---: | ---: | --- | ---: |
| Windows 11 build 26200 | 12th Gen Intel Core i7-12650H | 10 / 16 | 16 GB (15.7 GiB) | SIFT1M, 128-D, 1M base, 10k queries, L2 | Quiver 1 / 1; FAISS 1 / 1; hnswlib 1 / 1 |

FAISS and hnswlib were explicitly configured with one thread (`omp_set_num_threads(1)`,
`set_num_threads(1)`, and `num_threads=1`), so raw and single-thread-controlled
build numbers are identical. Quiver is single-threaded by API design. Build time
is index construction only; Quiver fsyncs a CRC32 WAL record for every insert,
whereas the competitors build in memory and serialize afterward.

## Build comparison (raw and controlled)

| M | efConstruction | Engine | Build s (raw) | Build s (1-thread controlled) | Recall@10 @ efSearch=100 |
|---:|---:|---|---:|---:|---:|
| 16 | 100 | Quiver (2026-07-28) | 783.1 | 783.1 | 0.9243 |
| 16 | 100 | Quiver (2026-08-22) | 1051.4 | 1051.4 | 0.9837 |
| 16 | 100 | FAISS | 193.9 | 193.9 | 0.9796 |
| 16 | 100 | hnswlib | 175.7 | 175.7 | 0.9772 |
| 16 | 200 | Quiver (2026-07-28) | 1090.2 | 1090.2 | 0.9379 |
| 16 | 200 | Quiver (2026-08-22) | 1347.5 | 1347.5 | 0.9896 |
| 16 | 200 | FAISS | 377.2 | 377.2 | 0.9868 |
| 16 | 200 | hnswlib | 330.3 | 330.3 | 0.9829 |
| 32 | 200 | Quiver (2026-07-28) | 2423.5 | 2423.5 | 0.9595 |
| 32 | 200 | Quiver (2026-08-22) | 2507.7 | 2507.7 | 0.9961 |
| 32 | 200 | FAISS | 559.8 | 559.8 | 0.9922 |
| 32 | 200 | hnswlib | 566.4 | 566.4 | 0.9920 |

The 2026-08-22 rerun is the effect of the diversified neighbor-selection and
packed `u32` adjacency commits. Recall at the headline M=32/efConstruction=200
config rose from 0.9595 to **0.9961**, which now exceeds both FAISS (0.9922)
and hnswlib (0.9920) on this host. Build time rose modestly where the
heuristic is a large fraction of insert cost (M=16/efc100: 1.34x) and barely
at the headline config (M=32/efc200: 1.03x), because the ef_construction
search dominates there.

## efSearch sweep (M=32, efConstruction=200)

Each cell is `Recall@10 / Recall@100`; p50/p99 latency (ms) follows in the
second table. Values are measured over all 10,000 queries.

| Engine | ef=10 | ef=50 | ef=100 | ef=200 | ef=400 |
|---|---|---|---|---|---|
| Quiver (2026-07-28) | 0.7795 / — | 0.9364 / — | 0.9595 / 0.9334 | 0.9696 / 0.9598 | 0.9735 / 0.9708 |
| Quiver (2026-08-22) | 0.8593 / — | 0.9852 / — | 0.9961 / 0.9798 | 0.9988 / 0.9954 | 0.9993 / 0.9992 |
| FAISS | 0.7813 / — | 0.9722 / — | 0.9922 / 0.9582 | 0.9980 / 0.9900 | 0.9991 / 0.9982 |
| hnswlib | 0.7783 / — | 0.9718 / — | 0.9920 / 0.9576 | 0.9979 / 0.9898 | 0.9990 / 0.9981 |

After the rerun, Quiver's recall exceeds both FAISS and hnswlib at every
ef_search level — at ef=100, Recall@10 is 0.9961 vs 0.9922 (FAISS) and 0.9920
(hnswlib), and Recall@100 is 0.9798 vs 0.9582 / 0.9576.

Recall@100 is not run for ef<100 because `efSearch` must be at least k.

| Engine | ef | p50 / p99 ms (k=10) | p50 / p99 ms (k=100) |
|---|---:|---:|---:|
| Quiver (2026-07-28) | 100 | 0.9822 / 2.1804 | 0.9722 / 2.1950 |
| Quiver (2026-07-28) | 200 | 1.7233 / 3.6072 | 1.7575 / 3.7447 |
| Quiver (2026-07-28) | 400 | 3.1355 / 6.2091 | 3.0768 / 6.0650 |
| Quiver (2026-08-22) | 100 | 1.2833 / 2.8432 | 1.0956 / 2.6115 |
| Quiver (2026-08-22) | 200 | 1.9531 / 4.5375 | 1.8544 / 4.3608 |
| Quiver (2026-08-22) | 400 | 3.4117 / 7.4191 | 3.4421 / 7.2205 |
| FAISS | 100 | 0.3820 / 0.9059 | 0.3888 / 0.9802 |
| FAISS | 200 | 0.6623 / 1.4414 | 0.6853 / 1.4365 |
| FAISS | 400 | 1.2371 / 2.4580 | 1.2691 / 2.3778 |
| hnswlib | 100 | 0.3180 / 0.7362 | 0.3371 / 0.7164 |
| hnswlib | 200 | 0.5809 / 1.2400 | 0.6042 / 1.4282 |
| hnswlib | 400 | 1.1126 / 2.1862 | 1.1102 / 2.6400 |

The rerun's search latency is ~1.1–1.3x the July baseline at the headline
config (ef=100 p50: 0.98 -> 1.28 ms). This is a real, newly introduced
search-speed regression traced to the packed-adjacency accessor: `neighbors()`
in `hnsw.rs` allocates and copies a `Vec<usize>` on every call inside the
beam-search loop. The effect is largest where the copy dominates (higher M,
lower ef_search). Returning a borrowed `&[u32]` slice instead is the primary
Phase-2 search-speed fix; see the follow-up section below.

## SQ8 recall and throughput

SQ8 uses a full flat scan with an asymmetric L2 lookup table. The comparison
below uses all 1,000,000 SIFT base vectors and all 10,000 queries at k=10.
HNSW rows use the existing M=32, efConstruction=200, efSearch=100 runs.
SQ8 here is exhaustive (it checks every vector), so its recall advantage over
Quiver's HNSW is exact search with quantization error versus approximate search
with graph and implementation gaps, not evidence that "SQ8 is better."
SQ8's low QPS is structural because it performs an O(n) scan; combining
quantization with an index structure, as IVF-PQ actually does, is how to get
both memory savings and speed.

| Engine | Search parameters | Recall@10 | QPS |
|---|---|---:|---:|
| Quiver SQ8 flat | exhaustive scan | 0.9889 | 14.6 |
| Quiver HNSW (2026-07-28) | M=32, efC=200, ef=100 | 0.9595 | 945.3 |
| Quiver HNSW (2026-08-22) | M=32, efC=200, ef=100 | 0.9961 | 727.8 |
| FAISS HNSW | M=32, efC=200, ef=100 | 0.9922 | 2336.3 |
| hnswlib HNSW | M=32, efC=200, ef=100 | 0.9920 | 2832.4 |

At the headline config the rerun raises Recall@10 above both competitors
(0.9961 vs 0.9922 / 0.9920) but throughput drops to 727.8 QPS, ~3x behind
FAISS and hnswlib. Search speed is now the primary remaining gap; recall and
memory are no longer it.

## RSS / serialized size (index growth)

The clearest before/after signal for the packed-adjacency change is Quiver's
own **peak RSS** (peak working set, same measurement method on both runs).
The July graph used a nested `Vec<Vec<usize>>` per node; the rerun uses a
packed contiguous `u32` adjacency arena.

| M / efC | Quiver peak RSS MB (2026-07-28) | Quiver peak RSS MB (2026-08-22) | Reduction |
|---|---:|---:|---:|
| 16 / 100 | 1831.0 | 775.2 | 2.36x |
| 16 / 200 | 1849.3 | 775.4 | 2.38x |
| 32 / 200 | 3058.0 | 902.5 | 3.39x |

At the headline M=32 config, peak RSS fell from 3058 MB to 902.5 MB (3.4x).
The ~488 MB f32 vector payload is mmap-backed and its residency varies, so the
graph representation is the dominant term in this reduction. Peak RSS is used
(rather than the end-of-build RSS delta) because the mmap'd payload is not
fully resident at the sampling instant, which would understate the footprint.

Cross-engine reference, unchanged 2026-07-28 numbers. FAISS/hnswlib RSS is the
process working-set increase after loading queries and ground truth; Quiver
additionally keeps its durable vector store/WAL mapping resident, so absolute
cross-engine RSS is approximate, not a precise like-for-like metric.

| M / efC | FAISS RSS MB | hnswlib RSS MB |
|---|---:|---:|
| 16 / 100 | 633.8 | 784.7 |
| 16 / 200 | 633.8 | 785.2 |
| 32 / 200 | 755.9 | 906.3 |

The SQ8 and full-precision brute-force measurements below use separate
single-threaded processes at the same 1,000,000-vector count. RSS means process
working-set growth after queries and ground truth are loaded; the HNSW row is
the 2026-08-22 M=32, efConstruction=200 peak RSS for consistency with the
table above.

| Quiver index | Index RSS MB | Vector payload MB |
|---|---:|---:|
| SQ8 flat | 123.2 | 122.1 |
| Full-precision brute-force | 504.8 | 488.3 |
| HNSW, M=32 / efC=200 (peak, 2026-08-22) | 902.5 | 488.3 plus graph |

SQ8 reduced measured RSS by 4.10x versus the mmap-backed brute-force process,
consistent with its exactly 4x-smaller encoded vector payload. The original
1.98x figure understated the reduction because RSS was sampled while the
brute-force mmap was only partially resident, not because SQ8 fell short of its
compression ratio; the corrected benchmark scans every vector first.

The July 2026 M=32 Quiver RSS gap came from the graph representation, not just
vector storage: every node owned a `Vec<Vec<usize>>` (one heap allocation for
the outer layer list and generally one per populated neighbor list), with each
neighbor a pointer-width `usize`. FAISS and hnswlib store graph links in
compact contiguous native buffers. The 2026-08-22 rerun confirms the fix: the
packed contiguous `u32` adjacency arena cut peak RSS by 3.4x at M=32, bringing
Quiver's graph footprint in line with the compact native-buffer approach.

## Recall-gap investigation (resolved by the 2026-08-22 rerun)

The July 2026 Quiver benchmark did **not** implement the original HNSW
heuristic/diversified neighbor-selection rule. In `hnsw.rs`, insertion selected
the first `M` nearest construction candidates, and `prune_connections` sorted by
distance to the node and truncated to capacity. FAISS and hnswlib use
diversified pruning. This was identified as the plausible primary cause of the
matched M=32/efC=200 recall gap (Quiver 0.9595 versus FAISS 0.9922 and hnswlib
0.9920 at efSearch=100) — an algorithmic difference, not an unexplained
benchmark anomaly.

The 2026-08-22 same-host rerun confirms the hypothesis. With diversified
selection, Quiver's M=32/efC=200 Recall@10 at efSearch=100 rose from 0.9595 to
**0.9961**, which now exceeds FAISS (0.9922) and hnswlib (0.9920) on this host.
The recall gap is not merely closed but reversed. The same rerun also confirms
the packed-adjacency memory fix (peak RSS 3.4x lower at M=32).

## Phase 3 measurement status (rerun complete)

The HNSW implementation uses diversified neighbor selection for both new-node
links and reverse-link pruning, plus fixed-capacity packed `u32` adjacency
blocks. The same-host SIFT1M rerun is **complete** (2026-08-22, this host,
single-threaded): raw JSON is committed under
[`results/2026-08-22-i7-12650h/raw`](results/2026-08-22-i7-12650h/raw), and the
tables above now carry both the 2026-07-28 baseline and the 2026-08-22 rerun.
FAISS and hnswlib were not rerun and remain the 2026-07-28 reference.

## Follow-up: newly measured search-speed regression

The rerun surfaced a search-speed regression introduced alongside the memory
fix: at the headline config, p50 latency at efSearch=100 rose from 0.98 to
1.28 ms and QPS fell from 945 to 728. The dominant cause is the
packed-adjacency accessor `neighbors()` in `hnsw.rs`, which allocates and
copies a `Vec<usize>` on every call inside the beam-search loop; the cost is
largest at higher M and lower ef_search, where the copy dominates the distance
math. Secondary per-distance overheads remain: runtime AVX2/FMA feature
detection on every `compute_distance` call, a per-query `HashSet` visited set,
and recomputing the cosine query norm per candidate. These are the Phase-2
search-speed targets; recall and memory are no longer the binding gap.

## Distance-kernel measurements

Criterion results in `target/criterion` report scalar versus AVX2/FMA dispatch.
At 128 dimensions, mean nanoseconds per call were: L2 7.12 SIMD vs 41.49
scalar (5.8x), dot product 5.63 vs 37.52 (6.7x), and cosine 17.42 vs 77.81
(4.5x). A flamegraph comparison was not captured: `samply` was attempted via
`cargo install samply --locked`, but the install did not complete before the
two-minute command limit and no `samply` executable was installed. The
reproducible Criterion scalar/SIMD numbers are committed and provide the
actionable kernel comparison.

## Run it

See `config.json`, `export_sift.py`, and `competitor_benchmark.py` for the
dataset export and FAISS/hnswlib commands. To rerun the Quiver side on SIFT1M
after exporting to fvecs/ivecs:

```bash
cargo run --release -p quiver-core --bin sift_benchmark -- \
  --base sift_base.fvecs --queries sift_query.fvecs \
  --groundtruth sift_groundtruth.ivecs \
  --work-dir <empty-dir> --output results/<date>-<host>/raw/quiver-m32-efc200.json \
  --m 32 --ef-construction 200 --base-limit 1000000 --query-limit 10000
```

`--work-dir` must be empty (the benchmark refuses a non-empty dir). To diff two
runs (build time, RSS, and per-(k, ef_search) recall/QPS/latency):

```bash
python benchmarks/compare_runs.py <baseline-raw-dir> <new-raw-dir>
```
