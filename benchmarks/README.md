# Reproducible SIFT1M benchmark suite

This directory contains a single-threaded SIFT1M comparison of Quiver HNSW,
FAISS `IndexHNSWFlat`, and hnswlib. Raw JSON under
[`results/2026-07-28-i7-12650h/raw`](results/2026-07-28-i7-12650h/raw) is the
source of truth.

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
| 16 | 100 | Quiver | 783.1 | 783.1 | 0.9243 |
| 16 | 100 | FAISS | 193.9 | 193.9 | 0.9796 |
| 16 | 100 | hnswlib | 175.7 | 175.7 | 0.9772 |
| 16 | 200 | Quiver | 1090.2 | 1090.2 | 0.9379 |
| 16 | 200 | FAISS | 377.2 | 377.2 | 0.9868 |
| 16 | 200 | hnswlib | 330.3 | 330.3 | 0.9829 |
| 32 | 200 | Quiver | 2423.5 | 2423.5 | 0.9595 |
| 32 | 200 | FAISS | 559.8 | 559.8 | 0.9922 |
| 32 | 200 | hnswlib | 566.4 | 566.4 | 0.9920 |

## efSearch sweep (M=32, efConstruction=200)

Each cell is `Recall@10 / Recall@100`; p50/p99 latency (ms) follows in the
second table. Values are measured over all 10,000 queries.

| Engine | ef=10 | ef=50 | ef=100 | ef=200 | ef=400 |
|---|---|---|---|---|---|
| Quiver | 0.7795 / — | 0.9364 / — | 0.9595 / 0.9334 | 0.9696 / 0.9598 | 0.9735 / 0.9708 |
| FAISS | 0.7813 / — | 0.9722 / — | 0.9922 / 0.9582 | 0.9980 / 0.9900 | 0.9991 / 0.9982 |
| hnswlib | 0.7783 / — | 0.9718 / — | 0.9920 / 0.9576 | 0.9979 / 0.9898 | 0.9990 / 0.9981 |

Recall@100 is not run for ef<100 because `efSearch` must be at least k.

| Engine | ef | p50 / p99 ms (k=10) | p50 / p99 ms (k=100) |
|---|---:|---:|---:|
| Quiver | 100 | 0.9822 / 2.1804 | 0.9722 / 2.1950 |
| Quiver | 200 | 1.7233 / 3.6072 | 1.7575 / 3.7447 |
| Quiver | 400 | 3.1355 / 6.2091 | 3.0768 / 6.0650 |
| FAISS | 100 | 0.3820 / 0.9059 | 0.3888 / 0.9802 |
| FAISS | 200 | 0.6623 / 1.4414 | 0.6853 / 1.4365 |
| FAISS | 400 | 1.2371 / 2.4580 | 1.2691 / 2.3778 |
| hnswlib | 100 | 0.3180 / 0.7362 | 0.3371 / 0.7164 |
| hnswlib | 200 | 0.5809 / 1.2400 | 0.6042 / 1.4282 |
| hnswlib | 400 | 1.1126 / 2.1862 | 1.1102 / 2.6400 |

## RSS / serialized size (index growth)

`index_rss_delta_bytes` is the process RSS increase after loading queries and
ground truth. It is the comparable index-growth metric in these runs.

| M / efC | Quiver RSS MB | FAISS RSS MB | hnswlib RSS MB |
|---|---:|---:|---:|
| 16 / 100 | 1227.6 | 633.8 | 784.7 |
| 16 / 200 | 1245.1 | 633.8 | 785.2 |
| 32 / 200 | 2397.7 | 755.9 | 906.3 |

The M=32 Quiver RSS gap is expected from the current graph representation, not
just vector storage: every node owns a `Vec<Vec<usize>>` (one heap allocation
for the outer layer list and generally one for each populated neighbor list),
and each neighbor is a pointer-width `usize`. FAISS and hnswlib store graph
links in compact contiguous native buffers. Quiver also keeps its durable vector
store/WAL mapping resident. Replacing the nested per-node allocations with a
packed contiguous adjacency arena and 32-bit node IDs is a known memory
optimization opportunity; it should materially reduce allocator overhead and
the ~2.4 GB versus ~0.75-0.9 GB M=32 RSS gap.

## Recall-gap investigation

Quiver does **not** implement the original HNSW heuristic/diversified
neighbor-selection rule. In `hnsw.rs`, insertion selects the first `M` nearest
construction candidates, and `prune_connections` sorts by distance to the node
and truncates to capacity. FAISS and hnswlib use diversified pruning. This is a
plausible primary cause of the matched M=32/efC=200 recall gap (Quiver 0.9595
versus FAISS 0.9922 and hnswlib 0.9920 at efSearch=100); the gap is therefore
explained as an algorithmic difference, not an unexplained benchmark anomaly.

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

See `config.json`, `export_sift.py`, and `competitor_benchmark.py` for commands.
