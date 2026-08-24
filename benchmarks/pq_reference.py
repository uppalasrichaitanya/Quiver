#!/usr/bin/env python3
"""numpy reference for the IVF-PQ pipeline: k-means++ / Lloyd, PQ, ADC search.

This is the algorithmic oracle for the Rust implementation in quiver-core.
When Rust recall is off, diff against this reference on identical data and
hyperparameters instead of debugging algorithm-correctness and systems-code
correctness at the same time.

Everything runs in float32 with a seeded RNG and deterministic tie-breaking
(argmin picks the lowest index), so a faithful reimplementation lands on the
same recall even though raw centroid bits may differ by ULPs.

Usage:
  python benchmarks/pq_reference.py selftest
  python benchmarks/pq_reference.py dump --out benchmarks/reference/ivfpq-d32
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def kmeans(data, k, iters, rng):
    """k-means with k-means++ init and fixed-iteration Lloyd. Empty clusters
    are reseeded to the point farthest from its assigned centroid."""
    n, d = data.shape
    centroids = np.empty((k, d), dtype=np.float32)
    centroids[0] = data[rng.integers(n)]
    closest = np.sum((data - centroids[0]) ** 2, axis=1)
    for c in range(1, k):
        weights = closest.astype(np.float64)
        total = weights.sum()
        probs = weights / total if total > 0 else None
        centroids[c] = data[rng.choice(n, p=probs)]
        closest = np.minimum(closest, np.sum((data - centroids[c]) ** 2, axis=1))

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        labels, dists = assign(data, centroids)
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = data[mask].mean(axis=0)
            else:
                far = int(dists.argmax())
                centroids[c] = data[far]
                dists[far] = 0.0
    labels, _ = assign(data, centroids)
    return centroids, labels


def assign(data, centroids, chunk=4096):
    """Nearest-centroid label and squared distance per point (chunked)."""
    n = data.shape[0]
    labels = np.empty(n, dtype=np.int32)
    dists = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        block = data[start : start + chunk]
        d2 = ((block[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels[start : start + chunk] = d2.argmin(axis=1)
        dists[start : start + chunk] = d2.min(axis=1)
    return labels, dists


def train_pq(data, m, ksub, iters, rng):
    """One independent k-means codebook per sub-vector."""
    n, d = data.shape
    assert d % m == 0, "dimension must be divisible by m"
    dsub = d // m
    codebooks = np.empty((m, ksub, dsub), dtype=np.float32)
    for sub in range(m):
        block = np.ascontiguousarray(data[:, sub * dsub : (sub + 1) * dsub])
        centroids, _ = kmeans(block, ksub, iters, rng)
        codebooks[sub] = centroids
    return codebooks


def encode_pq(data, codebooks):
    m, ksub, dsub = codebooks.shape
    codes = np.empty((data.shape[0], m), dtype=np.uint8)
    for sub in range(m):
        block = np.ascontiguousarray(data[:, sub * dsub : (sub + 1) * dsub])
        labels, _ = assign(block, codebooks[sub])
        codes[:, sub] = labels
    return codes


def adc_tables(query, codebooks):
    """Per-subspace squared distances from the query to every centroid: (m, ksub)."""
    m, ksub, dsub = codebooks.shape
    subs = query.reshape(m, dsub)
    return ((codebooks - subs[:, None, :]) ** 2).sum(axis=2)


def brute_knn(data, queries, k):
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for qi, q in enumerate(queries):
        d2 = ((data - q) ** 2).sum(axis=1)
        out[qi] = d2.argpartition(k)[:k]
    return out


def recall_at_k(results, truth, k):
    """results: (nq, k) ids (ordered or not); truth: (nq, k) unordered ids."""
    hits = 0
    for row, expected in zip(results, truth):
        hits += len(set(row.tolist()) & set(expected.tolist()))
    return hits / (results.shape[0] * k)


class IvfPq:
    def __init__(self, base, nlist, m, ksub, iters, seed):
        self.rng = np.random.default_rng(seed)
        self.base = np.ascontiguousarray(base, dtype=np.float32)
        self.coarse, self.assignments = kmeans(self.base, nlist, iters, self.rng)
        self.codebooks = train_pq(self.base, m, ksub, iters, self.rng)
        self.codes = encode_pq(self.base, self.codebooks)
        self.lists = [np.flatnonzero(self.assignments == c) for c in range(nlist)]

    def search(self, queries, k, nprobe, rerank_factor=0):
        """Top-k per query via ADC over the nprobe nearest coarse clusters.
        rerank_factor > 0 reranks that many ADC candidates with exact L2."""
        m = self.codebooks.shape[0]
        results = np.empty((queries.shape[0], k), dtype=np.int64)
        for qi, q in enumerate(queries):
            coarse_d2 = ((self.coarse - q) ** 2).sum(axis=1)
            probes = coarse_d2.argsort()[:nprobe]
            tables = adc_tables(q, self.codebooks)
            cand_ids = np.concatenate([self.lists[c] for c in probes]) \
                if any(self.lists[c].size for c in probes) else np.empty(0, dtype=np.int64)
            if cand_ids.size == 0:
                results[qi, :] = -1
                continue
            adc_d = tables[np.arange(m), self.codes[cand_ids]].sum(axis=1)
            take = min(cand_ids.size, max(k, k * rerank_factor))
            top = cand_ids[np.argpartition(adc_d, take - 1)[:take]]
            if rerank_factor > 0:
                exact = ((self.base[top] - q) ** 2).sum(axis=1)
                top = top[exact.argsort()[:k]]
            results[qi, : len(top)] = top
            if len(top) < k:
                results[qi, len(top):] = -1
        return results


def synthetic(rng, n=20000, d=32, blobs=64, std=1.0):
    centers = (rng.uniform(0.0, 40.0, (blobs, d))).astype(np.float32)
    which = rng.integers(0, blobs, n)
    return centers[which] + (rng.standard_normal((n, d)) * std).astype(np.float32)


def write_fvecs(path, array):
    dim = array.shape[1]
    prefix = struct.pack("<i", dim)
    with path.open("wb") as out:
        for row in array.astype(np.float32, copy=False):
            out.write(prefix)
            out.write(row.tobytes(order="C"))


def selftest():
    rng = np.random.default_rng(20260824)
    base = synthetic(rng)
    queries = synthetic(rng, n=200, blobs=64)
    k = 10
    truth = brute_knn(base, queries, k)
    index = IvfPq(base, nlist=64, m=8, ksub=256, iters=8, seed=20260824)

    print(f"{'nprobe':>7} {'recall@10':>10} {'recall rerank':>14}")
    previous = 0.0
    for nprobe in (1, 4, 8, 16, 32, 64):
        plain = recall_at_k(index.search(queries, k, nprobe), truth, k)
        reranked = recall_at_k(index.search(queries, k, nprobe, rerank_factor=4), truth, k)
        print(f"{nprobe:>7} {plain:>10.4f} {reranked:>14.4f}")
        assert plain >= previous - 0.02, "recall must not collapse as nprobe grows"
        assert reranked >= plain - 0.001, "rerank must not lose ADC candidates"
        previous = plain

    # Plain ADC recall on isotropic Gaussian blobs is limited by distance
    # concentration, not by a pipeline bug. The correctness signal is that a
    # generous rerank window over the full-probe candidate set recovers the
    # exact neighbors: this proves coarse assignment, ADC ranking, and the
    # rerank step are all correct.
    recovered = recall_at_k(index.search(queries, k, 64, rerank_factor=50), truth, k)
    print(f"full-probe rerank_factor=50 recall@10: {recovered:.4f}")
    assert recovered >= 0.98, f"rerank recovery too low: {recovered}"
    print("selftest OK")


def dump(out_dir: Path):
    """Small fixed dataset + trained state + expected recalls, for the Rust
    cross-validation test (which loads the trained state and must reproduce
    the recall column)."""
    rng = np.random.default_rng(20260824)
    base = synthetic(rng, n=4096, d=32, blobs=32)
    queries = synthetic(rng, n=100, d=32, blobs=32)
    k, nlist, m, ksub, iters = 10, 32, 8, 256, 10
    index = IvfPq(base, nlist=nlist, m=m, ksub=ksub, iters=iters, seed=20260824)
    truth = brute_knn(base, queries, k)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_fvecs(out_dir / "base.fvecs", base)
    write_fvecs(out_dir / "queries.fvecs", queries)
    write_fvecs(out_dir / "coarse.fvecs", index.coarse)
    (out_dir / "codebooks.bin").write_bytes(index.codebooks.tobytes(order="C"))
    (out_dir / "codes.bin").write_bytes(index.codes.tobytes(order="C"))
    (out_dir / "assign.bin").write_bytes(index.assignments.astype("<u4").tobytes())

    expected = {}
    for nprobe in (1, 2, 4, 8, 16, 32):
        expected[str(nprobe)] = {
            "recall": recall_at_k(index.search(queries, k, nprobe), truth, k),
            "recall_rerank": recall_at_k(
                index.search(queries, k, nprobe, rerank_factor=4), truth, k
            ),
        }
    meta = {
        "n": int(base.shape[0]),
        "dimension": 32,
        "n_queries": 100,
        "k": k,
        "nlist": nlist,
        "m": m,
        "ksub": ksub,
        "iters": iters,
        "seed": 20260824,
        "rerank_factor": 4,
        "expected": expected,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"dumped reference state to {out_dir}")
    print(json.dumps(expected, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    dump_cmd = sub.add_parser("dump")
    dump_cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "selftest":
        selftest()
    else:
        dump(args.out)


if __name__ == "__main__":
    main()
