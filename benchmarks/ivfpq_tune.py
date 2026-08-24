#!/usr/bin/env python3
"""Config tuner for the IVF-PQ SIFT1M benchmark.

Loads a SIFT subset, computes brute-force ground truth *on that subset* (the
shipped groundtruth file is for the full 1M and would be wrong here), builds the
numpy reference IvfPq, and reports recall@10 vs nprobe for plain ADC and for
exact rerank. Use it to pick nlist/m/ksub/iters before the slow Rust full run.

Usage:
  python benchmarks/ivfpq_tune.py --base sift1m/sift_base.fvecs \
      --queries sift1m/sift_query.fvecs --n 100000 --nq 1000 \
      --nlist 256 --m 32 --ksub 256 --iters 12
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pq_reference import IvfPq, recall_at_k


def read_fvecs(path, limit):
    rows = []
    with open(path, "rb") as f:
        while len(rows) < limit:
            head = f.read(4)
            if len(head) < 4:
                break
            (dim,) = struct.unpack("<i", head)
            data = f.read(dim * 4)
            rows.append(np.frombuffer(data, dtype=np.float32).copy())
    return np.stack(rows).astype(np.float32)


def brute_knn(data, queries, k):
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for qi, q in enumerate(queries):
        d2 = ((data - q) ** 2).sum(axis=1)
        out[qi] = d2.argpartition(k)[:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--ksub", type=int, default=256)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    base = read_fvecs(args.base, args.n)
    queries = read_fvecs(args.queries, args.nq)
    print(f"base {base.shape} queries {queries.shape}")

    t0 = time.time()
    truth = brute_knn(base, queries, args.k)
    print(f"subset ground truth in {time.time()-t0:.1f}s")

    t0 = time.time()
    index = IvfPq(base, args.nlist, args.m, args.ksub, args.iters, args.seed)
    print(f"build in {time.time()-t0:.1f}s")

    nlist = args.nlist
    print(f"{'nprobe':>7} {'plain':>8} {'rerank4':>9} {'rerank16':>9}")
    for nprobe in (1, 4, 8, 16, 32, 64, nlist):
        nprobe = min(nprobe, nlist)
        plain = recall_at_k(index.search(queries, args.k, nprobe), truth, args.k)
        r4 = recall_at_k(index.search(queries, args.k, nprobe, 4), truth, args.k)
        r16 = recall_at_k(index.search(queries, args.k, nprobe, 16), truth, args.k)
        print(f"{nprobe:>7} {plain:>8.4f} {r4:>9.4f} {r16:>9.4f}")


if __name__ == "__main__":
    main()
