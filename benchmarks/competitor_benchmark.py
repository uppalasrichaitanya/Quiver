#!/usr/bin/env python3
"""Run the same single-query SIFT1M sweep against FAISS or hnswlib."""

import argparse
import importlib.metadata
import json
import os
import platform
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import psutil


class PeakRssSampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRssSampler":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def build_faiss(train: h5py.Dataset, vector_count: int, m: int, ef_construction: int):
    import faiss

    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(train.shape[1], m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    for start in range(0, vector_count, 10_000):
        index.add(np.asarray(train[start : start + 10_000], dtype=np.float32))
        if (start + 10_000) % 100_000 == 0:
            print(f"added {min(start + 10_000, vector_count)} vectors", flush=True)
    return index


def build_hnswlib(train: h5py.Dataset, vector_count: int, m: int, ef_construction: int):
    import hnswlib

    index = hnswlib.Index(space="l2", dim=train.shape[1])
    index.init_index(
        max_elements=vector_count,
        ef_construction=ef_construction,
        M=m,
        random_seed=42,
    )
    index.set_num_threads(1)
    for start in range(0, vector_count, 10_000):
        rows = np.asarray(train[start : start + 10_000], dtype=np.float32)
        labels = np.arange(start, start + rows.shape[0], dtype=np.int64)
        index.add_items(rows, labels, num_threads=1)
        if (start + 10_000) % 100_000 == 0:
            print(f"added {min(start + 10_000, vector_count)} vectors", flush=True)
    return index


def search_one(engine: str, index, query: np.ndarray, k: int, ef_search: int) -> np.ndarray:
    if engine == "faiss":
        index.hnsw.efSearch = ef_search
        _, labels = index.search(query.reshape(1, -1), k)
    else:
        index.set_ef(ef_search)
        labels, _ = index.knn_query(query.reshape(1, -1), k=k, num_threads=1)
    return labels[0]


def percentile_ms(latencies_ns: list[int], percentile: float) -> float:
    values = sorted(latencies_ns)
    position = max(0, min(len(values) - 1, int(np.ceil(len(values) * percentile)) - 1))
    return values[position] / 1_000_000.0


def run_search(engine: str, index, queries: np.ndarray, truth: np.ndarray, k: int, ef: int) -> dict:
    for query in queries[:100]:
        search_one(engine, index, query, k, ef)

    latencies = []
    recall_sum = 0.0
    total_started = time.perf_counter()
    for query, expected in zip(queries, truth, strict=True):
        started = time.perf_counter_ns()
        found = search_one(engine, index, query, k, ef)
        latencies.append(time.perf_counter_ns() - started)
        recall_sum += len(set(map(int, found)).intersection(map(int, expected[:k]))) / k
    total_seconds = time.perf_counter() - total_started
    return {
        "k": k,
        "ef_search": ef,
        "recall": recall_sum / len(queries),
        "qps": len(queries) / total_seconds,
        "p50_latency_ms": percentile_ms(latencies, 0.50),
        "p99_latency_ms": percentile_ms(latencies, 0.99),
        "total_seconds": total_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("faiss", "hnswlib"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--ef-construction", type=int, required=True)
    parser.add_argument("--base-limit", type=int, default=1_000_000)
    parser.add_argument("--query-limit", type=int, default=10_000)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    process = psutil.Process()
    with h5py.File(args.dataset, "r") as source:
        train = source["train"]
        vector_count = min(args.base_limit, train.shape[0])
        queries = np.asarray(source["test"][: args.query_limit], dtype=np.float32)
        truth = np.asarray(source["neighbors"][: args.query_limit], dtype=np.int32)
        baseline_rss = process.memory_info().rss

        print(
            f"building {args.engine}: M={args.m} ef_construction={args.ef_construction} "
            f"vectors={len(train)}",
            flush=True,
        )
        with PeakRssSampler() as sampler:
            started = time.perf_counter()
            if args.engine == "faiss":
                index = build_faiss(train, vector_count, args.m, args.ef_construction)
            else:
                index = build_hnswlib(train, vector_count, args.m, args.ef_construction)
            build_seconds = time.perf_counter() - started

        rss_after_build = process.memory_info().rss
        serialized_path = args.work_dir / f"{args.engine}.index"
        if args.engine == "faiss":
            import faiss

            faiss.write_index(index, str(serialized_path))
        else:
            index.save_index(str(serialized_path))

        search = []
        for k, values in ((10, (10, 50, 100, 200, 400)), (100, (100, 200, 400))):
            for ef_search in values:
                search.append(run_search(args.engine, index, queries, truth, k, ef_search))

    result = {
        "engine": args.engine,
        "engine_version": importlib.metadata.version(
            "faiss" if args.engine == "faiss" else "hnswlib"
        ),
        "dataset": "SIFT1M",
        "dimension": int(queries.shape[1]),
        "base_vectors": int(vector_count),
        "queries": int(len(queries)),
        "thread_count": 1,
        "random_seed": 42,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "build_seconds": build_seconds,
        "baseline_rss_bytes": baseline_rss,
        "rss_after_build_bytes": rss_after_build,
        "index_rss_delta_bytes": max(0, rss_after_build - baseline_rss),
        "peak_rss_bytes": sampler.peak,
        "serialized_index_bytes": serialized_path.stat().st_size,
        "build_persistence": "in-memory build; serialization time excluded",
        "platform": platform.platform(),
        "search": search,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
