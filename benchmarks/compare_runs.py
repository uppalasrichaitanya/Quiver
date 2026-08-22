#!/usr/bin/env python3
"""Compare two Quiver SIFT1M benchmark runs (raw JSON directories).

Usage:
    python compare_runs.py <baseline_dir> <new_dir>

Prints build-time, RSS, and per-(k, ef_search) recall/QPS/latency deltas for
each matching quiver-m*-efc*.json pair, so reruns after an optimization can be
checked against the committed baseline without hand-built tables.
"""

import argparse
import json
from pathlib import Path


def load_runs(directory: Path) -> dict[str, dict]:
    runs = {}
    for path in sorted(directory.glob("quiver-m*-efc*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        key = f"m{data['m']}-efc{data['ef_construction']}"
        runs[key] = data
    return runs


def fmt_delta(new: float, old: float, better_when_lower: bool) -> str:
    if old == 0:
        return "n/a"
    ratio = new / old
    improved = ratio < 1.0 if better_when_lower else ratio > 1.0
    arrow = "better" if improved else "worse"
    return f"{ratio:.2f}x ({arrow})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    args = parser.parse_args()

    baseline = load_runs(args.baseline_dir)
    new = load_runs(args.new_dir)

    missing = sorted(set(baseline) - set(new))
    if missing:
        print(f"configs present in baseline but not in new run: {', '.join(missing)}")

    for key in sorted(set(baseline) & set(new)):
        old = baseline[key]
        cur = new[key]
        print(f"\n=== {key} ===")
        print(
            f"build: {old['build_seconds']:.1f}s -> {cur['build_seconds']:.1f}s "
            f"[{fmt_delta(cur['build_seconds'], old['build_seconds'], True)}]"
        )
        rss_key = (
            "index_rss_delta_bytes"
            if "index_rss_delta_bytes" in old and "index_rss_delta_bytes" in cur
            else "rss_after_build_bytes"
        )
        rss_label = "rss delta" if rss_key == "index_rss_delta_bytes" else "rss after build"
        old_rss = old.get(rss_key, 0) / 1e6
        cur_rss = cur.get(rss_key, 0) / 1e6
        print(
            f"{rss_label}: {old_rss:.1f}MB -> {cur_rss:.1f}MB "
            f"[{fmt_delta(cur_rss, old_rss, True)}]"
        )

        old_search = {(s["k"], s["ef_search"]): s for s in old["search"]}
        for row in cur["search"]:
            lookup = (row["k"], row["ef_search"])
            prev = old_search.get(lookup)
            if prev is None:
                continue
            print(
                f"k={row['k']:>3} ef={row['ef_search']:>3}: "
                f"recall {prev['recall']:.4f} -> {row['recall']:.4f} | "
                f"qps {prev['qps']:.0f} -> {row['qps']:.0f} "
                f"[{fmt_delta(row['qps'], prev['qps'], False)}] | "
                f"p50 {prev['p50_latency_ms']:.3f} -> {row['p50_latency_ms']:.3f} ms "
                f"[{fmt_delta(row['p50_latency_ms'], prev['p50_latency_ms'], True)}]"
            )


if __name__ == "__main__":
    main()
