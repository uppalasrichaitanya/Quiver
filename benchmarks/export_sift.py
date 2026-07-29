#!/usr/bin/env python3
"""Export ANN-Benchmarks SIFT1M HDF5 arrays to fvecs/ivecs for Rust."""

import argparse
import hashlib
import struct
from pathlib import Path

import h5py
import numpy as np


EXPECTED_SHA256 = "DD6F0A6ED6B7EBB8934680F861A33ED01FF33991EAEE4FD60914D854A0CA5984"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def export_rows(dataset: h5py.Dataset, destination: Path, dtype: np.dtype) -> None:
    dimension = dataset.shape[1]
    prefix = struct.pack("<i", dimension)
    with destination.open("wb") as output:
        for start in range(0, dataset.shape[0], 8192):
            rows = np.asarray(dataset[start : start + 8192], dtype=dtype)
            for row in rows:
                output.write(prefix)
                output.write(row.astype(dtype, copy=False).tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    actual_hash = sha256(args.dataset)
    if actual_hash != EXPECTED_SHA256:
        raise SystemExit(f"dataset hash mismatch: {actual_hash}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as source:
        export_rows(source["train"], args.output_dir / "sift_base.fvecs", np.dtype("<f4"))
        export_rows(source["test"], args.output_dir / "sift_query.fvecs", np.dtype("<f4"))
        export_rows(source["neighbors"], args.output_dir / "sift_groundtruth.ivecs", np.dtype("<i4"))


if __name__ == "__main__":
    main()
