//! Cross-validation of the Rust IVF-PQ implementation against the numpy
//! reference in `benchmarks/pq_reference.py`.
//!
//! The reference dumps a fixed trained state (base vectors, queries, coarse
//! centroids, PQ codebooks, PQ codes, coarse assignments) plus the recall it
//! achieves at several `nprobe` values. This test loads that state into
//! `IvfPqIndex::from_trained`, recomputes exact ground truth locally, and
//! asserts the Rust index reproduces the reference's recall. A mismatch here
//! means the Rust ADC / probing / rerank logic diverges from the reference.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use quiver_core::distance::l2_squared;
use quiver_core::index::ivfpq::IvfPqIndex;
use quiver_core::quantization::ProductQuantizer;

#[derive(serde::Deserialize)]
struct Meta {
    dimension: usize,
    k: usize,
    nlist: usize,
    m: usize,
    ksub: usize,
    rerank_factor: usize,
    expected: std::collections::HashMap<String, ExpectedRecall>,
}

#[derive(serde::Deserialize)]
struct ExpectedRecall {
    recall: f32,
    recall_rerank: f32,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/reference/ivfpq-d32")
}

fn read_fvecs(path: &Path) -> Vec<Vec<f32>> {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    let mut out = Vec::new();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let dim = i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()));
            i += 4;
        }
        out.push(v);
    }
    out
}

fn read_f32_le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "file not a multiple of 4 bytes"
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_u32_le(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(
        bytes.len().is_multiple_of(4),
        "file not a multiple of 4 bytes"
    );
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn brute_truth(base: &[Vec<f32>], query: &[f32], k: usize) -> HashSet<usize> {
    let mut scored: Vec<(usize, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_squared(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.iter().take(k).map(|(i, _)| *i).collect()
}

fn recall(results: &[quiver_core::index::SearchResult], truth: &HashSet<usize>, k: usize) -> f32 {
    let found: HashSet<usize> = results.iter().map(|r| r.slot).collect();
    found.intersection(truth).count() as f32 / k as f32
}

#[test]
fn rust_ivfpq_matches_numpy_reference_recall() {
    let dir = fixture_dir();
    let meta: Meta =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();

    let base = read_fvecs(&dir.join("base.fvecs"));
    let queries = read_fvecs(&dir.join("queries.fvecs"));
    // coarse is stored as fvecs (dim prefix per centroid); flatten to raw f32.
    let coarse_vecs = read_fvecs(&dir.join("coarse.fvecs"));
    let coarse: Vec<f32> = coarse_vecs.iter().flatten().copied().collect();
    let codebooks = read_f32_le(&dir.join("codebooks.bin"));
    let codes = std::fs::read(dir.join("codes.bin")).unwrap();
    let assignments = read_u32_le(&dir.join("assign.bin"));

    assert_eq!(base.len(), assignments.len());
    assert_eq!(codes.len(), base.len() * meta.m);
    assert_eq!(coarse_vecs.len(), meta.nlist);
    assert_eq!(coarse.len(), meta.nlist * meta.dimension);
    assert_eq!(
        codebooks.len(),
        meta.m * meta.ksub * (meta.dimension / meta.m)
    );

    let dsub = meta.dimension / meta.m;
    let pq = ProductQuantizer::from_codebooks(meta.m, meta.ksub, dsub, codebooks).unwrap();
    let index = IvfPqIndex::from_trained(
        meta.dimension,
        meta.nlist,
        pq,
        coarse,
        assignments,
        codes,
        base.clone(),
    )
    .unwrap();

    // Exact ground truth computed locally (matches the reference's brute force).
    let truths: Vec<HashSet<usize>> = queries
        .iter()
        .map(|q| brute_truth(&base, q, meta.k))
        .collect();

    // Recall is quantized in steps of 1/(n_queries*k); allow a few items of slack
    // for float summation-order differences between numpy and the SIMD kernels.
    let tolerance = 0.01;

    let mut nprobes: Vec<(usize, &ExpectedRecall)> = meta
        .expected
        .iter()
        .map(|(k, v)| (k.parse::<usize>().unwrap(), v))
        .collect();
    nprobes.sort_by_key(|(nprobe, _)| *nprobe);

    for (nprobe, expected) in nprobes {
        let mut plain = 0.0f32;
        let mut reranked = 0.0f32;
        for (query, truth) in queries.iter().zip(&truths) {
            plain += recall(
                &index.search(query, meta.k, nprobe, 0).unwrap(),
                truth,
                meta.k,
            );
            reranked += recall(
                &index
                    .search(query, meta.k, nprobe, meta.rerank_factor)
                    .unwrap(),
                truth,
                meta.k,
            );
        }
        let n = queries.len() as f32;
        let (plain, reranked) = (plain / n, reranked / n);

        assert!(
            (plain - expected.recall).abs() <= tolerance,
            "nprobe={nprobe}: plain recall {plain} diverges from reference {} (tol {tolerance})",
            expected.recall
        );
        assert!(
            (reranked - expected.recall_rerank).abs() <= tolerance,
            "nprobe={nprobe}: reranked recall {reranked} diverges from reference {} (tol {tolerance})",
            expected.recall_rerank
        );
    }
}
