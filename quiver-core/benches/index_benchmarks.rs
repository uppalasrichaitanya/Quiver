//! Criterion benchmarks for index operations.
//!
//! Measures HNSW and brute-force insert/search throughput and recall characteristics.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use std::collections::HashSet;

use quiver_core::distance::Metric;
use quiver_core::index::brute_force::BruteForceIndex;
use quiver_core::index::hnsw::{HnswConfig, HnswIndex};

/// Generate a dataset of random vectors.
fn generate_dataset(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect()
}

fn bench_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");
    group.sample_size(10); // Fewer samples since each iteration builds a full index

    let dim = 128;
    let n = 5000;
    let vectors = generate_dataset(n, dim);

    group.bench_function(BenchmarkId::new("M=16", n), |bencher| {
        bencher.iter(|| {
            let dir = tempfile::TempDir::new().unwrap();
            let data_path = dir.path().join("bench_hnsw.qvdb");
            let wal_path = dir.path().join("bench_hnsw.wal");
            let config = HnswConfig::new(16).with_ef_construction(100);
            let mut index =
                HnswIndex::create(&data_path, &wal_path, dim as u32, Metric::L2, config).unwrap();
            for v in &vectors {
                index.insert(black_box(v)).unwrap();
            }
        });
    });

    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    let dim = 128;
    let n = 10_000;
    let k = 10;
    let vectors = generate_dataset(n, dim);
    let mut rng = rand::rng();

    // Build the index once
    let dir = tempfile::TempDir::new().unwrap();
    let data_path = dir.path().join("bench_hnsw_search.qvdb");
    let wal_path = dir.path().join("bench_hnsw_search.wal");
    let config = HnswConfig::new(16).with_ef_construction(200);
    let mut index =
        HnswIndex::create(&data_path, &wal_path, dim as u32, Metric::L2, config).unwrap();
    for v in &vectors {
        index.insert(v).unwrap();
    }

    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();

    for ef_search in [10, 50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("ef_search", ef_search),
            &ef_search,
            |bencher, &ef| {
                let mut qi = 0;
                bencher.iter(|| {
                    let q = &queries[qi % queries.len()];
                    qi += 1;
                    index.search(black_box(q), k, ef).unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_brute_force_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("brute_force_search");

    let dim = 128;
    let k = 10;
    let mut rng = rand::rng();

    for n in [1000, 5000, 10_000] {
        let vectors = generate_dataset(n, dim);
        let dir = tempfile::TempDir::new().unwrap();
        let data_path = dir.path().join("bench_bf.qvdb");
        let wal_path = dir.path().join("bench_bf.wal");
        let mut index =
            BruteForceIndex::create(&data_path, &wal_path, dim as u32, Metric::L2).unwrap();
        for v in &vectors {
            index.insert(v).unwrap();
        }

        let queries: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();

        group.bench_with_input(BenchmarkId::new("n", n), &n, |bencher, _| {
            let mut qi = 0;
            bencher.iter(|| {
                let q = &queries[qi % queries.len()];
                qi += 1;
                index.search(black_box(q), k).unwrap()
            });
        });
    }

    group.finish();
}

/// Measure recall@10 at various ef_search settings (not a benchmark per se,
/// but useful to see the recall/speed tradeoff in the same harness).
fn bench_recall_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_sweep");
    group.sample_size(10);

    let dim = 128;
    let n = 5000;
    let k = 10;
    let num_queries = 50;

    let vectors = generate_dataset(n, dim);

    // Build HNSW
    let dir = tempfile::TempDir::new().unwrap();
    let data_path = dir.path().join("bench_recall.qvdb");
    let wal_path = dir.path().join("bench_recall.wal");
    let config = HnswConfig::new(16).with_ef_construction(200);
    let mut index =
        HnswIndex::create(&data_path, &wal_path, dim as u32, Metric::L2, config).unwrap();
    for v in &vectors {
        index.insert(v).unwrap();
    }

    let mut rng = rand::rng();
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect())
        .collect();

    for ef_search in [10, 20, 50, 100, 200, 400] {
        group.bench_with_input(
            BenchmarkId::new("ef_search", ef_search),
            &ef_search,
            |bencher, &ef| {
                bencher.iter(|| {
                    let mut total_recall = 0.0;
                    for query in &queries {
                        // Ground truth via brute force
                        let mut dists: Vec<(usize, f32)> = vectors
                            .iter()
                            .enumerate()
                            .map(|(i, v)| (i, quiver_core::distance::l2_squared(query, v)))
                            .collect();
                        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        let gt: HashSet<usize> = dists.iter().take(k).map(|(i, _)| *i).collect();

                        // HNSW search
                        let results = index.search(query, k, ef).unwrap();
                        let found: HashSet<usize> = results.iter().map(|r| r.slot).collect();
                        total_recall += gt.intersection(&found).count() as f64 / k as f64;
                    }
                    black_box(total_recall / num_queries as f64)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_insert,
    bench_hnsw_search,
    bench_brute_force_search,
    bench_recall_sweep
);
criterion_main!(benches);
