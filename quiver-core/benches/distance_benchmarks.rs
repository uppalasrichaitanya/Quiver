//! Criterion benchmarks for distance metric throughput.
//!
//! Measures scalar and SIMD implementations of L2, dot product, and cosine similarity
//! at various vector dimensions, plus explicit scalar-vs-SIMD comparison.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;

fn generate_random_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rand::rng();
    let a: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    (a, b)
}

fn bench_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");
    for dim in [128, 256, 512, 768, 1024, 1536] {
        let (a, b) = generate_random_vectors(dim);
        group.bench_with_input(BenchmarkId::new("dispatch", dim), &dim, |bencher, _| {
            bencher.iter(|| quiver_core::distance::l2_squared(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                quiver_core::distance::l2_squared_scalar(black_box(&a), black_box(&b))
            });
        });
    }
    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    for dim in [128, 256, 512, 768, 1024, 1536] {
        let (a, b) = generate_random_vectors(dim);
        group.bench_with_input(BenchmarkId::new("dispatch", dim), &dim, |bencher, _| {
            bencher.iter(|| quiver_core::distance::dot_product(black_box(&a), black_box(&b)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                quiver_core::distance::dot_product_scalar(black_box(&a), black_box(&b))
            });
        });
    }
    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    for dim in [128, 256, 512, 768, 1024, 1536] {
        let (a, b) = generate_random_vectors(dim);
        group.bench_with_input(BenchmarkId::new("dispatch", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                quiver_core::distance::cosine_similarity(black_box(&a), black_box(&b))
            });
        });
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                quiver_core::distance::cosine_similarity_scalar(black_box(&a), black_box(&b))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_l2, bench_dot_product, bench_cosine);
criterion_main!(benches);
