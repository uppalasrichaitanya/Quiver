//! k-means clustering with k-means++ initialization and Lloyd iterations.
//!
//! This is the shared training primitive for both the IVF coarse quantizer and
//! the per-subspace product-quantizer codebooks. It operates on a row-major
//! flat buffer of `n` points with `d` dimensions each.
//!
//! The implementation mirrors the numpy reference in `benchmarks/pq_reference.py`:
//! k-means++ seeding with D^2-weighted sampling, a fixed number of Lloyd
//! iterations, lowest-index tie-breaking on assignment, and empty-cluster
//! reseeding to the point farthest from its assigned centroid.

use crate::distance::l2_squared;
use crate::error::{QuiverError, Result};
use rand::Rng;

/// Cluster row-major `data` (`n` points of `d` dimensions) into `k` clusters.
///
/// Returns `(centroids, labels)` where `centroids` is a row-major `k * d`
/// buffer and `labels[i]` is the index of the nearest centroid for point `i`.
/// Runs k-means++ initialization followed by `iters` Lloyd iterations.
pub fn kmeans<R: Rng>(
    data: &[f32],
    n: usize,
    d: usize,
    k: usize,
    iters: usize,
    rng: &mut R,
) -> Result<(Vec<f32>, Vec<u32>)> {
    if n == 0 || d == 0 {
        return Err(QuiverError::EmptyIndex);
    }
    if let Some(expected) = n.checked_mul(d)
        && expected != data.len()
    {
        return Err(QuiverError::DimensionMismatch {
            expected: expected.min(u32::MAX as usize) as u32,
            actual: data.len().min(u32::MAX as usize) as u32,
        });
    }
    if k == 0 {
        return Err(QuiverError::InvalidFormat(
            "k-means requires at least one cluster".to_owned(),
        ));
    }
    if !data.iter().all(|value| value.is_finite()) {
        return Err(QuiverError::InvalidFormat(
            "k-means data must contain only finite values".to_owned(),
        ));
    }

    // More clusters than points is degenerate; clamp so every centroid is real.
    let k = k.min(n);

    let mut centroids = vec![0.0f32; k * d];
    centroids[..d].copy_from_slice(&data[rng.random_range(0..n) * d..][..d]);

    // closest[i] = squared distance from point i to its nearest chosen centroid.
    let mut closest: Vec<f32> = (0..n)
        .map(|i| l2_squared(&data[i * d..(i + 1) * d], &centroids[..d]))
        .collect();

    for c in 1..k {
        let chosen = sample_d2(&closest, n, rng);
        centroids[c * d..(c + 1) * d].copy_from_slice(&data[chosen * d..(chosen + 1) * d]);
        for (i, nearest) in closest.iter_mut().enumerate().take(n) {
            let dist = l2_squared(&data[i * d..(i + 1) * d], &centroids[c * d..(c + 1) * d]);
            if dist < *nearest {
                *nearest = dist;
            }
        }
    }

    let mut labels = vec![0u32; n];
    let mut dists = vec![0.0f32; n];
    for _ in 0..iters {
        assign(data, n, d, &centroids, k, &mut labels, &mut dists);

        let mut sums = vec![0.0f64; k * d];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i] as usize;
            counts[c] += 1;
            for j in 0..d {
                sums[c * d + j] += data[i * d + j] as f64;
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    centroids[c * d + j] = (sums[c * d + j] / counts[c] as f64) as f32;
                }
            } else {
                // Reseed an empty cluster with the point farthest from its own
                // centroid, then zero that point's distance so a second empty
                // cluster does not pick it again.
                let far = dists
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                centroids[c * d..(c + 1) * d].copy_from_slice(&data[far * d..(far + 1) * d]);
                dists[far] = 0.0;
            }
        }
    }

    assign(data, n, d, &centroids, k, &mut labels, &mut dists);
    Ok((centroids, labels))
}

/// D^2-weighted sampling: pick an index with probability proportional to
/// `closest[i]`. Falls back to uniform sampling when all distances are zero,
/// and to the farthest point when squared distances overflow to infinity
/// (finite-but-huge inputs) where a proportional draw is ill-defined.
fn sample_d2<R: Rng>(closest: &[f32], n: usize, rng: &mut R) -> usize {
    let total: f64 = closest.iter().take(n).map(|&value| value as f64).sum();
    if total <= 0.0 {
        return rng.random_range(0..n);
    }
    if !total.is_finite() {
        // At least one distance overflowed to +inf; it carries all the
        // sampling mass. Pick the first such index (lowest-index tie-break,
        // matching the rest of the implementation).
        return closest
            .iter()
            .position(|&value| value.is_infinite())
            .unwrap_or_else(|| {
                closest
                    .iter()
                    .take(n)
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });
    }
    let threshold = rng.random_range(0.0..total);
    let mut acc = 0.0f64;
    for (i, &dist) in closest.iter().enumerate().take(n) {
        acc += dist as f64;
        if acc >= threshold {
            return i;
        }
    }
    n - 1
}

/// Nearest-centroid assignment with lowest-index tie-breaking.
fn assign(
    data: &[f32],
    n: usize,
    d: usize,
    centroids: &[f32],
    k: usize,
    labels: &mut [u32],
    dists: &mut [f32],
) {
    for i in 0..n {
        let point = &data[i * d..(i + 1) * d];
        let mut best = 0usize;
        let mut best_dist = f32::INFINITY;
        for c in 0..k {
            let dist = l2_squared(point, &centroids[c * d..(c + 1) * d]);
            if dist < best_dist {
                best_dist = dist;
                best = c;
            }
        }
        labels[i] = best as u32;
        dists[i] = best_dist;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn blob_points(rng: &mut rand::rngs::StdRng, centers: &[(f32, f32)], per: usize) -> Vec<f32> {
        let mut data = Vec::new();
        for &(cx, cy) in centers {
            for _ in 0..per {
                let x = cx + rng.random_range(-0.1..0.1);
                let y = cy + rng.random_range(-0.1..0.1);
                data.push(x);
                data.push(y);
            }
        }
        data
    }

    #[test]
    fn separates_well_spaced_blobs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let centers = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)];
        let data = blob_points(&mut rng, &centers, 40);
        let (centroids, labels) = kmeans(&data, 120, 2, 3, 10, &mut rng).unwrap();

        // Each blob must map to exactly one cluster, and all three clusters used.
        let mut cluster_by_blob = [None::<u32>; 3];
        for (blob, slot) in labels.chunks(40).enumerate() {
            let unique: HashSet<u32> = slot.iter().copied().collect();
            assert_eq!(unique.len(), 1, "blob {} split across clusters", blob);
            cluster_by_blob[blob] = Some(*unique.iter().next().unwrap());
        }
        let used: HashSet<u32> = cluster_by_blob.iter().map(|c| c.unwrap()).collect();
        assert_eq!(used.len(), 3, "expected three distinct clusters");

        // Centroids land near the true blob centers.
        for c in 0..3 {
            let centroid = &centroids[c * 2..(c + 1) * 2];
            let near = centers
                .iter()
                .any(|&(cx, cy)| (centroid[0] - cx).abs() < 0.5 && (centroid[1] - cy).abs() < 0.5);
            assert!(near, "centroid {:?} not near any true center", centroid);
        }
    }

    #[test]
    fn labels_match_nearest_centroid() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let data: Vec<f32> = (0..200)
            .flat_map(|_| [rng.random_range(-5.0..5.0); 1])
            .collect();
        let (centroids, labels) = kmeans(&data, 200, 1, 4, 8, &mut rng).unwrap();
        for i in 0..200 {
            let point = &data[i..i + 1];
            let mut best = f32::INFINITY;
            let mut best_c = 0;
            for c in 0..4 {
                let dist = l2_squared(point, &centroids[c..c + 1]);
                if dist < best {
                    best = dist;
                    best_c = c;
                }
            }
            assert_eq!(labels[i] as usize, best_c, "point {} mislabeled", i);
        }
    }

    #[test]
    fn clamps_k_to_point_count() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let data = vec![0.0f32, 0.0, 1.0, 1.0];
        let (centroids, labels) = kmeans(&data, 2, 2, 10, 4, &mut rng).unwrap();
        assert_eq!(centroids.len(), 2 * 2, "k clamped to n=2");
        assert!(labels.iter().all(|&l| l < 2));
    }

    #[test]
    fn single_cluster_returns_mean() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let data = vec![0.0f32, 2.0, 4.0];
        let (centroids, labels) = kmeans(&data, 3, 1, 1, 5, &mut rng).unwrap();
        assert!((centroids[0] - 2.0).abs() < 1e-5);
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        assert!(kmeans(&[], 0, 2, 2, 4, &mut rng).is_err());
        assert!(kmeans(&[0.0, 1.0], 1, 2, 0, 4, &mut rng).is_err());
        assert!(kmeans(&[0.0, f32::NAN], 1, 2, 1, 4, &mut rng).is_err());
        // data length inconsistent with n*d
        assert!(kmeans(&[0.0, 1.0, 2.0], 2, 2, 1, 4, &mut rng).is_err());
    }

    #[test]
    fn huge_finite_values_do_not_panic_d2_sampling() {
        // Components near f32 max make squared distances overflow to +inf;
        // D^2 sampling must not feed a non-finite range to the RNG.
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        let data = [1e30f32, -1e30, 0.0, 0.0];
        let (centroids, labels) = kmeans(&data, 2, 2, 2, 3, &mut rng).unwrap();
        assert_eq!(centroids.len(), 4);
        assert!(labels.iter().all(|&l| l < 2));
    }
}
