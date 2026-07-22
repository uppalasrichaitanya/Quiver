//! Distance metrics for vector similarity computation.
//!
//! Provides scalar and SIMD-optimized implementations of L2 (Euclidean),
//! dot product, and cosine similarity.
//!
//! # SIMD Support
//!
//! On x86_64 CPUs with AVX2+FMA support, the distance functions automatically
//! dispatch to SIMD kernels that process 8 floats per cycle. Runtime feature
//! detection ensures correct fallback to scalar code on older hardware.
//!
//! # Design Note
//!
//! If vectors are normalized to unit length at insert time (common for embedding models),
//! cosine similarity reduces to a dot product — one fast kernel covers two of the three metrics.

/// The distance metric to use for similarity computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum Metric {
    /// L2 (Euclidean) distance. Lower is more similar.
    L2 = 0,
    /// Dot product. Higher (more positive) is more similar.
    DotProduct = 1,
    /// Cosine similarity. Higher is more similar. Equivalent to dot product on unit vectors.
    Cosine = 2,
}

impl Metric {
    /// Convert a raw byte to a Metric variant.
    pub fn from_u8(value: u8) -> Option<Metric> {
        match value {
            0 => Some(Metric::L2),
            1 => Some(Metric::DotProduct),
            2 => Some(Metric::Cosine),
            _ => None,
        }
    }
}

// ── Scalar implementations ──────────────────────────────────────────────────

/// Compute the squared L2 (Euclidean) distance between two vectors (scalar).
///
/// Returns the sum of squared differences. We skip the final sqrt because
/// it's monotonic — ranking is preserved without it, and it saves a costly operation.
#[inline]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Compute the dot product of two vectors (scalar).
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the cosine similarity between two vectors (scalar).
///
/// Returns dot(a, b) / (||a|| * ||b||). If either vector has zero magnitude,
/// returns 0.0.
#[inline]
pub fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let magnitude = (norm_a * norm_b).sqrt();
    if magnitude == 0.0 {
        0.0
    } else {
        dot / magnitude
    }
}

// ── AVX2 SIMD implementations ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2+FMA optimized L2 squared distance.
    ///
    /// Processes 8 floats per iteration using 256-bit SIMD registers.
    /// Handles the tail (dimensions not divisible by 8) with scalar code.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available (use `is_x86_feature_detected!`).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        // SAFETY: caller guarantees AVX2+FMA are available
        let mut sum = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            unsafe {
                let va = _mm256_loadu_ps(a_ptr.add(offset));
                let vb = _mm256_loadu_ps(b_ptr.add(offset));
                let diff = _mm256_sub_ps(va, vb);
                sum = _mm256_fmadd_ps(diff, diff, sum);
            }
        }

        let mut result = unsafe { hsum256_ps(sum) };

        // Scalar tail
        let tail_start = chunks * 8;
        for i in 0..remainder {
            let diff = a[tail_start + i] - b[tail_start + i];
            result += diff * diff;
        }

        result
    }

    /// AVX2+FMA optimized dot product.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut sum = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            unsafe {
                let va = _mm256_loadu_ps(a_ptr.add(offset));
                let vb = _mm256_loadu_ps(b_ptr.add(offset));
                sum = _mm256_fmadd_ps(va, vb, sum);
            }
        }

        let mut result = unsafe { hsum256_ps(sum) };

        // Scalar tail
        let tail_start = chunks * 8;
        for i in 0..remainder {
            result += a[tail_start + i] * b[tail_start + i];
        }

        result
    }

    /// AVX2+FMA optimized cosine similarity.
    ///
    /// Computes dot product and both norms in a single pass over the data,
    /// minimizing cache misses.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA are available.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut dot_acc = _mm256_setzero_ps();
        let mut norm_a_acc = _mm256_setzero_ps();
        let mut norm_b_acc = _mm256_setzero_ps();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            unsafe {
                let va = _mm256_loadu_ps(a_ptr.add(offset));
                let vb = _mm256_loadu_ps(b_ptr.add(offset));
                dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
                norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
                norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
            }
        }

        let mut dot = unsafe { hsum256_ps(dot_acc) };
        let mut norm_a = unsafe { hsum256_ps(norm_a_acc) };
        let mut norm_b = unsafe { hsum256_ps(norm_b_acc) };

        // Scalar tail
        let tail_start = chunks * 8;
        for i in 0..remainder {
            let x = a[tail_start + i];
            let y = b[tail_start + i];
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }

        let magnitude = (norm_a * norm_b).sqrt();
        if magnitude == 0.0 {
            0.0
        } else {
            dot / magnitude
        }
    }

    /// Horizontal sum of all 8 floats in an __m256.
    ///
    /// Uses the standard hadd + permute pattern:
    /// 1. Add high 128 bits to low 128 bits
    /// 2. Two horizontal adds to reduce to a single float
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hsum256_ps(v: __m256) -> f32 {
        // vlow = [a0+a4, a1+a5, a2+a6, a3+a7]
        let vlow = _mm256_castps256_ps128(v);
        let vhigh = _mm256_extractf128_ps(v, 1);
        let vsum = _mm_add_ps(vlow, vhigh);
        // hadd: [a0+a4+a1+a5, a2+a6+a3+a7, ...]
        let vsum = _mm_hadd_ps(vsum, vsum);
        // hadd again: [total, ...]
        let vsum = _mm_hadd_ps(vsum, vsum);
        _mm_cvtss_f32(vsum)
    }
}

// ── Dispatching public API ──────────────────────────────────────────────────

/// Compute the squared L2 (Euclidean) distance between two vectors.
///
/// Automatically dispatches to AVX2+FMA if available, otherwise falls back to scalar.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified AVX2+FMA are available.
            return unsafe { avx2::l2_squared_avx2(a, b) };
        }
    }
    l2_squared_scalar(a, b)
}

/// Compute the dot product of two vectors.
///
/// Automatically dispatches to AVX2+FMA if available, otherwise falls back to scalar.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::dot_product_avx2(a, b) };
        }
    }
    dot_product_scalar(a, b)
}

/// Compute the cosine similarity between two vectors.
///
/// Automatically dispatches to AVX2+FMA if available, otherwise falls back to scalar.
///
/// **Performance note:** If vectors are pre-normalized to unit length,
/// use [`dot_product`] directly instead — it produces the same result without
/// the magnitude computation overhead.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::cosine_similarity_avx2(a, b) };
        }
    }
    cosine_similarity_scalar(a, b)
}

/// Compute distance between two vectors using the specified metric.
///
/// For L2 and dot product, the result is a "distance" where lower values mean
/// more similar vectors. For cosine similarity, the result is negated so that
/// lower values also mean more similar (enabling uniform min-heap usage).
#[inline]
pub fn compute_distance(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::L2 => l2_squared(a, b),
        // Negate so that "lower is better" holds uniformly across all metrics.
        Metric::DotProduct => -dot_product(a, b),
        Metric::Cosine => -cosine_similarity(a, b),
    }
}

/// Check whether AVX2+FMA SIMD acceleration is available on this CPU.
pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < TOLERANCE
    }

    #[test]
    fn test_l2_squared_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(l2_squared(&v, &v), 0.0));
    }

    #[test]
    fn test_l2_squared_known() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        // (1-0)^2 + (0-1)^2 + (0-0)^2 = 2.0
        assert!(approx_eq(l2_squared(&a, &b), 2.0));
    }

    #[test]
    fn test_l2_squared_negative() {
        let a = vec![-1.0, -2.0];
        let b = vec![1.0, 2.0];
        // (-1-1)^2 + (-2-2)^2 = 4 + 16 = 20
        assert!(approx_eq(l2_squared(&a, &b), 20.0));
    }

    #[test]
    fn test_dot_product_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(approx_eq(dot_product(&a, &b), 0.0));
    }

    #[test]
    fn test_dot_product_parallel() {
        let a = vec![2.0, 3.0];
        let b = vec![4.0, 5.0];
        // 2*4 + 3*5 = 23
        assert!(approx_eq(dot_product(&a, &b), 23.0));
    }

    #[test]
    fn test_cosine_identical_direction() {
        let a = vec![1.0, 0.0];
        let b = vec![5.0, 0.0]; // same direction, different magnitude
        assert!(approx_eq(cosine_similarity(&a, &b), 1.0));
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0));
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!(approx_eq(cosine_similarity(&a, &b), -1.0));
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert!(approx_eq(cosine_similarity(&a, &b), 0.0));
    }

    #[test]
    fn test_cosine_equals_dot_for_unit_vectors() {
        // Verify the optimization claim: cosine(a, b) == dot(a, b) when ||a|| = ||b|| = 1
        let a = vec![0.6, 0.8]; // ||a|| = 1.0
        let b = vec![0.0, 1.0]; // ||b|| = 1.0
        let cos = cosine_similarity(&a, &b);
        let dot = dot_product(&a, &b);
        assert!(
            approx_eq(cos, dot),
            "For unit vectors, cosine and dot product should be equal: cosine={cos}, dot={dot}"
        );
    }

    #[test]
    fn test_compute_distance_l2() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(approx_eq(compute_distance(&a, &b, Metric::L2), 2.0));
    }

    #[test]
    fn test_compute_distance_dot_negated() {
        let a = vec![2.0, 3.0];
        let b = vec![4.0, 5.0];
        // dot = 23.0, negated = -23.0
        assert!(approx_eq(compute_distance(&a, &b, Metric::DotProduct), -23.0));
    }

    #[test]
    fn test_compute_distance_cosine_negated() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        // cosine = 1.0, negated = -1.0
        assert!(approx_eq(compute_distance(&a, &b, Metric::Cosine), -1.0));
    }

    /// Validate that SIMD and scalar produce identical results on random vectors.
    #[test]
    fn test_l2_against_naive_reference() {
        use rand::Rng;
        let mut rng = rand::rng();

        for _ in 0..100 {
            let dim = 128;
            let a: Vec<f32> = (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect();

            // Naive reference
            let mut expected = 0.0_f32;
            for i in 0..dim {
                let diff = a[i] - b[i];
                expected += diff * diff;
            }

            let result = l2_squared(&a, &b);
            assert!(
                (result - expected).abs() < 1e-2,
                "L2 mismatch: got {result}, expected {expected}"
            );
        }
    }

    /// Validate dot product against a naive reference on random vectors.
    #[test]
    fn test_dot_product_against_naive_reference() {
        use rand::Rng;
        let mut rng = rand::rng();

        for _ in 0..100 {
            let dim = 128;
            let a: Vec<f32> = (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect();

            let mut expected = 0.0_f32;
            for i in 0..dim {
                expected += a[i] * b[i];
            }

            let result = dot_product(&a, &b);
            assert!(
                (result - expected).abs() < 1e-2,
                "Dot product mismatch: got {result}, expected {expected}"
            );
        }
    }

    /// Validate SIMD cosine against scalar on random vectors.
    #[test]
    fn test_cosine_simd_vs_scalar() {
        use rand::Rng;
        let mut rng = rand::rng();

        for _ in 0..100 {
            let dim = 128;
            let a: Vec<f32> = (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect();

            let scalar = cosine_similarity_scalar(&a, &b);
            let dispatched = cosine_similarity(&a, &b);

            assert!(
                (scalar - dispatched).abs() < 1e-5,
                "Cosine mismatch: scalar={scalar}, dispatched={dispatched}"
            );
        }
    }

    /// Test SIMD on dimensions that aren't multiples of 8 (exercises tail handling).
    #[test]
    fn test_simd_non_aligned_dimensions() {
        use rand::Rng;
        let mut rng = rand::rng();

        for dim in [1, 3, 5, 7, 9, 13, 15, 17, 31, 33, 127, 129, 255] {
            let a: Vec<f32> = (0..dim).map(|_| rng.random_range(-5.0..5.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.random_range(-5.0..5.0)).collect();

            // Use relative tolerance: FMA has fewer rounding steps than scalar,
            // so absolute error grows with accumulated magnitude at higher dims.
            let rel_tol = 1e-5_f32;

            let l2_scalar = l2_squared_scalar(&a, &b);
            let l2_dispatch = l2_squared(&a, &b);
            let l2_tol = rel_tol * l2_scalar.abs().max(1.0);
            assert!(
                (l2_scalar - l2_dispatch).abs() < l2_tol,
                "L2 dim={dim}: scalar={l2_scalar}, dispatch={l2_dispatch}, tol={l2_tol}"
            );

            let dot_scalar = dot_product_scalar(&a, &b);
            let dot_dispatch = dot_product(&a, &b);
            let dot_tol = rel_tol * dot_scalar.abs().max(1.0);
            assert!(
                (dot_scalar - dot_dispatch).abs() < dot_tol,
                "Dot dim={dim}: scalar={dot_scalar}, dispatch={dot_dispatch}, tol={dot_tol}"
            );

            let cos_scalar = cosine_similarity_scalar(&a, &b);
            let cos_dispatch = cosine_similarity(&a, &b);
            assert!(
                (cos_scalar - cos_dispatch).abs() < 1e-4,
                "Cosine dim={dim}: scalar={cos_scalar}, dispatch={cos_dispatch}"
            );
        }
    }

    /// Verify that SIMD detection reports correctly.
    #[test]
    fn test_simd_available() {
        let avx2 = simd_available();
        // Just ensure it doesn't panic; the actual value depends on the CPU.
        println!("AVX2+FMA available: {avx2}");
    }
}
