//! Per-dimension scalar quantization (SQ8).
//!
//! Each dimension is independently mapped from its observed minimum/maximum
//! range to an unsigned byte. Queries remain full-precision.

use crate::error::{QuiverError, Result};

/// Trained per-dimension parameters for mapping `f32` values to `u8` codes.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    mins: Vec<f32>,
    scales: Vec<f64>,
}

impl ScalarQuantizer {
    /// Train quantization parameters from a non-empty collection of vectors.
    pub fn train(vectors: &[Vec<f32>]) -> Result<Self> {
        let first = vectors.first().ok_or(QuiverError::EmptyIndex)?;
        if first.is_empty() {
            return Err(QuiverError::DimensionMismatch {
                expected: 1,
                actual: 0,
            });
        }

        let dimension = first.len();
        let mut mins = vec![f32::INFINITY; dimension];
        let mut maxs = vec![f32::NEG_INFINITY; dimension];
        for vector in vectors {
            if vector.len() != dimension {
                return Err(QuiverError::DimensionMismatch {
                    expected: dimension as u32,
                    actual: vector.len() as u32,
                });
            }
            for (dim, &value) in vector.iter().enumerate() {
                if !value.is_finite() {
                    return Err(QuiverError::InvalidFormat(
                        "SQ8 training vectors must contain only finite values".to_owned(),
                    ));
                }
                mins[dim] = mins[dim].min(value);
                maxs[dim] = maxs[dim].max(value);
            }
        }

        let scales = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&min, &max)| (max as f64 - min as f64) / u8::MAX as f64)
            .collect();
        Ok(Self { mins, scales })
    }

    /// Quantize one vector, clamping values outside the training range.
    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u8>> {
        self.validate_dimension(vector)?;
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(QuiverError::InvalidFormat(
                "SQ8 vectors must contain only finite values".to_owned(),
            ));
        }
        Ok(vector
            .iter()
            .enumerate()
            .map(|(dim, &value)| {
                let scale = self.scales[dim];
                if scale == 0.0 {
                    0
                } else {
                    ((value as f64 - self.mins[dim] as f64) / scale)
                        .round()
                        .clamp(0.0, u8::MAX as f64) as u8
                }
            })
            .collect())
    }

    /// Reconstruct a full-precision approximation from quantized codes.
    pub fn dequantize(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if codes.len() != self.dimension() {
            return Err(QuiverError::DimensionMismatch {
                expected: self.dimension() as u32,
                actual: codes.len() as u32,
            });
        }
        Ok(codes
            .iter()
            .enumerate()
            .map(|(dim, &code)| self.reconstruct(dim, code))
            .collect())
    }

    /// Number of represented dimensions.
    pub fn dimension(&self) -> usize {
        self.mins.len()
    }

    /// Borrow the per-dimension minima.
    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    /// Borrow the per-dimension scales.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Rebuild a quantizer from stored calibration. Validates shape and finiteness.
    pub fn from_parts(mins: Vec<f32>, scales: Vec<f64>) -> Result<Self> {
        if mins.is_empty() || mins.len() != scales.len() {
            return Err(QuiverError::InvalidFormat(
                "SQ8 calibration length mismatch".to_owned(),
            ));
        }
        if mins.iter().any(|v| !v.is_finite()) || scales.iter().any(|v| !v.is_finite()) {
            return Err(QuiverError::InvalidFormat(
                "SQ8 calibration must contain only finite values".to_owned(),
            ));
        }
        if scales.iter().any(|&s| s < 0.0) {
            return Err(QuiverError::InvalidFormat(
                "SQ8 scales must be non-negative".to_owned(),
            ));
        }
        Ok(Self { mins, scales })
    }

    #[inline]
    pub(crate) fn reconstruct(&self, dimension: usize, code: u8) -> f32 {
        (self.mins[dimension] as f64 + code as f64 * self.scales[dimension])
            .clamp(f32::MIN as f64, f32::MAX as f64) as f32
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension() {
            return Err(QuiverError::DimensionMismatch {
                expected: self.dimension() as u32,
                actual: vector.len() as u32,
            });
        }
        Ok(())
    }
}

/// Product quantizer: split each vector into `m` sub-vectors and quantize each
/// independently against its own `ksub`-centroid codebook trained by k-means.
///
/// Codes are one byte per subspace, so a vector compresses to `m` bytes. Query
/// distances are approximated with asymmetric distance computation (ADC): the
/// query stays full-precision and is compared against the stored centroids.
#[derive(Debug, Clone)]
pub struct ProductQuantizer {
    m: usize,
    ksub: usize,
    dsub: usize,
    /// Row-major `(m, ksub, dsub)` centroid buffer.
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Train one codebook per subspace from a non-empty collection of vectors.
    pub fn train<R: rand::Rng>(
        vectors: &[Vec<f32>],
        m: usize,
        ksub: usize,
        iters: usize,
        rng: &mut R,
    ) -> Result<Self> {
        let n = vectors.len();
        if n == 0 {
            return Err(QuiverError::EmptyIndex);
        }
        let dimension = vectors[0].len();
        Self::validate_shape(dimension, m, ksub)?;
        if ksub > n {
            return Err(QuiverError::InvalidFormat(format!(
                "PQ requires at least ksub={ksub} training vectors, got {n}"
            )));
        }
        let dsub = dimension / m;

        // Every vector must have the full dimension; a total-length check
        // alone would silently misalign sub-vectors for mixed-length input.
        for vector in vectors {
            if vector.len() != dimension {
                return Err(QuiverError::DimensionMismatch {
                    expected: dimension as u32,
                    actual: vector.len() as u32,
                });
            }
        }
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        if !flat.iter().all(|value| value.is_finite()) {
            return Err(QuiverError::InvalidFormat(
                "PQ training vectors must contain only finite values".to_owned(),
            ));
        }

        let mut codebooks = vec![0.0f32; m * ksub * dsub];
        let mut sub = vec![0.0f32; n * dsub];
        for s in 0..m {
            for i in 0..n {
                sub[i * dsub..(i + 1) * dsub].copy_from_slice(
                    &flat[i * dimension + s * dsub..i * dimension + (s + 1) * dsub],
                );
            }
            let (centroids, _) = crate::kmeans::kmeans(&sub, n, dsub, ksub, iters, rng)?;
            codebooks[s * ksub * dsub..(s + 1) * ksub * dsub].copy_from_slice(&centroids);
        }
        Ok(Self {
            m,
            ksub,
            dsub,
            codebooks,
        })
    }

    /// Wrap pre-trained codebooks (used by cross-validation against the numpy
    /// reference, which supplies its own trained state).
    pub fn from_codebooks(m: usize, ksub: usize, dsub: usize, codebooks: Vec<f32>) -> Result<Self> {
        if m == 0 || ksub == 0 || ksub > 256 || dsub == 0 {
            return Err(QuiverError::InvalidFormat(
                "PQ requires m >= 1, dsub >= 1, and 1 <= ksub <= 256".to_owned(),
            ));
        }
        if codebooks.len() != m * ksub * dsub {
            return Err(QuiverError::InvalidFormat(
                "PQ codebook length does not match m * ksub * dsub".to_owned(),
            ));
        }
        if codebooks.iter().any(|value| !value.is_finite()) {
            return Err(QuiverError::InvalidFormat(
                "PQ codebooks must contain only finite values".to_owned(),
            ));
        }
        Ok(Self {
            m,
            ksub,
            dsub,
            codebooks,
        })
    }

    /// Encode one vector into `m` sub-vector codes.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        self.validate_dimension(vector)?;
        let mut codes = vec![0u8; self.m];
        for s in 0..self.m {
            codes[s] = self.nearest_code(&vector[s * self.dsub..(s + 1) * self.dsub], s);
        }
        Ok(codes)
    }

    /// Encode a batch, returning a flat row-major `n * m` code buffer.
    pub fn encode_batch(&self, vectors: &[Vec<f32>]) -> Result<Vec<u8>> {
        let mut codes = vec![0u8; vectors.len() * self.m];
        for (i, vector) in vectors.iter().enumerate() {
            self.validate_dimension(vector)?;
            for s in 0..self.m {
                codes[i * self.m + s] =
                    self.nearest_code(&vector[s * self.dsub..(s + 1) * self.dsub], s);
            }
        }
        Ok(codes)
    }

    /// ADC lookup table for a query: squared L2 distance from each query
    /// sub-vector to every centroid, as a row-major `(m, ksub)` buffer. The
    /// approximate distance to an encoded vector is the sum of
    /// `table[s * ksub + code[s]]` over all subspaces.
    pub fn adc_table(&self, query: &[f32]) -> Result<Vec<f32>> {
        self.validate_dimension(query)?;
        let mut table = vec![0.0f32; self.m * self.ksub];
        for s in 0..self.m {
            let qsub = &query[s * self.dsub..(s + 1) * self.dsub];
            let book = &self.codebooks[s * self.ksub * self.dsub..(s + 1) * self.ksub * self.dsub];
            for c in 0..self.ksub {
                let mut acc = 0.0f32;
                let centroid = &book[c * self.dsub..(c + 1) * self.dsub];
                for j in 0..self.dsub {
                    let diff = qsub[j] - centroid[j];
                    acc += diff * diff;
                }
                table[s * self.ksub + c] = acc;
            }
        }
        Ok(table)
    }

    /// Number of sub-vectors per vector.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of centroids per subspace codebook.
    pub fn ksub(&self) -> usize {
        self.ksub
    }

    /// Dimension of each sub-vector.
    pub fn dsub(&self) -> usize {
        self.dsub
    }

    /// Full represented dimension (`m * dsub`).
    pub fn dimension(&self) -> usize {
        self.m * self.dsub
    }

    /// Borrow the raw row-major `(m, ksub, dsub)` codebook buffer.
    pub fn codebooks(&self) -> &[f32] {
        &self.codebooks
    }

    /// Reconstruct a full-precision approximation from `m` sub-vector codes.
    pub fn reconstruct(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if codes.len() != self.m {
            return Err(QuiverError::DimensionMismatch {
                expected: self.m as u32,
                actual: codes.len() as u32,
            });
        }
        let mut out = vec![0.0f32; self.dimension()];
        for s in 0..self.m {
            let code = codes[s] as usize;
            if code >= self.ksub {
                return Err(QuiverError::InvalidFormat(format!(
                    "PQ code {code} out of range for ksub={}",
                    self.ksub
                )));
            }
            let start = s * self.ksub * self.dsub + code * self.dsub;
            let centroid = &self.codebooks[start..start + self.dsub];
            out[s * self.dsub..(s + 1) * self.dsub].copy_from_slice(centroid);
        }
        Ok(out)
    }

    fn nearest_code(&self, sub_vector: &[f32], subspace: usize) -> u8 {
        let book = &self.codebooks
            [subspace * self.ksub * self.dsub..(subspace + 1) * self.ksub * self.dsub];
        let mut best = 0usize;
        let mut best_dist = f32::INFINITY;
        for c in 0..self.ksub {
            let centroid = &book[c * self.dsub..(c + 1) * self.dsub];
            let mut acc = 0.0f32;
            for j in 0..self.dsub {
                let diff = sub_vector[j] - centroid[j];
                acc += diff * diff;
            }
            if acc < best_dist {
                best_dist = acc;
                best = c;
            }
        }
        best as u8
    }

    fn validate_shape(dimension: usize, m: usize, ksub: usize) -> Result<()> {
        if m == 0 || ksub == 0 || ksub > 256 {
            return Err(QuiverError::InvalidFormat(
                "PQ requires m >= 1 and 1 <= ksub <= 256".to_owned(),
            ));
        }
        if dimension == 0 || !dimension.is_multiple_of(m) {
            return Err(QuiverError::InvalidFormat(format!(
                "PQ dimension {dimension} must be positive and divisible by m={m}"
            )));
        }
        Ok(())
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension() {
            return Err(QuiverError::DimensionMismatch {
                expected: self.dimension() as u32,
                actual: vector.len() as u32,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_map_to_full_byte_range() {
        let quantizer = ScalarQuantizer::train(&[vec![-2.0, 10.0], vec![3.0, 20.0]]).unwrap();
        assert_eq!(quantizer.quantize(&[-2.0, 10.0]).unwrap(), vec![0, 0]);
        assert_eq!(quantizer.quantize(&[3.0, 20.0]).unwrap(), vec![255, 255]);
    }

    #[test]
    fn round_trip_error_is_bounded_by_half_a_bin() {
        let quantizer = ScalarQuantizer::train(&[vec![-10.0], vec![10.0]]).unwrap();
        let reconstructed = quantizer
            .dequantize(&quantizer.quantize(&[1.234]).unwrap())
            .unwrap();
        let half_bin = 20.0 / 255.0 / 2.0;
        assert!((reconstructed[0] - 1.234).abs() <= half_bin + f32::EPSILON);
    }

    #[test]
    fn constant_dimensions_round_trip_exactly() {
        let quantizer = ScalarQuantizer::train(&[vec![4.5, 1.0], vec![4.5, 8.0]]).unwrap();
        let codes = quantizer.quantize(&[4.5, 5.0]).unwrap();
        assert_eq!(codes[0], 0);
        assert_eq!(quantizer.dequantize(&codes).unwrap()[0], 4.5);
    }

    #[test]
    fn extreme_finite_range_does_not_overflow_calibration() {
        let quantizer = ScalarQuantizer::train(&[vec![f32::MIN], vec![f32::MAX]]).unwrap();
        assert_eq!(quantizer.quantize(&[f32::MIN]).unwrap(), vec![0]);
        assert_eq!(quantizer.quantize(&[f32::MAX]).unwrap(), vec![255]);

        let minimum = quantizer.dequantize(&[0]).unwrap()[0];
        let maximum = quantizer.dequantize(&[255]).unwrap()[0];
        assert_eq!(minimum, f32::MIN);
        assert_eq!(maximum, f32::MAX);
    }

    #[test]
    fn subnormal_range_preserves_distinct_endpoints() {
        let quantizer = ScalarQuantizer::train(&[vec![0.0], vec![f32::MIN_POSITIVE]]).unwrap();
        assert_eq!(quantizer.quantize(&[0.0]).unwrap(), vec![0]);
        assert_eq!(quantizer.quantize(&[f32::MIN_POSITIVE]).unwrap(), vec![255]);
    }

    #[test]
    fn rejects_bad_inputs() {
        let quantizer = ScalarQuantizer::train(&[vec![0.0, 1.0]]).unwrap();
        assert!(quantizer.quantize(&[0.0]).is_err());
        assert!(quantizer.quantize(&[0.0, f32::NAN]).is_err());
    }

    // ── ProductQuantizer ────────────────────────────────────────────────────

    use rand::{Rng, SeedableRng};

    fn clustered_vectors(rng: &mut rand::rngs::StdRng, n: usize, d: usize) -> Vec<Vec<f32>> {
        // A handful of well-separated blobs give PQ structure to latch onto.
        let blobs = 8;
        let centers: Vec<Vec<f32>> = (0..blobs)
            .map(|_| (0..d).map(|_| rng.random_range(-20.0..20.0)).collect())
            .collect();
        (0..n)
            .map(|_| {
                let center = &centers[rng.random_range(0..blobs)];
                center
                    .iter()
                    .map(|&c| c + rng.random_range(-0.5..0.5))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn pq_encode_produces_m_codes_in_range() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let vectors = clustered_vectors(&mut rng, 300, 16);
        let pq = ProductQuantizer::train(&vectors, 4, 64, 6, &mut rng).unwrap();
        assert_eq!(pq.m(), 4);
        assert_eq!(pq.ksub(), 64);
        assert_eq!(pq.dsub(), 4);
        assert_eq!(pq.dimension(), 16);

        let codes = pq.encode(&vectors[0]).unwrap();
        assert_eq!(codes.len(), 4);
        assert!(codes.iter().all(|&c| (c as usize) < 64));
    }

    #[test]
    fn pq_adc_table_matches_manual_distances() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let vectors = clustered_vectors(&mut rng, 200, 8);
        let pq = ProductQuantizer::train(&vectors, 2, 16, 5, &mut rng).unwrap();
        let query = &vectors[3];
        let table = pq.adc_table(query).unwrap();
        assert_eq!(table.len(), 2 * 16);

        // Verify a few entries by hand.
        for s in 0..2 {
            let qsub = &query[s * 4..(s + 1) * 4];
            for c in [0usize, 5, 15] {
                let centroid = &pq.codebooks()[s * 16 * 4 + c * 4..s * 16 * 4 + (c + 1) * 4];
                let expected: f32 = qsub
                    .iter()
                    .zip(centroid)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                assert!(
                    (table[s * 16 + c] - expected).abs() < 1e-4,
                    "table[{},{}] = {} expected {}",
                    s,
                    c,
                    table[s * 16 + c],
                    expected
                );
            }
        }
    }

    #[test]
    fn pq_adc_distance_approximates_true_l2() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let vectors = clustered_vectors(&mut rng, 500, 16);
        let pq = ProductQuantizer::train(&vectors, 8, 256, 8, &mut rng).unwrap();
        let codes = pq.encode_batch(&vectors).unwrap();

        let query = &vectors[10];
        let table = pq.adc_table(query).unwrap();
        let mut max_rel_error = 0.0f32;
        for i in [0usize, 50, 250, 499] {
            let code = &codes[i * 8..(i + 1) * 8];
            let adc: f32 = (0..8).map(|s| table[s * 256 + code[s] as usize]).sum();
            let true_dist: f32 = query
                .iter()
                .zip(&vectors[i])
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            let scale = true_dist.max(1.0);
            max_rel_error = max_rel_error.max((adc - true_dist).abs() / scale);
        }
        assert!(
            max_rel_error < 0.25,
            "ADC relative error too large: {max_rel_error}"
        );
    }

    #[test]
    fn pq_reconstruction_error_is_small_on_clustered_data() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let vectors = clustered_vectors(&mut rng, 400, 16);
        let pq = ProductQuantizer::train(&vectors, 8, 256, 8, &mut rng).unwrap();

        let mut total_err = 0.0f32;
        let mut total_norm = 0.0f32;
        for vector in vectors.iter().take(50) {
            let codes = pq.encode(vector).unwrap();
            let recon = pq.reconstruct(&codes).unwrap();
            let err: f32 = vector
                .iter()
                .zip(&recon)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = vector.iter().map(|a| a * a).sum::<f32>().sqrt();
            total_err += err;
            total_norm += norm;
        }
        let ratio = total_err / total_norm;
        assert!(ratio < 0.1, "mean reconstruction ratio too large: {ratio}");
    }

    #[test]
    fn pq_from_codebooks_round_trips() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let vectors = clustered_vectors(&mut rng, 100, 8);
        let pq = ProductQuantizer::train(&vectors, 2, 32, 4, &mut rng).unwrap();

        let rebuilt = ProductQuantizer::from_codebooks(2, 32, 4, pq.codebooks().to_vec()).unwrap();
        assert_eq!(
            rebuilt.encode(&vectors[0]).unwrap(),
            pq.encode(&vectors[0]).unwrap()
        );
        assert_eq!(
            rebuilt.adc_table(&vectors[1]).unwrap(),
            pq.adc_table(&vectors[1]).unwrap()
        );
    }

    #[test]
    fn pq_rejects_bad_shapes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(6);
        let vectors = clustered_vectors(&mut rng, 50, 8);
        // dimension not divisible by m
        assert!(ProductQuantizer::train(&vectors, 3, 16, 4, &mut rng).is_err());
        // ksub out of byte range
        assert!(ProductQuantizer::train(&vectors, 2, 257, 4, &mut rng).is_err());
        // empty input
        assert!(ProductQuantizer::train(&[], 2, 16, 4, &mut rng).is_err());

        let pq = ProductQuantizer::train(&vectors, 2, 16, 4, &mut rng).unwrap();
        assert!(pq.encode(&[0.0; 7]).is_err()); // wrong dimension
        assert!(pq.reconstruct(&[0u8; 3]).is_err()); // wrong code count
        assert!(ProductQuantizer::from_codebooks(2, 16, 4, vec![0.0; 5]).is_err());
    }

    #[test]
    fn pq_rejects_ksub_larger_than_vector_count() {
        // k-means clamps k to n, so this used to panic inside the codebook
        // copy; it must be a clean error instead.
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        let vectors = clustered_vectors(&mut rng, 4, 8);
        assert!(ProductQuantizer::train(&vectors, 2, 64, 4, &mut rng).is_err());
    }

    #[test]
    fn pq_rejects_mixed_length_vectors() {
        // 8 + 7 + 9 = 24 = 3 * 8, so a total-length check alone would pass
        // and silently train on shifted sub-vectors.
        let vectors = vec![vec![0.0f32; 8], vec![1.0f32; 7], vec![2.0f32; 9]];
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        assert!(ProductQuantizer::train(&vectors, 2, 16, 4, &mut rng).is_err());
    }

    #[test]
    fn pq_from_codebooks_rejects_non_finite() {
        let mut book = vec![0.0f32; 2 * 16 * 4];
        book[0] = f32::NAN;
        assert!(ProductQuantizer::from_codebooks(2, 16, 4, book).is_err());
    }
}
