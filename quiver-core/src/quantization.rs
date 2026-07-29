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
}
