//! Static flat index backed by SQ8-compressed vectors.
//!
//! Building is batch-only because changing calibration ranges requires
//! requantizing the existing collection.

use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::distance::Metric;
use crate::error::{QuiverError, Result};
use crate::index::SearchResult;
use crate::quantization::ScalarQuantizer;

/// A batch-built, in-memory SQ8 flat index.
pub struct Sq8Index {
    quantizer: ScalarQuantizer,
    vectors: Vec<u8>,
    metric: Metric,
}

impl Sq8Index {
    /// Train and build an SQ8 index from full-precision vectors.
    pub fn build(vectors: &[Vec<f32>], metric: Metric) -> Result<Self> {
        let quantizer = ScalarQuantizer::train(vectors)?;
        let mut encoded = Vec::with_capacity(vectors.len() * quantizer.dimension());
        for vector in vectors {
            encoded.extend(quantizer.quantize(vector)?);
        }
        Ok(Self {
            quantizer,
            vectors: encoded,
            metric,
        })
    }

    /// Search the compressed collection and return closest results first.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension() {
            return Err(QuiverError::DimensionMismatch {
                expected: self.dimension() as u32,
                actual: query.len() as u32,
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(QuiverError::InvalidFormat(
                "SQ8 queries must contain only finite values".to_owned(),
            ));
        }

        let k = k.min(self.len());
        if k == 0 {
            return Ok(Vec::new());
        }
        let l2_lookup = (self.metric == Metric::L2).then(|| self.l2_lookup(query));
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for (slot, codes) in self.vectors.chunks_exact(self.dimension()).enumerate() {
            let distance = match &l2_lookup {
                Some(lookup) => lookup
                    .chunks_exact(usize::from(u8::MAX) + 1)
                    .zip(codes)
                    .map(|(dimension, &code)| dimension[usize::from(code)])
                    .sum(),
                None => self.asymmetric_distance(query, codes),
            };
            let result = SearchResult {
                slot,
                vector_id: slot as u64 + 1,
                distance,
            };
            if heap.len() < k {
                heap.push(result);
            } else if heap.peek().is_some_and(|worst| distance < worst.distance) {
                heap.pop();
                heap.push(result);
            }
        }
        Ok(heap.into_sorted_vec())
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.vectors.len() / self.dimension()
    }

    /// Whether the collection is empty. A built index is always non-empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Vector dimension.
    pub fn dimension(&self) -> usize {
        self.quantizer.dimension()
    }

    /// Distance metric used by this index.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Bytes used by compressed vector payloads, excluding calibration arrays.
    pub fn vector_bytes(&self) -> usize {
        self.vectors.len()
    }

    /// Save the index to `path` with CRC32 integrity checks (atomic tmp+rename).
    ///
    /// Format `QVSQ` v1: header (magic, version, metric, dim, count + header CRC)
    /// then body (mins f32, scales f64, codes u8 + body CRC).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        const MAGIC: &[u8; 4] = b"QVSQ";
        let path = path.as_ref();
        let mut buf: Vec<u8> = Vec::new();
        buf.write_all(MAGIC).unwrap();
        buf.write_u8(1).unwrap();
        buf.write_u8(self.metric as u8).unwrap();
        buf.write_all(&[0u8; 2]).unwrap();
        let dim = self.dimension();
        let count = self.len();
        let dim_u32 = u32::try_from(dim)
            .map_err(|_| QuiverError::InvalidFormat("SQ8 dimension exceeds u32".to_owned()))?;
        let count_u64 = count as u64;
        buf.write_u32::<LittleEndian>(dim_u32).unwrap();
        buf.write_u64::<LittleEndian>(count_u64).unwrap();
        let header_crc = crc32fast::hash(&buf);
        buf.write_u32::<LittleEndian>(header_crc).unwrap();

        let body_start = buf.len();
        for &v in self.quantizer.mins() {
            buf.write_f32::<LittleEndian>(v).unwrap();
        }
        for &s in self.quantizer.scales() {
            buf.write_f64::<LittleEndian>(s).unwrap();
        }
        buf.write_all(&self.vectors).unwrap();
        let body_crc = crc32fast::hash(&buf[body_start..]);
        buf.write_u32::<LittleEndian>(body_crc).unwrap();

        let mut tmp = PathBuf::from(path.as_os_str());
        tmp.as_mut_os_string().push(".tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        let _ = fs::remove_file(path);
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load an index saved with [`Sq8Index::save`]. Rejects bad magic, version,
    /// metric, CRC mismatches, and truncated bodies before allocating payloads.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        const MAGIC: &[u8; 4] = b"QVSQ";
        const HEADER_LEN: usize = 4 + 1 + 1 + 2 + 4 + 8 + 4;
        let data = fs::read(path.as_ref())?;
        if data.len() < HEADER_LEN + 4 {
            return Err(QuiverError::InvalidFormat(
                "SQ8 snapshot too short".to_owned(),
            ));
        }
        let mut cur = Cursor::new(&data);
        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(QuiverError::InvalidFormat(
                "invalid SQ8 snapshot magic".to_owned(),
            ));
        }
        let version = cur.read_u8()?;
        if version != 1 {
            return Err(QuiverError::InvalidFormat(format!(
                "unsupported SQ8 snapshot version: {version}"
            )));
        }
        let metric_byte = cur.read_u8()?;
        let mut reserved = [0u8; 2];
        cur.read_exact(&mut reserved)?;
        let dim = cur.read_u32::<LittleEndian>()? as usize;
        let count = cur.read_u64::<LittleEndian>()? as usize;
        let header_crc = cur.read_u32::<LittleEndian>()?;
        if crc32fast::hash(&data[..HEADER_LEN - 4]) != header_crc {
            return Err(QuiverError::InvalidFormat(
                "SQ8 snapshot header checksum mismatch".to_owned(),
            ));
        }
        let metric =
            Metric::from_u8(metric_byte).ok_or(QuiverError::UnsupportedMetric(metric_byte))?;
        if dim == 0 {
            return Err(QuiverError::InvalidFormat(
                "SQ8 snapshot has zero dimension".to_owned(),
            ));
        }
        let body_bytes: u128 = dim as u128 * 4 + dim as u128 * 8 + count as u128 * dim as u128;
        let capacity = (data.len() as u128)
            .saturating_sub(HEADER_LEN as u128)
            .saturating_sub(4);
        if body_bytes > capacity {
            return Err(QuiverError::InvalidFormat(
                "SQ8 snapshot body truncated".to_owned(),
            ));
        }
        let crc_start = HEADER_LEN + body_bytes as usize;
        let body = &data[HEADER_LEN..crc_start];
        let body_crc = (&data[crc_start..crc_start + 4]).read_u32::<LittleEndian>()?;
        if crc32fast::hash(body) != body_crc {
            return Err(QuiverError::InvalidFormat(
                "SQ8 snapshot body checksum mismatch".to_owned(),
            ));
        }
        let mut bcur = Cursor::new(body);
        let mut mins = vec![0.0f32; dim];
        for v in &mut mins {
            *v = bcur.read_f32::<LittleEndian>()?;
        }
        let mut scales = vec![0.0f64; dim];
        for s in &mut scales {
            *s = bcur.read_f64::<LittleEndian>()?;
        }
        let mut vectors = vec![0u8; count * dim];
        bcur.read_exact(&mut vectors)?;
        let quantizer = ScalarQuantizer::from_parts(mins, scales)?;
        if quantizer.dimension() != dim {
            return Err(QuiverError::InvalidFormat(
                "SQ8 calibration dimension mismatch".to_owned(),
            ));
        }
        Ok(Self {
            quantizer,
            vectors,
            metric,
        })
    }

    fn l2_lookup(&self, query: &[f32]) -> Vec<f32> {
        let codes_per_dimension = usize::from(u8::MAX) + 1;
        let mut lookup = Vec::with_capacity(self.dimension() * codes_per_dimension);
        for (dimension, &query_value) in query.iter().enumerate() {
            for code in u8::MIN..=u8::MAX {
                let difference = query_value - self.quantizer.reconstruct(dimension, code);
                lookup.push(difference * difference);
            }
        }
        lookup
    }

    #[inline]
    fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        match self.metric {
            Metric::L2 => query
                .iter()
                .enumerate()
                .map(|(dim, &q)| {
                    let diff = q - self.quantizer.reconstruct(dim, codes[dim]);
                    diff * diff
                })
                .sum(),
            Metric::DotProduct => -query
                .iter()
                .enumerate()
                .map(|(dim, &q)| q * self.quantizer.reconstruct(dim, codes[dim]))
                .sum::<f32>(),
            Metric::Cosine => {
                let mut dot = 0.0;
                let mut query_norm = 0.0;
                let mut vector_norm = 0.0;
                for (dim, &q) in query.iter().enumerate() {
                    let value = self.quantizer.reconstruct(dim, codes[dim]);
                    dot += q * value;
                    query_norm += q * q;
                    vector_norm += value * value;
                }
                let magnitude = (query_norm * vector_norm).sqrt();
                if magnitude == 0.0 {
                    0.0
                } else {
                    -(dot / magnitude)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::distance::compute_distance;

    #[test]
    fn l2_search_returns_nearest_vector() {
        let vectors = vec![vec![0.0, 0.0], vec![5.0, 5.0], vec![10.0, 10.0]];
        let index = Sq8Index::build(&vectors, Metric::L2).unwrap();
        let results = index.search(&[4.9, 5.1], 2).unwrap();
        assert_eq!(results[0].slot, 1);
        assert!(results[0].distance <= results[1].distance);
    }

    #[test]
    fn supports_dot_product_and_cosine() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        for metric in [Metric::DotProduct, Metric::Cosine] {
            let index = Sq8Index::build(&vectors, metric).unwrap();
            assert_eq!(index.search(&[1.0, 0.05], 1).unwrap()[0].slot, 0);
        }
    }

    #[test]
    fn payload_uses_one_byte_per_dimension() {
        let vectors = vec![vec![0.0; 128]; 100];
        let index = Sq8Index::build(&vectors, Metric::L2).unwrap();
        assert_eq!(index.vector_bytes(), 100 * 128);
        assert_eq!(index.vector_bytes() * 4, 100 * 128 * size_of::<f32>());
    }

    #[test]
    fn zero_k_returns_no_results() {
        let index = Sq8Index::build(&[vec![1.0, 2.0]], Metric::L2).unwrap();
        assert!(index.search(&[1.0, 2.0], 0).unwrap().is_empty());
    }

    #[test]
    fn rejects_empty_and_non_finite_inputs() {
        assert!(Sq8Index::build(&[], Metric::L2).is_err());
        assert!(Sq8Index::build(&[vec![f32::INFINITY]], Metric::L2).is_err());

        let index = Sq8Index::build(&[vec![1.0]], Metric::L2).unwrap();
        assert!(index.search(&[f32::NAN], 1).is_err());
    }

    #[test]
    fn recall_at_10_stays_high_on_random_data() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..32).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        let index = Sq8Index::build(&vectors, Metric::L2).unwrap();
        let mut recall = 0.0;
        for _ in 0..25 {
            let query: Vec<f32> = (0..32).map(|_| rng.random_range(-1.0..1.0)).collect();
            let mut exact: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(slot, vector)| (slot, compute_distance(&query, vector, Metric::L2)))
                .collect();
            exact.sort_by(|a, b| a.1.total_cmp(&b.1));
            let expected: HashSet<usize> = exact.iter().take(10).map(|item| item.0).collect();
            let actual: HashSet<usize> = index
                .search(&query, 10)
                .unwrap()
                .into_iter()
                .map(|result| result.slot)
                .collect();
            recall += expected.intersection(&actual).count() as f32 / 10.0;
        }
        assert!(recall / 25.0 >= 0.95, "SQ8 recall was {}", recall / 25.0);
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let vectors = vec![vec![0.0, 0.0], vec![5.0, 5.0], vec![10.0, 10.0]];
        let index = Sq8Index::build(&vectors, Metric::L2).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "quiver-sq8-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.qvsq");
        index.save(&path).unwrap();
        let loaded = Sq8Index::load(&path).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dimension(), 2);
        let a = index.search(&[4.9, 5.1], 2).unwrap();
        let b = loaded.search(&[4.9, 5.1], 2).unwrap();
        assert_eq!(a, b);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_corrupt_snapshot() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let index = Sq8Index::build(&vectors, Metric::L2).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "quiver-sq8-bad-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("index.qvsq");
        index.save(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert!(Sq8Index::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
