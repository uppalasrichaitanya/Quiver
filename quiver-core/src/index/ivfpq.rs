//! IVF-PQ: inverted-file index with product-quantized vectors.
//!
//! A coarse k-means quantizer partitions the dataset into `nlist` Voronoi
//! cells; at query time only the `nprobe` nearest cells are scanned. Within
//! each cell, vectors are stored as `m`-byte product-quantization codes and
//! scored with asymmetric distance computation (ADC), so the scan touches a
//! small compressed payload instead of full-precision vectors. An optional
//! rerank step re-scores the top ADC candidates against the stored
//! full-precision vectors to recover recall.
//!
//! This is a batch-built, in-memory, L2-only index — the memory-efficient
//! alternative to HNSW in the project plan. It is validated against the numpy
//! reference in `benchmarks/pq_reference.py` (see the cross-validation test).

use std::collections::BinaryHeap;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::distance::l2_squared;
use crate::error::{QuiverError, Result};
use crate::index::SearchResult;
use crate::quantization::ProductQuantizer;
use rand::{Rng, SeedableRng};

/// Build-time parameters for [`IvfPqIndex`].
#[derive(Debug, Clone)]
pub struct IvfPqConfig {
    /// Number of coarse Voronoi cells (inverted lists).
    pub nlist: usize,
    /// Number of product-quantization sub-vectors per vector.
    pub m: usize,
    /// Number of centroids per sub-vector codebook (max 256).
    pub ksub: usize,
    /// Lloyd iterations for both coarse and PQ k-means.
    pub kmeans_iters: usize,
    /// Number of vectors sampled to train the quantizers (capped at `n`).
    pub training_size: usize,
    /// Whether to keep full-precision vectors for the rerank step.
    pub store_vectors: bool,
    /// RNG seed for k-means initialization and training subsampling.
    pub seed: u64,
}

impl IvfPqConfig {
    pub fn new(nlist: usize, m: usize, ksub: usize) -> Self {
        Self {
            nlist,
            m,
            ksub,
            kmeans_iters: 8,
            training_size: 65_536,
            store_vectors: true,
            seed: 0xC0FFEE,
        }
    }
}

/// A batch-built, in-memory IVF-PQ index (L2 only).
pub struct IvfPqIndex {
    dimension: usize,
    nlist: usize,
    pq: ProductQuantizer,
    /// Row-major `(nlist, dimension)` coarse centroids.
    coarse: Vec<f32>,
    /// Per-cell point slots (indices into the original build input).
    list_slots: Vec<Vec<u32>>,
    /// Per-cell flat PQ codes, `list_slots[c].len() * m` bytes each.
    list_codes: Vec<Vec<u8>>,
    /// Full-precision vectors for rerank, row-major `(n, dimension)`; empty when
    /// built with `store_vectors = false`.
    vectors: Vec<f32>,
    len: usize,
}

impl IvfPqIndex {
    /// Train the quantizers and build the index from full-precision vectors.
    pub fn build(vectors: &[Vec<f32>], config: &IvfPqConfig) -> Result<Self> {
        let n = vectors.len();
        if n == 0 {
            return Err(QuiverError::EmptyIndex);
        }
        let dimension = vectors[0].len();
        if config.m == 0 {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ requires m >= 1".to_owned(),
            ));
        }
        if dimension == 0 || !dimension.is_multiple_of(config.m) {
            return Err(QuiverError::InvalidFormat(format!(
                "IVF-PQ dimension {dimension} must be positive and divisible by m={}",
                config.m
            )));
        }
        if config.nlist == 0 {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ requires nlist >= 1".to_owned(),
            ));
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        // Training subsample (distinct random indices).
        let train_n = config.training_size.min(n).max(1);
        let sample: Vec<Vec<f32>> = if train_n >= n {
            vectors.to_vec()
        } else {
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..train_n {
                let j = rng.random_range(i..n);
                indices.swap(i, j);
            }
            indices[..train_n]
                .iter()
                .map(|&i| vectors[i].clone())
                .collect()
        };

        // Coarse quantizer.
        let nlist = config.nlist.min(train_n);
        let sample_flat: Vec<f32> = sample.iter().flatten().copied().collect();
        let (coarse, _) = crate::kmeans::kmeans(
            &sample_flat,
            train_n,
            dimension,
            nlist,
            config.kmeans_iters,
            &mut rng,
        )?;

        // Product quantizer.
        let pq = ProductQuantizer::train(
            &sample,
            config.m,
            config.ksub,
            config.kmeans_iters,
            &mut rng,
        )?;

        // Assign and encode every vector.
        let mut assignments = vec![0u32; n];
        for (i, vector) in vectors.iter().enumerate() {
            assignments[i] = nearest_coarse(vector, &coarse, nlist, dimension);
        }
        let codes = pq.encode_batch(vectors)?;

        let vectors_flat = if config.store_vectors {
            vectors.iter().flatten().copied().collect()
        } else {
            Vec::new()
        };

        Self::assemble(
            dimension,
            nlist,
            pq,
            coarse,
            assignments,
            codes,
            vectors_flat,
            n,
        )
    }

    /// Build an index from an already-trained state. Used by the cross-validation
    /// test, which loads the numpy reference's quantizers and must reproduce its
    /// recall exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn from_trained(
        dimension: usize,
        nlist: usize,
        pq: ProductQuantizer,
        coarse: Vec<f32>,
        assignments: Vec<u32>,
        codes: Vec<u8>,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Self> {
        let n = assignments.len();
        if coarse.len() != nlist * dimension {
            return Err(QuiverError::InvalidFormat(
                "coarse centroid buffer does not match nlist * dimension".to_owned(),
            ));
        }
        if codes.len() != n * pq.m() {
            return Err(QuiverError::InvalidFormat(
                "code buffer does not match n * m".to_owned(),
            ));
        }
        if vectors.len() != n {
            return Err(QuiverError::InvalidFormat(
                "vector count does not match assignment count".to_owned(),
            ));
        }
        if nlist == 0 {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ requires nlist >= 1".to_owned(),
            ));
        }
        if pq.dimension() != dimension {
            return Err(QuiverError::InvalidFormat(format!(
                "PQ dimension {} does not match index dimension {dimension}",
                pq.dimension()
            )));
        }
        for vector in &vectors {
            if vector.len() != dimension {
                return Err(QuiverError::DimensionMismatch {
                    expected: dimension as u32,
                    actual: vector.len() as u32,
                });
            }
        }
        // Codes are looked up without a range check in the search hot path, so
        // external state must be validated up front.
        for &code in &codes {
            if code as usize >= pq.ksub() {
                return Err(QuiverError::InvalidFormat(format!(
                    "PQ code {code} out of range for ksub={}",
                    pq.ksub()
                )));
            }
        }
        let vectors_flat = vectors.iter().flatten().copied().collect();
        Self::assemble(
            dimension,
            nlist,
            pq,
            coarse,
            assignments,
            codes,
            vectors_flat,
            n,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        dimension: usize,
        nlist: usize,
        pq: ProductQuantizer,
        coarse: Vec<f32>,
        assignments: Vec<u32>,
        codes: Vec<u8>,
        vectors_flat: Vec<f32>,
        n: usize,
    ) -> Result<Self> {
        let m = pq.m();
        let mut counts = vec![0usize; nlist];
        for &cell in &assignments {
            if cell as usize >= nlist {
                return Err(QuiverError::InvalidFormat(
                    "assignment references a coarse cell outside nlist".to_owned(),
                ));
            }
            counts[cell as usize] += 1;
        }
        let mut list_slots: Vec<Vec<u32>> = counts.iter().map(|&c| Vec::with_capacity(c)).collect();
        let mut list_codes: Vec<Vec<u8>> =
            counts.iter().map(|&c| Vec::with_capacity(c * m)).collect();
        for (slot, &cell) in assignments.iter().enumerate() {
            let cell = cell as usize;
            let slot = u32::try_from(slot).map_err(|_| {
                QuiverError::InvalidFormat("IVF-PQ supports at most 2^32 vectors".to_owned())
            })?;
            list_slots[cell].push(slot);
            list_codes[cell].extend_from_slice(&codes[slot as usize * m..(slot as usize + 1) * m]);
        }
        Ok(Self {
            dimension,
            nlist,
            pq,
            coarse,
            list_slots,
            list_codes,
            vectors: vectors_flat,
            len: n,
        })
    }

    /// Search the index, returning up to `k` results closest-first.
    ///
    /// `nprobe` coarse cells are scanned. When `rerank_factor` is non-zero, the
    /// top `k * rerank_factor` ADC candidates are re-scored with exact L2
    /// against the stored full-precision vectors (requires `store_vectors`).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        rerank_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(QuiverError::DimensionMismatch {
                expected: self.dimension as u32,
                actual: query.len() as u32,
            });
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(QuiverError::InvalidInput(
                "IVF-PQ queries must contain only finite values".to_owned(),
            ));
        }
        if self.len == 0 {
            return Err(QuiverError::EmptyIndex);
        }
        let k = k.min(self.len);
        if k == 0 {
            return Ok(Vec::new());
        }
        if rerank_factor > 0 && self.vectors.is_empty() {
            return Err(QuiverError::InvalidFormat(
                "rerank requires the index to be built with store_vectors = true".to_owned(),
            ));
        }

        let nprobe = nprobe.clamp(1, self.nlist);
        let probes = self.nearest_cells(query, nprobe);
        let table = self.pq.adc_table(query)?;
        let m = self.pq.m();
        let ksub = self.pq.ksub();

        let want = if rerank_factor > 0 {
            k.saturating_mul(rerank_factor).min(self.len)
        } else {
            k
        };

        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(want + 1);
        for &cell in &probes {
            let slots = &self.list_slots[cell];
            let codes = &self.list_codes[cell];
            for (idx, &slot) in slots.iter().enumerate() {
                let code = &codes[idx * m..(idx + 1) * m];
                let mut adc = 0.0f32;
                for s in 0..m {
                    adc += table[s * ksub + code[s] as usize];
                }
                let result = SearchResult {
                    slot: slot as usize,
                    vector_id: slot as u64 + 1,
                    distance: adc,
                };
                if heap.len() < want {
                    heap.push(result);
                } else if heap.peek().is_some_and(|worst| adc < worst.distance) {
                    heap.pop();
                    heap.push(result);
                }
            }
        }

        let mut candidates = heap.into_vec();
        if rerank_factor > 0 {
            for candidate in &mut candidates {
                let start = candidate.slot * self.dimension;
                candidate.distance =
                    l2_squared(query, &self.vectors[start..start + self.dimension]);
            }
        }
        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        candidates.truncate(k);
        Ok(candidates)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of coarse cells.
    pub fn nlist(&self) -> usize {
        self.nlist
    }

    /// Bytes of PQ codes (the compressed searchable payload), `n * m`.
    pub fn code_bytes(&self) -> usize {
        self.len * self.pq.m()
    }

    /// Bytes of stored full-precision vectors (0 when `store_vectors` is false).
    pub fn vector_bytes(&self) -> usize {
        self.vectors.len() * size_of::<f32>()
    }

    /// Save the index to `path` with CRC32 integrity checks (atomic tmp+rename).
    ///
    /// Format `QVPQ` v1: header (magic, version, dims, counts + header CRC),
    /// then body (coarse f32, codebooks f32, per-cell slot/code runs,
    /// full-precision vectors + body CRC).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        const MAGIC: &[u8; 4] = b"QVPQ";
        let path = path.as_ref();
        let store_flag: u8 = u8::from(!self.vectors.is_empty());
        let mut buf: Vec<u8> = Vec::new();
        buf.write_all(MAGIC).unwrap();
        buf.write_u8(1).unwrap();
        buf.write_u8(store_flag).unwrap();
        buf.write_all(&[0u8; 2]).unwrap();
        for v in [
            self.dimension as u64,
            self.nlist as u64,
            self.pq.m() as u64,
            self.pq.ksub() as u64,
            self.pq.dsub() as u64,
            self.len as u64,
        ] {
            let w = u32::try_from(v)
                .map_err(|_| QuiverError::InvalidFormat("IVF-PQ value exceeds u32".to_owned()))?;
            buf.write_u32::<LittleEndian>(w).unwrap();
        }
        let coarse_len = self.coarse.len() as u64;
        let codebook_len = self.pq.codebooks().len() as u64;
        let vectors_len = self.vectors.len() as u64;
        buf.write_u64::<LittleEndian>(coarse_len).unwrap();
        buf.write_u64::<LittleEndian>(codebook_len).unwrap();
        buf.write_u64::<LittleEndian>(vectors_len).unwrap();
        let header_crc = crc32fast::hash(&buf);
        buf.write_u32::<LittleEndian>(header_crc).unwrap();

        let body_start = buf.len();
        for &v in &self.coarse {
            buf.write_f32::<LittleEndian>(v).unwrap();
        }
        for &v in self.pq.codebooks() {
            buf.write_f32::<LittleEndian>(v).unwrap();
        }
        for cell in 0..self.nlist {
            let slots = &self.list_slots[cell];
            buf.write_u64::<LittleEndian>(slots.len() as u64).unwrap();
            for &s in slots {
                buf.write_u32::<LittleEndian>(s).unwrap();
            }
            buf.write_all(&self.list_codes[cell]).unwrap();
        }
        for &v in &self.vectors {
            buf.write_f32::<LittleEndian>(v).unwrap();
        }
        let body_crc = crc32fast::hash(&buf[body_start..]);
        buf.write_u32::<LittleEndian>(body_crc).unwrap();

        let tmp = path.with_extension("tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        let _ = fs::remove_file(path);
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load an index saved with [`IvfPqIndex::save`]. Validates magic, version,
    /// shape, code ranges, and CRCs with 128-bit pre-allocation bounds.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        const MAGIC: &[u8; 4] = b"QVPQ";
        const HEADER_LEN: usize = 4 + 1 + 1 + 2 + 6 * 4 + 3 * 8 + 4;
        let data = fs::read(path.as_ref())?;
        if data.len() < HEADER_LEN + 4 {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ snapshot too short".to_owned(),
            ));
        }
        let mut cur = Cursor::new(&data);
        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(QuiverError::InvalidFormat(
                "invalid IVF-PQ snapshot magic".to_owned(),
            ));
        }
        let version = cur.read_u8()?;
        if version != 1 {
            return Err(QuiverError::InvalidFormat(format!(
                "unsupported IVF-PQ snapshot version: {version}"
            )));
        }
        let store_flag = cur.read_u8()?;
        if store_flag > 1 {
            return Err(QuiverError::InvalidFormat(
                "invalid IVF-PQ store_vectors flag".to_owned(),
            ));
        }
        let mut reserved = [0u8; 2];
        cur.read_exact(&mut reserved)?;
        let dimension = cur.read_u32::<LittleEndian>()? as usize;
        let nlist = cur.read_u32::<LittleEndian>()? as usize;
        let m = cur.read_u32::<LittleEndian>()? as usize;
        let ksub = cur.read_u32::<LittleEndian>()? as usize;
        let dsub = cur.read_u32::<LittleEndian>()? as usize;
        let len = cur.read_u32::<LittleEndian>()? as usize;
        let coarse_len = cur.read_u64::<LittleEndian>()? as usize;
        let codebook_len = cur.read_u64::<LittleEndian>()? as usize;
        let vectors_len = cur.read_u64::<LittleEndian>()? as usize;
        let header_crc = cur.read_u32::<LittleEndian>()?;
        if crc32fast::hash(&data[..HEADER_LEN - 4]) != header_crc {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ snapshot header checksum mismatch".to_owned(),
            ));
        }
        if dimension == 0 || nlist == 0 || m == 0 || ksub == 0 || ksub > 256 || dsub == 0 {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ snapshot has invalid shape".to_owned(),
            ));
        }
        if dimension != m * dsub {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ dimension must equal m * dsub".to_owned(),
            ));
        }
        if coarse_len != nlist * dimension || codebook_len != m * ksub * dsub {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ snapshot buffer length mismatch".to_owned(),
            ));
        }
        let want_vectors = if store_flag == 1 { len * dimension } else { 0 };
        if vectors_len != want_vectors {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ snapshot vector buffer mismatch".to_owned(),
            ));
        }
        // Pre-allocation bound: fixed buffers + per-cell runs (len u64 +
        // slots u32 + codes u8*m per vector) + vectors + trailing CRC.
        let runs_fixed: u128 = nlist as u128 * 8;
        let payload: u128 = len as u128 * (4 + m as u128);
        let fixed: u128 =
            coarse_len as u128 * 4 + codebook_len as u128 * 4 + vectors_len as u128 * 4;
        let capacity = (data.len() as u128)
            .saturating_sub(HEADER_LEN as u128)
            .saturating_sub(4);
        if runs_fixed + payload + fixed > capacity {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ snapshot body truncated".to_owned(),
            ));
        }
        let crc_start = data.len() - 4;
        let body = &data[HEADER_LEN..crc_start];
        let body_crc = (&data[crc_start..]).read_u32::<LittleEndian>()?;
        if crc32fast::hash(body) != body_crc {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ snapshot body checksum mismatch".to_owned(),
            ));
        }
        let mut bcur = Cursor::new(body);
        let mut coarse = vec![0.0f32; coarse_len];
        for v in &mut coarse {
            *v = bcur.read_f32::<LittleEndian>()?;
        }
        if coarse.iter().any(|v| !v.is_finite()) {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ coarse centroids must be finite".to_owned(),
            ));
        }
        let mut codebooks = vec![0.0f32; codebook_len];
        for v in &mut codebooks {
            *v = bcur.read_f32::<LittleEndian>()?;
        }
        let pq = ProductQuantizer::from_codebooks(m, ksub, dsub, codebooks)?;
        let mut list_slots: Vec<Vec<u32>> = Vec::with_capacity(nlist);
        let mut list_codes: Vec<Vec<u8>> = Vec::with_capacity(nlist);
        let mut total = 0usize;
        for _ in 0..nlist {
            let cell_len = bcur.read_u64::<LittleEndian>()? as usize;
            if cell_len > len || total + cell_len > len {
                return Err(QuiverError::InvalidFormat(
                    "IVF-PQ cell run exceeds index length".to_owned(),
                ));
            }
            let mut slots = vec![0u32; cell_len];
            for s in &mut slots {
                *s = bcur.read_u32::<LittleEndian>()?;
                if *s as usize >= len {
                    return Err(QuiverError::InvalidFormat(
                        "IVF-PQ slot out of range".to_owned(),
                    ));
                }
            }
            let mut codes = vec![0u8; cell_len * m];
            bcur.read_exact(&mut codes)?;
            for &c in &codes {
                if c as usize >= ksub {
                    return Err(QuiverError::InvalidFormat(
                        "IVF-PQ code out of range".to_owned(),
                    ));
                }
            }
            total += cell_len;
            list_slots.push(slots);
            list_codes.push(codes);
        }
        if total != len {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ cell runs do not sum to index length".to_owned(),
            ));
        }
        let mut vectors = vec![0.0f32; vectors_len];
        for v in &mut vectors {
            *v = bcur.read_f32::<LittleEndian>()?;
        }
        if vectors.iter().any(|v| !v.is_finite()) {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ stored vectors must be finite".to_owned(),
            ));
        }
        if bcur.position() != body.len() as u64 {
            return Err(QuiverError::InvalidFormat(
                "IVF-PQ snapshot has trailing bytes".to_owned(),
            ));
        }
        Ok(Self {
            dimension,
            nlist,
            pq,
            coarse,
            list_slots,
            list_codes,
            vectors,
            len,
        })
    }

    fn nearest_cells(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut dists: Vec<(f32, usize)> = (0..self.nlist)
            .map(|cell| {
                let start = cell * self.dimension;
                (
                    l2_squared(query, &self.coarse[start..start + self.dimension]),
                    cell,
                )
            })
            .collect();
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        dists
            .into_iter()
            .take(nprobe)
            .map(|(_, cell)| cell)
            .collect()
    }
}

fn nearest_coarse(vector: &[f32], coarse: &[f32], nlist: usize, dimension: usize) -> u32 {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for cell in 0..nlist {
        let start = cell * dimension;
        let dist = l2_squared(vector, &coarse[start..start + dimension]);
        if dist < best_dist {
            best_dist = dist;
            best = cell;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::distance::compute_distance;
    use rand::Rng;
    use std::collections::HashSet;

    fn blobs(rng: &mut rand::rngs::StdRng, n: usize, d: usize, centers: usize) -> Vec<Vec<f32>> {
        let centroids: Vec<Vec<f32>> = (0..centers)
            .map(|_| (0..d).map(|_| rng.random_range(-30.0..30.0)).collect())
            .collect();
        (0..n)
            .map(|_| {
                let c = &centroids[rng.random_range(0..centers)];
                c.iter().map(|&v| v + rng.random_range(-1.0..1.0)).collect()
            })
            .collect()
    }

    fn recall(results: &[SearchResult], truth: &HashSet<usize>, k: usize) -> f32 {
        let found: HashSet<usize> = results.iter().map(|r| r.slot).collect();
        found.intersection(truth).count() as f32 / k as f32
    }

    fn brute_truth(vectors: &[Vec<f32>], query: &[f32], k: usize) -> HashSet<usize> {
        let mut scored: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, compute_distance(query, v, Metric::L2)))
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.iter().take(k).map(|(i, _)| *i).collect()
    }

    #[test]
    fn build_and_search_returns_nearest() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let vectors = blobs(&mut rng, 600, 16, 6);
        let config = IvfPqConfig::new(8, 4, 64);
        let index = IvfPqIndex::build(&vectors, &config).unwrap();
        assert_eq!(index.len(), 600);
        assert_eq!(index.dimension(), 16);

        let results = index.search(&vectors[42], 5, 8, 0).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(
            results[0].slot, 42,
            "exact query should return itself first"
        );
        assert!(results.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn recall_improves_with_nprobe_and_rerank() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let vectors = blobs(&mut rng, 2000, 32, 16);
        let config = IvfPqConfig::new(16, 8, 256);
        let index = IvfPqIndex::build(&vectors, &config).unwrap();

        let queries: Vec<Vec<f32>> = (0..20)
            .map(|_| blobs(&mut rng, 1, 32, 16)[0].clone())
            .collect();
        let k = 10;

        let mut low = 0.0f32;
        let mut high = 0.0f32;
        let mut reranked = 0.0f32;
        for query in &queries {
            let truth = brute_truth(&vectors, query, k);
            low += recall(&index.search(query, k, 1, 0).unwrap(), &truth, k);
            high += recall(&index.search(query, k, 16, 0).unwrap(), &truth, k);
            reranked += recall(&index.search(query, k, 16, 4).unwrap(), &truth, k);
        }
        let n = queries.len() as f32;
        let (low, high, reranked) = (low / n, high / n, reranked / n);
        assert!(
            high >= low - 0.01,
            "more probes should not reduce recall: {low} -> {high}"
        );
        assert!(
            reranked >= high - 0.01,
            "rerank should not reduce recall: {high} -> {reranked}"
        );
        assert!(
            reranked >= 0.8,
            "full-probe reranked recall too low: {reranked}"
        );
    }

    #[test]
    fn full_probe_rerank_recovers_exact_neighbors() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let vectors = blobs(&mut rng, 1500, 32, 12);
        let config = IvfPqConfig::new(12, 8, 256);
        let index = IvfPqIndex::build(&vectors, &config).unwrap();

        let mut total = 0.0f32;
        for i in (0..1500).step_by(100) {
            let truth = brute_truth(&vectors, &vectors[i], 10);
            let results = index.search(&vectors[i], 10, 12, 10).unwrap();
            total += recall(&results, &truth, 10);
        }
        let mean = total / 15.0;
        assert!(mean >= 0.95, "full-probe rerank recall too low: {mean}");
    }

    #[test]
    fn code_payload_is_m_bytes_per_vector() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let vectors = blobs(&mut rng, 500, 32, 8);
        let config = IvfPqConfig::new(8, 8, 256);
        let index = IvfPqIndex::build(&vectors, &config).unwrap();
        assert_eq!(index.code_bytes(), 500 * 8);
        // 8 bytes per vector vs 128 bytes of f32 => 16x smaller payload.
        assert_eq!(index.code_bytes() * 16, 500 * 32 * size_of::<f32>());
    }

    #[test]
    fn store_vectors_false_disables_rerank_but_searches() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let vectors = blobs(&mut rng, 400, 16, 6);
        let mut config = IvfPqConfig::new(6, 4, 64);
        config.store_vectors = false;
        let index = IvfPqIndex::build(&vectors, &config).unwrap();
        assert_eq!(index.vector_bytes(), 0);
        assert!(index.search(&vectors[0], 3, 6, 0).is_ok());
        assert!(index.search(&vectors[0], 3, 6, 2).is_err());
    }

    #[test]
    fn rejects_bad_inputs() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(6);
        let vectors = blobs(&mut rng, 100, 16, 4);
        assert!(IvfPqIndex::build(&[], &IvfPqConfig::new(4, 4, 16)).is_err());
        // dimension not divisible by m
        assert!(IvfPqIndex::build(&vectors, &IvfPqConfig::new(4, 3, 16)).is_err());
        // nlist zero
        assert!(IvfPqIndex::build(&vectors, &IvfPqConfig::new(0, 4, 16)).is_err());
        // m zero used to panic in is_multiple_of(0); it must be a clean error
        assert!(IvfPqIndex::build(&vectors, &IvfPqConfig::new(4, 0, 16)).is_err());
        // fewer training vectors than ksub used to panic inside PQ::train
        let small = blobs(&mut rng, 10, 16, 2);
        assert!(IvfPqIndex::build(&small, &IvfPqConfig::new(4, 4, 64)).is_err());

        let index = IvfPqIndex::build(&vectors, &IvfPqConfig::new(4, 4, 16)).unwrap();
        assert!(index.search(&[0.0; 15], 1, 1, 0).is_err()); // wrong dim
        assert!(index.search(&[f32::NAN; 16], 1, 1, 0).is_err()); // non-finite
        assert!(index.search(&vectors[0], 0, 1, 0).unwrap().is_empty());
    }

    #[test]
    fn from_trained_validates_external_state() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let vectors = blobs(&mut rng, 16, 8, 2);
        let pq =
            crate::quantization::ProductQuantizer::train(&vectors, 2, 16, 4, &mut rng).unwrap();
        let dim = 8;
        let nlist = 2;
        let n = vectors.len();
        let coarse = vec![0.0f32; nlist * dim];
        let assignments = vec![0u32; n];
        let codes = vec![0u8; n * 2];

        // Short stored vector: rerank used to index past the end.
        let mut short = vectors.clone();
        short[1].pop();
        let result = IvfPqIndex::from_trained(
            dim,
            nlist,
            pq.clone(),
            coarse.clone(),
            assignments.clone(),
            codes.clone(),
            short,
        );
        assert!(result.is_err());

        // PQ trained on a different dimension than the index claims.
        let result = IvfPqIndex::from_trained(
            16,
            nlist,
            pq.clone(),
            coarse.clone(),
            assignments.clone(),
            codes.clone(),
            vectors.clone(),
        );
        assert!(result.is_err());

        // Out-of-range code byte: the ADC lookup used to panic.
        let mut bad_codes = codes.clone();
        bad_codes[0] = 99;
        let result =
            IvfPqIndex::from_trained(dim, nlist, pq, coarse, assignments, bad_codes, vectors);
        assert!(result.is_err());
    }

    fn tmp_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let vectors = blobs(&mut rng, 300, 16, 4);
        let config = IvfPqConfig::new(4, 4, 16);
        let index = IvfPqIndex::build(&vectors, &config).unwrap();
        let dir = tmp_dir("quiver-ivfpq");
        let path = dir.join("index.qvpq");
        index.save(&path).unwrap();
        let loaded = IvfPqIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 300);
        assert_eq!(loaded.dimension(), 16);
        let a = index.search(&vectors[7], 5, 4, 2).unwrap();
        let b = loaded.search(&vectors[7], 5, 4, 2).unwrap();
        assert_eq!(a, b);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_rejects_corrupt_snapshot() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        let vectors = blobs(&mut rng, 100, 8, 2);
        let index = IvfPqIndex::build(&vectors, &IvfPqConfig::new(2, 2, 8)).unwrap();
        let dir = tmp_dir("quiver-ivfpq-bad");
        let path = dir.join("index.qvpq");
        index.save(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert!(IvfPqIndex::load(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
