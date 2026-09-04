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
            return Err(QuiverError::InvalidFormat(
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
}
