//! Brute-force (flat) index — exact nearest neighbor search.
//!
//! Performs a linear scan over all stored vectors and returns the top-K
//! closest matches using a binary max-heap. This is the simplest possible
//! index and serves two purposes:
//!
//! 1. **Ground truth** for measuring recall of approximate indexes (HNSW, IVF-PQ).
//! 2. **Fallback** for small datasets where the overhead of graph construction isn't justified.
//!
//! ## Complexity
//!
//! - **Search**: O(n·d) where n = vector count, d = dimension
//! - **Insert**: O(d) amortized (append to mmap'd file)

use std::collections::BinaryHeap;
use std::path::Path;

use crate::distance::{compute_distance, Metric};
use crate::error::{QuiverError, Result};
use crate::index::SearchResult;
use crate::storage::vecstore::VectorStore;

/// A brute-force (flat) index that performs exact nearest neighbor search.
pub struct BruteForceIndex {
    store: VectorStore,
}

impl BruteForceIndex {
    /// Create a new brute-force index backed by files at the given paths.
    pub fn create(
        data_path: impl AsRef<Path>,
        wal_path: impl AsRef<Path>,
        dimension: u32,
        metric: Metric,
    ) -> Result<Self> {
        let store = VectorStore::create(data_path, wal_path, dimension, metric)?;
        Ok(Self { store })
    }

    /// Open an existing brute-force index, replaying the WAL for crash recovery.
    pub fn open(
        data_path: impl AsRef<Path>,
        wal_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let store = VectorStore::open(data_path, wal_path)?;
        Ok(Self { store })
    }

    /// Insert a vector into the index.
    ///
    /// Returns the assigned vector ID.
    pub fn insert(&mut self, vector: &[f32]) -> Result<u64> {
        self.store.insert(vector)
    }

    /// Search for the `k` nearest neighbors of the query vector.
    ///
    /// Returns results sorted by distance (closest first).
    /// Returns an error if the index is empty.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.store.dimension() as usize {
            return Err(QuiverError::DimensionMismatch {
                expected: self.store.dimension(),
                actual: query.len() as u32,
            });
        }

        if self.store.is_empty() {
            return Err(QuiverError::EmptyIndex);
        }

        let metric = self.store.metric();
        let k = k.min(self.store.len()); // Can't return more than we have

        // Max-heap of size k: the worst (largest distance) result sits on top.
        // When we find something better, we pop the worst and push the new one.
        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k + 1);

        for (slot, vector) in self.store.iter() {
            let distance = compute_distance(query, vector, metric);

            if heap.len() < k {
                heap.push(SearchResult {
                    slot,
                    vector_id: slot as u64 + 1, // IDs are 1-based
                    distance,
                });
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(SearchResult {
                        slot,
                        vector_id: slot as u64 + 1,
                        distance,
                    });
                }
            }
        }

        // Drain the heap into a sorted vec (closest first)
        let mut results: Vec<SearchResult> = heap.into_sorted_vec();
        // into_sorted_vec returns ascending order by our Ord impl, which is
        // lowest distance first — exactly what we want.
        // However, BinaryHeap::into_sorted_vec sorts ascending by the Ord,
        // which puts smallest distance first. That's correct for us.
        results.truncate(k);
        Ok(results)
    }

    /// Return the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Return the vector dimension.
    pub fn dimension(&self) -> u32 {
        self.store.dimension()
    }

    /// Return the distance metric.
    pub fn metric(&self) -> Metric {
        self.store.metric()
    }

    /// Flush the index to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.store.flush()
    }

    /// Get a vector by slot index (for testing/verification).
    pub fn get_vector(&self, slot: usize) -> Result<&[f32]> {
        self.store.get_vector(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup(dim: u32, metric: Metric) -> (TempDir, BruteForceIndex) {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("bf_vectors.qvdb");
        let wal_path = dir.path().join("bf_vectors.wal");
        let index = BruteForceIndex::create(data_path, wal_path, dim, metric).unwrap();
        (dir, index)
    }

    #[test]
    fn test_create_empty() {
        let (_dir, index) = setup(3, Metric::L2);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.dimension(), 3);
    }

    #[test]
    fn test_insert_and_len() {
        let (_dir, mut index) = setup(3, Metric::L2);
        let id1 = index.insert(&[1.0, 0.0, 0.0]).unwrap();
        let id2 = index.insert(&[0.0, 1.0, 0.0]).unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_search_empty_returns_error() {
        let (_dir, index) = setup(3, Metric::L2);
        let result = index.search(&[1.0, 0.0, 0.0], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let (_dir, mut index) = setup(3, Metric::L2);
        index.insert(&[1.0, 0.0, 0.0]).unwrap();
        let result = index.search(&[1.0, 0.0], 5); // wrong dim
        assert!(result.is_err());
    }

    #[test]
    fn test_search_l2_returns_closest() {
        let (_dir, mut index) = setup(3, Metric::L2);

        // Insert vectors at known positions
        index.insert(&[1.0, 0.0, 0.0]).unwrap(); // slot 0: on x-axis
        index.insert(&[0.0, 1.0, 0.0]).unwrap(); // slot 1: on y-axis
        index.insert(&[0.0, 0.0, 1.0]).unwrap(); // slot 2: on z-axis
        index.insert(&[0.5, 0.5, 0.0]).unwrap(); // slot 3: between x and y

        // Query near x-axis — closest should be slot 0, then slot 3
        let results = index.search(&[0.9, 0.1, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].slot, 0); // [1,0,0] is closest to [0.9, 0.1, 0]
        assert_eq!(results[1].slot, 3); // [0.5,0.5,0] is second closest
    }

    #[test]
    fn test_search_cosine_returns_most_similar() {
        let (_dir, mut index) = setup(2, Metric::Cosine);

        index.insert(&[1.0, 0.0]).unwrap();  // slot 0: pointing right
        index.insert(&[0.0, 1.0]).unwrap();  // slot 1: pointing up
        index.insert(&[-1.0, 0.0]).unwrap(); // slot 2: pointing left

        // Query pointing right — most similar should be slot 0
        let results = index.search(&[1.0, 0.1], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].slot, 0);
    }

    #[test]
    fn test_search_k_larger_than_n() {
        let (_dir, mut index) = setup(2, Metric::L2);
        index.insert(&[1.0, 0.0]).unwrap();
        index.insert(&[0.0, 1.0]).unwrap();

        // Ask for k=10 but only 2 vectors exist
        let results = index.search(&[0.5, 0.5], 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_results_sorted_by_distance() {
        let (_dir, mut index) = setup(2, Metric::L2);

        for i in 0..20 {
            index.insert(&[i as f32, 0.0]).unwrap();
        }

        let results = index.search(&[10.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 5);

        // Verify distances are non-decreasing (sorted closest first)
        for window in results.windows(2) {
            assert!(
                window[0].distance <= window[1].distance,
                "Results not sorted: {} > {}",
                window[0].distance,
                window[1].distance
            );
        }
    }

    #[test]
    fn test_exact_recall_at_10() {
        // Brute-force must have perfect recall — it's exact search.
        // Generate a random dataset and verify that search returns the actual
        // nearest neighbors (compared against a naive reference).
        use rand::Rng;
        let mut rng = rand::rng();

        let dim = 32;
        let n = 500;
        let k = 10;

        let (_dir, mut index) = setup(dim, Metric::L2);

        // Generate and insert random vectors
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            index.insert(&v).unwrap();
            vectors.push(v);
        }

        // Generate a random query
        let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

        // Compute ground truth: sort all vectors by L2 distance
        let mut ground_truth: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let dist = crate::distance::l2_squared(&query, v);
                (i, dist)
            })
            .collect();
        ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let expected_slots: Vec<usize> = ground_truth.iter().take(k).map(|(i, _)| *i).collect();

        // Search
        let results = index.search(&query, k).unwrap();
        let result_slots: Vec<usize> = results.iter().map(|r| r.slot).collect();

        assert_eq!(
            result_slots, expected_slots,
            "Brute-force recall must be 100% — it's exact search"
        );
    }

    #[test]
    fn test_dot_product_search() {
        let (_dir, mut index) = setup(3, Metric::DotProduct);

        index.insert(&[1.0, 0.0, 0.0]).unwrap(); // slot 0
        index.insert(&[0.0, 1.0, 0.0]).unwrap(); // slot 1
        index.insert(&[1.0, 1.0, 0.0]).unwrap(); // slot 2

        // Query [1, 1, 0]: dot with [1,1,0] = 2, dot with [1,0,0] = 1, dot with [0,1,0] = 1
        // compute_distance negates dot, so [1,1,0] has distance -2 (best)
        let results = index.search(&[1.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(results[0].slot, 2);
    }

    #[test]
    fn test_persistence_and_reopen() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("bf_persist.qvdb");
        let wal_path = dir.path().join("bf_persist.wal");

        // Create, insert, flush
        {
            let mut index =
                BruteForceIndex::create(&data_path, &wal_path, 3, Metric::L2).unwrap();
            index.insert(&[1.0, 0.0, 0.0]).unwrap();
            index.insert(&[0.0, 1.0, 0.0]).unwrap();
            index.flush().unwrap();
        }

        // Reopen and verify search still works
        {
            let index = BruteForceIndex::open(&data_path, &wal_path).unwrap();
            assert_eq!(index.len(), 2);
            let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
            assert_eq!(results[0].slot, 0);
        }
    }
}
