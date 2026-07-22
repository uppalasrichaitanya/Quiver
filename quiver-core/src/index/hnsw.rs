//! HNSW (Hierarchical Navigable Small World) index.
//!
//! Implements the HNSW algorithm from:
//! Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
//! using Hierarchical Navigable Small World graphs" (2016, revised 2018).
//!
//! ## Architecture
//!
//! - Multi-layer skip-list-like graph with exponential-decay layer assignment
//! - Each node stores its neighbor lists contiguously for cache-friendly traversal
//! - Insert: greedy search from entry point down through layers, then connect to M nearest per layer
//! - Search: greedy descent to layer 0, then beam search with `ef_search` candidates
//! - Delete: tombstone approach with compaction trigger at configurable tombstone ratio
//!
//! ## Concurrency model (v1)
//!
//! Single-writer, multi-reader via `parking_lot::RwLock`. Lock-free reads are a stretch goal.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;

use parking_lot::RwLock;
use rand::Rng;

use crate::distance::{compute_distance, Metric};
use crate::error::{QuiverError, Result};
use crate::index::SearchResult;
use crate::storage::vecstore::VectorStore;

/// HNSW tuning parameters.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Maximum number of connections per node per layer (M in the paper).
    /// Higher M = better recall but more memory and slower insert.
    pub m: usize,
    /// Maximum connections for layer 0 (typically 2*M).
    pub m_max0: usize,
    /// Size of the dynamic candidate list during construction.
    /// Higher = better graph quality but slower insert.
    pub ef_construction: usize,
    /// Normalization factor for layer assignment: 1/ln(M).
    pub ml: f64,
    /// Maximum tombstone ratio before triggering compaction (e.g., 0.2 = 20%).
    pub max_tombstone_ratio: f64,
}

impl HnswConfig {
    /// Create a config with the given M parameter. Other parameters are derived.
    pub fn new(m: usize) -> Self {
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ml: 1.0 / (m as f64).ln(),
            max_tombstone_ratio: 0.2,
        }
    }

    /// Set ef_construction (builder pattern).
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self::new(16)
    }
}

/// A node in the HNSW graph.
#[derive(Debug, Clone)]
struct HnswNode {
    /// The slot index of this vector in the VectorStore.
    slot: usize,
    /// The vector ID as assigned at insert time.
    vector_id: u64,
    /// The maximum layer this node belongs to (0-indexed).
    max_layer: usize,
    /// Neighbor lists for each layer. neighbors[l] contains the neighbor node indices
    /// (indices into HnswIndex::nodes, not slot indices).
    neighbors: Vec<Vec<usize>>,
    /// Whether this node has been deleted (tombstone).
    deleted: bool,
}

/// A candidate during search: node index + distance.
#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    node_idx: usize,
    distance: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// The HNSW index.
pub struct HnswIndex {
    /// The underlying vector storage.
    store: VectorStore,
    /// The HNSW graph nodes.
    nodes: Vec<HnswNode>,
    /// Index of the current entry point node (topmost layer).
    entry_point: Option<usize>,
    /// Maximum layer currently in the graph.
    max_level: usize,
    /// Configuration parameters.
    config: HnswConfig,
    /// Number of tombstoned (deleted) nodes.
    tombstone_count: usize,
    /// RwLock for concurrent access (v1: wraps the whole index).
    _lock: RwLock<()>,
}

impl HnswIndex {
    /// Create a new HNSW index.
    pub fn create(
        data_path: impl AsRef<Path>,
        wal_path: impl AsRef<Path>,
        dimension: u32,
        metric: Metric,
        config: HnswConfig,
    ) -> Result<Self> {
        let store = VectorStore::create(data_path, wal_path, dimension, metric)?;
        Ok(Self {
            store,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            config,
            tombstone_count: 0,
            _lock: RwLock::new(()),
        })
    }

    /// Open an existing HNSW index and rebuild the graph from stored vectors.
    ///
    /// Since the HNSW graph structure is not persisted (v1 simplification),
    /// this re-inserts all vectors into a fresh graph after WAL replay.
    pub fn open(
        data_path: impl AsRef<Path>,
        wal_path: impl AsRef<Path>,
        config: HnswConfig,
    ) -> Result<Self> {
        let store = VectorStore::open(data_path, wal_path)?;
        let mut index = Self {
            store,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            config,
            tombstone_count: 0,
            _lock: RwLock::new(()),
        };

        // Rebuild graph from stored vectors
        let n = index.store.len();
        if n > 0 {
            tracing::info!(count = n, "Rebuilding HNSW graph from stored vectors");
            for slot in 0..n {
                let vector = index.store.get_vector(slot)?.to_vec();
                let vector_id = slot as u64 + 1;
                index.insert_into_graph(slot, vector_id, &vector);
            }
        }

        Ok(index)
    }

    /// Insert a vector into the index.
    ///
    /// Returns the assigned vector ID.
    pub fn insert(&mut self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.store.dimension() as usize {
            return Err(QuiverError::DimensionMismatch {
                expected: self.store.dimension(),
                actual: vector.len() as u32,
            });
        }

        // Write to storage (with WAL)
        let vector_id = self.store.insert(vector)?;
        let slot = self.store.len() - 1;

        // Insert into the HNSW graph
        self.insert_into_graph(slot, vector_id, vector);

        Ok(vector_id)
    }

    /// Search for the `k` nearest neighbors of the query vector.
    ///
    /// `ef_search` controls the size of the dynamic candidate list. Higher values
    /// give better recall at the cost of speed. Must be >= k.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.store.dimension() as usize {
            return Err(QuiverError::DimensionMismatch {
                expected: self.store.dimension(),
                actual: query.len() as u32,
            });
        }

        let entry_point = match self.entry_point {
            Some(ep) => ep,
            None => return Err(QuiverError::EmptyIndex),
        };

        let ef = ef_search.max(k);

        // Phase 1: Greedy descent from the entry point through upper layers
        let mut current = entry_point;
        let metric = self.store.metric();

        let ep_vector = self.store.get_vector(self.nodes[current].slot)?;
        let mut current_dist = compute_distance(query, ep_vector, metric);

        for level in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = &self.nodes[current].neighbors[level];
                for &neighbor_idx in neighbors {
                    if self.nodes[neighbor_idx].deleted {
                        continue;
                    }
                    let neighbor_vec = self.store.get_vector(self.nodes[neighbor_idx].slot)?;
                    let dist = compute_distance(query, neighbor_vec, metric);
                    if dist < current_dist {
                        current = neighbor_idx;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
        }

        // Phase 2: Beam search at layer 0 with ef candidates
        let candidates = self.search_layer(query, current, ef, 0, metric)?;

        // Take top-k results
        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|c| !self.nodes[c.node_idx].deleted)
            .take(k)
            .map(|c| SearchResult {
                slot: self.nodes[c.node_idx].slot,
                vector_id: self.nodes[c.node_idx].vector_id,
                distance: c.distance,
            })
            .collect();

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(results)
    }

    /// Mark a vector as deleted (tombstone).
    pub fn delete(&mut self, vector_id: u64) -> Result<()> {
        let node_idx = self
            .nodes
            .iter()
            .position(|n| n.vector_id == vector_id && !n.deleted)
            .ok_or(QuiverError::NotFound(vector_id))?;

        self.nodes[node_idx].deleted = true;
        self.tombstone_count += 1;

        // Check if compaction is needed
        if !self.nodes.is_empty() {
            let ratio = self.tombstone_count as f64 / self.nodes.len() as f64;
            if ratio > self.config.max_tombstone_ratio {
                tracing::info!(
                    tombstone_ratio = ratio,
                    threshold = self.config.max_tombstone_ratio,
                    "Tombstone ratio exceeded threshold — compaction recommended"
                );
                // Full compaction (rebuild from live vectors) is deferred to a
                // future PR — for now we just log the recommendation.
            }
        }

        Ok(())
    }

    /// Return the number of live (non-deleted) vectors.
    pub fn len(&self) -> usize {
        self.nodes.iter().filter(|n| !n.deleted).count()
    }

    /// Return true if the index has no live vectors.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the vector dimension.
    pub fn dimension(&self) -> u32 {
        self.store.dimension()
    }

    /// Return the metric.
    pub fn metric(&self) -> Metric {
        self.store.metric()
    }

    /// Flush the underlying storage to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.store.flush()
    }

    /// Return the total number of nodes (including tombstoned).
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Return the current max level of the graph.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Assign a random layer for a new node using the exponential decay formula.
    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let r: f64 = rng.random();
        let level = (-r.ln() * self.config.ml).floor() as usize;
        level
    }

    /// Insert a vector into the HNSW graph (assumes it's already in the store).
    fn insert_into_graph(&mut self, slot: usize, vector_id: u64, vector: &[f32]) {
        let new_level = self.random_level();
        let metric = self.store.metric();

        // Create the new node
        let m_max0 = self.config.m_max0;
        let m = self.config.m;
        let mut neighbors = Vec::with_capacity(new_level + 1);
        for _ in 0..=new_level {
            neighbors.push(Vec::new());
        }
        let new_node = HnswNode {
            slot,
            vector_id,
            max_layer: new_level,
            neighbors,
            deleted: false,
        };
        let new_node_idx = self.nodes.len();
        self.nodes.push(new_node);

        // First node — set as entry point and return
        if self.entry_point.is_none() {
            self.entry_point = Some(new_node_idx);
            self.max_level = new_level;
            return;
        }

        let entry_point = self.entry_point.unwrap();
        let mut current = entry_point;

        // Compute distance from the new vector to the current entry point
        let ep_vec = match self.store.get_vector(self.nodes[current].slot) {
            Ok(v) => v,
            Err(_) => return,
        };
        let mut current_dist = compute_distance(vector, ep_vec, metric);

        // Phase 1: Greedy descent through layers above the new node's level
        for level in (new_level + 1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                // Clone the neighbor list to avoid borrow conflict
                let neighbors: Vec<usize> =
                    if level <= self.nodes[current].max_layer {
                        self.nodes[current].neighbors[level].clone()
                    } else {
                        Vec::new()
                    };
                for neighbor_idx in neighbors {
                    let n_vec = match self.store.get_vector(self.nodes[neighbor_idx].slot) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    let dist = compute_distance(vector, n_vec, metric);
                    if dist < current_dist {
                        current = neighbor_idx;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
        }

        // Phase 2: Insert at each layer from min(new_level, max_level) down to 0
        let insert_from = new_level.min(self.max_level);
        for level in (0..=insert_from).rev() {
            let ef = self.config.ef_construction;
            let candidates = match self.search_layer_for_insert(vector, current, ef, level, metric) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Select M nearest non-deleted neighbors
            let m_level = if level == 0 { m_max0 } else { m };
            let selected: Vec<usize> = candidates
                .iter()
                .filter(|c| !self.nodes[c.node_idx].deleted && c.node_idx != new_node_idx)
                .take(m_level)
                .map(|c| c.node_idx)
                .collect();

            // Set forward connections (new_node -> selected neighbors)
            self.nodes[new_node_idx].neighbors[level] = selected.clone();

            // Set reverse connections (each selected neighbor -> new_node)
            for &neighbor_idx in &selected {
                if level <= self.nodes[neighbor_idx].max_layer {
                    self.nodes[neighbor_idx].neighbors[level].push(new_node_idx);

                    // Prune if over capacity
                    let max_conn = if level == 0 { m_max0 } else { m };
                    if self.nodes[neighbor_idx].neighbors[level].len() > max_conn {
                        self.prune_connections(neighbor_idx, level, max_conn, vector, metric);
                    }
                }
            }

            // Update the entry point for the next layer
            if let Some(best) = candidates.first() {
                current = best.node_idx;
            }
        }

        // Update entry point if new node has a higher layer
        if new_level > self.max_level {
            self.entry_point = Some(new_node_idx);
            self.max_level = new_level;
        }
    }

    /// Beam search at a given layer. Returns candidates sorted by distance (closest first).
    fn search_layer(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        level: usize,
        metric: Metric,
    ) -> Result<Vec<Candidate>> {
        let ep_vec = self.store.get_vector(self.nodes[entry_point].slot)?;
        let ep_dist = compute_distance(query, ep_vec, metric);

        let mut visited = HashSet::new();
        visited.insert(entry_point);

        // Min-heap: closest candidates to explore
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        candidates.push(Reverse(Candidate {
            node_idx: entry_point,
            distance: ep_dist,
        }));

        // Max-heap: best results found so far (worst on top for eviction)
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        results.push(Candidate {
            node_idx: entry_point,
            distance: ep_dist,
        });

        while let Some(Reverse(current)) = candidates.pop() {
            // If the closest candidate is further than the worst result, stop
            let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::MAX);
            if current.distance > worst_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            let neighbors = if level <= self.nodes[current.node_idx].max_layer {
                &self.nodes[current.node_idx].neighbors[level]
            } else {
                continue;
            };

            for &neighbor_idx in neighbors {
                if !visited.insert(neighbor_idx) {
                    continue;
                }

                let n_vec = self.store.get_vector(self.nodes[neighbor_idx].slot)?;
                let dist = compute_distance(query, n_vec, metric);
                let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::MAX);

                if dist < worst_dist || results.len() < ef {
                    candidates.push(Reverse(Candidate {
                        node_idx: neighbor_idx,
                        distance: dist,
                    }));
                    results.push(Candidate {
                        node_idx: neighbor_idx,
                        distance: dist,
                    });
                    if results.len() > ef {
                        results.pop(); // evict worst
                    }
                }
            }
        }

        // Convert to sorted vec (closest first)
        let mut sorted: Vec<Candidate> = results.into_vec();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(sorted)
    }

    /// Search layer for insertion — same as search_layer but without the Result overhead
    /// for the internal insert path.
    fn search_layer_for_insert(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        level: usize,
        metric: Metric,
    ) -> Result<Vec<Candidate>> {
        self.search_layer(query, entry_point, ef, level, metric)
    }

    /// Prune the connection list of a node at the given layer to max_conn connections.
    /// Keeps the closest neighbors by distance to the node's own vector.
    fn prune_connections(
        &mut self,
        node_idx: usize,
        level: usize,
        max_conn: usize,
        _new_vector: &[f32],
        metric: Metric,
    ) {
        let node_slot = self.nodes[node_idx].slot;
        let node_vec = match self.store.get_vector(node_slot) {
            Ok(v) => v.to_vec(),
            Err(_) => return,
        };

        // Score all current neighbors by distance to this node
        let mut scored: Vec<(usize, f32)> = self.nodes[node_idx].neighbors[level]
            .iter()
            .filter_map(|&n_idx| {
                let n_vec = self.store.get_vector(self.nodes[n_idx].slot).ok()?;
                let dist = compute_distance(&node_vec, n_vec, metric);
                Some((n_idx, dist))
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(max_conn);

        self.nodes[node_idx].neighbors[level] = scored.into_iter().map(|(idx, _)| idx).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup(dim: u32, metric: Metric, m: usize) -> (TempDir, HnswIndex) {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("hnsw_vectors.qvdb");
        let wal_path = dir.path().join("hnsw_vectors.wal");
        let config = HnswConfig::new(m).with_ef_construction(100);
        let index = HnswIndex::create(data_path, wal_path, dim, metric, config).unwrap();
        (dir, index)
    }

    #[test]
    fn test_create_empty() {
        let (_dir, index) = setup(3, Metric::L2, 16);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert_eq!(index.dimension(), 3);
    }

    #[test]
    fn test_insert_single() {
        let (_dir, mut index) = setup(3, Metric::L2, 16);
        let id = index.insert(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(id, 1);
        assert_eq!(index.len(), 1);
        assert!(index.entry_point.is_some());
    }

    #[test]
    fn test_insert_multiple() {
        let (_dir, mut index) = setup(3, Metric::L2, 4);
        for i in 0..20 {
            index.insert(&[i as f32, 0.0, 0.0]).unwrap();
        }
        assert_eq!(index.len(), 20);
    }

    #[test]
    fn test_search_empty() {
        let (_dir, index) = setup(3, Metric::L2, 16);
        assert!(index.search(&[1.0, 0.0, 0.0], 5, 50).is_err());
    }

    #[test]
    fn test_search_basic_l2() {
        let (_dir, mut index) = setup(3, Metric::L2, 16);

        index.insert(&[1.0, 0.0, 0.0]).unwrap();
        index.insert(&[0.0, 1.0, 0.0]).unwrap();
        index.insert(&[0.0, 0.0, 1.0]).unwrap();
        index.insert(&[0.5, 0.5, 0.0]).unwrap();

        let results = index.search(&[0.9, 0.1, 0.0], 2, 50).unwrap();
        assert_eq!(results.len(), 2);
        // Closest should be [1, 0, 0] (slot 0)
        assert_eq!(results[0].slot, 0);
    }

    #[test]
    fn test_search_cosine() {
        let (_dir, mut index) = setup(2, Metric::Cosine, 16);

        index.insert(&[1.0, 0.0]).unwrap();
        index.insert(&[0.0, 1.0]).unwrap();
        index.insert(&[-1.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.1], 1, 50).unwrap();
        assert_eq!(results[0].slot, 0);
    }

    #[test]
    fn test_delete_tombstone() {
        let (_dir, mut index) = setup(3, Metric::L2, 16);

        let id1 = index.insert(&[1.0, 0.0, 0.0]).unwrap();
        index.insert(&[0.0, 1.0, 0.0]).unwrap();
        index.insert(&[0.0, 0.0, 1.0]).unwrap();

        assert_eq!(index.len(), 3);
        index.delete(id1).unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.total_nodes(), 3); // Node still exists as tombstone

        // Searching should not return the deleted vector
        let results = index.search(&[1.0, 0.0, 0.0], 3, 50).unwrap();
        assert!(results.iter().all(|r| r.vector_id != id1));
    }

    #[test]
    fn test_delete_nonexistent() {
        let (_dir, mut index) = setup(3, Metric::L2, 16);
        index.insert(&[1.0, 0.0, 0.0]).unwrap();
        assert!(index.delete(999).is_err());
    }

    #[test]
    fn test_recall_against_brute_force() {
        // The core correctness test: HNSW recall@10 must exceed 95% against
        // brute-force ground truth on a random 1000-vector dataset.
        use rand::Rng;
        let mut rng = rand::rng();

        let dim = 32;
        let n = 1000;
        let k = 10;
        let ef_search = 100; // generous ef for high recall
        let num_queries = 50;

        let (_dir, mut index) = setup(dim as u32, Metric::L2, 16);

        // Insert random vectors
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            index.insert(&v).unwrap();
            vectors.push(v);
        }

        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();

            // Brute-force ground truth
            let mut ground_truth: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist = crate::distance::l2_squared(&query, v);
                    (i, dist)
                })
                .collect();
            ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let gt_slots: HashSet<usize> = ground_truth.iter().take(k).map(|(i, _)| *i).collect();

            // HNSW search
            let results = index.search(&query, k, ef_search).unwrap();
            let result_slots: HashSet<usize> = results.iter().map(|r| r.slot).collect();

            let hits = gt_slots.intersection(&result_slots).count();
            total_recall += hits as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.95,
            "HNSW recall@{k} must exceed 95%, got {:.1}% (ef_search={ef_search}, n={n})",
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_graph_structure() {
        // Verify the graph has the expected multi-layer structure
        let (_dir, mut index) = setup(4, Metric::L2, 4);

        for i in 0..100 {
            index.insert(&[i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        assert_eq!(index.total_nodes(), 100);
        assert!(index.max_level() >= 1, "With 100 nodes, expect at least 2 layers");
        assert!(index.entry_point.is_some());

        // Verify all nodes have valid neighbor references
        for (idx, node) in index.nodes.iter().enumerate() {
            for (level, neighbors) in node.neighbors.iter().enumerate() {
                for &neighbor_idx in neighbors {
                    assert!(
                        neighbor_idx < index.nodes.len(),
                        "Node {idx} at level {level} has invalid neighbor {neighbor_idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_dimension_mismatch_search() {
        let (_dir, mut index) = setup(3, Metric::L2, 16);
        index.insert(&[1.0, 0.0, 0.0]).unwrap();
        assert!(index.search(&[1.0, 0.0], 1, 50).is_err());
    }

    #[test]
    fn test_persistence_reopen() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("hnsw_persist.qvdb");
        let wal_path = dir.path().join("hnsw_persist.wal");
        let config = HnswConfig::new(8).with_ef_construction(50);

        // Create and insert
        {
            let mut index =
                HnswIndex::create(&data_path, &wal_path, 3, Metric::L2, config.clone()).unwrap();
            index.insert(&[1.0, 0.0, 0.0]).unwrap();
            index.insert(&[0.0, 1.0, 0.0]).unwrap();
            index.insert(&[0.0, 0.0, 1.0]).unwrap();
            index.flush().unwrap();
        }

        // Reopen (graph rebuilt from stored vectors)
        {
            let index = HnswIndex::open(&data_path, &wal_path, config).unwrap();
            assert_eq!(index.len(), 3);
            let results = index.search(&[1.0, 0.0, 0.0], 1, 50).unwrap();
            assert_eq!(results[0].slot, 0);
        }
    }
}
