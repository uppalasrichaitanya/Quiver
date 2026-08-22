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
//! The public API is single-threaded in v1: mutation requires `&mut self`, and the index does not
//! provide an internal synchronization wrapper. Concurrent access is a future API concern.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::distance::{Metric, compute_distance};
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
    /// Seed used for deterministic HNSW layer assignment.
    pub random_seed: u64,
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
            random_seed: 42,
        }
    }

    /// Set ef_construction (builder pattern).
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the deterministic random seed used for layer assignment.
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self::new(16)
    }
}

/// A node in the HNSW graph.
///
/// Level metadata lives in the shared [`HnswIndex::level_blocks`] arena: this
/// node's `max_layer + 1` blocks start at `levels_offset`. Packing every node's
/// level list into one arena avoids a per-node heap allocation (one small `Vec`
/// per node) and keeps level metadata contiguous.
#[derive(Debug, Clone)]
struct HnswNode {
    /// The slot index of this vector in the VectorStore.
    slot: usize,
    /// The vector ID as assigned at insert time.
    vector_id: u64,
    /// The maximum layer this node belongs to (0-indexed).
    max_layer: usize,
    /// Offset of this node's first `LevelBlock` in the shared arena.
    levels_offset: u32,
    /// Whether this node has been deleted (tombstone).
    deleted: bool,
}

/// One layer's neighbor-list location within the packed adjacency arena.
///
/// `u32` fields are sufficient: offsets index a `Vec<u32>` adjacency arena and
/// lengths/capacities are bounded by `m_max0`, all far below `u32::MAX`.
#[derive(Debug, Clone, Copy)]
struct LevelBlock {
    offset: u32,
    len: u32,
    capacity: u32,
}

/// A candidate during search: node index + distance.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Generation-stamped visited set reused across searches on the same thread.
///
/// Replaces the per-query `HashSet`: each search bumps a generation counter and
/// stamps visited node indices with it, so membership is a single array load
/// and there is no hashing or per-query allocation. Kept thread-local so the
/// index itself stays `Sync` (search takes `&self`).
struct VisitedPool {
    marks: Vec<u32>,
    generation: u32,
}

impl VisitedPool {
    fn new() -> Self {
        Self {
            marks: Vec::new(),
            generation: 0,
        }
    }

    /// Prepare for a new search over `n` nodes.
    fn begin(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped around — clear all marks and resume from generation 1.
            self.marks.fill(0);
            self.generation = 1;
        }
    }

    /// Mark `idx` visited. Returns true if it was not yet visited this search.
    #[inline]
    fn visit(&mut self, idx: usize) -> bool {
        if self.marks[idx] == self.generation {
            false
        } else {
            self.marks[idx] = self.generation;
            true
        }
    }
}

thread_local! {
    static VISITED_POOL: std::cell::RefCell<VisitedPool> =
        std::cell::RefCell::new(VisitedPool::new());
}

/// The HNSW index.
pub struct HnswIndex {
    /// The underlying vector storage.
    store: VectorStore,
    /// The HNSW graph nodes.
    nodes: Vec<HnswNode>,
    /// Fixed-capacity adjacency blocks containing 32-bit node IDs.
    adjacency_links: Vec<u32>,
    /// Packed per-node level metadata. Node `i` owns the contiguous range
    /// `levels_offset .. levels_offset + max_layer + 1` in this arena.
    level_blocks: Vec<LevelBlock>,
    /// Index of the current entry point node (topmost layer).
    entry_point: Option<usize>,
    /// Maximum layer currently in the graph.
    max_level: usize,
    /// Configuration parameters.
    config: HnswConfig,
    /// Number of tombstoned (deleted) nodes.
    tombstone_count: usize,
    /// Deterministic random generator for layer assignment.
    rng: StdRng,
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
        let rng = StdRng::seed_from_u64(config.random_seed);
        Ok(Self {
            store,
            nodes: Vec::new(),
            adjacency_links: Vec::new(),
            level_blocks: Vec::new(),
            entry_point: None,
            max_level: 0,
            config,
            tombstone_count: 0,
            rng,
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
        let rng = StdRng::seed_from_u64(config.random_seed);
        let mut index = Self {
            store,
            nodes: Vec::new(),
            adjacency_links: Vec::new(),
            level_blocks: Vec::new(),
            entry_point: None,
            max_level: 0,
            config,
            tombstone_count: 0,
            rng,
        };

        // Rebuild graph from stored vectors
        let n = index.store.len();
        if n > 0 {
            tracing::info!(count = n, "Rebuilding HNSW graph from stored vectors");
            for slot in 0..n {
                let vector = index.store.get_vector(slot)?.to_vec();
                let vector_id = index.store.vector_id(slot)?;
                index.insert_into_graph(slot, vector_id, &vector);
            }

            for node in &mut index.nodes {
                if index.store.is_deleted(node.vector_id) {
                    node.deleted = true;
                    index.tombstone_count += 1;
                }
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

    /// Insert a batch of vectors.
    ///
    /// Storage is made durable with a single group-committed WAL fsync for the
    /// whole batch (see [`VectorStore::insert_batch`]); graph insertion remains
    /// sequential because each insert depends on the prior graph state. Returns
    /// the assigned vector IDs in input order.
    pub fn insert_batch(&mut self, batch: &[&[f32]]) -> Result<Vec<u64>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let ids = self.store.insert_batch(batch)?;
        let base_slot = self.store.len() - ids.len();
        for (i, vector) in batch.iter().enumerate() {
            self.insert_into_graph(base_slot + i, ids[i], vector);
        }
        Ok(ids)
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

        let ep_vector = self.store.get_vector_unchecked(self.nodes[current].slot);
        let mut current_dist = compute_distance(query, ep_vector, metric);

        for level in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if level > self.nodes[current].max_layer {
                    break;
                }
                let neighbors = self.neighbors(current, level);
                for &neighbor_link in neighbors {
                    let neighbor_idx = neighbor_link as usize;
                    if self.nodes[neighbor_idx].deleted {
                        continue;
                    }
                    let neighbor_vec =
                        self.store.get_vector_unchecked(self.nodes[neighbor_idx].slot);
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
        let candidates = self.search_layer(query, current, ef, 0, metric);

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

        // Persist the tombstone before making it visible in the graph.
        self.store.delete(vector_id)?;
        self.nodes[node_idx].deleted = true;
        self.tombstone_count += 1;

        // Check if compaction is needed
        if !self.nodes.is_empty() {
            let ratio = self.tombstone_count as f64 / self.nodes.len() as f64;
            if ratio > self.config.max_tombstone_ratio {
                tracing::info!(
                    tombstone_ratio = ratio,
                    threshold = self.config.max_tombstone_ratio,
                    "Tombstone ratio exceeded threshold — compacting index"
                );
                self.compact()?;
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

    /// Rewrite storage with live vectors only and rebuild the HNSW graph.
    pub fn compact(&mut self) -> Result<()> {
        self.store.compact()?;
        self.nodes.clear();
        self.adjacency_links.clear();
        self.level_blocks.clear();
        self.entry_point = None;
        self.max_level = 0;
        self.tombstone_count = 0;

        for slot in 0..self.store.len() {
            let vector = self.store.get_vector(slot)?.to_vec();
            let vector_id = self.store.vector_id(slot)?;
            self.insert_into_graph(slot, vector_id, &vector);
        }

        Ok(())
    }

    /// Return the current WAL size in bytes.
    pub fn wal_len(&self) -> Result<u64> {
        self.store.wal_len()
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
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.random();
        (-r.ln() * self.config.ml).floor() as usize
    }

    /// Borrow the neighbor list for a node at a layer.
    ///
    /// Returns a slice into the packed adjacency arena — no allocation. Callers
    /// cast each `u32` link to `usize` when indexing `nodes`.
    #[inline]
    fn neighbors(&self, node_idx: usize, level: usize) -> &[u32] {
        let block = self.level_block(node_idx, level);
        let start = block.offset as usize;
        &self.adjacency_links[start..start + block.len as usize]
    }

    /// Read a node's level metadata from the packed arena.
    #[inline]
    fn level_block(&self, node_idx: usize, level: usize) -> LevelBlock {
        let node = &self.nodes[node_idx];
        self.level_blocks[node.levels_offset as usize + level]
    }

    /// Write a node's level metadata back into the packed arena.
    #[inline]
    fn set_level_block_len(&mut self, node_idx: usize, level: usize, len: u32) {
        let offset = self.nodes[node_idx].levels_offset as usize + level;
        self.level_blocks[offset].len = len;
    }

    fn replace_neighbors(&mut self, node_idx: usize, level: usize, neighbors: &[usize]) {
        let block = self.level_block(node_idx, level);
        assert!(
            neighbors.len() <= block.capacity as usize,
            "neighbor list exceeds fixed capacity"
        );
        let start = block.offset as usize;
        for (slot, &neighbor_idx) in self.adjacency_links[start..start + neighbors.len()]
            .iter_mut()
            .zip(neighbors)
        {
            *slot = u32::try_from(neighbor_idx).expect("node index exceeds u32 capacity");
        }
        self.set_level_block_len(node_idx, level, neighbors.len() as u32);
    }

    /// Insert a vector into the HNSW graph (assumes it's already in the store).
    fn insert_into_graph(&mut self, slot: usize, vector_id: u64, vector: &[f32]) {
        let new_level = self.random_level();
        let metric = self.store.metric();

        // Create the new node
        let m_max0 = self.config.m_max0;
        let m = self.config.m;
        let levels_offset = u32::try_from(self.level_blocks.len())
            .expect("level block arena exceeds u32 capacity");
        for level in 0..=new_level {
            let capacity = if level == 0 { m_max0 } else { m };
            let offset = u32::try_from(self.adjacency_links.len())
                .expect("adjacency arena exceeds u32 capacity");
            self.adjacency_links
                .resize(offset as usize + capacity, 0);
            self.level_blocks.push(LevelBlock {
                offset,
                len: 0,
                capacity: capacity as u32,
            });
        }
        let new_node = HnswNode {
            slot,
            vector_id,
            max_layer: new_level,
            levels_offset,
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
        let ep_vec = self.store.get_vector_unchecked(self.nodes[current].slot);
        let mut current_dist = compute_distance(vector, ep_vec, metric);

        // Phase 1: Greedy descent through layers above the new node's level
        for level in (new_level + 1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if level > self.nodes[current].max_layer {
                    continue;
                }
                let neighbors = self.neighbors(current, level);
                for &neighbor_link in neighbors {
                    let neighbor_idx = neighbor_link as usize;
                    let n_vec = self.store.get_vector_unchecked(self.nodes[neighbor_idx].slot);
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
            let candidates = self.search_layer_for_insert(vector, current, ef, level, metric);

            let m_level = if level == 0 { m_max0 } else { m };
            let candidates: Vec<Candidate> = candidates
                .into_iter()
                .filter(|c| !self.nodes[c.node_idx].deleted && c.node_idx != new_node_idx)
                .collect();
            let selected = self.select_neighbors_heuristic(vector, &candidates, m_level, metric);

            // Set forward connections (new_node -> selected neighbors)
            self.replace_neighbors(new_node_idx, level, &selected);

            // Set reverse connections (each selected neighbor -> new_node)
            for &neighbor_idx in &selected {
                if level <= self.nodes[neighbor_idx].max_layer {
                    let max_conn = if level == 0 { m_max0 } else { m };
                    let existing = self.neighbors(neighbor_idx, level);
                    let mut reverse_neighbors: Vec<usize> =
                        Vec::with_capacity(existing.len() + 1);
                    reverse_neighbors.extend(existing.iter().map(|&link| link as usize));
                    reverse_neighbors.push(new_node_idx);
                    if reverse_neighbors.len() > max_conn {
                        self.prune_connections(
                            neighbor_idx,
                            level,
                            max_conn,
                            &reverse_neighbors,
                            metric,
                        );
                    } else {
                        self.replace_neighbors(neighbor_idx, level, &reverse_neighbors);
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
    ) -> Vec<Candidate> {
        VISITED_POOL.with(|cell| {
            let mut pool = cell.borrow_mut();
            pool.begin(self.nodes.len());
            self.search_layer_inner(query, entry_point, ef, level, metric, &mut pool)
        })
    }

    fn search_layer_inner(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        level: usize,
        metric: Metric,
        pool: &mut VisitedPool,
    ) -> Vec<Candidate> {
        let ep_vec = self.store.get_vector_unchecked(self.nodes[entry_point].slot);
        let ep_dist = compute_distance(query, ep_vec, metric);

        pool.visit(entry_point);

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

            if level > self.nodes[current.node_idx].max_layer {
                continue;
            }
            let neighbors = self.neighbors(current.node_idx, level);

            for (i, &neighbor_link) in neighbors.iter().enumerate() {
                let neighbor_idx = neighbor_link as usize;

                // Prefetch the next neighbor's vector while we process this one.
                if let Some(&next_link) = neighbors.get(i + 1) {
                    self.store
                        .prefetch_vector(self.nodes[next_link as usize].slot);
                }

                if !pool.visit(neighbor_idx) {
                    continue;
                }

                let n_vec = self.store.get_vector_unchecked(self.nodes[neighbor_idx].slot);
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
        sorted
    }

    /// Search layer for insertion — same beam search used during construction.
    fn search_layer_for_insert(
        &self,
        query: &[f32],
        entry_point: usize,
        ef: usize,
        level: usize,
        metric: Metric,
    ) -> Vec<Candidate> {
        self.search_layer(query, entry_point, ef, level, metric)
    }

    /// Select links using the HNSW diversified-neighbor heuristic. A candidate
    /// is accepted only when it is not closer to an already selected neighbor
    /// than it is to the query; rejected candidates fill unused capacity.
    fn select_neighbors_heuristic(
        &self,
        query: &[f32],
        candidates: &[Candidate],
        max_connections: usize,
        metric: Metric,
    ) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(max_connections);
        let mut discarded: Vec<usize> = Vec::new();

        for candidate in candidates {
            // Slots come from live graph nodes, so unchecked access is safe here.
            let candidate_vec = self
                .store
                .get_vector_unchecked(self.nodes[candidate.node_idx].slot);
            let diverse = selected.iter().all(|&selected_idx| {
                let selected_vec = self
                    .store
                    .get_vector_unchecked(self.nodes[selected_idx].slot);
                compute_distance(candidate_vec, selected_vec, metric) >= candidate.distance
            });
            if diverse && selected.len() < max_connections {
                selected.push(candidate.node_idx);
            } else {
                discarded.push(candidate.node_idx);
            }
        }

        selected.extend(
            discarded
                .into_iter()
                .take(max_connections.saturating_sub(selected.len())),
        );
        let _ = query;
        selected
    }

    /// Prune the connection list of a node at the given layer to max_conn connections.
    /// Keeps the closest neighbors by distance to the node's own vector.
    fn prune_connections(
        &mut self,
        node_idx: usize,
        level: usize,
        max_conn: usize,
        neighbors: &[usize],
        metric: Metric,
    ) {
        let node_slot = self.nodes[node_idx].slot;
        // Slots come from live graph nodes, so unchecked access is safe here.
        let node_vec = self.store.get_vector_unchecked(node_slot);

        let mut candidates: Vec<Candidate> = neighbors
            .iter()
            .copied()
            .map(|n_idx| {
                let n_vec = self.store.get_vector_unchecked(self.nodes[n_idx].slot);
                let dist = compute_distance(node_vec, n_vec, metric);
                Candidate {
                    node_idx: n_idx,
                    distance: dist,
                }
            })
            .collect();
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let selected = self.select_neighbors_heuristic(node_vec, &candidates, max_conn, metric);
        self.replace_neighbors(node_idx, level, &selected);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
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
    fn test_insert_batch_matches_single_insert() {
        let (_dir, mut index) = setup(3, Metric::L2, 8);
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32, (i % 7) as f32, 0.0])
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let ids = index.insert_batch(&refs).unwrap();
        assert_eq!(ids.len(), 50);
        assert_eq!(index.len(), 50);
        // IDs are sequential starting at 1.
        assert_eq!(ids.first().copied(), Some(1));
        assert_eq!(ids.last().copied(), Some(50));
        // The batch-built graph must be searchable with good recall.
        let results = index.search(&[25.0, 4.0, 0.0], 5, 50).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].vector_id, 26); // vector index 25 -> id 26
    }

    #[test]
    fn test_insert_batch_empty() {
        let (_dir, mut index) = setup(3, Metric::L2, 4);
        let ids = index.insert_batch(&[]).unwrap();
        assert!(ids.is_empty());
        assert!(index.is_empty());
    }

    #[test]
    fn diversified_selection_keeps_a_farther_bridge_candidate() {
        let (_dir, mut index) = setup(1, Metric::L2, 4);
        index.insert(&[1.0]).unwrap();
        index.insert(&[1.1]).unwrap();
        index.insert(&[-2.0]).unwrap();

        let candidates = vec![
            Candidate {
                node_idx: 0,
                distance: 1.0,
            },
            Candidate {
                node_idx: 1,
                distance: 1.21,
            },
            Candidate {
                node_idx: 2,
                distance: 4.0,
            },
        ];

        let selected = index.select_neighbors_heuristic(&[0.0], &candidates, 2, Metric::L2);

        assert_eq!(selected, vec![0, 2]);
    }

    #[test]
    fn packed_adjacency_uses_fixed_u32_blocks() {
        let (_dir, mut index) = setup(2, Metric::L2, 4);
        index.insert(&[0.0, 0.0]).unwrap();

        let node = &index.nodes[0];
        let blocks = &index.level_blocks
            [node.levels_offset as usize..node.levels_offset as usize + node.max_layer + 1];
        assert_eq!(blocks.len(), node.max_layer + 1);
        assert_eq!(blocks[0].capacity as usize, index.config.m_max0);
        assert_eq!(index.adjacency_links.len(), index.config.m_max0);
        assert_eq!(std::mem::size_of_val(&index.adjacency_links[0]), 4);
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
        index.config.max_tombstone_ratio = 1.0;

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
    fn test_compaction_triggers_only_above_threshold() {
        let (dir, mut index) = setup(3, Metric::L2, 8);
        index.config.max_tombstone_ratio = 0.25;

        let id1 = index.insert(&[1.0, 0.0, 0.0]).unwrap();
        let id2 = index.insert(&[0.0, 1.0, 0.0]).unwrap();
        let id3 = index.insert(&[0.0, 0.0, 1.0]).unwrap();
        let id4 = index.insert(&[-1.0, 0.0, 0.0]).unwrap();

        index.delete(id1).unwrap();
        assert_eq!(index.total_nodes(), 4);
        assert_eq!(index.tombstone_count, 1);
        assert!(index.wal_len().unwrap() > 0);

        index.delete(id2).unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.total_nodes(), 2);
        assert_eq!(index.tombstone_count, 0);
        assert_eq!(index.wal_len().unwrap(), 0);

        let results = index.search(&[1.0, 0.0, 0.0], 4, 50).unwrap();
        assert!(results.iter().all(|result| result.vector_id != id1));
        assert!(results.iter().all(|result| result.vector_id != id2));
        assert!(results.iter().any(|result| result.vector_id == id3));
        assert!(results.iter().any(|result| result.vector_id == id4));

        let config = index.config.clone();
        drop(index);
        let reopened = HnswIndex::open(
            dir.path().join("hnsw_vectors.qvdb"),
            dir.path().join("hnsw_vectors.wal"),
            config,
        )
        .unwrap();
        let results = reopened.search(&[1.0, 0.0, 0.0], 4, 50).unwrap();
        assert_eq!(reopened.len(), 2);
        assert!(results.iter().all(|result| result.vector_id != id1));
        assert!(results.iter().all(|result| result.vector_id != id2));
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
        assert!(
            index.max_level() >= 1,
            "With 100 nodes, expect at least 2 layers"
        );
        assert!(index.entry_point.is_some());

        // Verify all nodes have valid neighbor references
        for (idx, node) in index.nodes.iter().enumerate() {
            for level in 0..=node.max_layer {
                for &neighbor_link in index.neighbors(idx, level) {
                    let neighbor_idx = neighbor_link as usize;
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

    #[test]
    fn test_delete_persists_after_reopen() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("hnsw_delete_persist.qvdb");
        let wal_path = dir.path().join("hnsw_delete_persist.wal");
        let mut config = HnswConfig::new(8).with_ef_construction(50);
        config.max_tombstone_ratio = 1.0;

        let deleted_id;
        {
            let mut index =
                HnswIndex::create(&data_path, &wal_path, 3, Metric::L2, config.clone()).unwrap();
            deleted_id = index.insert(&[1.0, 0.0, 0.0]).unwrap();
            index.insert(&[0.0, 1.0, 0.0]).unwrap();
            index.insert(&[0.0, 0.0, 1.0]).unwrap();
            index.delete(deleted_id).unwrap();
            index.flush().unwrap();
        }

        let index = HnswIndex::open(&data_path, &wal_path, config).unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(index.tombstone_count, 1);

        let results = index.search(&[1.0, 0.0, 0.0], 3, 50).unwrap();
        assert!(results.iter().all(|result| result.vector_id != deleted_id));
    }

    #[test]
    fn test_delete_recovers_after_unflushed_drop() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("hnsw_delete_recovery.qvdb");
        let wal_path = dir.path().join("hnsw_delete_recovery.wal");
        let mut config = HnswConfig::new(8).with_ef_construction(50);
        config.max_tombstone_ratio = 1.0;

        let deleted_id;
        {
            let mut index =
                HnswIndex::create(&data_path, &wal_path, 3, Metric::L2, config.clone()).unwrap();
            deleted_id = index.insert(&[1.0, 0.0, 0.0]).unwrap();
            index.insert(&[0.0, 1.0, 0.0]).unwrap();
            index.delete(deleted_id).unwrap();
            // No index.flush(): both the inserts and delete must recover from WAL.
        }

        let index = HnswIndex::open(&data_path, &wal_path, config).unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index.tombstone_count, 1);

        let results = index.search(&[1.0, 0.0, 0.0], 2, 50).unwrap();
        assert!(results.iter().all(|result| result.vector_id != deleted_id));
    }
}
