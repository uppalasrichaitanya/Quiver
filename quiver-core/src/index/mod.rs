//! Index implementations for vector similarity search.
//!
//! ## Implementations
//!
//! - [`BruteForceIndex`](brute_force::BruteForceIndex) — Exact nearest neighbor via
//!   linear scan. Used as ground truth for recall measurement and as a fallback for
//!   small datasets.
//! - [`HnswIndex`](hnsw::HnswIndex) — Approximate nearest neighbor via hierarchical
//!   navigable small world graph. The primary ANN structure.
//! - [`Sq8Index`](sq8::Sq8Index) provides batch-built flat search over vectors
//!   compressed to one byte per dimension.

pub mod brute_force;
pub mod hnsw;
pub mod sq8;

/// A single search result: the vector's internal ID and its distance from the query.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The slot index of the vector in the store (0-based).
    pub slot: usize,
    /// The vector ID as assigned at insert time.
    pub vector_id: u64,
    /// The distance to the query vector (lower is more similar for all metrics,
    /// because cosine/dot are negated by `compute_distance`).
    pub distance: f32,
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower distance = better, so we use the natural float ordering.
        // For use in a max-heap (BinaryHeap), we want the *worst* result on top
        // so we can efficiently evict it. This natural ordering achieves that:
        // BinaryHeap pops the largest, which is the worst match.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
