//! # Quiver Core
//!
//! Core vector search engine library implementing HNSW and IVF-PQ from scratch.
//!
//! This is a portfolio-grade, single-node, embeddable vector search engine.
//! It is explicitly **not** a production service — see the README for positioning.
//!
//! ## Architecture
//!
//! The library is organized into the following modules, built in dependency order:
//!
//! - [`storage`] — Custom memory-mapped file format with versioned header and WAL
//! - [`distance`] — Distance metrics (L2, cosine, dot product) with scalar and SIMD implementations
//! - [`index`] — Index implementations (brute-force, HNSW, IVF-PQ)
//! - [`quantization`] — Vector compression (SQ8, Product Quantization)
//! - [`filter`] — Metadata storage and filtered search
//! - [`error`] — Unified error types

pub mod distance;
pub mod error;
pub mod index;
pub mod storage;

// Future modules (uncomment as implemented):
// pub mod quantization;
// pub mod filter;
