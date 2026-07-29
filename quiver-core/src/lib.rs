//! # Quiver Core
//!
//! Core vector search engine library implementing durable mmap storage and HNSW from scratch.
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
//! - [`index`] — Index implementations (brute-force and HNSW)
//! - [`error`] — Unified error types

pub mod distance;
pub mod error;
pub mod index;
pub mod quantization;
pub mod storage;

// Future modules (uncomment only when implemented):
// pub mod filter;
