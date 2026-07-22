//! Unified error types for quiver-core.

use thiserror::Error;

/// Top-level error type for all quiver-core operations.
#[derive(Debug, Error)]
pub enum QuiverError {
    /// An I/O error occurred during storage operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The file format is invalid or corrupted.
    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    /// The WAL entry failed checksum validation.
    #[error("WAL checksum mismatch at offset {offset}: expected {expected:#010x}, got {actual:#010x}")]
    WalChecksumMismatch {
        offset: u64,
        expected: u32,
        actual: u32,
    },

    /// A dimension mismatch was detected (e.g., inserting a 128-d vector into a 256-d index).
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: u32, actual: u32 },

    /// The requested vector ID was not found.
    #[error("Vector ID {0} not found")]
    NotFound(u64),

    /// The index is empty and cannot be searched.
    #[error("Index is empty")]
    EmptyIndex,

    /// An unsupported metric type was encountered.
    #[error("Unsupported metric type: {0}")]
    UnsupportedMetric(u8),
}

/// Convenience type alias for Results using [`QuiverError`].
pub type Result<T> = std::result::Result<T, QuiverError>;
