//! Storage engine for Quiver.
//!
//! This module provides the foundational persistence layer:
//!
//! - **File Format**: Custom binary format with a versioned header (format version,
//!   dimension, count, metric type) and fixed-size vector records, accessed via memory mapping.
//! - **WAL (Write-Ahead Log)**: Length-prefixed, checksummed entries for crash recovery.
//!   Recovery replays until the first checksum failure and truncates there.
//!   This is deliberately *not* ARIES-style recovery.
//! - **Flush**: Periodic full mmap flush + fsync on a timer.

pub mod format;
pub mod header;
pub mod vecstore;
pub mod wal;
