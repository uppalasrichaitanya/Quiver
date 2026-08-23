//! Versioned binary file header for the Quiver vector store.
//!
//! ## File Format Layout
//!
//! ```text
//! Offset  Size  Field
//! ------  ----  -----
//! 0       4     Magic bytes: b"QVDB"
//! 4       1     Format version (currently 3)
//! 5       1     Metric type (0 = L2, 1 = DotProduct, 2 = Cosine)
//! 6       2     Reserved (padding, zeroed)
//! 8       4     Vector dimension (u32, little-endian)
//! 12      8     Vector count (u64, little-endian)
//! 20      8     Max vector ID assigned so far (u64, little-endian)
//! 28      36    Reserved for future use (zeroed)
//! ------  ----
//! 64      Total header size
//! ```
//!
//! The version byte is present from day one so format changes later
//! (e.g., metadata support) don't break old indexes.
//!
//! Version 3 records are byte-identical to version 2; the bump signals that
//! vectors may carry metadata (persisted out-of-band in the WAL and a `.meta`
//! snapshot, never inline in the fixed-size records). Binaries predating
//! metadata reject version 3 via the version check below.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::distance::Metric;
use crate::error::{QuiverError, Result};

/// Magic bytes identifying a Quiver database file.
pub const MAGIC: &[u8; 4] = b"QVDB";

/// Legacy format with raw f32 records and implicit `slot + 1` vector IDs.
pub const LEGACY_FORMAT_VERSION: u8 = 1;

/// Current format: explicit u64 vector ID at the start of each record, plus
/// (since version 3) optional per-vector metadata stored out-of-band.
pub const FORMAT_VERSION: u8 = 3;

/// Total size of the file header in bytes.
pub const HEADER_SIZE: usize = 64;

/// The file header, read from and written to the start of every Quiver data file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeader {
    /// File format version.
    pub version: u8,
    /// The distance metric used by this index.
    pub metric: Metric,
    /// Dimensionality of stored vectors.
    pub dimension: u32,
    /// Number of vectors currently stored.
    pub vector_count: u64,
    /// Maximum vector ID assigned so far (monotonically increasing).
    pub max_vector_id: u64,
}

impl FileHeader {
    /// Create a new header for a fresh index.
    pub fn new(dimension: u32, metric: Metric) -> Self {
        Self {
            version: FORMAT_VERSION,
            metric,
            dimension,
            vector_count: 0,
            max_vector_id: 0,
        }
    }

    /// Serialize the header to a byte buffer of exactly [`HEADER_SIZE`] bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE);
        buf.write_all(MAGIC).unwrap();
        buf.write_u8(self.version).unwrap();
        buf.write_u8(self.metric as u8).unwrap();
        buf.write_all(&[0u8; 2]).unwrap(); // reserved padding
        buf.write_u32::<LittleEndian>(self.dimension).unwrap();
        buf.write_u64::<LittleEndian>(self.vector_count).unwrap();
        buf.write_u64::<LittleEndian>(self.max_vector_id).unwrap();
        // Pad the rest to reach HEADER_SIZE
        let remaining = HEADER_SIZE - buf.len();
        buf.write_all(&vec![0u8; remaining]).unwrap();
        debug_assert_eq!(buf.len(), HEADER_SIZE);
        buf
    }

    /// Deserialize a header from a byte slice.
    ///
    /// Returns an error if the magic bytes or version are invalid.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_SIZE {
            return Err(QuiverError::InvalidFormat(format!(
                "Header too short: {} bytes, expected at least {}",
                data.len(),
                HEADER_SIZE
            )));
        }

        let mut cursor = Cursor::new(data);

        // Read and validate magic bytes
        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .map_err(|e| QuiverError::InvalidFormat(format!("Failed to read magic: {e}")))?;
        if &magic != MAGIC {
            return Err(QuiverError::InvalidFormat(format!(
                "Invalid magic bytes: expected {:?}, got {:?}",
                MAGIC, magic
            )));
        }

        // Read and validate version
        let version = cursor
            .read_u8()
            .map_err(|e| QuiverError::InvalidFormat(format!("Failed to read version: {e}")))?;
        if !(LEGACY_FORMAT_VERSION..=FORMAT_VERSION).contains(&version) {
            return Err(QuiverError::InvalidFormat(format!(
                "Unsupported format version: {version} (max supported: {FORMAT_VERSION})"
            )));
        }

        // Read metric type
        let metric_byte = cursor
            .read_u8()
            .map_err(|e| QuiverError::InvalidFormat(format!("Failed to read metric: {e}")))?;
        let metric =
            Metric::from_u8(metric_byte).ok_or(QuiverError::UnsupportedMetric(metric_byte))?;

        // Skip reserved padding (2 bytes)
        let mut _reserved = [0u8; 2];
        cursor.read_exact(&mut _reserved)?;

        // Read dimension
        let dimension = cursor.read_u32::<LittleEndian>()?;

        // Read vector count
        let vector_count = cursor.read_u64::<LittleEndian>()?;

        // Read max vector ID
        let max_vector_id = cursor.read_u64::<LittleEndian>()?;

        Ok(Self {
            version,
            metric,
            dimension,
            vector_count,
            max_vector_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = FileHeader::new(128, Metric::L2);
        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);

        let parsed = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, parsed);
    }

    #[test]
    fn test_header_roundtrip_all_metrics() {
        for metric in [Metric::L2, Metric::DotProduct, Metric::Cosine] {
            let header = FileHeader::new(256, metric);
            let bytes = header.to_bytes();
            let parsed = FileHeader::from_bytes(&bytes).unwrap();
            assert_eq!(header, parsed);
        }
    }

    #[test]
    fn test_header_with_counts() {
        let mut header = FileHeader::new(64, Metric::Cosine);
        header.vector_count = 1000;
        header.max_vector_id = 1500;

        let bytes = header.to_bytes();
        let parsed = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.vector_count, 1000);
        assert_eq!(parsed.max_vector_id, 1500);
    }

    #[test]
    fn test_header_invalid_magic() {
        let mut bytes = FileHeader::new(128, Metric::L2).to_bytes();
        bytes[0] = b'X'; // corrupt magic
        assert!(FileHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_header_invalid_version() {
        let mut bytes = FileHeader::new(128, Metric::L2).to_bytes();
        bytes[4] = 255; // future version
        assert!(FileHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_legacy_version_is_still_accepted() {
        let mut bytes = FileHeader::new(128, Metric::L2).to_bytes();
        bytes[4] = LEGACY_FORMAT_VERSION;
        let header = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header.version, LEGACY_FORMAT_VERSION);
    }

    #[test]
    fn test_version_2_is_still_accepted() {
        let mut bytes = FileHeader::new(128, Metric::L2).to_bytes();
        bytes[4] = 2;
        let header = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header.version, 2);
    }

    #[test]
    fn test_new_headers_are_version_3() {
        assert_eq!(FORMAT_VERSION, 3);
        assert_eq!(FileHeader::new(128, Metric::L2).version, 3);
    }

    #[test]
    fn test_header_invalid_metric() {
        let mut bytes = FileHeader::new(128, Metric::L2).to_bytes();
        bytes[5] = 99; // invalid metric
        assert!(FileHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_header_too_short() {
        let bytes = vec![0u8; 10];
        assert!(FileHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_header_size_is_64() {
        assert_eq!(HEADER_SIZE, 64);
        let bytes = FileHeader::new(128, Metric::L2).to_bytes();
        assert_eq!(bytes.len(), 64);
    }
}
