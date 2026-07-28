//! Bounds-checked parser for the mmap vector-file format.

use crate::error::{QuiverError, Result};
use crate::storage::header::{FORMAT_VERSION, FileHeader, HEADER_SIZE, LEGACY_FORMAT_VERSION};

pub(crate) const VECTOR_ID_SIZE: usize = std::mem::size_of::<u64>();

/// Validated layout information used when opening a vector store.
pub(crate) struct ParsedFile {
    pub header: FileHeader,
    pub record_size: usize,
    pub vector_ids: Vec<u64>,
}

/// Validate a complete vector data file without dereferencing unchecked offsets.
///
/// This is public so the dedicated `cargo fuzz` target can exercise the exact
/// parser used by [`crate::storage::vecstore::VectorStore::open`].
pub fn validate_file_bytes(data: &[u8]) -> Result<()> {
    parse_file_bytes(data).map(|_| ())
}

pub(crate) fn parse_file_bytes(data: &[u8]) -> Result<ParsedFile> {
    let header_bytes = data.get(..HEADER_SIZE).ok_or_else(|| {
        QuiverError::InvalidFormat(format!(
            "File too short: {} bytes, expected at least {HEADER_SIZE}",
            data.len()
        ))
    })?;
    let header = FileHeader::from_bytes(header_bytes)?;

    if !(LEGACY_FORMAT_VERSION..=FORMAT_VERSION).contains(&header.version) {
        return Err(QuiverError::InvalidFormat(format!(
            "Unsupported format version: {}",
            header.version
        )));
    }
    if header.dimension == 0 {
        return Err(QuiverError::InvalidFormat(
            "Vector dimension must be greater than zero".to_string(),
        ));
    }

    let vector_bytes = (header.dimension as usize)
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| QuiverError::InvalidFormat("Vector record size overflow".to_string()))?;
    let record_size = if header.version == LEGACY_FORMAT_VERSION {
        vector_bytes
    } else {
        VECTOR_ID_SIZE
            .checked_add(vector_bytes)
            .ok_or_else(|| QuiverError::InvalidFormat("Vector record size overflow".to_string()))?
    };
    let vector_count = usize::try_from(header.vector_count).map_err(|_| {
        QuiverError::InvalidFormat("Vector count does not fit this platform".to_string())
    })?;
    let records_len = vector_count
        .checked_mul(record_size)
        .ok_or_else(|| QuiverError::InvalidFormat("Vector data size overflow".to_string()))?;
    let required_len = HEADER_SIZE
        .checked_add(records_len)
        .ok_or_else(|| QuiverError::InvalidFormat("Vector file size overflow".to_string()))?;

    if data.len() < required_len {
        return Err(QuiverError::InvalidFormat(format!(
            "Truncated vector data: {} bytes, expected at least {required_len}",
            data.len()
        )));
    }
    if vector_count > 0 && header.max_vector_id == 0 {
        return Err(QuiverError::InvalidFormat(
            "Non-empty store has max_vector_id 0".to_string(),
        ));
    }

    let vector_ids = if header.version == LEGACY_FORMAT_VERSION {
        if header.max_vector_id < header.vector_count {
            return Err(QuiverError::InvalidFormat(format!(
                "Legacy max_vector_id {} is smaller than vector_count {}",
                header.max_vector_id, header.vector_count
            )));
        }
        (1..=header.vector_count).collect()
    } else {
        let mut ids = Vec::with_capacity(vector_count);
        let mut previous_id = 0;
        for slot in 0..vector_count {
            let offset = HEADER_SIZE
                .checked_add(slot.checked_mul(record_size).ok_or_else(|| {
                    QuiverError::InvalidFormat("Vector offset overflow".to_string())
                })?)
                .ok_or_else(|| QuiverError::InvalidFormat("Vector offset overflow".to_string()))?;
            let id_end = offset.checked_add(VECTOR_ID_SIZE).ok_or_else(|| {
                QuiverError::InvalidFormat("Vector ID offset overflow".to_string())
            })?;
            let id_bytes: [u8; VECTOR_ID_SIZE] = data
                .get(offset..id_end)
                .ok_or_else(|| QuiverError::InvalidFormat("Truncated vector ID field".to_string()))?
                .try_into()
                .map_err(|_| QuiverError::InvalidFormat("Invalid vector ID field".to_string()))?;
            let vector_id = u64::from_le_bytes(id_bytes);
            if vector_id == 0 || vector_id <= previous_id || vector_id > header.max_vector_id {
                return Err(QuiverError::InvalidFormat(format!(
                    "Invalid vector ID {vector_id} at slot {slot}"
                )));
            }
            ids.push(vector_id);
            previous_id = vector_id;
        }
        ids
    };

    Ok(ParsedFile {
        header,
        record_size,
        vector_ids,
    })
}
