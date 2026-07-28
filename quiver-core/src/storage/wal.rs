//! Write-Ahead Log (WAL) for crash recovery.
//!
//! ## WAL Entry Format
//!
//! Each entry in the WAL is structured as:
//!
//! ```text
//! Offset  Size       Field
//! ------  ---------  -----
//! 0       4          Entry length in bytes (u32 LE, excludes this field and checksum)
//! 4       1          Operation type (0 = Insert, 1 = Delete)
//! 5       8          Vector ID (u64 LE)
//! 13      N*4        Vector data (N f32s, LE) — only present for Insert ops
//! 13+N*4  4          CRC32 checksum of bytes [0..13+N*4)
//! ```
//!
//! ## Recovery
//!
//! On startup, the WAL is replayed entry-by-entry. If a checksum mismatch
//! is encountered, all remaining entries are considered incomplete/corrupt
//! and the WAL is truncated at that point. This is a deliberate simplification —
//! not ARIES-style recovery.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::Result;

/// The type of operation recorded in a WAL entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WalOp {
    /// Insert a vector with the given ID and data.
    Insert = 0,
    /// Delete the vector with the given ID.
    Delete = 1,
}

impl WalOp {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(WalOp::Insert),
            1 => Some(WalOp::Delete),
            _ => None,
        }
    }
}

/// A single WAL entry representing an insert or delete operation.
#[derive(Debug, Clone, PartialEq)]
pub struct WalEntry {
    /// The operation type.
    pub op: WalOp,
    /// The vector ID this operation applies to.
    pub vector_id: u64,
    /// The vector data (only present for Insert operations).
    pub vector_data: Option<Vec<f32>>,
}

/// The Write-Ahead Log writer/reader.
pub struct Wal {
    /// Path to the WAL file.
    path: PathBuf,
    /// Buffered writer for appending entries.
    writer: BufWriter<File>,
}

impl Wal {
    /// Open or create a WAL file at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let writer = BufWriter::new(file);
        Ok(Self { path, writer })
    }

    /// Append an insert entry to the WAL.
    pub fn log_insert(&mut self, vector_id: u64, data: &[f32]) -> Result<()> {
        let entry_body = Self::serialize_insert(vector_id, data);
        self.write_entry(&entry_body)?;
        Ok(())
    }

    /// Append a delete entry to the WAL.
    pub fn log_delete(&mut self, vector_id: u64) -> Result<()> {
        let entry_body = Self::serialize_delete(vector_id);
        self.write_entry(&entry_body)?;
        Ok(())
    }

    /// Flush the WAL to disk (fsync).
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Read all valid entries from the WAL file for replay.
    ///
    /// Stops at the first entry with a checksum mismatch and returns
    /// the truncation point (byte offset) so the caller can truncate.
    pub fn read_entries(path: impl AsRef<Path>) -> Result<(Vec<WalEntry>, u64)> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok((Vec::new(), 0));
        }

        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut valid_up_to: u64 = 0;

        loop {
            let current_pos = valid_up_to;

            // Try to read the entry length
            let entry_len = match reader.read_u32::<LittleEndian>() {
                Ok(len) => match usize::try_from(len) {
                    Ok(len) => len,
                    Err(_) => break,
                },
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            };

            // Check if there are enough bytes remaining
            let total_entry_len = match 4_usize
                .checked_add(entry_len)
                .and_then(|len| len.checked_add(4))
            {
                Some(len) => len,
                None => break,
            };
            let entry_end = match current_pos.checked_add(total_entry_len as u64) {
                Some(end) => end,
                None => break,
            };
            if entry_end > file_len {
                // Incomplete entry — truncate here
                break;
            }

            // Read the entry body
            let mut body = vec![0u8; entry_len];
            if reader.read_exact(&mut body).is_err() {
                break;
            }

            // Read the checksum
            let stored_checksum = match reader.read_u32::<LittleEndian>() {
                Ok(c) => c,
                Err(_) => break,
            };

            // Compute checksum over length prefix + body
            let mut hasher = Hasher::new();
            let mut len_bytes = Vec::with_capacity(4);
            len_bytes
                .write_u32::<LittleEndian>(entry_len as u32)
                .unwrap();
            hasher.update(&len_bytes);
            hasher.update(&body);
            let computed_checksum = hasher.finalize();

            if stored_checksum != computed_checksum {
                tracing::warn!(
                    offset = current_pos,
                    expected = format!("{stored_checksum:#010x}"),
                    actual = format!("{computed_checksum:#010x}"),
                    "WAL checksum mismatch — truncating here"
                );
                break;
            }

            // Parse the entry body
            match Self::parse_entry_body(&body) {
                Some(entry) => {
                    entries.push(entry);
                    valid_up_to = entry_end;
                }
                None => break,
            }
        }

        Ok((entries, valid_up_to))
    }

    /// Truncate the WAL file to the given byte offset (removing corrupt tail).
    pub fn truncate(path: impl AsRef<Path>, valid_up_to: u64) -> Result<()> {
        let file = OpenOptions::new().write(true).open(path.as_ref())?;
        file.set_len(valid_up_to)?;
        file.sync_all()?;
        Ok(())
    }

    /// Clear the WAL (e.g., after a successful checkpoint/flush of the main store).
    pub fn clear(&mut self) -> Result<()> {
        self.writer.flush()?;
        let file = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        self.writer = BufWriter::new(file);
        Ok(())
    }

    // ── Private helpers ──────────────────────────────────────────────────

    fn write_entry(&mut self, body: &[u8]) -> Result<()> {
        let entry_len = body.len() as u32;

        // Compute checksum over length prefix + body
        let mut hasher = Hasher::new();
        let mut len_bytes = Vec::with_capacity(4);
        len_bytes.write_u32::<LittleEndian>(entry_len).unwrap();
        hasher.update(&len_bytes);
        hasher.update(body);
        let checksum = hasher.finalize();

        // Write: length prefix + body + checksum
        self.writer.write_u32::<LittleEndian>(entry_len)?;
        self.writer.write_all(body)?;
        self.writer.write_u32::<LittleEndian>(checksum)?;

        Ok(())
    }

    fn serialize_insert(vector_id: u64, data: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1 + 8 + data.len() * 4);
        buf.write_u8(WalOp::Insert as u8).unwrap();
        buf.write_u64::<LittleEndian>(vector_id).unwrap();
        for &val in data {
            buf.write_f32::<LittleEndian>(val).unwrap();
        }
        buf
    }

    fn serialize_delete(vector_id: u64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(1 + 8);
        buf.write_u8(WalOp::Delete as u8).unwrap();
        buf.write_u64::<LittleEndian>(vector_id).unwrap();
        buf
    }

    fn parse_entry_body(body: &[u8]) -> Option<WalEntry> {
        if body.is_empty() {
            return None;
        }

        let op = WalOp::from_u8(body[0])?;
        if body.len() < 9 {
            return None; // Need at least op (1) + vector_id (8)
        }

        let mut cursor = io::Cursor::new(&body[1..]);
        let vector_id = cursor.read_u64::<LittleEndian>().ok()?;

        match op {
            WalOp::Insert => {
                let remaining = &body[9..];
                if remaining.len() % 4 != 0 {
                    return None; // Not aligned to f32
                }
                let mut data = Vec::with_capacity(remaining.len() / 4);
                let mut cursor = io::Cursor::new(remaining);
                while cursor.position() < remaining.len() as u64 {
                    data.push(cursor.read_f32::<LittleEndian>().ok()?);
                }
                Some(WalEntry {
                    op,
                    vector_id,
                    vector_data: Some(data),
                })
            }
            WalOp::Delete => Some(WalEntry {
                op,
                vector_id,
                vector_data: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use tempfile::TempDir;

    fn wal_path(dir: &TempDir) -> PathBuf {
        dir.path().join("test.wal")
    }

    #[test]
    fn test_wal_insert_and_read() {
        let dir = TempDir::new().unwrap();
        let path = wal_path(&dir);

        // Write entries
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_insert(1, &[1.0, 2.0, 3.0]).unwrap();
            wal.log_insert(2, &[4.0, 5.0, 6.0]).unwrap();
            wal.flush().unwrap();
        }

        // Read them back
        let (entries, _) = Wal::read_entries(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].op, WalOp::Insert);
        assert_eq!(entries[0].vector_id, 1);
        assert_eq!(entries[0].vector_data.as_ref().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(entries[1].vector_id, 2);
    }

    #[test]
    fn test_wal_delete_and_read() {
        let dir = TempDir::new().unwrap();
        let path = wal_path(&dir);

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_insert(1, &[1.0, 2.0]).unwrap();
            wal.log_delete(1).unwrap();
            wal.flush().unwrap();
        }

        let (entries, _) = Wal::read_entries(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].op, WalOp::Insert);
        assert_eq!(entries[1].op, WalOp::Delete);
        assert_eq!(entries[1].vector_id, 1);
        assert!(entries[1].vector_data.is_none());
    }

    #[test]
    fn test_wal_checksum_corruption_truncates() {
        let dir = TempDir::new().unwrap();
        let path = wal_path(&dir);

        // Write two valid entries
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_insert(1, &[1.0, 2.0]).unwrap();
            wal.log_insert(2, &[3.0, 4.0]).unwrap();
            wal.flush().unwrap();
        }

        // Corrupt the second entry's checksum (last 4 bytes of file)
        {
            let mut file = OpenOptions::new().write(true).open(&path).unwrap();
            let len = file.metadata().unwrap().len();
            file.seek(SeekFrom::Start(len - 1)).unwrap();
            file.write_all(&[0xFF]).unwrap();
            file.sync_all().unwrap();
        }

        // Read — should recover only the first entry
        let (entries, valid_up_to) = Wal::read_entries(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].vector_id, 1);

        // Truncate to valid portion
        Wal::truncate(&path, valid_up_to).unwrap();

        // Re-read — should still have exactly one entry
        let (entries2, _) = Wal::read_entries(&path).unwrap();
        assert_eq!(entries2.len(), 1);
    }

    #[test]
    fn test_wal_empty_file() {
        let dir = TempDir::new().unwrap();
        let path = wal_path(&dir);

        // Create empty file
        File::create(&path).unwrap();

        let (entries, valid_up_to) = Wal::read_entries(&path).unwrap();
        assert!(entries.is_empty());
        assert_eq!(valid_up_to, 0);
    }

    #[test]
    fn test_wal_nonexistent_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("does_not_exist.wal");

        let (entries, valid_up_to) = Wal::read_entries(&path).unwrap();
        assert!(entries.is_empty());
        assert_eq!(valid_up_to, 0);
    }

    #[test]
    fn test_wal_clear() {
        let dir = TempDir::new().unwrap();
        let path = wal_path(&dir);

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_insert(1, &[1.0]).unwrap();
            wal.flush().unwrap();
            wal.clear().unwrap();
        }

        let (entries, _) = Wal::read_entries(&path).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_wal_partial_write_truncation() {
        let dir = TempDir::new().unwrap();
        let path = wal_path(&dir);

        // Write one valid entry
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.log_insert(1, &[1.0, 2.0, 3.0]).unwrap();
            wal.flush().unwrap();
        }

        // Append garbage (simulating a partial write / crash mid-entry)
        {
            let mut file = OpenOptions::new().append(true).open(&path).unwrap();
            // Write a length prefix that claims 100 bytes, but don't write the body
            file.write_all(&[100, 0, 0, 0]).unwrap();
            file.sync_all().unwrap();
        }

        // Read — should recover only the first entry
        let (entries, _) = Wal::read_entries(&path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].vector_id, 1);
    }

    #[test]
    fn test_wal_absurd_length_prefix_does_not_allocate_or_panic() {
        let dir = TempDir::new().unwrap();
        let path = wal_path(&dir);
        std::fs::write(&path, u32::MAX.to_le_bytes()).unwrap();

        let (entries, valid_up_to) = Wal::read_entries(&path).unwrap();
        assert!(entries.is_empty());
        assert_eq!(valid_up_to, 0);
    }
}
