//! Memory-mapped vector storage.
//!
//! Stores vectors as fixed-size records in a memory-mapped file, preceded
//! by the [`FileHeader`]. Each record is a contiguous block of `dimension` f32 values.
//!
//! ## File Layout
//!
//! ```text
//! [ FileHeader (64 bytes) ][ Vector 0 (dim*4 bytes) ][ Vector 1 ][ ... ]
//! ```

use memmap2::{MmapMut, MmapOptions};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use crate::distance::Metric;
use crate::error::{QuiverError, Result};
use crate::storage::header::{FileHeader, HEADER_SIZE};
use crate::storage::wal::{Wal, WalOp};

/// A memory-mapped vector store backed by a single file + WAL.
pub struct VectorStore {
    /// Path to the main data file.
    _path: PathBuf,
    /// The mutable memory-mapped region.
    mmap: MmapMut,
    /// The backing file handle (kept open for resizing).
    file: File,
    /// The parsed file header (kept in sync with the mmap'd copy).
    header: FileHeader,
    /// The WAL for crash recovery.
    wal: Wal,
    /// Size of a single vector record in bytes (dimension * 4).
    record_size: usize,
    /// Vector IDs durably tombstoned by delete records in the WAL.
    deleted_ids: HashSet<u64>,
}

impl VectorStore {
    /// Create a new, empty vector store at the given path.
    pub fn create(
        data_path: impl AsRef<Path>,
        wal_path: impl AsRef<Path>,
        dimension: u32,
        metric: Metric,
    ) -> Result<Self> {
        let data_path = data_path.as_ref().to_path_buf();
        let record_size = dimension as usize * std::mem::size_of::<f32>();

        // Create the data file with just the header
        let header = FileHeader::new(dimension, metric);
        let header_bytes = header.to_bytes();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&data_path)?;
        file.set_len(HEADER_SIZE as u64)?;

        // Memory-map the file
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        mmap[..HEADER_SIZE].copy_from_slice(&header_bytes);
        mmap.flush()?;

        let mut wal = Wal::open(wal_path)?;
        // `create` defines a fresh store. Any WAL history from a previous
        // database at these paths must not be replayed into it.
        wal.clear()?;

        Ok(Self {
            _path: data_path,
            mmap,
            file,
            header,
            wal,
            record_size,
            deleted_ids: HashSet::new(),
        })
    }

    /// Open an existing vector store, replaying the WAL for crash recovery.
    pub fn open(
        data_path: impl AsRef<Path>,
        wal_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let data_path = data_path.as_ref().to_path_buf();
        let wal_path_buf = wal_path.as_ref().to_path_buf();

        // Open and read the header
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&data_path)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        let header = FileHeader::from_bytes(&mmap[..HEADER_SIZE])?;
        let record_size = header.dimension as usize * std::mem::size_of::<f32>();

        let mut store = Self {
            _path: data_path,
            mmap,
            file,
            header,
            wal: Wal::open(&wal_path_buf)?,
            record_size,
            deleted_ids: HashSet::new(),
        };

        // Replay WAL. Insert replay is idempotent: IDs at or below the
        // checkpointed max ID are already present in the mmap file. Delete
        // entries remain in the WAL as the durable tombstone source until
        // compaction can checkpoint them into a rewritten store.
        let (entries, valid_up_to) = Wal::read_entries(&wal_path_buf)?;
        if !entries.is_empty() {
            tracing::info!(count = entries.len(), "Replaying WAL entries");
            let mut recovered_inserts = false;
            for entry in &entries {
                match entry.op {
                    WalOp::Insert => {
                        if entry.vector_id > store.header.max_vector_id
                            && let Some(ref data) = entry.vector_data
                        {
                            store.insert_raw(entry.vector_id, data)?;
                            recovered_inserts = true;
                        }
                    }
                    WalOp::Delete => {
                        store.deleted_ids.insert(entry.vector_id);
                    }
                }
            }
            // Truncate any corrupt tail
            Wal::truncate(&wal_path_buf, valid_up_to)?;
            if recovered_inserts {
                store.flush()?;
            }
        }

        Ok(store)
    }

    /// Insert a vector into the store.
    ///
    /// Returns the assigned vector ID.
    pub fn insert(&mut self, data: &[f32]) -> Result<u64> {
        if data.len() != self.header.dimension as usize {
            return Err(QuiverError::DimensionMismatch {
                expected: self.header.dimension,
                actual: data.len() as u32,
            });
        }

        let vector_id = self.header.max_vector_id + 1;

        // Log to WAL first (durability guarantee)
        self.wal.log_insert(vector_id, data)?;
        self.wal.flush()?;

        // Then write to the main store
        self.insert_raw(vector_id, data)?;

        Ok(vector_id)
    }

    /// Durably mark a vector ID as deleted.
    ///
    /// The delete record is fsynced to the WAL before the in-memory tombstone
    /// is updated, so a crash cannot acknowledge a delete that is then lost.
    pub fn delete(&mut self, vector_id: u64) -> Result<()> {
        if vector_id == 0
            || vector_id > self.header.max_vector_id
            || self.deleted_ids.contains(&vector_id)
        {
            return Err(QuiverError::NotFound(vector_id));
        }

        self.wal.log_delete(vector_id)?;
        self.wal.flush()?;
        self.deleted_ids.insert(vector_id);
        Ok(())
    }

    /// Return whether a vector ID has been durably tombstoned.
    pub fn is_deleted(&self, vector_id: u64) -> bool {
        self.deleted_ids.contains(&vector_id)
    }

    /// Read a vector by its slot index (0-based).
    pub fn get_vector(&self, slot: usize) -> Result<&[f32]> {
        if slot >= self.header.vector_count as usize {
            return Err(QuiverError::NotFound(slot as u64));
        }

        let offset = HEADER_SIZE + slot * self.record_size;
        let end = offset + self.record_size;

        if end > self.mmap.len() {
            return Err(QuiverError::InvalidFormat(
                "Vector offset exceeds file size".to_string(),
            ));
        }

        let bytes = &self.mmap[offset..end];
        // SAFETY: f32 has alignment of 4, and our records are naturally aligned
        // after a 64-byte header. The data is valid because we wrote it.
        let floats = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                self.header.dimension as usize,
            )
        };
        Ok(floats)
    }

    /// Return the number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.header.vector_count as usize
    }

    /// Return true if the store contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.header.vector_count == 0
    }

    /// Return the vector dimension.
    pub fn dimension(&self) -> u32 {
        self.header.dimension
    }

    /// Return the metric type.
    pub fn metric(&self) -> Metric {
        self.header.metric
    }

    /// Flush the memory-mapped file to disk (fsync).
    ///
    /// Insert replay is idempotent, while delete entries remain in the WAL as
    /// durable tombstones. A later compaction pass can rewrite the live vectors
    /// and safely reset the WAL.
    pub fn flush(&mut self) -> Result<()> {
        // Update the header in the mmap
        let header_bytes = self.header.to_bytes();
        self.mmap[..HEADER_SIZE].copy_from_slice(&header_bytes);
        self.mmap.flush()?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Iterate over all stored vectors as (slot_index, &[f32]) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[f32])> {
        (0..self.header.vector_count as usize).map(move |i| {
            let offset = HEADER_SIZE + i * self.record_size;
            let bytes = &self.mmap[offset..offset + self.record_size];
            let floats = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const f32,
                    self.header.dimension as usize,
                )
            };
            (i, floats)
        })
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Write a vector directly into the mmap'd file (no WAL logging).
    /// Used both by `insert` and by WAL replay.
    fn insert_raw(&mut self, vector_id: u64, data: &[f32]) -> Result<()> {
        let slot = self.header.vector_count as usize;
        let required_size = HEADER_SIZE + (slot + 1) * self.record_size;

        // Grow the file if needed
        if required_size > self.mmap.len() {
            // Grow by at least 2x to amortize resize cost
            let new_size = required_size.max(self.mmap.len() * 2).max(HEADER_SIZE + self.record_size * 64);
            self.file.set_len(new_size as u64)?;
            self.mmap = unsafe { MmapOptions::new().map_mut(&self.file)? };
        }

        // Write vector data
        let offset = HEADER_SIZE + slot * self.record_size;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        self.mmap[offset..offset + self.record_size].copy_from_slice(bytes);

        // Update header
        self.header.vector_count += 1;
        if vector_id > self.header.max_vector_id {
            self.header.max_vector_id = vector_id;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup(dim: u32) -> (TempDir, VectorStore) {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("vectors.qvdb");
        let wal_path = dir.path().join("vectors.wal");
        let store = VectorStore::create(data_path, wal_path, dim, Metric::L2).unwrap();
        (dir, store)
    }

    #[test]
    fn test_create_empty_store() {
        let (_dir, store) = setup(128);
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
        assert_eq!(store.dimension(), 128);
        assert_eq!(store.metric(), Metric::L2);
    }

    #[test]
    fn test_insert_and_read() {
        let (_dir, mut store) = setup(3);
        let data = vec![1.0, 2.0, 3.0];
        let id = store.insert(&data).unwrap();
        assert_eq!(id, 1);
        assert_eq!(store.len(), 1);

        let read_back = store.get_vector(0).unwrap();
        assert_eq!(read_back, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_insert_multiple() {
        let (_dir, mut store) = setup(2);
        store.insert(&[1.0, 2.0]).unwrap();
        store.insert(&[3.0, 4.0]).unwrap();
        store.insert(&[5.0, 6.0]).unwrap();

        assert_eq!(store.len(), 3);
        assert_eq!(store.get_vector(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(store.get_vector(1).unwrap(), &[3.0, 4.0]);
        assert_eq!(store.get_vector(2).unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let (_dir, mut store) = setup(3);
        let result = store.insert(&[1.0, 2.0]); // wrong dimension
        assert!(result.is_err());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let (_dir, store) = setup(3);
        let result = store.get_vector(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush_and_reopen() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("vectors.qvdb");
        let wal_path = dir.path().join("vectors.wal");

        // Create and insert
        {
            let mut store =
                VectorStore::create(&data_path, &wal_path, 3, Metric::Cosine).unwrap();
            store.insert(&[1.0, 2.0, 3.0]).unwrap();
            store.insert(&[4.0, 5.0, 6.0]).unwrap();
            store.flush().unwrap();
        }

        // Reopen
        {
            let store = VectorStore::open(&data_path, &wal_path).unwrap();
            assert_eq!(store.len(), 2);
            assert_eq!(store.metric(), Metric::Cosine);
            assert_eq!(store.get_vector(0).unwrap(), &[1.0, 2.0, 3.0]);
            assert_eq!(store.get_vector(1).unwrap(), &[4.0, 5.0, 6.0]);
        }
    }

    #[test]
    fn test_wal_recovery() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("vectors.qvdb");
        let wal_path = dir.path().join("vectors.wal");

        // Create store and write to WAL but DON'T flush the main store properly
        {
            let mut store =
                VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            // Insert goes through WAL
            store.insert(&[1.0, 2.0]).unwrap();
            store.insert(&[3.0, 4.0]).unwrap();
            // Simulate crash: drop without proper flush/clear
            // The WAL has entries but the main file header says 0 vectors
            // We need to reset the header to simulate the crash
        }

        // Simulate crash: rewrite the header to show 0 vectors
        // (as if the mmap header flush didn't happen)
        {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&data_path)
                .unwrap();
            let mut mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
            let mut header = FileHeader::new(2, Metric::L2);
            header.vector_count = 0;
            header.max_vector_id = 0;
            let bytes = header.to_bytes();
            mmap[..HEADER_SIZE].copy_from_slice(&bytes);
            mmap.flush().unwrap();
        }

        // Reopen — WAL should replay the 2 inserts
        {
            let store = VectorStore::open(&data_path, &wal_path).unwrap();
            assert_eq!(store.len(), 2);
            assert_eq!(store.get_vector(0).unwrap(), &[1.0, 2.0]);
            assert_eq!(store.get_vector(1).unwrap(), &[3.0, 4.0]);
        }
    }

    #[test]
    fn test_delete_persists_after_flush_and_reopen() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("delete_persist.qvdb");
        let wal_path = dir.path().join("delete_persist.wal");

        let deleted_id;
        let live_id;
        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            deleted_id = store.insert(&[1.0, 2.0]).unwrap();
            live_id = store.insert(&[3.0, 4.0]).unwrap();
            store.delete(deleted_id).unwrap();
            store.flush().unwrap();
        }

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 2);
        assert!(store.is_deleted(deleted_id));
        assert!(!store.is_deleted(live_id));
    }

    #[test]
    fn test_delete_recovers_without_store_flush() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("delete_recovery.qvdb");
        let wal_path = dir.path().join("delete_recovery.wal");

        let deleted_id;
        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            deleted_id = store.insert(&[1.0, 2.0]).unwrap();
            store.insert(&[3.0, 4.0]).unwrap();
            store.delete(deleted_id).unwrap();
            // Drop without flushing the mmap header, simulating recovery from
            // operations that were durable only in the WAL.
        }

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 2);
        assert!(store.is_deleted(deleted_id));
        assert_eq!(store.get_vector(1).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn test_create_resets_previous_wal_history() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("recreate.qvdb");
        let wal_path = dir.path().join("recreate.wal");

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            let id = store.insert(&[1.0, 2.0]).unwrap();
            store.delete(id).unwrap();
        }

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            let id = store.insert(&[3.0, 4.0]).unwrap();
            store.flush().unwrap();
            assert_eq!(id, 1);
            assert!(!store.is_deleted(id));
        }

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_deleted(1));
        assert_eq!(store.get_vector(0).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn test_iterate_vectors() {
        let (_dir, mut store) = setup(2);
        store.insert(&[1.0, 2.0]).unwrap();
        store.insert(&[3.0, 4.0]).unwrap();

        let collected: Vec<(usize, Vec<f32>)> = store
            .iter()
            .map(|(i, v)| (i, v.to_vec()))
            .collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], (0, vec![1.0, 2.0]));
        assert_eq!(collected[1], (1, vec![3.0, 4.0]));
    }
}
