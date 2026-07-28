//! Memory-mapped vector storage.
//!
//! Stores vectors as fixed-size records in a memory-mapped file, preceded
//! by the [`FileHeader`]. Version 2 records keep a stable vector ID alongside
//! each contiguous block of `dimension` f32 values.
//!
//! ## File Layout
//!
//! ```text
//! [ FileHeader (64 bytes) ][ ID (u64) + Vector 0 ][ ID + Vector 1 ][ ... ]
//! ```

use memmap2::{MmapMut, MmapOptions};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::distance::Metric;
use crate::error::{QuiverError, Result};
use crate::storage::format::{VECTOR_ID_SIZE, parse_file_bytes};
use crate::storage::header::{FileHeader, HEADER_SIZE, LEGACY_FORMAT_VERSION};
use crate::storage::wal::{Wal, WalOp};

const COMPACTION_PREPARED: &[u8] = b"prepared";
const COMPACTION_OLD_MOVED: &[u8] = b"old_moved";
const COMPACTION_INSTALLED: &[u8] = b"installed";

struct CompactionPaths {
    temp_data: PathBuf,
    temp_wal: PathBuf,
    backup_data: PathBuf,
    backup_wal: PathBuf,
    marker: PathBuf,
}

impl CompactionPaths {
    fn new(data_path: &Path, wal_path: &Path) -> Self {
        Self {
            temp_data: sidecar_path(data_path, ".compact.tmp"),
            temp_wal: sidecar_path(wal_path, ".compact.tmp"),
            backup_data: sidecar_path(data_path, ".compact.bak"),
            backup_wal: sidecar_path(wal_path, ".compact.bak"),
            marker: sidecar_path(data_path, ".compact.marker"),
        }
    }
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

/// A memory-mapped vector store backed by a single file + WAL.
pub struct VectorStore {
    /// Path to the main data file.
    path: PathBuf,
    /// Path to the write-ahead log.
    wal_path: PathBuf,
    /// The mutable memory-mapped region.
    mmap: Option<MmapMut>,
    /// The backing file handle (kept open for resizing).
    file: Option<File>,
    /// The parsed file header (kept in sync with the mmap'd copy).
    header: FileHeader,
    /// The WAL for crash recovery.
    wal: Option<Wal>,
    /// Size of a single vector record in bytes.
    record_size: usize,
    /// Vector IDs durably tombstoned by delete records in the WAL.
    deleted_ids: HashSet<u64>,
    /// Stable vector ID for each physical slot.
    vector_ids: Vec<u64>,
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
        let wal_path = wal_path.as_ref().to_path_buf();
        Self::recover_compaction(&data_path, &wal_path)?;
        let record_size = VECTOR_ID_SIZE + dimension as usize * std::mem::size_of::<f32>();

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

        let mut wal = Wal::open(&wal_path)?;
        // `create` defines a fresh store. Any WAL history from a previous
        // database at these paths must not be replayed into it.
        wal.clear()?;

        Ok(Self {
            path: data_path,
            wal_path,
            mmap: Some(mmap),
            file: Some(file),
            header,
            wal: Some(wal),
            record_size,
            deleted_ids: HashSet::new(),
            vector_ids: Vec::new(),
        })
    }

    /// Open an existing vector store, replaying the WAL for crash recovery.
    pub fn open(data_path: impl AsRef<Path>, wal_path: impl AsRef<Path>) -> Result<Self> {
        let data_path = data_path.as_ref().to_path_buf();
        let wal_path_buf = wal_path.as_ref().to_path_buf();
        Self::recover_compaction(&data_path, &wal_path_buf)?;

        // Open and read the header
        let file = OpenOptions::new().read(true).write(true).open(&data_path)?;

        let file_len = file.metadata()?.len();
        if file_len < HEADER_SIZE as u64 {
            return Err(QuiverError::InvalidFormat(format!(
                "File too short: {file_len} bytes, expected at least {HEADER_SIZE}"
            )));
        }

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        let parsed = parse_file_bytes(&mmap)?;

        let mut store = Self {
            path: data_path,
            wal_path: wal_path_buf.clone(),
            mmap: Some(mmap),
            file: Some(file),
            header: parsed.header,
            wal: Some(Wal::open(&wal_path_buf)?),
            record_size: parsed.record_size,
            deleted_ids: HashSet::new(),
            vector_ids: parsed.vector_ids,
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
        let wal = self.wal.as_mut().expect("vector store WAL is open");
        wal.log_insert(vector_id, data)?;
        wal.flush()?;
        ordinary_write_failpoint("wal_durable");

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
            || !self.vector_ids.contains(&vector_id)
            || self.deleted_ids.contains(&vector_id)
        {
            return Err(QuiverError::NotFound(vector_id));
        }

        let wal = self.wal.as_mut().expect("vector store WAL is open");
        wal.log_delete(vector_id)?;
        wal.flush()?;
        self.deleted_ids.insert(vector_id);
        Ok(())
    }

    /// Return whether a vector ID has been durably tombstoned.
    pub fn is_deleted(&self, vector_id: u64) -> bool {
        self.deleted_ids.contains(&vector_id)
    }

    /// Return the stable vector ID stored in a physical slot.
    pub fn vector_id(&self, slot: usize) -> Result<u64> {
        self.vector_ids
            .get(slot)
            .copied()
            .ok_or(QuiverError::NotFound(slot as u64))
    }

    /// Read a vector by its slot index (0-based).
    pub fn get_vector(&self, slot: usize) -> Result<&[f32]> {
        if slot >= self.header.vector_count as usize {
            return Err(QuiverError::NotFound(slot as u64));
        }

        let offset = HEADER_SIZE + slot * self.record_size;
        let end = offset + self.record_size;

        let mmap = self.mmap.as_ref().expect("vector store mmap is open");
        if end > mmap.len() {
            return Err(QuiverError::InvalidFormat(
                "Vector offset exceeds file size".to_string(),
            ));
        }

        let vector_offset = offset + self.vector_data_offset();
        let bytes = &mmap[vector_offset..end];
        // SAFETY: f32 has alignment of 4, and our records are naturally aligned
        // after a 64-byte header. The data is valid because we wrote it.
        let floats = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.header.dimension as usize)
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
        let mmap = self.mmap.as_mut().expect("vector store mmap is open");
        mmap[..HEADER_SIZE].copy_from_slice(&header_bytes);
        mmap.flush()?;
        self.file
            .as_ref()
            .expect("vector store file is open")
            .sync_all()?;
        Ok(())
    }

    /// Return the current WAL size in bytes.
    pub fn wal_len(&self) -> Result<u64> {
        Ok(fs::metadata(&self.wal_path)?.len())
    }

    /// Rewrite the store with only live vectors and atomically install it.
    ///
    /// The existing data and WAL remain untouched until the replacement pair
    /// has been fully flushed. A small marker journals the two-file rename so
    /// `open` can roll back or roll forward after a crash at any swap step.
    pub fn compact(&mut self) -> Result<()> {
        let live_vectors: Vec<(u64, Vec<f32>)> = (0..self.len())
            .filter_map(|slot| {
                let vector_id = self.vector_ids[slot];
                (!self.deleted_ids.contains(&vector_id)).then(|| {
                    (
                        vector_id,
                        self.get_vector(slot).expect("valid slot").to_vec(),
                    )
                })
            })
            .collect();
        let max_vector_id = self.header.max_vector_id;
        let dimension = self.header.dimension;
        let metric = self.header.metric;
        let paths = CompactionPaths::new(&self.path, &self.wal_path);

        Self::recover_compaction(&self.path, &self.wal_path)?;
        remove_file_if_exists(&paths.temp_data)?;
        remove_file_if_exists(&paths.temp_wal)?;

        {
            let mut replacement =
                Self::create(&paths.temp_data, &paths.temp_wal, dimension, metric)?;
            for (vector_id, vector) in &live_vectors {
                replacement.insert_raw(*vector_id, vector)?;
            }
            replacement.header.max_vector_id = max_vector_id;
            replacement.flush()?;
            let wal = replacement.wal.as_mut().expect("replacement WAL is open");
            wal.clear()?;
            wal.flush()?;
        }
        sync_parent(&paths.temp_data)?;
        sync_parent(&paths.temp_wal)?;

        write_marker(&paths.marker, COMPACTION_PREPARED)?;
        compaction_failpoint("replacement_durable");

        // Windows cannot reliably rename an open mmap. Close every handle to
        // the old pair only after the replacement is durable.
        drop(self.mmap.take());
        drop(self.file.take());
        drop(self.wal.take());

        let swap_result = (|| -> Result<()> {
            fs::rename(&self.path, &paths.backup_data)?;
            sync_parent(&self.path)?;
            compaction_failpoint("data_backed_up");
            fs::rename(&self.wal_path, &paths.backup_wal)?;
            sync_parent(&self.wal_path)?;
            write_marker(&paths.marker, COMPACTION_OLD_MOVED)?;
            compaction_failpoint("old_pair_moved");

            fs::rename(&paths.temp_data, &self.path)?;
            sync_parent(&self.path)?;
            compaction_failpoint("new_data_installed");
            fs::rename(&paths.temp_wal, &self.wal_path)?;
            sync_parent(&self.wal_path)?;
            write_marker(&paths.marker, COMPACTION_INSTALLED)?;

            remove_file_if_exists(&paths.backup_data)?;
            remove_file_if_exists(&paths.backup_wal)?;
            remove_file_if_exists(&paths.marker)?;
            Ok(())
        })();

        if let Err(error) = swap_result {
            Self::recover_compaction(&self.path, &self.wal_path)?;
            let reopened = Self::open(&self.path, &self.wal_path)?;
            let installed = reopened.deleted_ids.is_empty()
                && reopened.vector_ids
                    == live_vectors
                        .iter()
                        .map(|(vector_id, _)| *vector_id)
                        .collect::<Vec<_>>();
            *self = reopened;
            return if installed { Ok(()) } else { Err(error) };
        }

        let reopened = Self::open(&self.path, &self.wal_path)?;
        *self = reopened;
        Ok(())
    }

    /// Iterate over all stored vectors as (slot_index, &[f32]) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[f32])> {
        (0..self.header.vector_count as usize).map(move |i| {
            let offset = HEADER_SIZE + i * self.record_size;
            let vector_offset = offset + self.vector_data_offset();
            let mmap = self.mmap.as_ref().expect("vector store mmap is open");
            let bytes = &mmap[vector_offset..offset + self.record_size];
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
        if required_size > self.mmap.as_ref().expect("vector store mmap is open").len() {
            // Grow by at least 2x to amortize resize cost
            let mmap_len = self.mmap.as_ref().expect("vector store mmap is open").len();
            let new_size = required_size
                .max(mmap_len * 2)
                .max(HEADER_SIZE + self.record_size * 64);
            let file = self.file.as_ref().expect("vector store file is open");
            file.set_len(new_size as u64)?;
            self.mmap = Some(unsafe { MmapOptions::new().map_mut(file)? });
        }

        // Write stable vector ID followed by vector data.
        let offset = HEADER_SIZE + slot * self.record_size;
        let vector_offset = offset + self.vector_data_offset();
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
        let mmap = self.mmap.as_mut().expect("vector store mmap is open");
        if self.header.version != LEGACY_FORMAT_VERSION {
            mmap[offset..vector_offset].copy_from_slice(&vector_id.to_le_bytes());
        }
        mmap[vector_offset..offset + self.record_size].copy_from_slice(bytes);

        // Update header
        self.header.vector_count += 1;
        self.vector_ids.push(vector_id);
        if vector_id > self.header.max_vector_id {
            self.header.max_vector_id = vector_id;
        }

        Ok(())
    }

    fn vector_data_offset(&self) -> usize {
        if self.header.version == LEGACY_FORMAT_VERSION {
            0
        } else {
            VECTOR_ID_SIZE
        }
    }

    fn recover_compaction(data_path: &Path, wal_path: &Path) -> Result<()> {
        let paths = CompactionPaths::new(data_path, wal_path);
        if !paths.marker.exists() {
            return Ok(());
        }

        let phase = fs::read(&paths.marker).unwrap_or_default();
        let data_exists = data_path.exists();
        let wal_exists = wal_path.exists();

        if data_exists && wal_exists {
            remove_file_if_exists(&paths.temp_data)?;
            remove_file_if_exists(&paths.temp_wal)?;
            remove_file_if_exists(&paths.backup_data)?;
            remove_file_if_exists(&paths.backup_wal)?;
            remove_file_if_exists(&paths.marker)?;
            return Ok(());
        }

        if phase == COMPACTION_OLD_MOVED || phase == COMPACTION_INSTALLED {
            if !data_exists && paths.temp_data.exists() {
                fs::rename(&paths.temp_data, data_path)?;
            }
            if !wal_exists && paths.temp_wal.exists() {
                fs::rename(&paths.temp_wal, wal_path)?;
            }
        } else {
            if !data_exists && paths.backup_data.exists() {
                fs::rename(&paths.backup_data, data_path)?;
            }
            if !wal_exists && paths.backup_wal.exists() {
                fs::rename(&paths.backup_wal, wal_path)?;
            }
        }

        if !data_path.exists() || !wal_path.exists() {
            return Err(QuiverError::InvalidFormat(
                "Unable to recover interrupted compaction".to_string(),
            ));
        }

        remove_file_if_exists(&paths.temp_data)?;
        remove_file_if_exists(&paths.temp_wal)?;
        remove_file_if_exists(&paths.backup_data)?;
        remove_file_if_exists(&paths.backup_wal)?;
        remove_file_if_exists(&paths.marker)?;
        Ok(())
    }
}

fn write_marker(path: &Path, phase: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    file.write_all(phase)?;
    file.sync_all()?;
    sync_parent(path)?;
    Ok(())
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> Result<()> {
    // Windows does not expose directory fsync through std. Each replacement
    // file and journal marker is still individually flushed before renames.
    Ok(())
}

#[cfg(test)]
fn compaction_failpoint(name: &str) {
    if std::env::var("QUIVER_COMPACTION_FAILPOINT").as_deref() != Ok(name) {
        return;
    }

    if let Ok(signal_path) = std::env::var("QUIVER_COMPACTION_SIGNAL") {
        fs::write(signal_path, name).unwrap();
    }

    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

#[cfg(not(test))]
fn compaction_failpoint(_name: &str) {}

#[cfg(test)]
fn ordinary_write_failpoint(name: &str) {
    if std::env::var("QUIVER_WRITE_FAILPOINT").as_deref() != Ok(name) {
        return;
    }

    if let Ok(signal_path) = std::env::var("QUIVER_WRITE_SIGNAL") {
        fs::write(signal_path, name).unwrap();
    }

    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

#[cfg(not(test))]
fn ordinary_write_failpoint(_name: &str) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    fn setup(dim: u32) -> (TempDir, VectorStore) {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("vectors.qvdb");
        let wal_path = dir.path().join("vectors.wal");
        let store = VectorStore::create(data_path, wal_path, dim, Metric::L2).unwrap();
        (dir, store)
    }

    fn assert_invalid_open(data_path: &Path, wal_path: &Path) {
        match VectorStore::open(data_path, wal_path) {
            Err(QuiverError::InvalidFormat(_)) => {}
            Err(error) => panic!("expected InvalidFormat, got {error:?}"),
            Ok(_) => panic!("corrupted file unexpectedly opened"),
        }
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
    fn test_open_zero_byte_file_returns_invalid_format() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("zero.qvdb");
        let wal_path = dir.path().join("zero.wal");
        File::create(&data_path).unwrap();
        File::create(&wal_path).unwrap();

        assert_invalid_open(&data_path, &wal_path);
    }

    #[test]
    fn test_open_short_header_returns_invalid_format() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("short.qvdb");
        let wal_path = dir.path().join("short.wal");
        fs::write(&data_path, b"QVDB\x02").unwrap();
        File::create(&wal_path).unwrap();

        assert_invalid_open(&data_path, &wal_path);
    }

    #[test]
    fn test_open_corrupted_header_field_returns_invalid_format() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("bad_dimension.qvdb");
        let wal_path = dir.path().join("bad_dimension.wal");
        let header = FileHeader::new(0, Metric::L2);
        fs::write(&data_path, header.to_bytes()).unwrap();
        File::create(&wal_path).unwrap();

        assert_invalid_open(&data_path, &wal_path);
    }

    #[test]
    fn test_open_truncated_final_record_returns_invalid_format_for_v1_and_v2() {
        for version in [
            LEGACY_FORMAT_VERSION,
            crate::storage::header::FORMAT_VERSION,
        ] {
            let dir = TempDir::new().unwrap();
            let data_path = dir.path().join(format!("truncated_v{version}.qvdb"));
            let wal_path = dir.path().join(format!("truncated_v{version}.wal"));
            let mut header = FileHeader::new(2, Metric::L2);
            header.version = version;
            header.vector_count = 1;
            header.max_vector_id = 1;

            let mut bytes = header.to_bytes();
            if version != LEGACY_FORMAT_VERSION {
                bytes.extend_from_slice(&1_u64.to_le_bytes());
            }
            bytes.extend_from_slice(&1.0_f32.to_le_bytes());
            fs::write(&data_path, bytes).unwrap();
            File::create(&wal_path).unwrap();

            assert_invalid_open(&data_path, &wal_path);
        }
    }

    #[test]
    fn test_flush_and_reopen() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("vectors.qvdb");
        let wal_path = dir.path().join("vectors.wal");

        // Create and insert
        {
            let mut store = VectorStore::create(&data_path, &wal_path, 3, Metric::Cosine).unwrap();
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
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
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
    fn test_compaction_keeps_only_live_vectors_and_resets_wal() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("compact.qvdb");
        let wal_path = dir.path().join("compact.wal");
        let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();

        let id1 = store.insert(&[1.0, 1.0]).unwrap();
        let id2 = store.insert(&[2.0, 2.0]).unwrap();
        let deleted_id = store.insert(&[3.0, 3.0]).unwrap();
        store.delete(deleted_id).unwrap();
        let wal_len_before = store.wal_len().unwrap();
        assert!(wal_len_before > 0);

        store.compact().unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.vector_id(0).unwrap(), id1);
        assert_eq!(store.vector_id(1).unwrap(), id2);
        assert_eq!(store.get_vector(0).unwrap(), &[1.0, 1.0]);
        assert_eq!(store.get_vector(1).unwrap(), &[2.0, 2.0]);
        assert!(!store.is_deleted(deleted_id));
        assert!(store.delete(deleted_id).is_err());
        assert_eq!(store.wal_len().unwrap(), 0);

        // Compaction must preserve the monotonic ID sequence even when the
        // highest existing slots were removed.
        let next_id = store.insert(&[4.0, 4.0]).unwrap();
        assert_eq!(next_id, 4);
    }

    #[test]
    fn test_compaction_migrates_legacy_records_and_preserves_ids() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("legacy.qvdb");
        let wal_path = dir.path().join("legacy.wal");

        let mut header = FileHeader::new(2, Metric::L2);
        header.version = LEGACY_FORMAT_VERSION;
        header.vector_count = 2;
        header.max_vector_id = 2;

        {
            let mut file = File::create(&data_path).unwrap();
            file.write_all(&header.to_bytes()).unwrap();
            for value in [1.0_f32, 2.0, 3.0, 4.0] {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            file.sync_all().unwrap();
            File::create(&wal_path).unwrap().sync_all().unwrap();
        }

        let mut store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.vector_id(0).unwrap(), 1);
        assert_eq!(store.vector_id(1).unwrap(), 2);
        assert_eq!(store.get_vector(1).unwrap(), &[3.0, 4.0]);

        store.delete(1).unwrap();
        store.compact().unwrap();
        assert_ne!(store.header.version, LEGACY_FORMAT_VERSION);
        assert_eq!(store.len(), 1);
        assert_eq!(store.vector_id(0).unwrap(), 2);
        assert_eq!(store.get_vector(0).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn compaction_kill_child() {
        if std::env::var("QUIVER_COMPACTION_CHILD").as_deref() != Ok("1") {
            return;
        }

        let data_path = PathBuf::from(std::env::var("QUIVER_COMPACTION_DATA").unwrap());
        let wal_path = PathBuf::from(std::env::var("QUIVER_COMPACTION_WAL").unwrap());
        let mut store = VectorStore::open(data_path, wal_path).unwrap();
        store.compact().unwrap();
    }

    #[test]
    fn test_kill_mid_compaction_recovers_without_data_loss() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("kill_compact.qvdb");
        let wal_path = dir.path().join("kill_compact.wal");
        let signal_path = dir.path().join("compaction.signal");

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            store.insert(&[1.0, 1.0]).unwrap();
            let deleted_id = store.insert(&[2.0, 2.0]).unwrap();
            store.insert(&[3.0, 3.0]).unwrap();
            store.delete(deleted_id).unwrap();
            store.flush().unwrap();
        }

        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("storage::vecstore::tests::compaction_kill_child")
            .arg("--nocapture")
            .env("QUIVER_COMPACTION_CHILD", "1")
            .env("QUIVER_COMPACTION_DATA", &data_path)
            .env("QUIVER_COMPACTION_WAL", &wal_path)
            .env("QUIVER_COMPACTION_FAILPOINT", "old_pair_moved")
            .env("QUIVER_COMPACTION_SIGNAL", &signal_path)
            .spawn()
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !signal_path.exists() && Instant::now() < deadline {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("compaction child exited before failpoint: {status}");
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        assert!(signal_path.exists(), "compaction failpoint was not reached");

        child.kill().unwrap();
        child.wait().unwrap();

        // Opening performs journal recovery. At this failpoint both old files
        // are backups and the durable replacement pair is waiting to install.
        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.vector_id(0).unwrap(), 1);
        assert_eq!(store.vector_id(1).unwrap(), 3);
        assert_eq!(store.get_vector(0).unwrap(), &[1.0, 1.0]);
        assert_eq!(store.get_vector(1).unwrap(), &[3.0, 3.0]);
        assert_eq!(store.wal_len().unwrap(), 0);
    }

    #[test]
    fn ordinary_insert_kill_child() {
        if std::env::var("QUIVER_WRITE_CHILD").as_deref() != Ok("1") {
            return;
        }

        let data_path = PathBuf::from(std::env::var("QUIVER_WRITE_DATA").unwrap());
        let wal_path = PathBuf::from(std::env::var("QUIVER_WRITE_WAL").unwrap());
        let mut store = VectorStore::open(data_path, wal_path).unwrap();
        store.insert(&[9.0, 10.0]).unwrap();
    }

    #[test]
    fn test_process_kill_after_wal_fsync_recovers_complete_insert() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("kill_insert.qvdb");
        let wal_path = dir.path().join("kill_insert.wal");
        let signal_path = dir.path().join("write.signal");

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            store.insert(&[1.0, 2.0]).unwrap();
            store.flush().unwrap();
        }

        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("storage::vecstore::tests::ordinary_insert_kill_child")
            .arg("--nocapture")
            .env("QUIVER_WRITE_CHILD", "1")
            .env("QUIVER_WRITE_DATA", &data_path)
            .env("QUIVER_WRITE_WAL", &wal_path)
            .env("QUIVER_WRITE_FAILPOINT", "wal_durable")
            .env("QUIVER_WRITE_SIGNAL", &signal_path)
            .spawn()
            .unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !signal_path.exists() && Instant::now() < deadline {
            if let Some(status) = child.try_wait().unwrap() {
                panic!("insert child exited before failpoint: {status}");
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        assert!(signal_path.exists(), "write failpoint was not reached");

        // `kill` is SIGKILL on Unix and TerminateProcess on Windows. The WAL
        // record is already fsynced, but insert_raw has not modified the mmap.
        child.kill().unwrap();
        child.wait().unwrap();

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.get_vector(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(store.get_vector(1).unwrap(), &[9.0, 10.0]);
        assert_eq!(store.vector_id(1).unwrap(), 2);

        // This kill point intentionally recovers the last complete record;
        // partial-tail truncation remains covered by the lower-level WAL tests.
        let (entries, valid_up_to) = Wal::read_entries(&wal_path).unwrap();
        assert_eq!(entries.last().unwrap().vector_id, 2);
        assert_eq!(valid_up_to, fs::metadata(&wal_path).unwrap().len());
    }

    #[test]
    fn test_iterate_vectors() {
        let (_dir, mut store) = setup(2);
        store.insert(&[1.0, 2.0]).unwrap();
        store.insert(&[3.0, 4.0]).unwrap();

        let collected: Vec<(usize, Vec<f32>)> =
            store.iter().map(|(i, v)| (i, v.to_vec())).collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0], (0, vec![1.0, 2.0]));
        assert_eq!(collected[1], (1, vec![3.0, 4.0]));
    }
}
