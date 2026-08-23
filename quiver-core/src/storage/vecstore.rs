//! Memory-mapped vector storage.
//!
//! Stores vectors as fixed-size records in a memory-mapped file, preceded
//! by the [`FileHeader`]. Version 2 records keep a stable vector ID alongside
//! each contiguous block of `dimension` f32 values. Version 3 records are
//! byte-identical; the version bump signals that vectors may carry metadata.
//!
//! ## File Layout
//!
//! ```text
//! [ FileHeader (64 bytes) ][ ID (u64) + Vector 0 ][ ID + Vector 1 ][ ... ]
//! ```
//!
//! ## Metadata
//!
//! Metadata never goes inline in the fixed-size records (that would break the
//! hot-path layout). It is kept in a slot-indexed in-memory vector, logged to
//! the WAL as `InsertMeta` entries, and checkpointed to a CRC32-protected
//! `<data_path>.meta` snapshot on `flush`/`compact`. On `open` the snapshot is
//! loaded when it validates; otherwise metadata is rebuilt from WAL replay.

use memmap2::{MmapMut, MmapOptions};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::distance::Metric;
use crate::error::{QuiverError, Result};
use crate::metadata::Metadata;
use crate::storage::format::{VECTOR_ID_SIZE, parse_file_bytes};
use crate::storage::header::{FORMAT_VERSION, FileHeader, HEADER_SIZE, LEGACY_FORMAT_VERSION};
use crate::storage::wal::{Wal, WalOp};

const COMPACTION_PREPARED: &[u8] = b"prepared";
const COMPACTION_OLD_MOVED: &[u8] = b"old_moved";
const COMPACTION_INSTALLED: &[u8] = b"installed";

/// Magic bytes identifying a persisted metadata snapshot.
const META_MAGIC: &[u8; 4] = b"QVMD";

/// Current metadata-snapshot format version.
const META_FORMAT_VERSION: u8 = 1;

/// Size of the metadata-snapshot header in bytes (through and including the
/// header CRC).
const META_HEADER_SIZE: usize = 4 + 1 + 3 + 8 + 8 + 8 + 4;

struct CompactionPaths {
    temp_data: PathBuf,
    temp_wal: PathBuf,
    temp_meta: PathBuf,
    backup_data: PathBuf,
    backup_wal: PathBuf,
    backup_meta: PathBuf,
    marker: PathBuf,
}

impl CompactionPaths {
    fn new(data_path: &Path, wal_path: &Path) -> Self {
        Self {
            temp_data: sidecar_path(data_path, ".compact.tmp"),
            temp_wal: sidecar_path(wal_path, ".compact.tmp"),
            // The replacement store writes its snapshot next to its own data
            // file (`<temp_data>.meta`), so the temp meta path must match.
            temp_meta: sidecar_path(data_path, ".compact.tmp.meta"),
            backup_data: sidecar_path(data_path, ".compact.bak"),
            backup_wal: sidecar_path(wal_path, ".compact.bak"),
            backup_meta: sidecar_path(data_path, ".meta.compact.bak"),
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
    /// Metadata attached to each physical slot, parallel to `vector_ids`.
    /// `None` for slots whose vector was inserted without metadata.
    metadata: Vec<Option<Metadata>>,
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
        // Same reasoning: a stale metadata snapshot from a previous database
        // at these paths must not be loaded by a later `open`.
        remove_file_if_exists(&sidecar_path(&data_path, ".meta"))?;

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
            metadata: Vec::new(),
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
        let metadata = vec![None; parsed.vector_ids.len()];

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
            metadata,
        };

        // Load the metadata snapshot when present and valid. A missing or
        // corrupt snapshot falls back to WAL replay, which still carries every
        // metadata entry logged since the last compaction.
        store.load_meta_snapshot();

        // Replay WAL. Insert replay is idempotent: IDs at or below the
        // checkpointed max ID are already present in the mmap file. Delete
        // entries remain in the WAL as the durable tombstone source until
        // compaction can checkpoint them into a rewritten store.
        let (entries, valid_up_to) = Wal::read_entries(&wal_path_buf)?;
        if !entries.is_empty() {
            tracing::info!(count = entries.len(), "Replaying WAL entries");
            let mut recovered = false;
            for entry in &entries {
                match entry.op {
                    WalOp::Insert => {
                        if entry.vector_id > store.header.max_vector_id
                            && let Some(ref data) = entry.vector_data
                        {
                            store.insert_raw(entry.vector_id, data, None)?;
                            recovered = true;
                        }
                    }
                    WalOp::InsertMeta => {
                        if entry.vector_id > store.header.max_vector_id
                            && let Some(ref data) = entry.vector_data
                        {
                            store.insert_raw(entry.vector_id, data, entry.metadata.clone())?;
                            recovered = true;
                        } else if let Some(slot) = store.slot_for_id(entry.vector_id)
                            && store.metadata[slot].is_none()
                            && let Some(ref metadata) = entry.metadata
                        {
                            // The vector itself is already in the data file;
                            // recover just its metadata (e.g. the snapshot was
                            // missing or corrupt).
                            store.metadata[slot] = Some(metadata.clone());
                            recovered = true;
                        }
                    }
                    WalOp::Delete => {
                        store.deleted_ids.insert(entry.vector_id);
                    }
                }
            }
            // Truncate any corrupt tail
            Wal::truncate(&wal_path_buf, valid_up_to)?;
            if recovered {
                store.flush()?;
            }
        }

        Ok(store)
    }

    /// Insert a vector into the store.
    ///
    /// Returns the assigned vector ID.
    pub fn insert(&mut self, data: &[f32]) -> Result<u64> {
        self.insert_inner(data, None)
    }

    /// Insert a vector with metadata into the store.
    ///
    /// Returns the assigned vector ID. The metadata is logged to the WAL as an
    /// `InsertMeta` entry before the vector is written, so it shares the
    /// vector's durability guarantee.
    pub fn insert_with_metadata(&mut self, data: &[f32], metadata: Metadata) -> Result<u64> {
        self.insert_inner(data, Some(metadata))
    }

    fn insert_inner(&mut self, data: &[f32], metadata: Option<Metadata>) -> Result<u64> {
        if data.len() != self.header.dimension as usize {
            return Err(QuiverError::DimensionMismatch {
                expected: self.header.dimension,
                actual: data.len() as u32,
            });
        }

        let vector_id = self.header.max_vector_id + 1;

        // Log to WAL first (durability guarantee)
        let wal = self.wal.as_mut().expect("vector store WAL is open");
        match &metadata {
            Some(metadata) => wal.log_insert_meta(vector_id, metadata, data)?,
            None => wal.log_insert(vector_id, data)?,
        }
        wal.flush()?;
        ordinary_write_failpoint("wal_durable");

        // Then write to the main store
        self.insert_raw(vector_id, data, metadata)?;

        Ok(vector_id)
    }

    /// Insert a batch of vectors with a single WAL fsync (group commit).
    ///
    /// All vectors are validated first, then logged to the WAL and made durable
    /// with one fsync for the whole batch, then written to the mmap. This
    /// amortizes the per-insert fsync cost for bulk ingestion while preserving
    /// the durability guarantee — the entire batch is durable before returning,
    /// and a crash mid-batch replays exactly the durable prefix.
    pub fn insert_batch(&mut self, batch: &[&[f32]]) -> Result<Vec<u64>> {
        self.insert_batch_inner(batch, None)
    }

    /// Insert a batch of vectors with per-vector metadata and a single WAL
    /// fsync (group commit).
    ///
    /// `metadata` must have exactly one entry per vector; `None` entries are
    /// inserted without metadata. See [`Self::insert_batch`] for the
    /// durability semantics.
    pub fn insert_batch_with_metadata(
        &mut self,
        batch: &[&[f32]],
        metadata: &[Option<Metadata>],
    ) -> Result<Vec<u64>> {
        if batch.len() != metadata.len() {
            return Err(QuiverError::InvalidFormat(format!(
                "Batch has {} vectors but {} metadata entries",
                batch.len(),
                metadata.len()
            )));
        }
        self.insert_batch_inner(batch, Some(metadata))
    }

    fn insert_batch_inner(
        &mut self,
        batch: &[&[f32]],
        metadata: Option<&[Option<Metadata>]>,
    ) -> Result<Vec<u64>> {
        // Validate every dimension up front so a bad vector cannot partially commit.
        for data in batch {
            if data.len() != self.header.dimension as usize {
                return Err(QuiverError::DimensionMismatch {
                    expected: self.header.dimension,
                    actual: data.len() as u32,
                });
            }
        }
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let base = self.header.max_vector_id;
        let wal = self.wal.as_mut().expect("vector store WAL is open");
        let mut ids = Vec::with_capacity(batch.len());
        for (i, data) in batch.iter().enumerate() {
            let vector_id = base + 1 + i as u64;
            match metadata.and_then(|slice| slice[i].as_ref()) {
                Some(meta) => wal.log_insert_meta(vector_id, meta, data)?,
                None => wal.log_insert(vector_id, data)?,
            }
            ids.push(vector_id);
        }
        // Single fsync for the whole batch (group commit).
        wal.flush()?;
        ordinary_write_failpoint("wal_durable");

        // Write all vectors to the mmap.
        for (i, data) in batch.iter().enumerate() {
            let vector_id = base + 1 + i as u64;
            let meta = metadata.and_then(|slice| slice[i].clone());
            self.insert_raw(vector_id, data, meta)?;
        }

        Ok(ids)
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

    /// Return the metadata attached to the vector at `slot`, if any.
    ///
    /// Slots whose vector was inserted without metadata return `None`.
    #[inline]
    pub fn metadata(&self, slot: usize) -> Option<&Metadata> {
        debug_assert!(
            slot < self.header.vector_count as usize,
            "slot {slot} out of range (count {})",
            self.header.vector_count
        );
        self.metadata.get(slot).and_then(|entry| entry.as_ref())
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

    /// Read a vector by slot without bounds checking or `Result` wrapping.
    ///
    /// This is the hot-path accessor used by graph traversal, where every slot
    /// is known to be valid because it came from the graph structure. It skips
    /// the bounds check and error-path overhead of [`Self::get_vector`].
    ///
    /// # Panics (debug only)
    /// Debug-asserts that `slot` is in range. In release builds an out-of-range
    /// slot is undefined behavior — callers must only pass slots `< len()`.
    #[inline]
    pub fn get_vector_unchecked(&self, slot: usize) -> &[f32] {
        debug_assert!(
            slot < self.header.vector_count as usize,
            "slot {slot} out of range (count {})",
            self.header.vector_count
        );
        let offset = HEADER_SIZE + slot * self.record_size;
        let vector_offset = offset + self.vector_data_offset();
        let mmap = self.mmap.as_ref().expect("vector store mmap is open");
        let bytes = &mmap[vector_offset..offset + self.record_size];
        // SAFETY: same alignment/validity argument as `get_vector`; the caller
        // guarantees `slot` is in range, so `offset + record_size` is in bounds.
        unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.header.dimension as usize)
        }
    }

    /// Prefetch the cache line(s) holding the vector at `slot`.
    ///
    /// A best-effort hint to start fetching the vector's first cache line before
    /// its distance is computed. No-op on non-x86_64 targets.
    #[inline]
    pub fn prefetch_vector(&self, slot: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            debug_assert!(slot < self.header.vector_count as usize);
            let offset = HEADER_SIZE + slot * self.record_size + self.vector_data_offset();
            let mmap = self.mmap.as_ref().expect("vector store mmap is open");
            let ptr = mmap[offset..].as_ptr();
            // SAFETY: prefetch is a non-faulting hint; the pointer is within the
            // mapping. SSE is part of the x86_64 baseline ISA, so `_mm_prefetch`
            // is always available here.
            unsafe {
                std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = slot;
        }
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
    ///
    /// Also checkpoints the metadata snapshot, after the data file is durable
    /// so a snapshot can never reference vectors that are not themselves
    /// durable. Snapshot failure only costs the checkpoint — the WAL still
    /// carries every metadata entry — so it is logged rather than returned.
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
        if let Err(e) = self.write_meta_snapshot() {
            tracing::warn!(error = %e, "failed to persist metadata snapshot");
        }
        Ok(())
    }

    /// Return the current WAL size in bytes.
    pub fn wal_len(&self) -> Result<u64> {
        Ok(fs::metadata(&self.wal_path)?.len())
    }

    /// Rewrite the store with only live vectors and atomically install it.
    ///
    /// The existing data, WAL, and metadata snapshot remain untouched until
    /// the replacement set has been fully flushed. A small marker journals the
    /// multi-file rename so `open` can roll back or roll forward after a crash
    /// at any swap step.
    pub fn compact(&mut self) -> Result<()> {
        let live_vectors: Vec<(u64, Vec<f32>, Option<Metadata>)> = (0..self.len())
            .filter_map(|slot| {
                let vector_id = self.vector_ids[slot];
                (!self.deleted_ids.contains(&vector_id)).then(|| {
                    (
                        vector_id,
                        self.get_vector(slot).expect("valid slot").to_vec(),
                        self.metadata[slot].clone(),
                    )
                })
            })
            .collect();
        let max_vector_id = self.header.max_vector_id;
        let dimension = self.header.dimension;
        let metric = self.header.metric;
        let paths = CompactionPaths::new(&self.path, &self.wal_path);
        let meta_path = self.meta_snapshot_path();

        Self::recover_compaction(&self.path, &self.wal_path)?;
        remove_file_if_exists(&paths.temp_data)?;
        remove_file_if_exists(&paths.temp_wal)?;
        remove_file_if_exists(&paths.temp_meta)?;

        {
            let mut replacement =
                Self::create(&paths.temp_data, &paths.temp_wal, dimension, metric)?;
            for (vector_id, vector, metadata) in &live_vectors {
                replacement.insert_raw(*vector_id, vector, metadata.clone())?;
            }
            replacement.header.max_vector_id = max_vector_id;
            // Flushes the replacement data file and, when any live vector has
            // metadata, writes the replacement snapshot to `paths.temp_meta`.
            replacement.flush()?;
            let wal = replacement.wal.as_mut().expect("replacement WAL is open");
            wal.clear()?;
            wal.flush()?;
        }
        sync_parent(&paths.temp_data)?;
        sync_parent(&paths.temp_wal)?;
        if paths.temp_meta.exists() {
            sync_parent(&paths.temp_meta)?;
        }

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
            if meta_path.exists() {
                fs::rename(&meta_path, &paths.backup_meta)?;
                sync_parent(&meta_path)?;
            }
            write_marker(&paths.marker, COMPACTION_OLD_MOVED)?;
            compaction_failpoint("old_pair_moved");

            fs::rename(&paths.temp_data, &self.path)?;
            sync_parent(&self.path)?;
            compaction_failpoint("new_data_installed");
            fs::rename(&paths.temp_wal, &self.wal_path)?;
            sync_parent(&self.wal_path)?;
            if paths.temp_meta.exists() {
                fs::rename(&paths.temp_meta, &meta_path)?;
                sync_parent(&meta_path)?;
            }
            write_marker(&paths.marker, COMPACTION_INSTALLED)?;

            remove_file_if_exists(&paths.backup_data)?;
            remove_file_if_exists(&paths.backup_wal)?;
            remove_file_if_exists(&paths.backup_meta)?;
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
                        .map(|(vector_id, _, _)| *vector_id)
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

    // ── Metadata snapshot persistence ────────────────────────────────────
    //
    // Per-vector metadata is checkpointed to `<data_path>.meta` on
    // `flush`/`compact` and loaded on `open`. Between checkpoints the WAL
    // carries every metadata entry, so a missing or corrupt snapshot falls
    // back to WAL replay. Compaction clears the WAL, so it installs the
    // replacement snapshot atomically with the compacted data pair.

    /// Path of the metadata snapshot derived from the vector data path.
    fn meta_snapshot_path(&self) -> PathBuf {
        sidecar_path(&self.path, ".meta")
    }

    /// Serialize live metadata to a byte buffer with CRC32 integrity checks.
    fn serialize_meta_snapshot(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_all(META_MAGIC).unwrap();
        buf.write_u8(META_FORMAT_VERSION).unwrap();
        buf.write_all(&[0u8; 3]).unwrap();
        let entry_count = self.metadata.iter().filter(|m| m.is_some()).count();
        buf.write_u64::<LittleEndian>(entry_count as u64).unwrap();
        buf.write_u64::<LittleEndian>(self.header.vector_count)
            .unwrap();
        buf.write_u64::<LittleEndian>(self.header.max_vector_id)
            .unwrap();
        let header_crc = crc32fast::hash(&buf);
        buf.write_u32::<LittleEndian>(header_crc).unwrap();

        let body_start = buf.len();
        for (slot, metadata) in self.metadata.iter().enumerate() {
            let Some(metadata) = metadata else { continue };
            buf.write_u64::<LittleEndian>(self.vector_ids[slot])
                .unwrap();
            let bytes = metadata.to_bytes();
            buf.write_u32::<LittleEndian>(bytes.len() as u32).unwrap();
            buf.write_all(&bytes).unwrap();
        }
        let body_crc = crc32fast::hash(&buf[body_start..]);
        buf.write_u32::<LittleEndian>(body_crc).unwrap();
        buf
    }

    /// Write the metadata snapshot atomically (temp file + fsync + rename).
    fn write_meta_snapshot(&self) -> Result<()> {
        if !self.metadata.iter().any(Option::is_some) {
            return Ok(());
        }
        let bytes = self.serialize_meta_snapshot();
        let final_path = self.meta_snapshot_path();
        let tmp_path = sidecar_path(&final_path, ".tmp");
        {
            let mut f = File::create(&tmp_path)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        // A missing or torn snapshot only triggers a safe WAL-replay fallback,
        // so removing the old file before the rename is crash-safe.
        let _ = fs::remove_file(&final_path);
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    }

    /// Best-effort load of the metadata snapshot. A missing, corrupt, or
    /// mismatched snapshot leaves the slots empty so WAL replay can refill
    /// them.
    fn load_meta_snapshot(&mut self) {
        let path = self.meta_snapshot_path();
        let data = match fs::read(&path) {
            Ok(d) => d,
            Err(_) => return,
        };
        match self.apply_meta_snapshot(&data) {
            Ok(count) => {
                if count > 0 {
                    tracing::info!(count, "Loaded vector metadata from snapshot");
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "metadata snapshot invalid; falling back to WAL replay"
                );
            }
        }
    }

    /// Parse and validate a snapshot, applying it to this store on success.
    fn apply_meta_snapshot(&mut self, data: &[u8]) -> Result<usize> {
        let invalid = |message: String| QuiverError::InvalidFormat(message);

        if data.len() < META_HEADER_SIZE {
            return Err(invalid("metadata snapshot too short".to_string()));
        }
        let mut cur = Cursor::new(data);
        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if &magic != META_MAGIC {
            return Err(invalid("invalid metadata snapshot magic".to_string()));
        }
        let version = cur.read_u8()?;
        if version != META_FORMAT_VERSION {
            return Err(invalid(format!(
                "unsupported metadata snapshot version: {version}"
            )));
        }
        let mut reserved = [0u8; 3];
        cur.read_exact(&mut reserved)?;
        let entry_count = cur.read_u64::<LittleEndian>()? as usize;
        let store_len = cur.read_u64::<LittleEndian>()? as usize;
        let max_vector_id = cur.read_u64::<LittleEndian>()?;
        let header_crc = cur.read_u32::<LittleEndian>()?;

        if crc32fast::hash(&data[..META_HEADER_SIZE - 4]) != header_crc {
            return Err(invalid(
                "metadata snapshot header checksum mismatch".to_string(),
            ));
        }
        if store_len != self.len() || max_vector_id != self.header.max_vector_id {
            return Err(invalid(
                "metadata snapshot does not match store".to_string(),
            ));
        }

        let body = &data[META_HEADER_SIZE..];
        if body.len() < 4 {
            return Err(invalid("metadata snapshot body truncated".to_string()));
        }
        let body_end = body.len() - 4;
        let stored_body_crc = (&body[body_end..]).read_u32::<LittleEndian>()?;
        if crc32fast::hash(&body[..body_end]) != stored_body_crc {
            return Err(invalid(
                "metadata snapshot body checksum mismatch".to_string(),
            ));
        }

        let mut cursor = Cursor::new(&body[..body_end]);
        let mut applied = 0usize;
        for _ in 0..entry_count {
            let vector_id = cursor
                .read_u64::<LittleEndian>()
                .map_err(|_| invalid("metadata snapshot entry truncated".to_string()))?;
            let meta_len = cursor
                .read_u32::<LittleEndian>()
                .ok()
                .and_then(|len| usize::try_from(len).ok())
                .ok_or_else(|| invalid("metadata snapshot entry length invalid".to_string()))?;
            let start = usize::try_from(cursor.position())
                .ok()
                .and_then(|pos| pos.checked_add(meta_len))
                .ok_or_else(|| invalid("metadata snapshot entry overflow".to_string()))?;
            let bytes = body
                .get(cursor.position() as usize..start)
                .ok_or_else(|| invalid("metadata snapshot entry truncated".to_string()))?;
            let metadata = Metadata::from_bytes(bytes)?;
            cursor.set_position(start as u64);

            let slot = self.slot_for_id(vector_id).ok_or_else(|| {
                invalid(format!(
                    "metadata snapshot references unknown vector {vector_id}"
                ))
            })?;
            if self.metadata[slot].is_some() {
                return Err(invalid(format!(
                    "metadata snapshot has duplicate entry for vector {vector_id}"
                )));
            }
            self.metadata[slot] = Some(metadata);
            applied += 1;
        }
        if cursor.position() as usize != body_end {
            return Err(invalid(
                "trailing bytes in metadata snapshot body".to_string(),
            ));
        }

        Ok(applied)
    }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Write a vector directly into the mmap'd file (no WAL logging).
    /// Used both by `insert` and by WAL replay.
    fn insert_raw(
        &mut self,
        vector_id: u64,
        data: &[f32],
        metadata: Option<Metadata>,
    ) -> Result<()> {
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

        // Once any metadata exists, the store must identify as v3 so
        // pre-metadata binaries (which cannot see the `.meta` snapshot or
        // replay `InsertMeta` entries) refuse to open it.
        if metadata.is_some() && self.header.version < FORMAT_VERSION {
            self.header.version = FORMAT_VERSION;
        }
        self.metadata.push(metadata);

        Ok(())
    }

    /// Locate the physical slot holding `vector_id`, if present.
    ///
    /// `vector_ids` is strictly increasing (enforced at parse time and by
    /// construction on insert), so binary search applies.
    fn slot_for_id(&self, vector_id: u64) -> Option<usize> {
        self.vector_ids.binary_search(&vector_id).ok()
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
        let meta_path = sidecar_path(data_path, ".meta");
        let meta_exists = meta_path.exists();

        // The metadata snapshot is optional: a store whose vectors carry no
        // metadata never writes one. Restore it from whichever sidecar
        // matches the surviving data pair, best-effort.
        let restore_meta = |preferred: &Path, fallback: &Path| -> Result<()> {
            if !meta_exists && preferred.exists() {
                fs::rename(preferred, &meta_path)?;
            } else if !meta_exists && fallback.exists() {
                fs::rename(fallback, &meta_path)?;
            }
            Ok(())
        };

        if data_exists && wal_exists {
            // The data pair is intact: either the swap never started (roll
            // back: the old snapshot lives in the backup slot) or an earlier
            // recovery pass restored it (roll forward: prefer the temp slot).
            if phase == COMPACTION_PREPARED {
                restore_meta(&paths.backup_meta, &paths.temp_meta)?;
            } else {
                restore_meta(&paths.temp_meta, &paths.backup_meta)?;
            }
            remove_file_if_exists(&paths.temp_data)?;
            remove_file_if_exists(&paths.temp_wal)?;
            remove_file_if_exists(&paths.temp_meta)?;
            remove_file_if_exists(&paths.backup_data)?;
            remove_file_if_exists(&paths.backup_wal)?;
            remove_file_if_exists(&paths.backup_meta)?;
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
            restore_meta(&paths.temp_meta, &paths.backup_meta)?;
        } else {
            if !data_exists && paths.backup_data.exists() {
                fs::rename(&paths.backup_data, data_path)?;
            }
            if !wal_exists && paths.backup_wal.exists() {
                fs::rename(&paths.backup_wal, wal_path)?;
            }
            restore_meta(&paths.backup_meta, &paths.temp_meta)?;
        }

        if !data_path.exists() || !wal_path.exists() {
            return Err(QuiverError::InvalidFormat(
                "Unable to recover interrupted compaction".to_string(),
            ));
        }

        remove_file_if_exists(&paths.temp_data)?;
        remove_file_if_exists(&paths.temp_wal)?;
        remove_file_if_exists(&paths.temp_meta)?;
        remove_file_if_exists(&paths.backup_data)?;
        remove_file_if_exists(&paths.backup_wal)?;
        remove_file_if_exists(&paths.backup_meta)?;
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
    fn test_insert_batch_assigns_sequential_ids_and_reads_back() {
        let (_dir, mut store) = setup(3);
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let c = [7.0, 8.0, 9.0];
        let ids = store.insert_batch(&[&a, &b, &c]).unwrap();
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(store.len(), 3);
        assert_eq!(store.get_vector(0).unwrap(), &a);
        assert_eq!(store.get_vector(1).unwrap(), &b);
        assert_eq!(store.get_vector(2).unwrap(), &c);
    }

    #[test]
    fn test_insert_batch_empty_is_noop() {
        let (_dir, mut store) = setup(3);
        let ids = store.insert_batch(&[]).unwrap();
        assert!(ids.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_insert_batch_rejects_bad_dimension_without_partial_commit() {
        let (_dir, mut store) = setup(3);
        let good = [1.0, 2.0, 3.0];
        let bad = [1.0, 2.0]; // wrong dimension
        let result = store.insert_batch(&[&good, &bad]);
        assert!(result.is_err());
        // Nothing should have been committed.
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_insert_batch_then_single_insert_continues_ids() {
        let (_dir, mut store) = setup(2);
        let ids = store.insert_batch(&[&[1.0, 1.0], &[2.0, 2.0]]).unwrap();
        assert_eq!(ids, vec![1, 2]);
        let next = store.insert(&[3.0, 3.0]).unwrap();
        assert_eq!(next, 3);
        assert_eq!(store.len(), 3);
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

    fn sample_metadata() -> Metadata {
        let mut metadata = Metadata::new();
        metadata.insert("category", "science");
        metadata.insert("year", 2024i64);
        metadata
    }

    fn other_metadata() -> Metadata {
        let mut metadata = Metadata::new();
        metadata.insert("category", "sports");
        metadata
    }

    #[test]
    fn test_insert_with_metadata_and_read_back() {
        let (_dir, mut store) = setup(2);
        let with_meta = store
            .insert_with_metadata(&[1.0, 2.0], sample_metadata())
            .unwrap();
        let without_meta = store.insert(&[3.0, 4.0]).unwrap();

        assert_eq!(with_meta, 1);
        assert_eq!(without_meta, 2);
        assert_eq!(store.metadata(0), Some(&sample_metadata()));
        assert_eq!(store.metadata(1), None);
    }

    #[test]
    fn test_insert_batch_with_metadata() {
        let (_dir, mut store) = setup(2);
        let a = [1.0, 1.0];
        let b = [2.0, 2.0];
        let metadata = [Some(sample_metadata()), None];
        let ids = store
            .insert_batch_with_metadata(&[&a, &b], &metadata)
            .unwrap();
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(store.metadata(0), Some(&sample_metadata()));
        assert_eq!(store.metadata(1), None);
    }

    #[test]
    fn test_insert_batch_with_metadata_rejects_length_mismatch() {
        let (_dir, mut store) = setup(2);
        let a = [1.0, 1.0];
        let result = store.insert_batch_with_metadata(&[&a], &[Some(sample_metadata()), None]);
        assert!(result.is_err());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_metadata_persists_across_flush_and_reopen() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("meta_flush.qvdb");
        let wal_path = dir.path().join("meta_flush.wal");

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            store
                .insert_with_metadata(&[1.0, 2.0], sample_metadata())
                .unwrap();
            store.insert(&[3.0, 4.0]).unwrap();
            store.flush().unwrap();
        }

        // The checkpointed snapshot lives next to the data file.
        assert!(dir.path().join("meta_flush.qvdb.meta").exists());

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.metadata(0), Some(&sample_metadata()));
        assert_eq!(store.metadata(1), None);
    }

    #[test]
    fn test_metadata_recovers_from_wal_without_flush() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("meta_wal.qvdb");
        let wal_path = dir.path().join("meta_wal.wal");

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            store
                .insert_with_metadata(&[1.0, 2.0], sample_metadata())
                .unwrap();
            // Drop without flushing the mmap header, simulating a crash after
            // the WAL fsync but before the data-file checkpoint.
        }

        // Reset the header to 0 vectors, as if the mmap flush never happened.
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

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.get_vector(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(store.metadata(0), Some(&sample_metadata()));
    }

    #[test]
    fn test_missing_meta_sidecar_falls_back_to_wal() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("meta_missing.qvdb");
        let wal_path = dir.path().join("meta_missing.wal");
        let meta_path = dir.path().join("meta_missing.qvdb.meta");

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            store
                .insert_with_metadata(&[1.0, 2.0], sample_metadata())
                .unwrap();
            store.flush().unwrap();
        }
        assert!(meta_path.exists());
        fs::remove_file(&meta_path).unwrap();

        // The WAL still carries the InsertMeta entry (flush does not clear
        // it), so metadata survives the lost snapshot.
        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.metadata(0), Some(&sample_metadata()));
    }

    #[test]
    fn test_corrupt_meta_sidecar_falls_back_to_wal() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("meta_corrupt.qvdb");
        let wal_path = dir.path().join("meta_corrupt.wal");
        let meta_path = dir.path().join("meta_corrupt.qvdb.meta");

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            store
                .insert_with_metadata(&[1.0, 2.0], sample_metadata())
                .unwrap();
            store.flush().unwrap();
        }

        // Flip a byte in the middle of the snapshot (body region).
        let mut bytes = fs::read(&meta_path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        fs::write(&meta_path, bytes).unwrap();

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.metadata(0), Some(&sample_metadata()));
    }

    #[test]
    fn test_compaction_preserves_metadata_and_drops_deleted() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("meta_compact.qvdb");
        let wal_path = dir.path().join("meta_compact.wal");
        let meta_path = dir.path().join("meta_compact.qvdb.meta");

        let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
        store
            .insert_with_metadata(&[1.0, 1.0], sample_metadata())
            .unwrap();
        let deleted_id = store
            .insert_with_metadata(&[2.0, 2.0], other_metadata())
            .unwrap();
        store.insert(&[3.0, 3.0]).unwrap();
        store.delete(deleted_id).unwrap();

        store.compact().unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.vector_id(0).unwrap(), 1);
        assert_eq!(store.vector_id(1).unwrap(), 3);
        assert_eq!(store.metadata(0), Some(&sample_metadata()));
        assert_eq!(store.metadata(1), None);
        assert_eq!(store.wal_len().unwrap(), 0);
        // Compaction installed a fresh snapshot covering only live vectors.
        assert!(meta_path.exists());

        drop(store);
        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 2);
        assert_eq!(store.metadata(0), Some(&sample_metadata()));
        assert_eq!(store.metadata(1), None);
    }

    #[test]
    fn test_create_removes_stale_meta_sidecar() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("meta_stale.qvdb");
        let wal_path = dir.path().join("meta_stale.wal");
        let meta_path = dir.path().join("meta_stale.qvdb.meta");

        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            store
                .insert_with_metadata(&[1.0, 2.0], sample_metadata())
                .unwrap();
            store.flush().unwrap();
        }
        assert!(meta_path.exists());

        // Recreating the store must not inherit the old snapshot (the new
        // store reuses vector ID 1 for a different vector).
        {
            let mut store = VectorStore::create(&data_path, &wal_path, 2, Metric::L2).unwrap();
            assert!(!meta_path.exists());
            store.insert(&[9.0, 9.0]).unwrap();
            store.flush().unwrap();
        }

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.metadata(0), None);
    }

    #[test]
    fn test_metadata_insert_into_v2_store_bumps_version_to_3() {
        let dir = TempDir::new().unwrap();
        let data_path = dir.path().join("v2_upgrade.qvdb");
        let wal_path = dir.path().join("v2_upgrade.wal");

        // Hand-craft a version-2 file with one vector.
        {
            let mut header = FileHeader::new(2, Metric::L2);
            header.version = 2;
            header.vector_count = 1;
            header.max_vector_id = 1;
            let mut bytes = header.to_bytes();
            bytes.extend_from_slice(&1_u64.to_le_bytes());
            bytes.extend_from_slice(&1.0_f32.to_le_bytes());
            bytes.extend_from_slice(&2.0_f32.to_le_bytes());
            fs::write(&data_path, bytes).unwrap();
            File::create(&wal_path).unwrap();
        }

        {
            let mut store = VectorStore::open(&data_path, &wal_path).unwrap();
            assert_eq!(fs::read(&data_path).unwrap()[4], 2);
            store
                .insert_with_metadata(&[3.0, 4.0], sample_metadata())
                .unwrap();
            store.flush().unwrap();
        }

        let store = VectorStore::open(&data_path, &wal_path).unwrap();
        assert_eq!(fs::read(&data_path).unwrap()[4], 3);
        assert_eq!(store.len(), 2);
        assert_eq!(store.metadata(0), None);
        assert_eq!(store.metadata(1), Some(&sample_metadata()));
    }
}
