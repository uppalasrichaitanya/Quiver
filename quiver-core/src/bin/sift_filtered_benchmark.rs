//! SIFT1M filtered-search benchmark.
//!
//! Assigns deterministic metadata categories to every base vector
//! (`cat100 = position % 100`, `cat10 = position % 10`,
//! `parity = position % 2`, position = 0-based insert order), builds a Quiver
//! HNSW index carrying that metadata, and measures `search_filtered` at
//! ~1%, ~10%, and ~50% selectivity against brute-force *filtered* ground
//! truth (the SIFT ground-truth file is unfiltered and therefore unusable
//! here). Reports Recall@10, QPS, and p50/p99 latency per selectivity and
//! ef_search.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use quiver_core::distance::{Metric, l2_squared};
use quiver_core::index::hnsw::{HnswConfig, HnswIndex};
use quiver_core::metadata::{Filter, MetaValue, Metadata};
use serde::Serialize;

const K: usize = 10;
const EF_SEARCH_VALUES: &[usize] = &[100, 200, 400];

#[derive(Debug)]
struct Args {
    base: PathBuf,
    queries: PathBuf,
    work_dir: PathBuf,
    output: PathBuf,
    m: usize,
    ef_construction: usize,
    base_limit: usize,
    query_limit: usize,
}

/// One selectivity scenario: an Eq filter on one of the category keys.
#[derive(Clone, Copy)]
struct Scenario {
    label: &'static str,
    key: &'static str,
    value: i64,
    modulus: usize,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        label: "1%",
        key: "cat100",
        value: 7,
        modulus: 100,
    },
    Scenario {
        label: "10%",
        key: "cat10",
        value: 3,
        modulus: 10,
    },
    Scenario {
        label: "50%",
        key: "parity",
        value: 0,
        modulus: 2,
    },
];

#[derive(Serialize)]
struct FilteredSearchRow {
    selectivity: &'static str,
    filter: Filter,
    matching_vectors: usize,
    k: usize,
    ef_search: usize,
    recall: f64,
    qps: f64,
    p50_latency_ms: f64,
    p99_latency_ms: f64,
    total_seconds: f64,
}

#[derive(Serialize)]
struct BenchmarkResult {
    engine: &'static str,
    engine_version: &'static str,
    dataset: &'static str,
    benchmark: &'static str,
    dimension: usize,
    base_vectors: usize,
    queries: usize,
    thread_count: usize,
    random_seed: u64,
    m: usize,
    ef_construction: usize,
    metadata_scheme: &'static str,
    build_seconds: f64,
    ground_truth_seconds: f64,
    baseline_rss_bytes: u64,
    rss_after_build_bytes: u64,
    index_rss_delta_bytes: u64,
    peak_rss_bytes: u64,
    data_file_bytes: u64,
    wal_file_bytes: u64,
    meta_file_bytes: u64,
    insert_durability: &'static str,
    filtered_search: Vec<FilteredSearchRow>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    fs::create_dir_all(&args.work_dir)?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    let queries = read_fvecs(&args.queries, args.query_limit)?;
    if queries.is_empty() {
        return Err("queries must be non-empty".into());
    }
    let dimension = queries[0].len();
    let baseline_rss_bytes = current_rss_bytes();

    let data_path = args.work_dir.join("quiver.qvdb");
    let wal_path = args.work_dir.join("quiver.wal");
    if data_path.exists() || wal_path.exists() {
        return Err(format!(
            "benchmark work directory is not empty: {}",
            args.work_dir.display()
        )
        .into());
    }

    let config = HnswConfig::new(args.m)
        .with_ef_construction(args.ef_construction)
        .with_random_seed(42);
    let mut index = HnswIndex::create(&data_path, &wal_path, dimension as u32, Metric::L2, config)?;

    println!(
        "building Quiver HNSW with metadata: M={} ef_construction={} vectors={}",
        args.m, args.ef_construction, args.base_limit
    );
    // Keep a contiguous copy of the base vectors for the brute-force filtered
    // ground-truth pass (the SIFT ground-truth file is unfiltered).
    let mut base: Vec<f32> = Vec::with_capacity(args.base_limit * dimension);
    let mut buffer: Vec<Vec<f32>> = Vec::with_capacity(BATCH);
    let mut inserted = 0_usize;
    let build_started = Instant::now();
    let streamed = stream_fvecs(&args.base, args.base_limit, |position, vector| {
        base.extend_from_slice(vector);
        buffer.push(vector.to_vec());
        if buffer.len() >= BATCH {
            flush_batch(&mut index, &mut buffer, inserted)?;
            inserted += BATCH;
        }
        if position.is_multiple_of(10_000) {
            println!("inserted {position} vectors");
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    flush_batch(&mut index, &mut buffer, inserted)?;
    inserted = streamed;
    index.flush()?;
    let build_seconds = build_started.elapsed().as_secs_f64();
    let rss_after_build_bytes = current_rss_bytes();
    println!("build complete in {build_seconds:.1}s");

    if inserted != base.len() / dimension {
        return Err("base vector bookkeeping mismatch".into());
    }

    // Matching slot lists per scenario (slot == 0-based insert position).
    let matching_slots: Vec<Vec<u32>> = SCENARIOS
        .iter()
        .map(|scenario| {
            (0..inserted)
                .filter(|position| position % scenario.modulus == scenario.value as usize)
                .map(|position| position as u32)
                .collect()
        })
        .collect();
    for (scenario, slots) in SCENARIOS.iter().zip(&matching_slots) {
        println!(
            "scenario {}: {} matching vectors ({} == {})",
            scenario.label,
            slots.len(),
            scenario.key,
            scenario.value
        );
    }

    println!("computing brute-force filtered ground truth (k={K})");
    let ground_truth_started = Instant::now();
    let ground_truth = filtered_ground_truth(&base, dimension, &queries, &matching_slots, K);
    let ground_truth_seconds = ground_truth_started.elapsed().as_secs_f64();
    println!("ground truth complete in {ground_truth_seconds:.1}s");

    let mut filtered_search = Vec::new();
    for (scenario_idx, scenario) in SCENARIOS.iter().enumerate() {
        let filter = Filter::Eq {
            key: scenario.key.to_owned(),
            value: MetaValue::Int(scenario.value),
        };
        for &ef_search in EF_SEARCH_VALUES {
            println!(
                "measuring scenario {} ef_search={ef_search}",
                scenario.label
            );
            filtered_search.push(run_filtered_search(
                &index,
                &queries,
                &ground_truth[scenario_idx],
                &filter,
                scenario.label,
                matching_slots[scenario_idx].len(),
                ef_search,
            )?);
        }
    }

    let meta_path = sidecar_path(&data_path, ".meta");
    let result = BenchmarkResult {
        engine: "quiver",
        engine_version: env!("CARGO_PKG_VERSION"),
        dataset: "SIFT1M",
        benchmark: "filtered-search",
        dimension,
        base_vectors: inserted,
        queries: queries.len(),
        thread_count: 1,
        random_seed: 42,
        m: args.m,
        ef_construction: args.ef_construction,
        metadata_scheme: "cat100 = position % 100, cat10 = position % 10, parity = position % 2 \
                          (0-based insert position); all values stored as integers",
        build_seconds,
        ground_truth_seconds,
        baseline_rss_bytes,
        rss_after_build_bytes,
        index_rss_delta_bytes: rss_after_build_bytes.saturating_sub(baseline_rss_bytes),
        peak_rss_bytes: peak_rss_bytes(),
        data_file_bytes: fs::metadata(&data_path)?.len(),
        wal_file_bytes: fs::metadata(&wal_path)?.len(),
        meta_file_bytes: fs::metadata(&meta_path).map(|m| m.len()).unwrap_or(0),
        insert_durability: "CRC32 WAL records group-committed (one fsync per batch) before acknowledgment",
        filtered_search,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&result)?)?;
    println!("wrote {}", args.output.display());
    Ok(())
}

const BATCH: usize = 1024;

/// Group-commit buffered vectors (with per-position metadata) in one batched
/// WAL fsync.
fn flush_batch(
    index: &mut HnswIndex,
    buffer: &mut Vec<Vec<f32>>,
    start_position: usize,
) -> io::Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }
    let refs: Vec<&[f32]> = buffer.iter().map(|v| v.as_slice()).collect();
    let metadata: Vec<Option<Metadata>> = (start_position..start_position + buffer.len())
        .map(|position| Some(position_metadata(position)))
        .collect();
    index
        .insert_batch_with_metadata(&refs, &metadata)
        .map_err(io::Error::other)?;
    buffer.clear();
    Ok(())
}

fn position_metadata(position: usize) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert("cat100", (position % 100) as i64);
    metadata.insert("cat10", (position % 10) as i64);
    metadata.insert("parity", (position % 2) as i64);
    metadata
}

/// Mirror of `VectorStore::meta_snapshot_path` for use from the benchmark.
fn sidecar_path(data_path: &Path, suffix: &str) -> PathBuf {
    let mut s = std::ffi::OsString::from(data_path.as_os_str());
    s.push(suffix);
    PathBuf::from(s)
}

/// A candidate for the brute-force top-k heap (max-heap by distance, so the
/// farthest candidate sits on top and is evicted first).
#[derive(Clone, Copy, PartialEq)]
struct TruthCandidate {
    distance: f32,
    slot: u32,
}

impl Eq for TruthCandidate {}

impl PartialOrd for TruthCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TruthCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Exact top-k within each scenario's matching slot list, for every query.
///
/// Returns `truth[scenario][query]` = up to `k` slots sorted closest-first.
fn filtered_ground_truth(
    base: &[f32],
    dimension: usize,
    queries: &[Vec<f32>],
    matching_slots: &[Vec<u32>],
    k: usize,
) -> Vec<Vec<Vec<u32>>> {
    let mut truth: Vec<Vec<Vec<u32>>> = matching_slots
        .iter()
        .map(|_| Vec::with_capacity(queries.len()))
        .collect();

    for query in queries {
        for (scenario_idx, slots) in matching_slots.iter().enumerate() {
            let mut heap: BinaryHeap<TruthCandidate> = BinaryHeap::with_capacity(k + 1);
            for &slot in slots {
                let start = slot as usize * dimension;
                let distance = l2_squared(query, &base[start..start + dimension]);
                if heap.len() < k {
                    heap.push(TruthCandidate { distance, slot });
                } else if let Some(worst) = heap.peek()
                    && distance < worst.distance
                {
                    heap.pop();
                    heap.push(TruthCandidate { distance, slot });
                }
            }
            let mut top = heap.into_vec();
            top.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(Ordering::Equal)
            });
            truth[scenario_idx].push(top.iter().map(|candidate| candidate.slot).collect());
        }
    }
    truth
}

fn run_filtered_search(
    index: &HnswIndex,
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
    filter: &Filter,
    selectivity: &'static str,
    matching_vectors: usize,
    ef_search: usize,
) -> Result<FilteredSearchRow, Box<dyn std::error::Error>> {
    for query in queries.iter().take(100) {
        let _ = index.search_filtered(query, K, ef_search, filter)?;
    }

    let mut latencies_ns = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0_f64;
    let total_started = Instant::now();
    for (query, expected) in queries.iter().zip(truth) {
        let started = Instant::now();
        let found = index.search_filtered(query, K, ef_search, filter)?;
        latencies_ns.push(started.elapsed().as_nanos() as u64);

        let expected: HashSet<u64> = expected.iter().map(|slot| *slot as u64).collect();
        let matches = found
            .iter()
            .filter(|result| expected.contains(&(result.vector_id - 1)))
            .count();
        recall_sum += matches as f64 / K as f64;
    }
    let total_seconds = total_started.elapsed().as_secs_f64();
    latencies_ns.sort_unstable();

    Ok(FilteredSearchRow {
        selectivity,
        filter: filter.clone(),
        matching_vectors,
        k: K,
        ef_search,
        recall: recall_sum / queries.len() as f64,
        qps: queries.len() as f64 / total_seconds,
        p50_latency_ms: percentile_ns(&latencies_ns, 0.50) / 1_000_000.0,
        p99_latency_ms: percentile_ns(&latencies_ns, 0.99) / 1_000_000.0,
        total_seconds,
    })
}

fn percentile_ns(values: &[u64], percentile: f64) -> f64 {
    let index = ((values.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[index] as f64
}

fn read_fvecs(path: &Path, limit: usize) -> io::Result<Vec<Vec<f32>>> {
    let mut rows = Vec::new();
    stream_fvecs(path, limit, |_, row| {
        rows.push(row.to_vec());
        Ok(())
    })?;
    Ok(rows)
}

fn stream_fvecs<F>(path: &Path, limit: usize, mut consume: F) -> io::Result<usize>
where
    F: FnMut(usize, &[f32]) -> io::Result<()>,
{
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut count = 0_usize;
    while count < limit {
        let Some(dimension) = read_dimension(&mut reader)? else {
            break;
        };
        let mut bytes = vec![0_u8; dimension * 4];
        reader.read_exact(&mut bytes)?;
        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte float")))
            .collect();
        count += 1;
        consume(count, &vector)?;
    }
    Ok(count)
}

fn read_dimension(reader: &mut impl Read) -> io::Result<Option<usize>> {
    let mut bytes = [0_u8; 4];
    match reader.read_exact(&mut bytes) {
        Ok(()) => {
            let dimension = i32::from_le_bytes(bytes);
            if dimension <= 0 || dimension > 65_536 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid vector dimension {dimension}"),
                ));
            }
            Ok(Some(dimension as usize))
        }
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(error) => Err(error),
    }
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut values = std::env::args().skip(1);
    let mut get = |expected: &str| -> Result<String, Box<dyn std::error::Error>> {
        let flag = values.next().ok_or_else(|| format!("missing {expected}"))?;
        if flag != expected {
            return Err(format!("expected {expected}, got {flag}").into());
        }
        values
            .next()
            .ok_or_else(|| format!("missing value for {expected}").into())
    };

    Ok(Args {
        base: PathBuf::from(get("--base")?),
        queries: PathBuf::from(get("--queries")?),
        work_dir: PathBuf::from(get("--work-dir")?),
        output: PathBuf::from(get("--output")?),
        m: get("--m")?.parse()?,
        ef_construction: get("--ef-construction")?.parse()?,
        base_limit: get("--base-limit")?.parse()?,
        query_limit: get("--query-limit")?.parse()?,
    })
}

#[cfg(target_os = "windows")]
fn memory_counters() -> (u64, u64) {
    use std::ffi::c_void;

    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCounters {
        cb: size_of::<ProcessMemoryCounters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    unsafe {
        let _ = K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            size_of::<ProcessMemoryCounters>() as u32,
        );
    }
    (
        counters.working_set_size as u64,
        counters.peak_working_set_size as u64,
    )
}

#[cfg(target_os = "linux")]
fn memory_counters() -> (u64, u64) {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let read_kib = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    (read_kib("VmRSS:"), read_kib("VmHWM:"))
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn memory_counters() -> (u64, u64) {
    (0, 0)
}

fn current_rss_bytes() -> u64 {
    memory_counters().0
}

fn peak_rss_bytes() -> u64 {
    memory_counters().1
}
