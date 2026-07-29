use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use quiver_core::distance::Metric;
use quiver_core::index::hnsw::{HnswConfig, HnswIndex};
use serde::Serialize;

#[derive(Debug)]
struct Args {
    base: PathBuf,
    queries: PathBuf,
    groundtruth: PathBuf,
    work_dir: PathBuf,
    output: PathBuf,
    m: usize,
    ef_construction: usize,
    base_limit: usize,
    query_limit: usize,
}

#[derive(Serialize)]
struct SearchResultRow {
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
    dimension: usize,
    base_vectors: usize,
    queries: usize,
    thread_count: usize,
    random_seed: u64,
    m: usize,
    ef_construction: usize,
    build_seconds: f64,
    baseline_rss_bytes: u64,
    rss_after_build_bytes: u64,
    index_rss_delta_bytes: u64,
    peak_rss_bytes: u64,
    data_file_bytes: u64,
    wal_file_bytes: u64,
    insert_durability: &'static str,
    search: Vec<SearchResultRow>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    fs::create_dir_all(&args.work_dir)?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }

    let queries = read_fvecs(&args.queries, args.query_limit)?;
    let groundtruth = read_ivecs(&args.groundtruth, args.query_limit)?;
    if queries.len() != groundtruth.len() || queries.is_empty() {
        return Err("queries and ground truth must be non-empty and have equal lengths".into());
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
        "building Quiver HNSW: M={} ef_construction={} vectors={}",
        args.m, args.ef_construction, args.base_limit
    );
    let build_started = Instant::now();
    let inserted = stream_fvecs(&args.base, args.base_limit, |position, vector| {
        index.insert(vector).map_err(io::Error::other)?;
        if position.is_multiple_of(10_000) {
            println!("inserted {position} vectors");
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    index.flush()?;
    let build_seconds = build_started.elapsed().as_secs_f64();
    let rss_after_build_bytes = current_rss_bytes();
    let peak_rss_bytes = peak_rss_bytes();

    let mut search = Vec::new();
    for &(k, ef_values) in &[
        (10_usize, &[10_usize, 50, 100, 200, 400][..]),
        (100_usize, &[100_usize, 200, 400][..]),
    ] {
        for &ef_search in ef_values {
            search.push(run_search(&index, &queries, &groundtruth, k, ef_search)?);
        }
    }

    let result = BenchmarkResult {
        engine: "quiver",
        engine_version: env!("CARGO_PKG_VERSION"),
        dataset: "SIFT1M",
        dimension,
        base_vectors: inserted,
        queries: queries.len(),
        thread_count: 1,
        random_seed: 42,
        m: args.m,
        ef_construction: args.ef_construction,
        build_seconds,
        baseline_rss_bytes,
        rss_after_build_bytes,
        index_rss_delta_bytes: rss_after_build_bytes.saturating_sub(baseline_rss_bytes),
        peak_rss_bytes,
        data_file_bytes: fs::metadata(&data_path)?.len(),
        wal_file_bytes: fs::metadata(&wal_path)?.len(),
        insert_durability: "CRC32 WAL record fsynced before every acknowledged insert",
        search,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&result)?)?;
    println!("wrote {}", args.output.display());
    Ok(())
}

fn run_search(
    index: &HnswIndex,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<u32>],
    k: usize,
    ef_search: usize,
) -> Result<SearchResultRow, Box<dyn std::error::Error>> {
    for query in queries.iter().take(100) {
        let _ = index.search(query, k, ef_search)?;
    }

    let mut latencies_ns = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0_f64;
    let total_started = Instant::now();
    for (query, expected) in queries.iter().zip(groundtruth) {
        let started = Instant::now();
        let found = index.search(query, k, ef_search)?;
        latencies_ns.push(started.elapsed().as_nanos() as u64);

        let expected: HashSet<u64> = expected.iter().take(k).map(|value| *value as u64).collect();
        let matches = found
            .iter()
            .filter(|result| expected.contains(&(result.vector_id - 1)))
            .count();
        recall_sum += matches as f64 / k as f64;
    }
    let total_seconds = total_started.elapsed().as_secs_f64();
    latencies_ns.sort_unstable();

    Ok(SearchResultRow {
        k,
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

fn read_ivecs(path: &Path, limit: usize) -> io::Result<Vec<Vec<u32>>> {
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);
    let mut rows = Vec::new();
    while rows.len() < limit {
        let Some(dimension) = read_dimension(&mut reader)? else {
            break;
        };
        let mut bytes = vec![0_u8; dimension * 4];
        reader.read_exact(&mut bytes)?;
        rows.push(
            bytes
                .chunks_exact(4)
                .map(|value| u32::from_le_bytes(value.try_into().expect("four-byte integer")))
                .collect(),
        );
    }
    Ok(rows)
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
        groundtruth: PathBuf::from(get("--groundtruth")?),
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
