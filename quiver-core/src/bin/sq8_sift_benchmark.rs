use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use quiver_core::distance::Metric;
use quiver_core::index::brute_force::BruteForceIndex;
use quiver_core::index::sq8::Sq8Index;
use serde::Serialize;

#[derive(Debug, Clone, Copy)]
enum Engine {
    Sq8,
    BruteForce,
}

#[derive(Debug)]
struct Args {
    engine: Engine,
    base: PathBuf,
    queries: PathBuf,
    groundtruth: PathBuf,
    work_dir: PathBuf,
    output: PathBuf,
    base_limit: usize,
    query_limit: usize,
}

#[derive(Serialize)]
struct SearchResultRow {
    k: usize,
    recall: f64,
    qps: f64,
    p50_latency_ms: f64,
    p99_latency_ms: f64,
    total_seconds: f64,
}

#[derive(Serialize)]
struct BenchmarkResult {
    engine: &'static str,
    dataset: &'static str,
    dimension: usize,
    base_vectors: usize,
    queries: usize,
    thread_count: usize,
    build_seconds: f64,
    baseline_rss_bytes: u64,
    rss_after_build_bytes: u64,
    index_rss_delta_bytes: u64,
    peak_rss_bytes: u64,
    vector_payload_bytes: Option<usize>,
    data_file_bytes: Option<u64>,
    wal_file_bytes: Option<u64>,
    search: Option<SearchResultRow>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
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

    let result = match args.engine {
        Engine::Sq8 => run_sq8(&args, &queries, &groundtruth, dimension, baseline_rss_bytes)?,
        Engine::BruteForce => run_brute_force(&args, dimension, baseline_rss_bytes)?,
    };

    fs::write(&args.output, serde_json::to_vec_pretty(&result)?)?;
    println!("wrote {}", args.output.display());
    Ok(())
}

fn run_sq8(
    args: &Args,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<u32>],
    dimension: usize,
    baseline_rss_bytes: u64,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("loading {} SIFT base vectors for SQ8", args.base_limit);
    let build_started = Instant::now();
    let vectors = read_fvecs_with_progress(&args.base, args.base_limit)?;
    let index = Sq8Index::build(&vectors, Metric::L2)?;
    let inserted = vectors.len();
    drop(vectors);
    let build_seconds = build_started.elapsed().as_secs_f64();
    let rss_after_build_bytes = current_rss_bytes();
    let peak_rss_bytes = peak_rss_bytes();
    let search = run_sq8_search(&index, queries, groundtruth, 10)?;

    Ok(BenchmarkResult {
        engine: "quiver-sq8",
        dataset: "SIFT1M",
        dimension,
        base_vectors: inserted,
        queries: queries.len(),
        thread_count: 1,
        build_seconds,
        baseline_rss_bytes,
        rss_after_build_bytes,
        index_rss_delta_bytes: rss_after_build_bytes.saturating_sub(baseline_rss_bytes),
        peak_rss_bytes,
        vector_payload_bytes: Some(index.vector_bytes()),
        data_file_bytes: None,
        wal_file_bytes: None,
        search: Some(search),
    })
}

fn run_brute_force(
    args: &Args,
    dimension: usize,
    baseline_rss_bytes: u64,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    fs::create_dir_all(&args.work_dir)?;
    let data_path = args.work_dir.join("quiver-flat.qvdb");
    let wal_path = args.work_dir.join("quiver-flat.wal");
    if data_path.exists() || wal_path.exists() {
        return Err(format!(
            "benchmark work directory is not empty: {}",
            args.work_dir.display()
        )
        .into());
    }

    let mut index = BruteForceIndex::create(&data_path, &wal_path, dimension as u32, Metric::L2)?;
    println!(
        "building Quiver brute-force index with {} vectors",
        args.base_limit
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

    // Insertion grows and remaps the backing file, so pages written through an
    // earlier mapping are not necessarily resident in the final mapping. Scan
    // every full vector before sampling RSS to make the mmap measurement
    // comparable with the fully resident in-memory SQ8 payload.
    println!("faulting the full brute-force mmap into the working set");
    let residency_query = vec![0.0_f32; dimension];
    let _ = index.search(&residency_query, 1)?;
    let rss_after_build_bytes = current_rss_bytes();

    Ok(BenchmarkResult {
        engine: "quiver-brute-force",
        dataset: "SIFT1M",
        dimension,
        base_vectors: inserted,
        queries: args.query_limit,
        thread_count: 1,
        build_seconds,
        baseline_rss_bytes,
        rss_after_build_bytes,
        index_rss_delta_bytes: rss_after_build_bytes.saturating_sub(baseline_rss_bytes),
        peak_rss_bytes: peak_rss_bytes(),
        vector_payload_bytes: Some(inserted * dimension * size_of::<f32>()),
        data_file_bytes: Some(fs::metadata(data_path)?.len()),
        wal_file_bytes: Some(fs::metadata(wal_path)?.len()),
        search: None,
    })
}

fn run_sq8_search(
    index: &Sq8Index,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<u32>],
    k: usize,
) -> Result<SearchResultRow, Box<dyn std::error::Error>> {
    for query in queries.iter().take(100) {
        let _ = index.search(query, k)?;
    }

    let mut latencies_ns = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0_f64;
    let total_started = Instant::now();
    for (query, expected) in queries.iter().zip(groundtruth) {
        let started = Instant::now();
        let found = index.search(query, k)?;
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

fn read_fvecs_with_progress(path: &Path, limit: usize) -> io::Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(limit);
    stream_fvecs(path, limit, |position, row| {
        rows.push(row.to_vec());
        if position.is_multiple_of(10_000) {
            println!("loaded {position} vectors");
            io::stdout().flush()?;
        }
        Ok(())
    })?;
    Ok(rows)
}

fn read_fvecs(path: &Path, limit: usize) -> io::Result<Vec<Vec<f32>>> {
    let mut rows = Vec::with_capacity(limit);
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
    let mut rows = Vec::with_capacity(limit);
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

    let engine = match get("--engine")?.as_str() {
        "sq8" => Engine::Sq8,
        "brute-force" => Engine::BruteForce,
        value => return Err(format!("unsupported engine {value}").into()),
    };
    Ok(Args {
        engine,
        base: PathBuf::from(get("--base")?),
        queries: PathBuf::from(get("--queries")?),
        groundtruth: PathBuf::from(get("--groundtruth")?),
        work_dir: PathBuf::from(get("--work-dir")?),
        output: PathBuf::from(get("--output")?),
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
