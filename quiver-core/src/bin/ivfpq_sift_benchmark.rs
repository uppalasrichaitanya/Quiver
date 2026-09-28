//! SIFT1M benchmark for the IVF-PQ index, comparable with the HNSW and SQ8
//! runs. Builds the index in memory, then sweeps `nprobe` (the IVF analogue of
//! `ef_search`) at k=10, reporting recall / QPS / latency both for pure ADC
//! (`rerank_factor=0`) and with exact-L2 rerank.
//!
//! The memory story is carried by `code_bytes` (the compressed `n * m` payload
//! that IVF-PQ actually searches) versus `vector_bytes` (the optional stored
//! full-precision vectors used only for rerank). Peak RSS includes whichever of
//! those are resident.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use quiver_core::index::SearchResult;
use quiver_core::index::ivfpq::{IvfPqConfig, IvfPqIndex};
use serde::Serialize;

#[derive(Debug)]
struct Args {
    base: PathBuf,
    queries: PathBuf,
    groundtruth: PathBuf,
    output: PathBuf,
    nlist: usize,
    m: usize,
    ksub: usize,
    kmeans_iters: usize,
    training_size: usize,
    rerank_factor: usize,
    base_limit: usize,
    query_limit: usize,
}

#[derive(Serialize)]
struct SearchResultRow {
    k: usize,
    nprobe: usize,
    rerank_factor: usize,
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
    nlist: usize,
    m: usize,
    ksub: usize,
    kmeans_iters: usize,
    training_size: usize,
    store_vectors: bool,
    build_seconds: f64,
    baseline_rss_bytes: u64,
    rss_after_build_bytes: u64,
    index_rss_delta_bytes: u64,
    peak_rss_bytes: u64,
    code_bytes: usize,
    vector_bytes: usize,
    search: Vec<SearchResultRow>,
}

const NPROBE_VALUES: &[usize] = &[1, 8, 16, 32, 64, 128, 256];

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

    println!("loading {} SIFT base vectors for IVF-PQ", args.base_limit);
    let vectors = read_fvecs_with_progress(&args.base, args.base_limit)?;

    let config = IvfPqConfig {
        nlist: args.nlist,
        m: args.m,
        ksub: args.ksub,
        kmeans_iters: args.kmeans_iters,
        training_size: args.training_size,
        store_vectors: true,
        seed: 42,
    };
    println!(
        "building Quiver IVF-PQ: nlist={} m={} ksub={} iters={} train_n={}",
        args.nlist, args.m, args.ksub, args.kmeans_iters, args.training_size
    );
    let build_started = Instant::now();
    let index = IvfPqIndex::build(&vectors, &config)?;
    let inserted = vectors.len();
    drop(vectors);
    let build_seconds = build_started.elapsed().as_secs_f64();
    println!("build complete in {build_seconds:.1}s");

    let rss_after_build_bytes = current_rss_bytes();
    let peak_rss_bytes = peak_rss_bytes();

    let mut search = Vec::new();
    let k = 10;
    for &nprobe in NPROBE_VALUES {
        // Pure ADC (the codes-only operating point) ...
        search.push(run_search(&index, &queries, &groundtruth, k, nprobe, 0)?);
        // ... and with exact rerank (the high-recall operating point).
        if args.rerank_factor > 0 {
            search.push(run_search(
                &index,
                &queries,
                &groundtruth,
                k,
                nprobe,
                args.rerank_factor,
            )?);
        }
    }

    let result = BenchmarkResult {
        engine: "quiver-ivfpq",
        engine_version: env!("CARGO_PKG_VERSION"),
        dataset: "SIFT1M",
        dimension,
        base_vectors: inserted,
        queries: queries.len(),
        thread_count: 1,
        random_seed: 42,
        nlist: args.nlist,
        m: args.m,
        ksub: args.ksub,
        kmeans_iters: args.kmeans_iters,
        training_size: args.training_size,
        store_vectors: true,
        build_seconds,
        baseline_rss_bytes,
        rss_after_build_bytes,
        index_rss_delta_bytes: rss_after_build_bytes.saturating_sub(baseline_rss_bytes),
        peak_rss_bytes,
        code_bytes: index.code_bytes(),
        vector_bytes: index.vector_bytes(),
        search,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&result)?)?;
    println!("wrote {}", args.output.display());
    Ok(())
}

fn run_search(
    index: &IvfPqIndex,
    queries: &[Vec<f32>],
    groundtruth: &[Vec<u32>],
    k: usize,
    nprobe: usize,
    rerank_factor: usize,
) -> Result<SearchResultRow, Box<dyn std::error::Error>> {
    for query in queries.iter().take(100) {
        let _ = index.search(query, k, nprobe, rerank_factor)?;
    }

    let mut latencies_ns = Vec::with_capacity(queries.len());
    let mut recall_sum = 0.0_f64;
    let total_started = Instant::now();
    for (query, expected) in queries.iter().zip(groundtruth) {
        let started = Instant::now();
        let found = index.search(query, k, nprobe, rerank_factor)?;
        latencies_ns.push(started.elapsed().as_nanos() as u64);
        recall_sum += recall_at_k(&found, expected, k);
    }
    let total_seconds = total_started.elapsed().as_secs_f64();
    latencies_ns.sort_unstable();

    println!(
        "nprobe={nprobe:>3} rerank={rerank_factor:>2} recall={} qps={:.1}",
        recall_sum / queries.len() as f64,
        queries.len() as f64 / total_seconds
    );

    Ok(SearchResultRow {
        k,
        nprobe,
        rerank_factor,
        recall: recall_sum / queries.len() as f64,
        qps: queries.len() as f64 / total_seconds,
        p50_latency_ms: percentile_ns(&latencies_ns, 0.50) / 1_000_000.0,
        p99_latency_ms: percentile_ns(&latencies_ns, 0.99) / 1_000_000.0,
        total_seconds,
    })
}

fn recall_at_k(found: &[SearchResult], expected: &[u32], k: usize) -> f64 {
    let expected: HashSet<u64> = expected.iter().take(k).map(|value| *value as u64).collect();
    let matches = found
        .iter()
        .filter(|result| expected.contains(&(result.vector_id - 1)))
        .count();
    matches as f64 / k as f64
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
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
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| u32::from_le_bytes(*value))
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
        output: PathBuf::from(get("--output")?),
        nlist: get("--nlist")?.parse()?,
        m: get("--m")?.parse()?,
        ksub: get("--ksub")?.parse()?,
        kmeans_iters: get("--kmeans-iters")?.parse()?,
        training_size: get("--training-size")?.parse()?,
        rerank_factor: get("--rerank-factor")?.parse()?,
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
