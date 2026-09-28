use std::{
    env,
    sync::{Arc, RwLock},
    time::Duration,
};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post, put},
};
use quiver_core::{
    distance::Metric,
    index::hnsw::{HnswConfig, HnswIndex},
    index::ivfpq::IvfPqIndex,
    index::sq8::Sq8Index,
    metadata::{Filter, Metadata},
};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

type SharedIndex = Arc<RwLock<HnswIndex>>;
type SharedSq8 = Arc<RwLock<Option<Sq8Index>>>;
type SharedIvfPq = Arc<RwLock<Option<IvfPqIndex>>>;

#[derive(Clone)]
struct AppState {
    index: SharedIndex,
    sq8: SharedSq8,
    ivfpq: SharedIvfPq,
    shutdown: Arc<tokio::sync::Notify>,
}

#[derive(Deserialize)]
struct InsertRequest {
    vector: Vec<f32>,
    #[serde(default)]
    metadata: Option<Metadata>,
}
#[derive(Deserialize)]
struct UpdateMetadataRequest {
    metadata: Metadata,
}
#[derive(Serialize)]
struct InsertResponse {
    id: u64,
}
#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    k: usize,
    ef_search: Option<usize>,
    #[serde(default)]
    filter: Option<Filter>,
}
#[derive(Deserialize)]
struct BatchSearchRequest {
    queries: Vec<BatchQuery>,
}
#[derive(Deserialize)]
struct BatchQuery {
    vector: Vec<f32>,
    k: usize,
    ef_search: Option<usize>,
    #[serde(default)]
    filter: Option<Filter>,
}
#[derive(Serialize)]
struct SearchHit {
    id: u64,
    distance: f32,
}
#[derive(Deserialize)]
struct Sq8SearchRequest {
    vector: Vec<f32>,
    k: usize,
}
#[derive(Deserialize)]
struct IvfPqSearchRequest {
    vector: Vec<f32>,
    k: usize,
    nprobe: Option<usize>,
    rerank_factor: Option<usize>,
}
#[derive(Serialize)]
struct MetricsResponse {
    len: usize,
    dimension: u32,
    metric: String,
    max_level: usize,
    sq8_len: Option<usize>,
    ivfpq_len: Option<usize>,
}
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// Largest `k` a single query may request. Bounds per-request work and
/// memory; the core also clamps `k`, but a public endpoint should reject
/// absurd values up front instead of silently doing more work.
const MAX_K: usize = 10_000;
/// Largest `ef_search` (HNSW beam width) a single query may request.
const MAX_EF_SEARCH: usize = 10_000;

type ApiError = (StatusCode, Json<ErrorResponse>);

fn bad_request(error: impl Into<String>) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: error.into(),
        }),
    )
}

/// Reject out-of-range `k` / `ef_search` before touching an index.
fn validate_search_params(k: usize, ef_search: Option<usize>) -> Result<(), ApiError> {
    if !(1..=MAX_K).contains(&k) {
        return Err(bad_request(format!("k must be between 1 and {MAX_K}")));
    }
    if let Some(ef) = ef_search
        && !(1..=MAX_EF_SEARCH).contains(&ef)
    {
        return Err(bad_request(format!(
            "ef_search must be between 1 and {MAX_EF_SEARCH}"
        )));
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("quiver=info".parse().unwrap()),
        )
        .json()
        .init();
    let data = env::var("QUIVER_DATA_PATH").unwrap_or_else(|_| "quiver-server.qvdb".into());
    let wal = env::var("QUIVER_WAL_PATH").unwrap_or_else(|_| "quiver-server.wal".into());
    let dimension = match env::var("QUIVER_DIMENSION") {
        Ok(value) => match value.parse::<u32>() {
            Ok(d) if d > 0 => d,
            _ => {
                eprintln!("QUIVER_DIMENSION must be a positive integer, got {value:?}");
                std::process::exit(1);
            }
        },
        Err(_) => 384,
    };
    let config = HnswConfig::new(16);
    let index = match std::path::Path::new(&data).exists() {
        true => HnswIndex::open(&data, &wal, config),
        false => HnswIndex::create(&data, &wal, dimension, Metric::Cosine, config),
    };
    let index = match index {
        Ok(index) => index,
        Err(e) => {
            eprintln!("failed to open or create server index at {data}: {e}");
            std::process::exit(1);
        }
    };
    if index.dimension() != dimension {
        eprintln!(
            "dimension mismatch: QUIVER_DIMENSION is {dimension} but the existing index at {data} has {} dimensions",
            index.dimension()
        );
        std::process::exit(1);
    }
    let bind = env::var("QUIVER_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = match tokio::net::TcpListener::bind(&bind).await {
        Ok(listener) => listener,
        Err(e) => {
            eprintln!("failed to bind {bind}: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!(address = %listener.local_addr().unwrap(), "Quiver server listening");
    let index = Arc::new(RwLock::new(index));
    let sq8 = Arc::new(RwLock::new(load_sq8_snapshot()));
    let ivfpq = Arc::new(RwLock::new(load_ivfpq_snapshot()));
    let shutdown = Arc::new(tokio::sync::Notify::new());
    spawn_auto_flush(Arc::clone(&index));
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/vectors", post(insert))
        .route("/search", post(search))
        .route("/search/batch", post(search_batch))
        .route("/sq8/search", post(search_sq8))
        .route("/ivfpq/search", post(search_ivfpq))
        .route("/vectors/{id}", delete(remove))
        .route(
            "/vectors/{id}/metadata",
            put(update_metadata).delete(clear_metadata),
        )
        .route("/shutdown", post(shutdown_handler))
        .with_state(AppState {
            index: Arc::clone(&index),
            sq8: Arc::clone(&sq8),
            ivfpq: Arc::clone(&ivfpq),
            shutdown: Arc::clone(&shutdown),
        });
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(shutdown))
        .await
        .expect("serve HTTP API");

    // Persist vectors and the graph-topology snapshot so the next start reopens
    // without rebuilding the HNSW graph.
    match index.write().unwrap().flush() {
        Ok(()) => tracing::info!("flushed index on shutdown"),
        Err(e) => tracing::error!(error = %e, "failed to flush index on shutdown"),
    }
}

async fn shutdown_signal(shutdown: Arc<tokio::sync::Notify>) {
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let notified = shutdown.notified();
    tokio::pin!(notified);
    tokio::select! {
        _ = &mut ctrl_c => tracing::info!("ctrl+c received; draining connections"),
        _ = &mut notified => tracing::info!("shutdown requested; draining connections"),
    }
}

/// Initiates a graceful shutdown. Useful on platforms where a running server
/// cannot receive a console Ctrl+C event (e.g. detached Windows processes).
async fn shutdown_handler(State(state): State<AppState>) -> StatusCode {
    state.shutdown.notify_one();
    StatusCode::ACCEPTED
}

/// Periodically flush vectors + graph snapshot so a hard kill loses at most
/// one interval of graph freshness instead of everything since startup.
///
/// `QUIVER_FLUSH_INTERVAL_SECS` sets the period (default 300); 0 disables.
/// Only the HNSW index is flushed — quantized snapshots are immutable.
fn spawn_auto_flush(index: SharedIndex) {
    let secs: u64 = env::var("QUIVER_FLUSH_INTERVAL_SECS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(300);
    if secs == 0 {
        return;
    }
    tracing::info!(interval_secs = secs, "auto-flush enabled");
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(secs));
        loop {
            ticker.tick().await;
            // The flush fsyncs under the write lock; run it on the blocking
            // pool so it never stalls an async worker thread.
            let index = Arc::clone(&index);
            let result = tokio::task::spawn_blocking(move || index.write().unwrap().flush()).await;
            match result {
                Ok(Ok(())) => tracing::debug!("auto-flush completed"),
                Ok(Err(e)) => tracing::error!(error = %e, "auto-flush failed"),
                Err(e) => tracing::error!(error = %e, "auto-flush task panicked"),
            }
        }
    });
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics(State(state): State<AppState>) -> Json<MetricsResponse> {
    let index = state.index.read().unwrap();
    Json(MetricsResponse {
        len: index.len(),
        dimension: index.dimension(),
        metric: format!("{:?}", index.metric()),
        max_level: index.max_level(),
        sq8_len: state.sq8.read().unwrap().as_ref().map(|s| s.len()),
        ivfpq_len: state.ivfpq.read().unwrap().as_ref().map(|i| i.len()),
    })
}

/// Load an optional SQ8 snapshot. `QUIVER_SQ8_PATH` unset or missing means
/// quantized search is disabled; a present-but-corrupt file is logged and
/// also disables the endpoint rather than refusing to start the HNSW server.
fn load_sq8_snapshot() -> Option<Sq8Index> {
    let path = env::var("QUIVER_SQ8_PATH").ok()?;
    if !std::path::Path::new(&path).exists() {
        return None;
    }
    match Sq8Index::load(&path) {
        Ok(index) => {
            tracing::info!(path = %path, len = index.len(), "loaded SQ8 snapshot");
            Some(index)
        }
        Err(e) => {
            tracing::warn!(path = %path, error = %e, "SQ8 snapshot invalid; endpoint disabled");
            None
        }
    }
}

/// Load an optional IVF-PQ snapshot. Same unset/missing/corrupt semantics as
/// [`load_sq8_snapshot`].
fn load_ivfpq_snapshot() -> Option<IvfPqIndex> {
    let path = env::var("QUIVER_IVFPQ_PATH").ok()?;
    if !std::path::Path::new(&path).exists() {
        return None;
    }
    match IvfPqIndex::load(&path) {
        Ok(index) => {
            tracing::info!(path = %path, len = index.len(), "loaded IVF-PQ snapshot");
            Some(index)
        }
        Err(e) => {
            tracing::warn!(path = %path, error = %e, "IVF-PQ snapshot invalid; endpoint disabled");
            None
        }
    }
}

async fn insert(
    State(state): State<AppState>,
    Json(request): Json<InsertRequest>,
) -> Result<(StatusCode, Json<InsertResponse>), (StatusCode, Json<ErrorResponse>)> {
    let mut index = state.index.write().unwrap();
    let id = match request.metadata {
        Some(metadata) => index.insert_with_metadata(&request.vector, metadata),
        None => index.insert(&request.vector),
    }
    .map_err(api_error)?;
    Ok((StatusCode::CREATED, Json(InsertResponse { id })))
}

/// Dispatch to filtered or unfiltered search depending on whether a filter was
/// supplied.
fn run_search(
    index: &HnswIndex,
    vector: &[f32],
    k: usize,
    ef_search: usize,
    filter: Option<&Filter>,
) -> Result<Vec<SearchHit>, quiver_core::error::QuiverError> {
    let hits = match filter {
        Some(filter) => index.search_filtered(vector, k, ef_search, filter)?,
        None => index.search(vector, k, ef_search)?,
    };
    Ok(hits
        .into_iter()
        .map(|hit| SearchHit {
            id: hit.vector_id,
            distance: hit.distance,
        })
        .collect())
}

async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<Vec<SearchHit>>, (StatusCode, Json<ErrorResponse>)> {
    validate_search_params(request.k, request.ef_search)?;
    let hits = run_search(
        &state.index.read().unwrap(),
        &request.vector,
        request.k,
        request.ef_search.unwrap_or(100),
        request.filter.as_ref(),
    )
    .map_err(api_error)?;
    Ok(Json(hits))
}

async fn search_batch(
    State(state): State<AppState>,
    Json(request): Json<BatchSearchRequest>,
) -> Result<Json<Vec<Vec<SearchHit>>>, (StatusCode, Json<ErrorResponse>)> {
    if request.queries.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "queries must not be empty".into(),
            }),
        ));
    }
    for query in &request.queries {
        validate_search_params(query.k, query.ef_search)?;
    }
    let index = state.index.read().unwrap();
    let mut results = Vec::with_capacity(request.queries.len());
    for query in &request.queries {
        let hits = run_search(
            &index,
            &query.vector,
            query.k,
            query.ef_search.unwrap_or(100),
            query.filter.as_ref(),
        )
        .map_err(api_error)?;
        results.push(hits);
    }
    Ok(Json(results))
}

async fn remove(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.index.write().unwrap().delete(id).map_err(api_error)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Replace the metadata attached to a live vector.
async fn update_metadata(
    State(state): State<AppState>,
    Path(id): Path<u64>,
    Json(request): Json<UpdateMetadataRequest>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state
        .index
        .write()
        .unwrap()
        .update_metadata(id, request.metadata)
        .map_err(api_error)?;
    Ok(StatusCode::NO_CONTENT)
}

/// Remove the metadata attached to a live vector.
async fn clear_metadata(
    State(state): State<AppState>,
    Path(id): Path<u64>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state
        .index
        .write()
        .unwrap()
        .clear_metadata(id)
        .map_err(api_error)?;
    Ok(StatusCode::NO_CONTENT)
}

fn quantized_unavailable(name: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: format!("{name} snapshot not loaded; set QUIVER_{name}_PATH"),
        }),
    )
}

/// Search a pre-built SQ8 snapshot. Filtered search is not supported on the
/// quantized path: SQ8 stores no metadata, so use `/search` with a filter.
async fn search_sq8(
    State(state): State<AppState>,
    Json(request): Json<Sq8SearchRequest>,
) -> Result<Json<Vec<SearchHit>>, (StatusCode, Json<ErrorResponse>)> {
    validate_search_params(request.k, None)?;
    let guard = state.sq8.read().unwrap();
    let index = guard.as_ref().ok_or_else(|| quantized_unavailable("SQ8"))?;
    let hits = index
        .search(&request.vector, request.k)
        .map_err(api_error)?;
    Ok(Json(
        hits.into_iter()
            .map(|hit| SearchHit {
                id: hit.vector_id,
                distance: hit.distance,
            })
            .collect(),
    ))
}

/// Search a pre-built IVF-PQ snapshot. L2-only with no metadata, so `nprobe`
/// (default 8) and `rerank_factor` (default 0, exact-L2 rerank off) are the
/// only quality knobs.
async fn search_ivfpq(
    State(state): State<AppState>,
    Json(request): Json<IvfPqSearchRequest>,
) -> Result<Json<Vec<SearchHit>>, (StatusCode, Json<ErrorResponse>)> {
    validate_search_params(request.k, None)?;
    let guard = state.ivfpq.read().unwrap();
    let index = guard
        .as_ref()
        .ok_or_else(|| quantized_unavailable("IVFPQ"))?;
    let hits = index
        .search(
            &request.vector,
            request.k,
            request.nprobe.unwrap_or(8),
            request.rerank_factor.unwrap_or(0),
        )
        .map_err(api_error)?;
    Ok(Json(
        hits.into_iter()
            .map(|hit| SearchHit {
                id: hit.vector_id,
                distance: hit.distance,
            })
            .collect(),
    ))
}

/// Map a core error to an HTTP status: client mistakes (bad dimension, empty
/// search target, missing ID) are 4xx; storage corruption and I/O failures are
/// 5xx so clients and alerts don't misclassify server-side faults.
fn api_error(error: quiver_core::error::QuiverError) -> (StatusCode, Json<ErrorResponse>) {
    let status = match &error {
        quiver_core::error::QuiverError::DimensionMismatch { .. }
        | quiver_core::error::QuiverError::InvalidInput(_)
        | quiver_core::error::QuiverError::EmptyIndex => StatusCode::BAD_REQUEST,
        quiver_core::error::QuiverError::NotFound(_) => StatusCode::NOT_FOUND,
        quiver_core::error::QuiverError::InvalidFormat(_)
        | quiver_core::error::QuiverError::Io(_)
        | quiver_core::error::QuiverError::WalChecksumMismatch { .. }
        | quiver_core::error::QuiverError::UnsupportedMetric(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    };
    (
        status,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}
