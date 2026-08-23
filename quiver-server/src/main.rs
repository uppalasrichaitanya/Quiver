use std::{
    env,
    sync::{Arc, RwLock},
};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
};
use quiver_core::{
    distance::Metric,
    index::hnsw::{HnswConfig, HnswIndex},
};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

type SharedIndex = Arc<RwLock<HnswIndex>>;

#[derive(Clone)]
struct AppState {
    index: SharedIndex,
    shutdown: Arc<tokio::sync::Notify>,
}

#[derive(Deserialize)]
struct InsertRequest {
    vector: Vec<f32>,
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
}
#[derive(Serialize)]
struct SearchHit {
    id: u64,
    distance: f32,
}
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
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
    let dimension = env::var("QUIVER_DIMENSION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(384);
    let config = HnswConfig::new(16);
    let index = if std::path::Path::new(&data).exists() {
        HnswIndex::open(&data, &wal, config)
    } else {
        HnswIndex::create(&data, &wal, dimension, Metric::Cosine, config)
    }
    .expect("open or create server index");
    let bind = env::var("QUIVER_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .expect("bind server listener");
    tracing::info!(address = %listener.local_addr().unwrap(), "Quiver server listening");
    let index = Arc::new(RwLock::new(index));
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let app = Router::new()
        .route("/health", get(health))
        .route("/vectors", post(insert))
        .route("/search", post(search))
        .route("/search/batch", post(search_batch))
        .route("/vectors/{id}", delete(remove))
        .route("/shutdown", post(shutdown_handler))
        .with_state(AppState {
            index: Arc::clone(&index),
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

async fn health() -> &'static str {
    "ok"
}

async fn insert(
    State(state): State<AppState>,
    Json(request): Json<InsertRequest>,
) -> Result<(StatusCode, Json<InsertResponse>), (StatusCode, Json<ErrorResponse>)> {
    let id = state
        .index
        .write()
        .unwrap()
        .insert(&request.vector)
        .map_err(api_error)?;
    Ok((StatusCode::CREATED, Json(InsertResponse { id })))
}

async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<Vec<SearchHit>>, (StatusCode, Json<ErrorResponse>)> {
    let hits = state
        .index
        .read()
        .unwrap()
        .search(&request.vector, request.k, request.ef_search.unwrap_or(100))
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
    if request.queries.iter().any(|query| query.k < 1) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "k must be at least 1 for every query".into(),
            }),
        ));
    }
    let index = state.index.read().unwrap();
    let mut results = Vec::with_capacity(request.queries.len());
    for query in &request.queries {
        let hits = index
            .search(&query.vector, query.k, query.ef_search.unwrap_or(100))
            .map_err(api_error)?;
        results.push(
            hits.into_iter()
                .map(|hit| SearchHit {
                    id: hit.vector_id,
                    distance: hit.distance,
                })
                .collect(),
        );
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

fn api_error(error: quiver_core::error::QuiverError) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: error.to_string(),
        }),
    )
}
