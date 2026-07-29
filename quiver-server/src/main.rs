use std::env;
use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::State, http::StatusCode, routing::{delete, get, post}};
use quiver_core::{distance::Metric, index::hnsw::{HnswConfig, HnswIndex}};
use serde::{Deserialize, Serialize};
use tracing_subscriber::EnvFilter;

type SharedIndex = Arc<Mutex<HnswIndex>>;

#[derive(Clone)]
struct AppState { index: SharedIndex }

#[derive(Deserialize)]
struct InsertRequest { vector: Vec<f32> }
#[derive(Serialize)]
struct InsertResponse { id: u64 }
#[derive(Deserialize)]
struct SearchRequest { vector: Vec<f32>, k: usize, ef_search: Option<usize> }
#[derive(Serialize)]
struct SearchHit { id: u64, distance: f32 }
#[derive(Serialize)]
struct ErrorResponse { error: String }

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env().add_directive("quiver=info".parse().unwrap())).json().init();
    let data = env::var("QUIVER_DATA_PATH").unwrap_or_else(|_| "quiver-server.qvdb".into());
    let wal = env::var("QUIVER_WAL_PATH").unwrap_or_else(|_| "quiver-server.wal".into());
    let dimension = env::var("QUIVER_DIMENSION").ok().and_then(|s| s.parse().ok()).unwrap_or(384);
    let index = HnswIndex::create(data, wal, dimension, Metric::Cosine, HnswConfig::new(16))
        .expect("create a new server index (choose unused QUIVER_*_PATH paths)");
    let app = Router::new()
        .route("/health", get(health))
        .route("/vectors", post(insert))
        .route("/search", post(search))
        .route("/vectors/{id}", delete(remove))
        .with_state(AppState { index: Arc::new(Mutex::new(index)) });
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    tracing::info!(address = %listener.local_addr().unwrap(), "Quiver server listening");
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> &'static str { "ok" }

async fn insert(State(state): State<AppState>, Json(request): Json<InsertRequest>) -> Result<Json<InsertResponse>, (StatusCode, Json<ErrorResponse>)> {
    let id = state.index.lock().unwrap().insert(&request.vector).map_err(api_error)?;
    Ok(Json(InsertResponse { id }))
}

async fn search(State(state): State<AppState>, Json(request): Json<SearchRequest>) -> Result<Json<Vec<SearchHit>>, (StatusCode, Json<ErrorResponse>)> {
    let hits = state.index.lock().unwrap().search(&request.vector, request.k, request.ef_search.unwrap_or(100)).map_err(api_error)?;
    Ok(Json(hits.into_iter().map(|hit| SearchHit { id: hit.vector_id, distance: hit.distance }).collect()))
}

async fn remove(State(state): State<AppState>, axum::extract::Path(id): axum::extract::Path<u64>) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    state.index.lock().unwrap().delete(id).map_err(api_error)?;
    Ok(StatusCode::NO_CONTENT)
}

fn api_error(error: quiver_core::error::QuiverError) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: error.to_string() }))
}
