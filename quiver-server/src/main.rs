use std::{env, io::{BufRead, BufReader, Read, Write}, net::{TcpListener, TcpStream}};

use quiver_core::{distance::Metric, index::hnsw::{HnswConfig, HnswIndex}};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)] struct InsertRequest { vector: Vec<f32> }
#[derive(Serialize)] struct InsertResponse { id: u64 }
#[derive(Deserialize)] struct SearchRequest { vector: Vec<f32>, k: usize, ef_search: Option<usize> }
#[derive(Serialize)] struct SearchHit { id: u64, distance: f32 }
#[derive(Serialize)] struct ErrorResponse { error: String }

fn main() {
    let data = env::var("QUIVER_DATA_PATH").unwrap_or_else(|_| "quiver-server.qvdb".into());
    let wal = env::var("QUIVER_WAL_PATH").unwrap_or_else(|_| "quiver-server.wal".into());
    let dimension = env::var("QUIVER_DIMENSION").ok().and_then(|s| s.parse().ok()).unwrap_or(384);
    let mut index = HnswIndex::create(data, wal, dimension, Metric::Cosine, HnswConfig::new(16))
        .expect("create a new server index (choose unused QUIVER_*_PATH paths)");
    let listener = TcpListener::bind("127.0.0.1:8080").expect("bind 127.0.0.1:8080");
    println!("Quiver server listening on http://127.0.0.1:8080 ({dimension} dimensions)");
    for stream in listener.incoming() {
        match stream { Ok(stream) => handle(stream, &mut index), Err(error) => eprintln!("connection error: {error}") }
    }
}

fn handle(mut stream: TcpStream, index: &mut HnswIndex) {
    let result = read_request(&stream).and_then(|(method, path, body)| route(index, &method, &path, &body));
    let (status, body) = match result {
        Ok(response) => response,
        Err(error) => ("400 Bad Request", serde_json::to_vec(&ErrorResponse { error }).unwrap()),
    };
    let response = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    let _ = stream.write_all(response.as_bytes()).and_then(|_| stream.write_all(&body));
}

fn read_request(stream: &TcpStream) -> Result<(String, String, Vec<u8>), String> {
    let mut reader = BufReader::new(stream);
    let mut first = String::new(); reader.read_line(&mut first).map_err(|e| e.to_string())?;
    let mut parts = first.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_owned();
    let path = parts.next().ok_or("missing path")?.to_owned();
    let mut length = 0;
    loop {
        let mut line = String::new(); reader.read_line(&mut line).map_err(|e| e.to_string())?;
        if line == "\r\n" || line.is_empty() { break; }
        if let Some(value) = line.strip_prefix("Content-Length:").or_else(|| line.strip_prefix("content-length:")) {
            length = value.trim().parse().map_err(|_| "invalid content length")?;
        }
    }
    let mut body = vec![0; length]; reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok((method, path, body))
}

fn route(index: &mut HnswIndex, method: &str, path: &str, body: &[u8]) -> Result<(&'static str, Vec<u8>), String> {
    match (method, path) {
        ("GET", "/health") => Ok(("200 OK", b"{\"status\":\"ok\"}".to_vec())),
        ("POST", "/vectors") => {
            let request: InsertRequest = serde_json::from_slice(body).map_err(|e| e.to_string())?;
            let id = index.insert(&request.vector).map_err(|e| e.to_string())?;
            Ok(("201 Created", serde_json::to_vec(&InsertResponse { id }).unwrap()))
        }
        ("POST", "/search") => {
            let request: SearchRequest = serde_json::from_slice(body).map_err(|e| e.to_string())?;
            let hits = index.search(&request.vector, request.k, request.ef_search.unwrap_or(100)).map_err(|e| e.to_string())?;
            let response: Vec<_> = hits.into_iter().map(|hit| SearchHit { id: hit.vector_id, distance: hit.distance }).collect();
            Ok(("200 OK", serde_json::to_vec(&response).unwrap()))
        }
        ("DELETE", _) if path.starts_with("/vectors/") => {
            let id = path[9..].parse().map_err(|_| "invalid vector id")?;
            index.delete(id).map_err(|e| e.to_string())?;
            Ok(("200 OK", b"{}".to_vec()))
        }
        _ => Ok(("404 Not Found", b"{\"error\":\"not found\"}".to_vec())),
    }
}
