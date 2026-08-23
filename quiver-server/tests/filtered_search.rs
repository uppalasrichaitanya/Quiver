use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

struct ServerProcess(Child);

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn unused_address() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

fn start_server(data: &Path, wal: &Path, address: SocketAddr) -> ServerProcess {
    let child = Command::new(env!("CARGO_BIN_EXE_quiver-server"))
        .env("QUIVER_DATA_PATH", data)
        .env("QUIVER_WAL_PATH", wal)
        .env("QUIVER_DIMENSION", "3")
        .env("QUIVER_BIND", address.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    ServerProcess(child)
}

fn request(address: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(address).unwrap();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    let status = head
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u16>()
        .unwrap();
    (status, body.to_owned())
}

fn wait_until_ready(address: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(address).is_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("server did not start on {address}");
}

fn insert_vector(address: SocketAddr, body: &str) -> u64 {
    let (status, response) = request(address, "POST", "/vectors", body);
    assert_eq!(status, 201, "insert response: {response}");
    let inserted: serde_json::Value = serde_json::from_str(&response).unwrap();
    inserted["id"].as_u64().unwrap()
}

fn hit_ids(body: &str) -> Vec<u64> {
    let hits: serde_json::Value = serde_json::from_str(body).unwrap();
    hits.as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["id"].as_u64().unwrap())
        .collect()
}

/// Insert the standard fixture set and return the four assigned IDs:
/// (science 2024, sports 2024, science 1999, no metadata).
fn insert_fixture(address: SocketAddr) -> (u64, u64, u64, u64) {
    let id_science_2024 = insert_vector(
        address,
        r#"{"vector":[1.0,0.0,0.0],"metadata":{"category":"science","year":2024}}"#,
    );
    let id_sports_2024 = insert_vector(
        address,
        r#"{"vector":[0.0,1.0,0.0],"metadata":{"category":"sports","year":2024}}"#,
    );
    let id_science_1999 = insert_vector(
        address,
        r#"{"vector":[0.0,0.0,1.0],"metadata":{"category":"science","year":1999}}"#,
    );
    // Closest to most queries, but carries no metadata and must never match.
    let id_bare = insert_vector(address, r#"{"vector":[0.9,0.1,0.0]}"#);
    (id_science_2024, id_sports_2024, id_science_1999, id_bare)
}

#[test]
fn filtered_search_returns_only_matching_vectors() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let address = unused_address();

    let server = start_server(&data, &wal, address);
    wait_until_ready(address);
    let (id_science_2024, id_sports_2024, id_science_1999, id_bare) = insert_fixture(address);

    // Eq filter: only the two "science" vectors qualify.
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":10,"ef_search":50,
            "filter":{"Eq":{"key":"category","value":"science"}}}"#,
    );
    assert_eq!(status, 200, "filtered search response: {body}");
    let ids = hit_ids(&body);
    assert_eq!(ids.len(), 2, "filtered search response: {body}");
    assert!(ids.contains(&id_science_2024), "response: {body}");
    assert!(ids.contains(&id_science_1999), "response: {body}");
    assert!(!ids.contains(&id_sports_2024), "response: {body}");
    assert!(!ids.contains(&id_bare), "response: {body}");

    // And filter: category == science AND year == 2024 narrows to one vector.
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":10,"ef_search":50,
            "filter":{"And":[
                {"Eq":{"key":"category","value":"science"}},
                {"Eq":{"key":"year","value":2024}}
            ]}}"#,
    );
    assert_eq!(status, 200, "And-filter search response: {body}");
    assert_eq!(hit_ids(&body), vec![id_science_2024], "response: {body}");

    // No filter: all four vectors come back (backward compatibility).
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":10,"ef_search":50}"#,
    );
    assert_eq!(status, 200, "unfiltered search response: {body}");
    let ids = hit_ids(&body);
    assert_eq!(ids.len(), 4, "unfiltered search response: {body}");

    // Filter that matches nothing returns an empty list, not an error.
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":10,
            "filter":{"Eq":{"key":"category","value":"cooking"}}}"#,
    );
    assert_eq!(status, 200, "no-match search response: {body}");
    assert!(hit_ids(&body).is_empty(), "response: {body}");
    drop(server);
}

#[test]
fn search_rejects_malformed_filter() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let address = unused_address();

    let server = start_server(&data, &wal, address);
    wait_until_ready(address);
    insert_fixture(address);

    // Unknown filter variant. Axum's Json extractor rejects deserialization
    // failures with 422 Unprocessable Entity.
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":1,"filter":{"Nope":{}}}"#,
    );
    assert_eq!(status, 422, "malformed filter response: {body}");

    // Metadata must be a JSON object.
    let (status, body) = request(
        address,
        "POST",
        "/vectors",
        r#"{"vector":[1.0,0.0,0.0],"metadata":[1,2]}"#,
    );
    assert_eq!(status, 422, "malformed metadata response: {body}");
    drop(server);
}

#[test]
fn batch_search_applies_per_query_filters() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let address = unused_address();

    let server = start_server(&data, &wal, address);
    wait_until_ready(address);
    let (id_science_2024, _id_sports_2024, id_science_1999, _id_bare) = insert_fixture(address);

    let (status, body) = request(
        address,
        "POST",
        "/search/batch",
        r#"{"queries":[
            {"vector":[1.0,0.0,0.0],"k":10,
             "filter":{"Eq":{"key":"category","value":"science"}}},
            {"vector":[1.0,0.0,0.0],"k":10}
        ]}"#,
    );
    assert_eq!(status, 200, "batch search response: {body}");
    let results: serde_json::Value = serde_json::from_str(&body).unwrap();
    let results = results.as_array().unwrap();
    assert_eq!(results.len(), 2, "batch response: {body}");

    let filtered: Vec<u64> = results[0]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["id"].as_u64().unwrap())
        .collect();
    assert_eq!(filtered.len(), 2, "filtered query response: {body}");
    assert!(filtered.contains(&id_science_2024), "response: {body}");
    assert!(filtered.contains(&id_science_1999), "response: {body}");

    assert_eq!(
        results[1].as_array().unwrap().len(),
        4,
        "unfiltered query should see all vectors: {body}"
    );
    drop(server);
}

#[test]
fn metadata_and_filter_survive_graceful_restart() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let address = unused_address();

    let mut first = start_server(&data, &wal, address);
    wait_until_ready(address);
    let (id_science_2024, _id_sports_2024, id_science_1999, _id_bare) = insert_fixture(address);
    let (status, _) = request(address, "POST", "/shutdown", "");
    assert_eq!(status, 202, "shutdown should be accepted");

    // The server must exit on its own (flushing the snapshot and metadata
    // sidecar) before we reopen it.
    let deadline = Instant::now() + Duration::from_secs(15);
    let exited = loop {
        match first.0.try_wait() {
            Ok(Some(_)) => break true,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            _ => break false,
        }
    };
    assert!(exited, "server should exit after /shutdown");
    std::mem::forget(first); // already reaped; skip the Drop kill

    let second = start_server(&data, &wal, address);
    wait_until_ready(address);
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":10,"ef_search":50,
            "filter":{"Eq":{"key":"category","value":"science"}}}"#,
    );
    assert_eq!(status, 200, "filtered search after restart: {body}");
    let ids = hit_ids(&body);
    assert_eq!(ids.len(), 2, "filtered search after restart: {body}");
    assert!(ids.contains(&id_science_2024), "response: {body}");
    assert!(ids.contains(&id_science_1999), "response: {body}");
    drop(second);
}
