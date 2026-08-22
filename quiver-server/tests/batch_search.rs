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

fn insert_vector(address: SocketAddr, vector: &str) -> u64 {
    let body = format!(r#"{{"vector":[{vector}]}}"#);
    let (status, response) = request(address, "POST", "/vectors", &body);
    assert_eq!(status, 201, "insert response: {response}");
    let inserted: serde_json::Value = serde_json::from_str(&response).unwrap();
    inserted["id"].as_u64().unwrap()
}

#[test]
fn batch_search_returns_results_in_input_order() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let address = unused_address();

    let server = start_server(&data, &wal, address);
    wait_until_ready(address);
    let id_x = insert_vector(address, "1.0,0.0,0.0");
    let id_y = insert_vector(address, "0.0,1.0,0.0");
    let id_z = insert_vector(address, "0.0,0.0,1.0");

    let (status, body) = request(
        address,
        "POST",
        "/search/batch",
        r#"{"queries":[
            {"vector":[1.0,0.0,0.0],"k":1},
            {"vector":[0.0,1.0,0.0],"k":2,"ef_search":10},
            {"vector":[0.0,0.0,1.0],"k":1}
        ]}"#,
    );
    assert_eq!(status, 200, "batch search response: {body}");
    let results: serde_json::Value = serde_json::from_str(&body).unwrap();
    let results = results.as_array().unwrap();
    assert_eq!(results.len(), 3, "batch response: {body}");

    assert_eq!(
        results[0][0]["id"].as_u64(),
        Some(id_x),
        "first query should match its nearest neighbor: {body}"
    );
    assert_eq!(
        results[1][0]["id"].as_u64(),
        Some(id_y),
        "second query should match its nearest neighbor first: {body}"
    );
    assert_eq!(
        results[1].as_array().unwrap().len(),
        2,
        "second query asked for k=2: {body}"
    );
    assert_eq!(
        results[2][0]["id"].as_u64(),
        Some(id_z),
        "third query should match its nearest neighbor: {body}"
    );
    for hit in results.iter().flat_map(|hits| hits.as_array().unwrap()) {
        assert!(hit["id"].is_u64(), "hit id shape: {body}");
        assert!(hit["distance"].is_number(), "hit distance shape: {body}");
    }
    drop(server);
}

#[test]
fn batch_search_rejects_empty_queries() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let address = unused_address();

    let server = start_server(&data, &wal, address);
    wait_until_ready(address);
    let (status, body) = request(address, "POST", "/search/batch", r#"{"queries":[]}"#);
    assert_eq!(status, 400, "empty batch response: {body}");
    let error: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        error["error"].is_string(),
        "expected ErrorResponse shape: {body}"
    );
    drop(server);
}

#[test]
fn batch_search_rejects_zero_k() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let address = unused_address();

    let server = start_server(&data, &wal, address);
    wait_until_ready(address);
    insert_vector(address, "1.0,0.0,0.0");
    let (status, body) = request(
        address,
        "POST",
        "/search/batch",
        r#"{"queries":[{"vector":[1.0,0.0,0.0],"k":1},{"vector":[1.0,0.0,0.0],"k":0}]}"#,
    );
    assert_eq!(status, 400, "zero-k batch response: {body}");
    let error: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        error["error"].is_string(),
        "expected ErrorResponse shape: {body}"
    );
    drop(server);
}
