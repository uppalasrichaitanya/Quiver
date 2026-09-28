use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use quiver_core::{
    distance::Metric,
    index::ivfpq::{IvfPqConfig, IvfPqIndex},
    index::sq8::Sq8Index,
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

fn start_server(
    data: &Path,
    wal: &Path,
    sq8: &Path,
    ivfpq: &Path,
    address: SocketAddr,
) -> ServerProcess {
    let child = Command::new(env!("CARGO_BIN_EXE_quiver-server"))
        .env("QUIVER_DATA_PATH", data)
        .env("QUIVER_WAL_PATH", wal)
        .env("QUIVER_SQ8_PATH", sq8)
        .env("QUIVER_IVFPQ_PATH", ivfpq)
        .env("QUIVER_DIMENSION", "4")
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

fn fixture_vectors() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![5.0, 5.0, 5.0, 5.0],
        vec![10.0, 10.0, 10.0, 10.0],
        vec![-5.0, -5.0, -5.0, -5.0],
    ]
}

#[test]
fn quantized_endpoints_serve_snapshot_search() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let sq8_path = directory.path().join("index.qvsq");
    let ivfpq_path = directory.path().join("index.qvpq");

    let vectors = fixture_vectors();
    Sq8Index::build(&vectors, Metric::L2)
        .unwrap()
        .save(&sq8_path)
        .unwrap();
    let mut config = IvfPqConfig::new(2, 2, 4);
    config.store_vectors = true;
    IvfPqIndex::build(&vectors, &config)
        .unwrap()
        .save(&ivfpq_path)
        .unwrap();

    let address = unused_address();
    let _server = start_server(&data, &wal, &sq8_path, &ivfpq_path, address);
    wait_until_ready(address);

    let (status, body) = request(
        address,
        "POST",
        "/sq8/search",
        r#"{"vector":[4.9,5.1,5.0,4.8],"k":1}"#,
    );
    assert_eq!(status, 200, "sq8 response: {body}");
    let hits: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(hits[0]["id"], 2);

    let (status, body) = request(
        address,
        "POST",
        "/ivfpq/search",
        r#"{"vector":[4.9,5.1,5.0,4.8],"k":1,"nprobe":2,"rerank_factor":2}"#,
    );
    assert_eq!(status, 200, "ivfpq response: {body}");
    let hits: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(hits[0]["id"], 2);

    let (status, body) = request(address, "GET", "/metrics", "");
    assert_eq!(status, 200, "metrics response: {body}");
    let metrics: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(metrics["sq8_len"], 4);
    assert_eq!(metrics["ivfpq_len"], 4);
}

#[test]
fn quantized_endpoints_unavailable_without_snapshots() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let missing_sq8 = directory.path().join("missing.qvsq");
    let missing_ivfpq = directory.path().join("missing.qvpq");

    let address = unused_address();
    let _server = start_server(&data, &wal, &missing_sq8, &missing_ivfpq, address);
    wait_until_ready(address);

    let (status, _) = request(
        address,
        "POST",
        "/sq8/search",
        r#"{"vector":[1.0,1.0,1.0,1.0],"k":1}"#,
    );
    assert_eq!(status, 503);
    let (status, _) = request(
        address,
        "POST",
        "/ivfpq/search",
        r#"{"vector":[1.0,1.0,1.0,1.0],"k":1}"#,
    );
    assert_eq!(status, 503);
}
