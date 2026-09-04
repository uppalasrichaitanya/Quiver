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

#[test]
fn test_error_mapping_and_validation() {
    let dir = tempfile::TempDir::new().unwrap();
    let data = dir.path().join("server.qvdb");
    let wal = dir.path().join("server.wal");
    let address = unused_address();
    let _server = start_server(&data, &wal, address);
    wait_until_ready(address);

    // k < 1 is a client mistake on both search endpoints.
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":0}"#,
    );
    assert_eq!(status, 400, "k=0 response: {body}");

    // A dimension-mismatched insert is a client mistake, not a server fault.
    let (status, body) = request(address, "POST", "/vectors", r#"{"vector":[1.0,0.0]}"#);
    assert_eq!(status, 400, "bad-dim response: {body}");

    // Non-finite vector components must be rejected (they would otherwise
    // produce NaN distances that used to panic the search sort and poison
    // the server's lock for every later request). JSON has no NaN literal,
    // but an out-of-range literal like 1e999 parses to infinity.
    let (status, body) = request(address, "POST", "/vectors", r#"{"vector":[1.0,0.0,1e999]}"#);
    assert_eq!(
        status, 400,
        "non-finite insert should be a client error, got {status}: {body}"
    );

    // Deleting an unknown ID is 404, and a valid delete is 204.
    let (status, body) = request(address, "DELETE", "/vectors/999", "");
    assert_eq!(status, 404, "delete-unknown response: {body}");

    let (status, body) = request(address, "POST", "/vectors", r#"{"vector":[1.0,0.0,0.0]}"#);
    assert_eq!(status, 201, "insert response: {body}");
    let id: serde_json::Value = serde_json::from_str(&body).unwrap();
    let id = id["id"].as_u64().unwrap();

    let (status, body) = request(address, "POST", "/vectors", r#"{"vector":[0.0,1.0,0.0]}"#);
    assert_eq!(status, 201, "second insert response: {body}");

    let (status, _) = request(address, "DELETE", &format!("/vectors/{id}"), "");
    assert_eq!(status, 204, "valid delete must be 204");

    // The server must still be usable after all of the above (a poisoned
    // lock would 500 every subsequent request).
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":1}"#,
    );
    assert_eq!(status, 200, "post-error search response: {body}");
}

#[test]
fn test_existing_index_dimension_mismatch_refuses_to_start() {
    let dir = tempfile::TempDir::new().unwrap();
    let data = dir.path().join("dim.qvdb");
    let wal = dir.path().join("dim.wal");
    let address = unused_address();

    // Start with dimension 3 and store a vector so the data file exists.
    {
        let _server = start_server(&data, &wal, address);
        wait_until_ready(address);
        let (status, body) = request(address, "POST", "/vectors", r#"{"vector":[1.0,0.0,0.0]}"#);
        assert_eq!(status, 201, "insert response: {body}");
    }

    // Restart with a different dimension: must refuse to start instead of
    // silently serving the 3-d index.
    let child = Command::new(env!("CARGO_BIN_EXE_quiver-server"))
        .env("QUIVER_DATA_PATH", &data)
        .env("QUIVER_WAL_PATH", &wal)
        .env("QUIVER_DIMENSION", "5")
        .env("QUIVER_BIND", address.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success(), "server must exit non-zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("dimension mismatch"),
        "startup error must explain the mismatch: {stderr}"
    );
}
