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
fn restart_reopens_existing_index_and_preserves_searchable_vectors() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("server.qvdb");
    let wal = directory.path().join("server.wal");
    let address = unused_address();

    let first = start_server(&data, &wal, address);
    wait_until_ready(address);
    let (status, body) = request(address, "POST", "/vectors", r#"{"vector":[1.0,0.0,0.0]}"#);
    assert_eq!(status, 201, "insert response: {body}");
    let inserted: serde_json::Value = serde_json::from_str(&body).unwrap();
    let inserted_id = inserted["id"].as_u64().unwrap();
    drop(first);

    let second = start_server(&data, &wal, address);
    wait_until_ready(address);
    let (status, body) = request(
        address,
        "POST",
        "/search",
        r#"{"vector":[1.0,0.0,0.0],"k":1,"ef_search":10}"#,
    );
    assert_eq!(status, 200, "search response: {body}");
    let hits: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        hits[0]["id"].as_u64(),
        Some(inserted_id),
        "search response: {body}"
    );
    drop(second);
}
