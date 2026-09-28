# quiver-server

The Axum HTTP API listens on `127.0.0.1:8080` by default. Set `QUIVER_BIND` to override the address.

- `GET /health`
- `POST /vectors` with `{"vector":[...],"metadata":{...}?}` returns `{"id":...}`
- `POST /search` with `{"vector":[...],"k":10,"ef_search":100?,"filter":{...}?}` returns hits
- `POST /search/batch` with `{"queries":[{"vector":[...],"k":10,"ef_search":100?,"filter":{...}?}]}` returns an array of hit-arrays in input order
- `DELETE /vectors/{id}`
- `POST /shutdown` triggers a graceful shutdown (flushes vectors + graph snapshot)
- `GET /metrics` returns `{"len":...,"dimension":...,"metric":...,"max_level":...}`

Set `QUIVER_DIMENSION` (default 384), `QUIVER_DATA_PATH`, and `QUIVER_WAL_PATH` before starting. The server opens an existing data path or creates a new index when it does not exist, so restarting preserves vectors.

Axum accepts concurrent HTTP connections around one `RwLock`-protected HNSW index (parallel reads, exclusive writes). Core mutations are serialized because `HnswIndex` currently uses a single-writer `&mut self` API.

On Windows GNU, make sure `C:\msys64\mingw64\bin` appears before any 32-bit `C:\MinGW\bin` entry in `PATH`. The incompatible 32-bit `dlltool.exe` fails to create 64-bit import libraries with `Invalid bfd target`.

`examples/semantic_search.py` exercises insertion, search, and deletion against a small text corpus using scikit-learn's `HashingVectorizer`.
