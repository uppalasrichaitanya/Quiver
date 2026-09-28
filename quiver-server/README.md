# quiver-server

The Axum HTTP API listens on `127.0.0.1:8080` by default. Set `QUIVER_BIND` to override the address.

- `GET /health`
- `POST /vectors` with `{"vector":[...],"metadata":{...}?}` returns `{"id":...}`
- `POST /search` with `{"vector":[...],"k":10,"ef_search":100?,"filter":{...}?}` returns hits
- `POST /search/batch` with `{"queries":[{"vector":[...],"k":10,"ef_search":100?,"filter":{...}?}]}` returns an array of hit-arrays in input order
- `POST /sq8/search` with `{"vector":[...],"k":10}` searches a pre-built SQ8 snapshot (503 when none loaded)
- `POST /ivfpq/search` with `{"vector":[...],"k":10,"nprobe":8?,"rerank_factor":0?}` searches a pre-built IVF-PQ snapshot (503 when none loaded)
- `DELETE /vectors/{id}`
- `POST /shutdown` triggers a graceful shutdown (flushes vectors + graph snapshot)
- `GET /metrics` returns `{"len":...,"dimension":...,"metric":...,"max_level":...,"sq8_len":...?,"ivfpq_len":...?}`

Set `QUIVER_SQ8_PATH` / `QUIVER_IVFPQ_PATH` to serve pre-built quantized snapshots (created via `Sq8Index::save` / `IvfPqIndex::save`); when unset, missing, or corrupt the quantized endpoints return 503 while HNSW keeps serving. Quantized search is L2-only with no metadata/filter support.

Set `QUIVER_DIMENSION` (default 384), `QUIVER_DATA_PATH`, and `QUIVER_WAL_PATH` before starting. The server opens an existing data path or creates a new index when it does not exist, so restarting preserves vectors.

Axum accepts concurrent HTTP connections around one `RwLock`-protected HNSW index (parallel reads, exclusive writes). Core mutations are serialized because `HnswIndex` currently uses a single-writer `&mut self` API.

On Windows GNU, make sure `C:\msys64\mingw64\bin` appears before any 32-bit `C:\MinGW\bin` entry in `PATH`. The incompatible 32-bit `dlltool.exe` fails to create 64-bit import libraries with `Invalid bfd target`.

`examples/semantic_search.py` exercises insertion, search, and deletion against a small text corpus using scikit-learn's `HashingVectorizer`.
