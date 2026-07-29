# quiver-server

The minimal HTTP API listens on `127.0.0.1:8080`:

- `GET /health`
- `POST /vectors` with `{"vector":[...]}` → `{"id":...}`
- `POST /search` with `{"vector":[...],"k":10,"ef_search":100}` → hits
- `DELETE /vectors/{id}`

Set `QUIVER_DIMENSION` (default 384), `QUIVER_DATA_PATH`, and `QUIVER_WAL_PATH`
before starting. `examples/semantic_search.py` exercises all three mutations
and search against a small text corpus using scikit-learn's
`HashingVectorizer`.
