# HANDOFF — Metadata + Filtered Search (Quiver)

> Read this first, then `PROGRESS.md`, then `quiver-project-plan-v2.md` (week-14 milestone).
> Goal: implement **metadata + filtered search**, the last missing Full-tier feature.
> Storage-format study is already DONE (below) — you can start designing/implementing immediately.

---

## 0. Environment quirks (READ — these cost hours to discover)

- **MSVC toolchain is unusable on this host.** No `link.exe` / Visual Studio / Build Tools / Windows Kits exist. Ignore any advice (including in the old `HANDOFF.md`) to use `cargo +stable-x86_64-pc-windows-msvc`. Use the default `stable-x86_64-pc-windows-gnu` target.
- **GNU builds need the MSYS2 PATH fix.** Prepend `C:\msys64\mingw64\bin` so the 64-bit MSYS2 `dlltool` shadows the broken 32-bit `C:\MinGW\bin\dlltool.exe` (which fails `Invalid bfd target` on `windows-sys`). Prefix every cargo command:
  ```
  set "PATH=C:\msys64\mingw64\bin;%PATH%" && cargo ...
  ```
- **Background subagents CANNOT run cargo/rustc** (permission wall: "background agents cannot prompt for confirmation"). They can read files, edit, and `git`, but the **main session must run all build/test/clippy/bench verification**. If you parallelize code-writing into subagents, plan to verify centrally.
- **`tracing_subscriber::fmt()` writes to STDOUT, not stderr.** When capturing `quiver-server` logs in tests, redirect stdout (`.stdout(Stdio::from(file))`) or the log file is empty.
- **Detached Windows processes can't receive Ctrl+C.** That's why `quiver-server` has a `POST /shutdown` endpoint; use it (not `taskkill`) for graceful-shutdown tests.
- **Benchmark builds have thermal/background variance on this laptop.** Anomalously *slow* build numbers (1.2–1.5x) are usually environmental, not regressions — re-run after a ~20 min idle cool-down. Search/recall numbers are stable. Don't run builds/tests/benches concurrently with a benchmark sweep.

## 1. Current state (all green, all pushed)

- `main` is clean and pushed to `origin` (github.com/uppalasrichaitanya/Quiver) through `1ba7900`. Git identity is the owner's (`uppalasrichaitanya`); push normally, never as a bot.
- **108 `quiver-core` tests + 5 `quiver-server` integration tests pass**, clippy `-D warnings` clean, `cargo fmt --check` clean.
- Recent commits this effort: `fb569a0` (HNSW hot-path + group commit), `1a448e5` (SIMD dispatch-once + 4-accumulator AVX2), `f6aff84` (server RwLock + `/search/batch`), `e03c191` (merge), `f24a777` (SIFT1M bench results), `200905d` (HNSW graph-topology snapshot persistence), `1ba7900` (graceful-shutdown flush).
- Headline SIFT1M (M=32/efc200/ef=100): **2680 QPS, p50 0.38 ms, Recall@10 0.9961** — on par with/above FAISS & hnswlib. Remaining gap vs competitors is **build time** (WAL fsync during build), not search.

## 2. Storage-format study (the groundwork — verified against the code)

### File header — `quiver-core/src/storage/header.rs`
- Fixed **64 bytes**: `QVDB` magic (4) · version u8 (off 4) · metric u8 (off 5) · reserved 2 (off 6) · dimension u32 LE (off 8) · vector_count u64 LE (off 12) · max_vector_id u64 LE (off 20) · reserved 36 (off 28).
- `LEGACY_FORMAT_VERSION = 1`, `FORMAT_VERSION = 2`. `FileHeader::from_bytes` **rejects** any version outside `1..=FORMAT_VERSION`. The version byte was explicitly reserved "so format changes later (e.g., metadata support) don't break old indexes" — **bumping to v3 is the intended extension path.**

### Data records — `quiver-core/src/storage/vecstore.rs`
- **Fixed-size records** in an mmap: `[FileHeader 64][u64 id + f32×dim][u64 id + f32×dim]...`.
- `record_size = VECTOR_ID_SIZE (=8) + dimension*4`. `vector_data_offset()` is `0` for legacy v1 else `VECTOR_ID_SIZE`.
- Hot path: `get_vector_unchecked(slot)` reads `HEADER_SIZE + slot*record_size + vector_data_offset` with no bounds check. **This fixed-size layout is why metadata must NOT go inline in the record.**
- `insert_raw(vector_id, data)` writes id+bytes, grows file ~2x, bumps `header.vector_count`, pushes to `vector_ids: Vec<u64>`, updates `max_vector_id`.
- `VectorStore` fields: `path, wal_path, mmap, file, header, wal, record_size, deleted_ids: HashSet<u64>, vector_ids: Vec<u64>`.
- `flush()` writes header into mmap + fsyncs; **does NOT clear the WAL** (deletes stay as durable tombstones). `compact()` rewrites live vectors into a temp store, atomically renames with a marker journal, then clears the WAL. `open()` replays the WAL (insert replay is idempotent for `id <= max_vector_id`).

### WAL — `quiver-core/src/storage/wal.rs`
- Entry: `[u32 len][body][crc32]`; CRC covers len-prefix + body. `read_entries` stops at first checksum mismatch, returns `(entries, valid_up_to)` for truncation.
- body: `[u8 op][u64 vector_id][f32×N (Insert only)]`. `WalOp`: `Insert=0`, `Delete=1`.
- **Constraint:** `parse_entry_body` for `Insert` requires the post-header remainder to be a multiple of 4 (pure f32s). Appending metadata to the existing Insert op would break this → **use a new op code** (e.g. `InsertMeta=2`). Old code never opens v3 stores (version gate), so a new op is safe.

### HNSW — `quiver-core/src/index/hnsw.rs`
- Wraps `VectorStore`. Already has **graph-topology snapshot persistence**: `flush`/`compact` write a CRC32-protected `<data_path>.graph`; `open` validates it (counts, per-node slot→vector_id, config, CRCs) and loads it, else rebuilds. **Mirror this exact pattern for a metadata sidecar** — it's proven and idiomatic here.
- Public API: `create`, `open`, `insert`, `insert_batch`, `search(&self, query, k, ef_search)`, `delete`, `flush`, `compact`, `len`, `dimension`, `metric`, `max_level`.

### Server — `quiver-server/src/main.rs`
- `SharedIndex = Arc<RwLock<HnswIndex>>`. Routes: `GET /health`, `POST /vectors`, `POST /search`, `POST /search/batch`, `DELETE /vectors/{id}`, `POST /shutdown`.
- Insert body `{"vector":[...]}`; search body `{"vector":[...],"k":N,"ef_search":M?}`; batch `{"queries":[...]}`. Integration tests spawn `CARGO_BIN_EXE_quiver-server` and capture **stdout**.

### Python — `quiver-py/`
- PyO3 `Index` API (create/open/insert/search/delete/flush). Extend with metadata + filter.

## 3. Recommended design (decision points flagged)

**Model** (new file, e.g. `quiver-core/src/metadata.rs`, re-export from `lib.rs`):
```rust
pub enum MetaValue { Str(String), Int(i64), Float(f64), Bool(bool) }  // + serde
pub struct Metadata { /* key -> value; BTreeMap<String, MetaValue> for determinism */ }
pub enum Filter {
    Eq { key: String, value: MetaValue },
    And(Vec<Filter>),
}
impl Filter { pub fn matches(&self, md: &Metadata) -> bool { /* ... */ } }
```
- **Decision 1 (scope):** start with `Eq` + `And` only (plan says "naive"). Defer `Or`/`In`/`Range` unless trivial. Keep `matches` total (missing key → `false`).

**Storage (durability-first, matching project ethos):**
- Bump data-file `FORMAT_VERSION` to **3**. v3 signals metadata may exist; new code still reads v1/v2 (empty metadata). Old binaries reject v3 via the existing version check.
- In-memory: add `metadata: Vec<Option<Metadata>>` to `VectorStore`, **indexed by slot** (parallel to `vector_ids`) — slot-indexed is fastest for post-filter during search. *(Decision 2: slot-`Vec` recommended over `HashMap<u64, Metadata>`.)*
- WAL: add `WalOp::InsertMeta = 2`, body `[op][u64 id][u32 meta_len][meta bytes][f32×dim]`. Metadata-less inserts keep `Insert=0`. Serialize `Metadata` with a small length-prefixed binary or serde_json bytes (serde_json is already a dep; binary is leaner — your call, but keep it versioned).
- Sidecar snapshot `<data_path>.meta`: CRC32-protected `(vector_id → metadata)` written on `flush`/`compact`, loaded+validated on `open` (skip rebuild), **exactly mirroring the `.graph` snapshot**. If missing/corrupt, rebuild metadata from WAL replay. *(Decision 3: sidecar+WAL recommended. Rejected: inline variable-length records (breaks fixed-size hot path) and in-memory-only (not durable).)*
- **Phasing option:** if you want momentum, you may land model + `search_filtered` + benchmark first with in-memory metadata, then add WAL/sidecar persistence as a follow-up commit. But the project standard is durable, so plan for persistence.

**Search (naive post-filter):**
- `HnswIndex::search_filtered(&self, query, k, ef_search, filter) -> Result<Vec<SearchResult>>`: over-fetch candidates from HNSW (raise ef so enough survive filtering), post-filter with `filter.matches`, sort, take top `k`. Handle "fewer than k match" gracefully. Keep plain `search` unchanged.
- **Decision 4 (over-fetch):** simplest correct start = search with `ef' = max(ef_search, k) * multiplier` or a fixed large ef, then filter. Measure how ef' must scale with selectivity.

**Benchmark (deliverable):**
- New binary (e.g. `quiver-core/src/bin/sift_filtered_benchmark.rs`) or a flag on `sift_benchmark`. Assign each SIFT vector a category (e.g. `id % buckets`) to hit **1% / 10% / 50%** selectivity; compute **brute-force filtered ground truth**; report filtered **Recall@10, QPS, p50/p99** per selectivity. Write JSON under `benchmarks/results/<date>-i7-12650h/raw/`. Follow `benchmarks/README.md` conventions and update it.

**Server + Python:**
- Insert accepts optional `"metadata": {k: v}`; search/batch accept optional `"filter"`. Map JSON ↔ `Metadata`/`Filter` (serde). Add integration tests (stdout capture). Extend `quiver-py` `Index.insert(..., metadata=...)` and `.search(..., filter=...)`.

## 4. Suggested execution order

1. `metadata.rs`: `MetaValue`/`Metadata`/`Filter` + serde + unit tests (`matches` truth table).
2. `VectorStore`: slot-indexed metadata, WAL `InsertMeta`, `.meta` sidecar save/load, v3 header. Unit tests incl. reopen-persistence + corrupt-sidecar-falls-back.
3. `HnswIndex::insert(..., metadata)` / `insert_batch`, `search_filtered`. Unit tests incl. recall across selectivities on random data.
4. Filtered SIFT benchmark binary; run 1%/10%/50%; record JSON.
5. Server endpoints + integration tests; `quiver-py` bindings.
6. Docs (`PROGRESS.md`, `benchmarks/README.md`, root `README.md`), full verification, commit, push.

## 5. Verification commands (run from repo root, main session)

```
set "PATH=C:\msys64\mingw64\bin;%PATH%" && cargo test -p quiver-core
set "PATH=C:\msys64\mingw64\bin;%PATH%" && cargo test -p quiver-server
set "PATH=C:\msys64\mingw64\bin;%PATH%" && cargo clippy --workspace --all-targets -- -D warnings
set "PATH=C:\msys64\mingw64\bin;%PATH%" && cargo fmt --all -- --check
```
Baseline to not regress: 108 core + 5 server tests green. Paste real output tails in your report; never fabricate results.

## 6. Definition of done

- Metadata persists across reopen (WAL replay and/or `.meta` sidecar), with a corrupt/missing-sidecar fallback test.
- `search_filtered` returns only matching vectors; filtered Recall@10 measured at 1%/10%/50% selectivity vs brute-force ground truth and recorded in `benchmarks/`.
- Server + Python expose metadata + filter; integration tests pass.
- All verification commands green; docs updated; committed and pushed as the owner.
