# File-format fuzzing

The `file_format` target calls the same bounds-checked parser used by
`VectorStore::open`. It exercises raw/truncated input plus synthesized files with
both v1 and v2 markers and fuzzer-controlled header fields, record data, IDs, and
offsets.

```bash
cargo install cargo-fuzz
cd fuzz
cargo +nightly fuzz run file_format -- -max_total_time=60
```

Run fuzz campaigns on Linux with the nightly toolchain. The Windows GNU C++
toolchain cannot compile `libfuzzer-sys`'s Windows LLVM shim; Windows requires a
compatible MSVC/LLVM setup.

The GitHub Actions workflow executes this target for 60 seconds on an Ubuntu
runner for every push and pull request.

Crash artifacts are written under `fuzz/artifacts/` and are intentionally ignored.
