# Criterion scalar vs AVX2/FMA

Run: `cargo bench -p quiver-core --bench distance_benchmarks` on the benchmark
host (i7-12650H, Windows 11, release profile). Criterion mean time per distance
call at 128 dimensions:

| Kernel | AVX2/FMA dispatch | Scalar | Speedup |
|---|---:|---:|---:|
| L2 squared | 7.12 ns | 41.49 ns | 5.8x |
| Dot product | 5.63 ns | 37.52 ns | 6.7x |
| Cosine similarity | 17.42 ns | 77.81 ns | 4.5x |

The complete Criterion distributions remain in `target/criterion` after a
local run. A sampling flamegraph could not be produced on this Windows host:
`cargo-flamegraph` and an equivalent sampling profiler were not installed.
