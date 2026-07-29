use std::hint::black_box;
use std::time::{Duration, Instant};

use quiver_core::distance::{l2_squared, simd_available};

fn main() {
    let seconds = std::env::args()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(20);
    let dimension = 128_usize;
    let vector_count = 16_384_usize;

    let query: Vec<f32> = (0..dimension)
        .map(|index| ((index * 17 % 251) as f32 - 125.0) / 125.0)
        .collect();
    let vectors: Vec<Vec<f32>> = (0..vector_count)
        .map(|row| {
            (0..dimension)
                .map(|column| (((row * 31 + column * 13) % 509) as f32 - 254.0) / 254.0)
                .collect()
        })
        .collect();

    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut operations = 0_u64;
    let mut checksum = 0.0_f32;
    while Instant::now() < deadline {
        for vector in &vectors {
            checksum += l2_squared(black_box(&query), black_box(vector));
            operations += 1;
        }
    }

    println!(
        "mode={} operations={} checksum={}",
        if simd_available() {
            "avx2-fma"
        } else {
            "scalar"
        },
        operations,
        black_box(checksum)
    );
}
