#![no_main]

use libfuzzer_sys::fuzz_target;
use quiver_core::storage::format::validate_file_bytes;

const HEADER_SIZE: usize = 64;

fn synthesized_file(input: &[u8], version: u8) -> Vec<u8> {
    let mut file = vec![0_u8; HEADER_SIZE];
    file[..4].copy_from_slice(b"QVDB");
    file[4] = version;
    file[5] = input.first().copied().unwrap_or(0);

    for (index, byte) in input.iter().copied().skip(1).take(22).enumerate() {
        file[6 + index] = byte;
    }
    file.extend_from_slice(input.get(23..).unwrap_or_default());
    file
}

fn mutated_valid_file(input: &[u8], version: u8) -> Vec<u8> {
    let mut file = vec![0_u8; HEADER_SIZE];
    file[..4].copy_from_slice(b"QVDB");
    file[4] = version;
    file[5] = 0;
    file[8..12].copy_from_slice(&2_u32.to_le_bytes());
    file[12..20].copy_from_slice(&1_u64.to_le_bytes());
    file[20..28].copy_from_slice(&1_u64.to_le_bytes());
    if version >= 2 {
        file.extend_from_slice(&1_u64.to_le_bytes());
    }
    file.extend_from_slice(&1.0_f32.to_le_bytes());
    file.extend_from_slice(&2.0_f32.to_le_bytes());

    for chunk in input.chunks_exact(2) {
        let index = chunk[0] as usize % file.len();
        file[index] ^= chunk[1];
    }
    file
}

fuzz_target!(|data: &[u8]| {
    // Raw input covers arbitrary truncation, bad magic, and corrupt headers.
    let _ = validate_file_bytes(data);

    if !data.is_empty() {
        let truncated_at = data[0] as usize % data.len();
        let _ = validate_file_bytes(&data[..truncated_at]);
    }

    // Force every supported version marker while leaving dimension, count,
    // max-ID, record bytes, and offsets under fuzzer control. Versions 2 and
    // 3 share the same record layout (v3 only adds out-of-band metadata).
    for version in [1_u8, 2_u8, 3_u8] {
        let candidate = synthesized_file(data, version);
        let _ = validate_file_bytes(&candidate);

        let truncated_at = data
            .first()
            .map_or(0, |byte| *byte as usize % candidate.len());
        let _ = validate_file_bytes(&candidate[..truncated_at]);

        let mutated = mutated_valid_file(data, version);
        let _ = validate_file_bytes(&mutated);
    }
});
