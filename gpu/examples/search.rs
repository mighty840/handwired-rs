//! Search 100,000 text records for a pattern on the GPU, from an ordinary
//! Rust function, and check every result against that function.
//!
//! cargo run --release -p handwired-rs-gpu --example search

use std::time::Instant;

use handwired_rs::handwired;
use handwired_rs::{BatchExecutor, LANES, TensorExecutor};
use handwired_rs_gpu::{Gpu, GpuBatch};

const TEXT: usize = 32;

/// Where `pattern` first occurs in `text`, as (found, position).
#[handwired]
fn find_pattern(text: [u8; 32], pattern: [u8; 3]) -> (bool, u8) {
    for start in 0..30 {
        if text[start] == pattern[0] && text[start + 1] == pattern[1] && text[start + 2] == pattern[2] {
            return (true, start as u8);
        }
    }
    (false, 0)
}

fn main() {
    let records = 100_000;
    let words: [&[u8]; 8] = [b"cat ", b"dog ", b"the ", b"cart ", b"scatter ", b"a ", b"category ", b"doge "];
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let inputs: Vec<([u8; TEXT], [u8; 3])> = (0..records)
        .map(|_| {
            let mut text = Vec::new();
            while text.len() < TEXT {
                text.extend_from_slice(words[(next() % 8) as usize]);
            }
            let text: [u8; TEXT] = text[..TEXT].try_into().unwrap();
            let pattern = if next() % 2 == 0 { *b"cat" } else { *b"dog" };
            (text, pattern)
        })
        .collect();

    let start = Instant::now();
    let compiled = find_pattern_circuit();
    println!(
        "circuit: {} live operations in {} layers, built in {:.1?}",
        compiled.program.live_operation_count(),
        compiled.program.layers.len(),
        start.elapsed()
    );

    let start = Instant::now();
    let native: Vec<(bool, u8)> = inputs.iter().map(|&(text, pattern)| find_pattern(text, pattern)).collect();
    let native_time = start.elapsed();

    let start = Instant::now();
    let mut executor = TensorExecutor::new(&compiled.program);
    let mut source = Vec::with_capacity(TEXT + 3);
    let cpu: Vec<(bool, u8)> = inputs
        .iter()
        .map(|record| {
            source.clear();
            handwired_rs::dsl::Params::encode(&(record.0, record.1), &mut source);
            let outputs = executor.run(&source);
            (outputs[0] == 1.0, outputs[1] as u8)
        })
        .collect();
    let cpu_time = start.elapsed();

    let mut sources = Vec::with_capacity(records * (TEXT + 3));
    for record in &inputs {
        handwired_rs::dsl::Params::encode(record, &mut sources);
    }
    let decode = |outputs: Vec<f32>| -> Vec<(bool, u8)> {
        outputs.chunks(2).map(|record| (record[0] == 1.0, record[1] as u8)).collect()
    };
    let start = Instant::now();
    let batched = decode(BatchExecutor::new(&compiled.program).run(&sources));
    let batch_time = start.elapsed();
    let start = Instant::now();
    let parallel = decode(BatchExecutor::run_parallel(&compiled.program, &sources, None));
    let parallel_time = start.elapsed();
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get());

    let gpu = Gpu::new().expect("a GPU adapter");
    let start = Instant::now();
    let batch = GpuBatch::new(&gpu, &compiled).unwrap();
    let shader_time = start.elapsed();
    let _ = batch.run(&gpu, &inputs[..1024]).unwrap();
    let start = Instant::now();
    let on_gpu = batch.run(&gpu, &inputs).unwrap();
    let gpu_time = start.elapsed();

    assert_eq!(cpu, native, "the CPU tensor executor disagrees with the function");
    assert_eq!(batched, native, "the batch executor disagrees with the function");
    assert_eq!(parallel, native, "the parallel batch executor disagrees with the function");
    assert_eq!(on_gpu, native, "the GPU disagrees with the function");
    let found = native.iter().filter(|result| result.0).count();
    println!("{records} records, {found} contain their pattern; every result matches the function");
    println!("plain Rust:           {native_time:>10.1?}");
    println!("CPU tensor executor:  {cpu_time:>10.1?}  (one record at a time)");
    println!("CPU batch executor:   {batch_time:>10.1?}  ({LANES} records at a time, one thread)");
    println!("CPU batch executor:   {parallel_time:>10.1?}  ({threads} threads)");
    println!(
        "GPU ({}, {:?}): {gpu_time:>10.1?}  (upload, run, read back; shader built once in {shader_time:.1?})",
        gpu.info.name, gpu.info.device_type
    );
}
