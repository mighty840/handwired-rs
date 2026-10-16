//! A fixed-capacity key-value store whose every operation is an exact circuit:
//! the store's state goes in, the next state and the answer come out.
//!
//! A `get` compares the key with every slot and picks the matching value,
//! which is hard attention: a one-hot match over keys, multiplied into the
//! values. A `put` overwrites the matching slot, or fills the first empty one,
//! as a select per slot. The function below is plain Rust, and the circuit is
//! checked against it.
//!
//! cargo run --release -p handwired-rs-gpu --example kv_store

use std::time::Instant;

use handwired_rs::TensorExecutor;
use handwired_rs::dsl::Params;
use handwired_rs::handwired;
use handwired_rs_gpu::{Gpu, GpuBatch};

const SLOTS: usize = 16;
const GET: u8 = 0;
const PUT: u8 = 1;
const DELETE: u8 = 2;

/// One operation on the store. A key of 0 marks an empty slot. Returns the
/// next keys and values, whether the key was present, and its value before
/// the operation.
///
/// Keys are declared within 0..15 here so that random tests hit present keys
/// often; any range within u16 works.
#[handwired]
fn kv_step(
    #[range(0, 15)] keys: [u16; 16],
    values: [u16; 16],
    #[range(0, 2)] op: u8,
    #[range(1, 15)] key: u16,
    value: u16,
) -> ([u16; 16], [u16; 16], bool, u16) {
    let mut keys = keys;
    let mut values = values;
    let mut found = false;
    let mut previous: u16 = 0;
    for slot in 0..16 {
        if keys[slot] == key {
            found = true;
            previous = values[slot];
            if op == PUT {
                values[slot] = value;
            }
            if op == DELETE {
                keys[slot] = 0;
                values[slot] = 0;
            }
            break;
        }
    }
    if op == PUT && !found {
        // Insert into the first empty slot; a full store drops the write.
        for slot in 0..16 {
            if keys[slot] == 0 {
                keys[slot] = key;
                values[slot] = value;
                break;
            }
        }
    }
    (keys, values, found, previous)
}

type State = ([u16; SLOTS], [u16; SLOTS]);
type Operation = ([u16; SLOTS], [u16; SLOTS], u8, u16, u16);

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

fn main() {
    let start = Instant::now();
    let compiled = kv_step_circuit();
    println!(
        "kv_step: {} live operations in {} layers, built in {:.1?}",
        compiled.program.live_operation_count(),
        compiled.program.layers.len(),
        start.elapsed()
    );
    let report = compiled.verify(20_000).unwrap();
    println!("verified against the function on {} sampled inputs", report.checked);

    // A session: the circuit holds the store's state from one operation to the next, like a database.
    let mut executor = TensorExecutor::new(&compiled.program);
    let mut circuit_state: State = ([0; SLOTS], [0; SLOTS]);
    let mut native_state: State = ([0; SLOTS], [0; SLOTS]);
    let script = [(PUT, 3, 300), (PUT, 7, 700), (GET, 3, 0), (PUT, 3, 333), (GET, 3, 0), (DELETE, 7, 0), (GET, 7, 0)];
    let mut source = Vec::new();
    for (op, key, value) in script {
        source.clear();
        (circuit_state.0, circuit_state.1, op, key, value).encode(&mut source);
        let outputs = executor.run(&source);
        let keys: [u16; SLOTS] = std::array::from_fn(|slot| outputs[slot] as u16);
        let values: [u16; SLOTS] = std::array::from_fn(|slot| outputs[SLOTS + slot] as u16);
        let (found, previous) = (outputs[2 * SLOTS] == 1.0, outputs[2 * SLOTS + 1] as u16);
        circuit_state = (keys, values);
        let native = kv_step(native_state.0, native_state.1, op, key, value);
        native_state = (native.0, native.1);
        assert_eq!((circuit_state, found, previous), (native_state, native.2, native.3));
        let name = ["get", "put", "delete"][op as usize];
        println!("  {name:6} {key:>2} {value:>4}  ->  found {found:5}, previous {previous}");
    }

    // 100,000 independent operations on 100,000 different stores, in one GPU dispatch.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let records: Vec<Operation> = (0..100_000)
        .map(|_| {
            let keys: [u16; SLOTS] = std::array::from_fn(|_| rng.below(16) as u16);
            let values: [u16; SLOTS] = std::array::from_fn(|_| rng.below(65_536) as u16);
            (keys, values, rng.below(3) as u8, 1 + rng.below(15) as u16, rng.below(65_536) as u16)
        })
        .collect();
    let start = Instant::now();
    let native: Vec<_> =
        records.iter().map(|&(keys, values, op, key, value)| kv_step(keys, values, op, key, value)).collect();
    let native_time = start.elapsed();
    let gpu = Gpu::new().expect("a GPU adapter");
    let batch = GpuBatch::new(&gpu, &compiled).unwrap();
    let _ = batch.run(&gpu, &records[..1024]).unwrap();
    let start = Instant::now();
    let on_gpu = batch.run(&gpu, &records).unwrap();
    let gpu_time = start.elapsed();
    assert_eq!(on_gpu, native, "the GPU disagrees with the function");
    println!("100,000 operations on separate stores; every result matches the function");
    println!("plain Rust: {native_time:>9.1?}");
    println!("GPU ({}, {:?}): {gpu_time:>9.1?}  (upload, run, read back)", gpu.info.name, gpu.info.device_type);
}
