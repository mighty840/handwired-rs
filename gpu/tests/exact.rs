//! The GPU reproduces the CPU reference evaluator bit for bit.

use handwired_rs::{
    Circuit, CompareOptions, CompileOptions, Lin, OutOfRange, Relation, Wire, emit_compare_with, handwired,
};
use handwired_rs_gpu::{Gpu, GpuBatch, GpuProgram};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
    fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (self.next() >> 11) as f64 / (1u64 << 53) as f64 * (high - low)
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(gpu) => {
            eprintln!("GPU: {} ({:?}, {:?})", gpu.info.name, gpu.info.device_type, gpu.info.backend);
            Some(gpu)
        }
        // CI sets HANDWIRED_REQUIRE_GPU, so a missing adapter fails instead of passing untested.
        Err(error) if std::env::var_os("HANDWIRED_REQUIRE_GPU").is_some() => panic!("{error}"),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

/// Run `records` on the GPU and check every root against the reference evaluator.
fn check(gpu: &Gpu, circuit: &Circuit, roots: &[Wire], records: &[Vec<f32>]) {
    let program = GpuProgram::new(gpu, circuit, roots).unwrap();
    let sources: Vec<f32> = records.concat();
    let outputs = program.run(gpu, &sources).unwrap();
    for (record, got) in records.iter().zip(outputs.chunks(roots.len())) {
        let values = circuit.evaluate_reference(record).unwrap();
        for (root, &got) in roots.iter().zip(got) {
            let want = values[root.index()];
            // == treats -0 and +0 as equal, like handwired's checks.
            assert!(got == want, "wire {} for {record:?}: GPU {got:e}, CPU {want:e}\n{}", root.index(), program.wgsl);
        }
    }
}

fn random_circuit(rng: &mut Rng, operations: usize) -> Circuit {
    let mut circuit = Circuit::new(6);
    for index in 0..4 {
        circuit.assume_range(circuit.source(index), -10.0, 10.0);
    }
    let bit = circuit.source(4);
    circuit.assume_bit(bit);
    let index = circuit.source(5);
    circuit.assume_range(index, 0.0, 7.0);
    circuit.assume_grid(index, 1.0);
    let coefficients = [1.0, -1.0, 0.5, 2.0, -0.25, 0.1, 3.7, -1.3];
    let mut wires: Vec<Wire> = (0..4).map(|index| circuit.source(index)).collect();
    while circuit.operations().len() < operations {
        let bounded: Vec<Wire> = wires
            .iter()
            .copied()
            .filter(|&wire| {
                let range = circuit.range(wire);
                range.is_finite() && range.min.abs().max(range.max.abs()) < 1e4
            })
            .collect();
        let form = |rng: &mut Rng| {
            let terms = (0..1 + rng.below(5)).map(|_| (rng.pick(&bounded), rng.pick(&coefficients))).collect();
            Lin::new(terms, rng.pick(&coefficients))
        };
        let wire = match rng.below(7) {
            0 => circuit.affine(form(rng)),
            1 => circuit.round_affine(form(rng)),
            2 => circuit.reglu(form(rng), form(rng)),
            3 => circuit.multiply(rng.pick(&bounded), rng.pick(&bounded)),
            4 => circuit.select(bit, rng.pick(&bounded), rng.pick(&bounded)),
            5 => {
                let (a, b) = (rng.pick(&bounded), rng.pick(&bounded));
                let relation = rng.pick(&[Relation::Greater, Relation::AtLeast, Relation::Equal]);
                emit_compare_with(&mut circuit, a, relation, b, CompareOptions { end_with_boolean: rng.below(2) == 0 })
            }
            _ => {
                let table: Vec<Wire> = (0..1 + rng.below(8)).map(|_| rng.pick(&bounded)).collect();
                circuit.index_read(index, &table, rng.pick(&[OutOfRange::Clamp, OutOfRange::Zero]))
            }
        };
        wires.push(wire);
    }
    circuit
}

#[test]
fn random_circuits_match_the_reference() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..40 {
        let circuit = random_circuit(&mut rng, 60);
        let roots: Vec<Wire> = (circuit.source_size()..circuit.width()).map(Wire::from_index).collect();
        let records: Vec<Vec<f32>> = (0..256)
            .map(|_| {
                let mut record: Vec<f32> = (0..4).map(|_| rng.uniform(-10.0, 10.0) as f32).collect();
                record.push(rng.below(2) as f32);
                record.push(rng.below(8) as f32);
                record
            })
            .collect();
        check(&gpu, &circuit, &roots, &records);
    }
}

/// Comparisons at their edges: equal values, one-ulp neighbors, grid neighbors.
#[test]
fn comparisons_match_native_f32() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(7);
    for (low, high, grid) in [(0.0, 16_384.0, None), (-8_192.0, 8_192.0, Some(1.0 / 64.0)), (-1.0, 1.0, None)] {
        for end_with_boolean in [false, true] {
            let mut circuit = Circuit::new(2);
            let (a, b) = (circuit.source(0), circuit.source(1));
            for wire in [a, b] {
                circuit.assume_range(wire, low, high);
                if let Some(step) = grid {
                    circuit.assume_grid(wire, step);
                }
            }
            let relations = [Relation::Greater, Relation::AtLeast, Relation::Less, Relation::AtMost, Relation::Equal];
            let bits: Vec<Wire> = relations
                .iter()
                .map(|&relation| emit_compare_with(&mut circuit, a, relation, b, CompareOptions { end_with_boolean }))
                .collect();
            let records: Vec<Vec<f32>> = (0..4096)
                .map(|_| {
                    let snap = |value: f64| match grid {
                        Some(step) => (value / step).round() * step,
                        None => value,
                    };
                    let x = snap(rng.uniform(low, high)) as f32;
                    let y = match rng.below(3) {
                        0 => x,
                        1 => match grid {
                            Some(step) => (x as f64 + step).min(high) as f32,
                            None => f32::from_bits(x.to_bits() + 1).min(high as f32),
                        },
                        _ => snap(rng.uniform(low, high)) as f32,
                    };
                    vec![x, y]
                })
                .collect();
            check(&gpu, &circuit, &bits, &records);
            // And against the native comparison itself.
            let program = GpuProgram::new(&gpu, &circuit, &bits).unwrap();
            let outputs = program.run(&gpu, &records.concat()).unwrap();
            for (record, got) in records.iter().zip(outputs.chunks(bits.len())) {
                for (relation, &bit) in relations.iter().zip(got) {
                    assert_eq!(bit == 1.0, relation.holds(record[0], record[1]), "{record:?} {}", relation.symbol());
                }
            }
        }
    }
}

#[handwired]
fn find(values: [u8; 8], target: u8) -> u8 {
    for (index, value) in values.into_iter().enumerate() {
        if value == target {
            return index as u8;
        }
    }
    255
}

#[handwired]
fn mix(#[range(-1e6, 1e6)] a: f32, #[range(-1.0, 1.0)] b: f32, #[range(-1e-3, 1e-3)] c: f32) -> (f32, f32, i32) {
    let left = a + b + c;
    let right = a + (b + c);
    (left * 0.1 - b * c, right, (left / 4.0) as i32)
}

#[test]
fn handwired_functions_run_as_batches() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(11);
    let compiled = find_circuit();
    let batch = GpuBatch::new(&gpu, &compiled).unwrap();
    let records: Vec<([u8; 8], u8)> = (0..10_000)
        .map(|_| {
            let values: [u8; 8] = std::array::from_fn(|_| rng.below(16) as u8);
            (values, rng.below(16) as u8)
        })
        .collect();
    let results = batch.run(&gpu, &records).unwrap();
    for (record, result) in records.iter().zip(results) {
        assert_eq!(result, find(record.0, record.1), "{record:?}");
    }

    let compiled = mix_circuit();
    let batch = GpuBatch::new(&gpu, &compiled).unwrap();
    let mut records: Vec<(f32, f32, f32)> = (0..10_000)
        .map(|_| (rng.uniform(-1e6, 1e6) as f32, rng.uniform(-1.0, 1.0) as f32, rng.uniform(-1e-3, 1e-3) as f32))
        .collect();
    // Where the two summation orders round differently.
    records.push((1e6, 0.03125, 0.001));
    let results = batch.run(&gpu, &records).unwrap();
    for (record, result) in records.iter().zip(results) {
        assert_eq!(result, mix(record.0, record.1, record.2), "{record:?}");
    }
}

/// Whether the GPU flushes subnormal results to zero: 2^-126 / 2.
fn flushes_subnormals(gpu: &Gpu) -> bool {
    let mut circuit = Circuit::new(1);
    let x = circuit.source(0);
    circuit.assume_range(x, 0.0, 1.0);
    let half = circuit.affine(x * 0.5);
    let program = GpuProgram::new(gpu, &circuit, &[half]).unwrap();
    program.run(gpu, &[f32::MIN_POSITIVE]).unwrap()[0] == 0.0
}

/// Comparisons near zero, where the operands or their difference are subnormal. A GPU that flushes subnormal
/// values to zero, like handwired's `flushSubnormals` test executor, reads a subnormal operand as 0 and a subnormal
/// difference as equality: every pair touching a subnormal value follows that model, and every other pair is exact.
#[test]
fn subnormal_comparisons_follow_the_flushing_model() {
    let Some(gpu) = gpu() else { return };
    let flushes = flushes_subnormals(&gpu);
    eprintln!("flushes subnormals: {flushes}");
    let smallest_normal = f32::MIN_POSITIVE;
    let subnormal = |value: f32| value != 0.0 && value.abs() < smallest_normal;
    let flush = |value: f32| if subnormal(value) { 0.0 } else { value };
    let touches_subnormal = |x: f32, y: f32| subnormal(x) || subnormal(y) || subnormal(x - y);
    let mut pairs = Vec::new();
    for x in [
        0.0,
        f32::from_bits(1),
        f32::from_bits(2),
        f32::from_bits(0x007F_FFFF),
        smallest_normal,
        f32::from_bits(smallest_normal.to_bits() + 1),
        smallest_normal * 3.0,
        1e-37,
    ] {
        for y in [x, f32::from_bits(x.to_bits() + 1), f32::from_bits(x.to_bits() + 2), x * 2.0, smallest_normal * 4.0] {
            for (x, y) in [(x, y), (y, x), (-x, -y), (-y, -x), (x, -y)] {
                pairs.push((x, y));
            }
        }
    }
    let mut flushed_pairs = 0;
    for end_with_boolean in [false, true] {
        let mut circuit = Circuit::new(2);
        let (a, b) = (circuit.source(0), circuit.source(1));
        for wire in [a, b] {
            circuit.assume_range(wire, -1e-36, 1e-36);
        }
        let relations = [Relation::Greater, Relation::AtLeast, Relation::Less, Relation::AtMost, Relation::Equal];
        let bits: Vec<Wire> = relations
            .iter()
            .map(|&relation| emit_compare_with(&mut circuit, a, relation, b, CompareOptions { end_with_boolean }))
            .collect();
        let exact: Vec<Vec<f32>> =
            pairs.iter().filter(|&&(x, y)| !flushes || !touches_subnormal(x, y)).map(|&(x, y)| vec![x, y]).collect();
        check(&gpu, &circuit, &bits, &exact);
        let program = GpuProgram::new(&gpu, &circuit, &bits).unwrap();
        let sources: Vec<f32> = pairs.iter().flat_map(|&(x, y)| [x, y]).collect();
        let outputs = program.run(&gpu, &sources).unwrap();
        for (&(x, y), got) in pairs.iter().zip(outputs.chunks(bits.len())) {
            let (seen_x, seen_y) = if flushes { (flush(x), flush(y)) } else { (x, y) };
            let equal = seen_x == seen_y || (flushes && subnormal(seen_x - seen_y));
            for (relation, &bit) in relations.iter().zip(got) {
                let want = match relation {
                    Relation::Equal => equal,
                    Relation::AtLeast | Relation::AtMost => equal || relation.holds(seen_x, seen_y),
                    Relation::Greater | Relation::Less => !equal && relation.holds(seen_x, seen_y),
                };
                assert_eq!(bit == 1.0, want, "{x:e} {} {y:e}, end_with_boolean {end_with_boolean}", relation.symbol());
            }
            if (got[4] == 1.0) != (x == y) {
                flushed_pairs += 1;
            }
        }
    }
    // A flushing GPU calls some unequal pairs equal, such as 0 and 2^-149.
    assert_eq!(flushed_pairs > 0, flushes);
}

/// Masked sums over one-hot groups, lowered to gathers, on the GPU: groups that allow all zeros, biases, signed and
/// chained reads, with build-time and recomputed depths.
#[test]
fn gathers_match_the_reference() {
    let Some(gpu) = gpu() else { return };
    let mut rng = Rng(23);
    let mut total_reads = 0;
    for circuit_number in 0..20 {
        let sizes = [1 + rng.below(6), 1 + rng.below(9)];
        let pool = 8;
        let mut circuit = Circuit::new(sizes[0] + sizes[1] + pool);
        let mut groups = Vec::new();
        let mut start = 0;
        for size in sizes {
            let members: Vec<Wire> = (start..start + size).map(|cell| circuit.source(cell)).collect();
            let allow_all_zero = rng.below(2) == 0;
            circuit.assume_one_hot(&members, allow_all_zero);
            groups.push((members, allow_all_zero));
            start += size;
        }
        for cell in start..start + pool {
            circuit.assume_range(circuit.source(cell), if cell % 2 == 0 { 0.0 } else { -1e4 }, 1e4);
        }
        let mut values: Vec<Wire> = (start..start + pool).map(|cell| circuit.source(cell)).collect();
        let mut roots = Vec::new();
        for _ in 0..4 {
            let (members, _) = &groups[rng.below(2)];
            let terms: Vec<(Wire, f64)> = members
                .iter()
                .map(|&member| {
                    let value = values[rng.below(values.len())];
                    (circuit.multiply(member, value), 1.0)
                })
                .collect();
            let read = circuit.affine(Lin::new(terms, rng.pick(&[0.0, 0.75, -3.0])));
            values.push(read);
            roots.push(read);
            roots.push(circuit.round_affine(read * 0.1 + 1.0));
        }
        let options = CompileOptions { recompute_depths: circuit_number % 2 == 0, ..Default::default() };
        let program = circuit.compile_with(Some(&roots), options);
        // Identical reads deduplicate, and a one-member group's bare read is its product.
        total_reads += program.gather_read_count();
        let records: Vec<Vec<f32>> = (0..256)
            .map(|_| {
                let mut record = vec![0.0f32; circuit.source_size()];
                let mut start = 0;
                for (members, allow_all_zero) in &groups {
                    let choice = rng.below(members.len() + usize::from(*allow_all_zero));
                    if choice < members.len() {
                        record[start + choice] = 1.0;
                    }
                    start += members.len();
                }
                for (offset, value) in record[start..].iter_mut().enumerate() {
                    let low = if (start + offset) % 2 == 0 { 0.0 } else { -1e4 };
                    *value = rng.uniform(low, 1e4) as f32;
                }
                record
            })
            .collect();
        let gpu_program = GpuProgram::from_program(&gpu, &program).unwrap();
        let outputs = gpu_program.run(&gpu, &records.concat()).unwrap();
        for (record, got) in records.iter().zip(outputs.chunks(roots.len())) {
            let values = circuit.evaluate_reference(record).unwrap();
            for (root, &got) in roots.iter().zip(got) {
                let want = values[root.index()];
                assert!(
                    got == want,
                    "wire {} for {record:?}: GPU {got:e}, CPU {want:e}\n{}",
                    root.index(),
                    gpu_program.wgsl
                );
            }
        }
    }
    assert!(total_reads >= 50, "only {total_reads} gather reads");
}

#[handwired]
fn bounded(#[range(0, 100)] a: u8) -> u8 {
    a / 2
}

#[test]
fn batches_check_declared_ranges() {
    let Some(gpu) = gpu() else { return };
    let batch = GpuBatch::new(&gpu, &bounded_circuit()).unwrap();
    assert_eq!(batch.run(&gpu, &[(0,), (99,), (100,)]).unwrap(), vec![0, 49, 50]);
    let error = batch.run(&gpu, &[(5,), (101,)]).unwrap_err();
    assert!(error.to_string().contains("record 1: source cell 0 is 101"), "{error}");
}
