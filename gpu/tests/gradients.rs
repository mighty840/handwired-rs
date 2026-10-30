//! GPU gradients against the CPU's.

use handwired_rs::{BackpropOptions, BooleanGradient, ComparisonGradient, IndexGradient, handwired};
use handwired_rs_gpu::{Gpu, GpuError, GpuGradients};

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

/// The balance example's fight, with four volleys.
#[handwired]
#[allow(clippy::too_many_arguments)]
fn skirmish(
    #[range(2, 20)] damage_a: f32,
    #[range(2, 20)] damage_b: f32,
    #[range(1, 200)] health_a: f32,
    #[range(1, 200)] health_b: f32,
    #[range(0, 2)] armor_a: f32,
    #[range(0, 2)] armor_b: f32,
    #[range(2, 6)] hits_on_a: u8,
    #[range(2, 6)] hits_on_b: u8,
    #[range(1, 6)] rolls_a: [u8; 4],
    #[range(1, 6)] rolls_b: [u8; 4],
) -> f32 {
    let mut taken_a: f32 = 0.0;
    let mut taken_b: f32 = 0.0;
    for volley in 0..4 {
        if rolls_a[volley] >= hits_on_a {
            let mut hit = damage_a - armor_b;
            if rolls_a[volley] == 6 {
                hit *= 2.0;
            }
            taken_b += hit;
        }
        if rolls_b[volley] >= hits_on_b {
            let mut hit = damage_b - armor_a;
            if rolls_b[volley] == 6 {
                hit *= 2.0;
            }
            taken_a += hit;
        }
    }
    (health_a - taken_a) * health_b - (health_b - taken_b) * health_a
}

/// A stats table read at a computed index, and a threshold on what it gives.
#[handwired]
fn table_rule(#[range(1, 20)] params: [f32; 6], #[range(0, 2)] faction: u8, #[range(0, 1)] stat: u8) -> f32 {
    let value = params[(faction * 2 + stat) as usize] * 3.0 - 1.0;
    if value > 20.0 { value * 0.5 } else { value.max(params[0]) }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn between(&mut self, low: u64, high: u64) -> u64 {
        low + self.next() % (high - low + 1)
    }
}

type Fight = (f32, f32, f32, f32, f32, f32, u8, u8, [u8; 4], [u8; 4]);

fn fights(count: usize) -> Vec<Fight> {
    let mut rng = Rng(0x5EED_0F16);
    (0..count)
        .map(|_| {
            let mut rolls = || [0; 4].map(|_: u8| rng.between(1, 6) as u8);
            let (a, b) = (rolls(), rolls());
            (
                rng.between(2, 20) as f32,
                rng.between(2, 20) as f32,
                rng.between(1, 200) as f32,
                rng.between(1, 200) as f32,
                rng.between(0, 2) as f32,
                rng.between(0, 2) as f32,
                rng.between(2, 6) as u8,
                rng.between(2, 6) as u8,
                a,
                b,
            )
        })
        .collect()
}

fn options(comparison_gradient: ComparisonGradient, index_gradient: IndexGradient) -> BackpropOptions {
    BackpropOptions { boolean_gradient: BooleanGradient::StraightThrough, index_gradient, comparison_gradient }
}

/// Each record's GPU gradients against the CPU tape's: bit for bit without surrogates; with them, within a
/// relative 1e-4 of the largest gradient of the record, since the GPU takes the exponential in f32.
fn check_records<Args: handwired_rs::dsl::Params>(
    gpu: &Gpu,
    gradients: &GpuGradients<Args>,
    sources: &[f32],
    exact: bool,
) {
    let cells = gradients.tape.source_size;
    let records = sources.len() / cells;
    let seeds: Vec<f32> = (0..records).map(|record| if record % 7 == 3 { 0.0 } else { 0.5 + record as f32 }).collect();
    let actual = gradients.per_record(gpu, sources, &seeds).unwrap();
    for (record, (source, &seed)) in sources.chunks(cells).zip(&seeds).enumerate() {
        let expected = gradients.tape.backpropagate(source, seed);
        let row = &actual[record * cells..(record + 1) * cells];
        let scale = expected.iter().fold(0f32, |largest, gradient| largest.max(gradient.abs()));
        for (cell, (&expected, &actual)) in expected.iter().zip(row).enumerate() {
            if exact {
                assert_eq!(expected.to_bits(), actual.to_bits(), "record {record} cell {cell}: {expected} vs {actual}");
            } else {
                assert!(
                    (expected - actual).abs() <= 1e-4 * scale,
                    "record {record} cell {cell}: {expected} vs {actual}"
                );
            }
        }
    }
}

#[test]
fn fight_gradients_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let compiled = skirmish_circuit();
    let records = fights(4096);
    let encoded = compiled.encode_records(&records).unwrap();
    for comparison in [
        ComparisonGradient::Exact,
        ComparisonGradient::Surrogate { width: 1.5 },
        ComparisonGradient::RelativeSurrogate { fraction: 0.1 },
    ] {
        let options = options(comparison, IndexGradient::Zero);
        let mut gradients = GpuGradients::new(&gpu, &compiled, 0, options).unwrap();
        let exact = comparison == ComparisonGradient::Exact;
        check_records(&gpu, &gradients, encoded.sources(), exact);
        // Small dispatches give the same rows.
        gradients.dispatch_records = Some(300);
        check_records(&gpu, &gradients, encoded.sources(), exact);
        // Summed with weights, against the CPU's sum: per-record values agree, the order of the f64 sums doesn't.
        let weights: Vec<f32> = (0..records.len()).map(|record| (record % 5) as f32 * 0.25).collect();
        let cpu = compiled.gradients_encoded(&encoded, 0, Some(&weights), options).unwrap();
        let ours = gradients.gradients(&gpu, &records, Some(&weights)).unwrap();
        for (cell, (&cpu, &ours)) in cpu.iter().zip(&ours).enumerate() {
            let tolerance = if exact { 1e-12 } else { 1e-4 } * cpu.abs().max(1.0);
            assert!((cpu - ours).abs() <= tolerance, "{comparison:?} cell {cell}: {cpu} vs {ours}");
        }
    }
}

#[test]
fn table_read_gradients_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let compiled = table_rule_circuit();
    let mut rng = Rng(0x7AB1E);
    let records: Vec<_> = (0..2048)
        .map(|_| ([0; 6].map(|_: u8| rng.between(1, 20) as f32), rng.between(0, 2) as u8, rng.between(0, 1) as u8))
        .collect();
    let encoded = compiled.encode_records(&records).unwrap();
    for index in [IndexGradient::Zero, IndexGradient::BackwardDifference] {
        let gradients = GpuGradients::new(&gpu, &compiled, 0, options(ComparisonGradient::Exact, index)).unwrap();
        check_records(&gpu, &gradients, encoded.sources(), true);
    }
}

#[test]
fn a_rejecting_rule_is_an_error_before_anything_runs() {
    let Some(gpu) = gpu() else { return };
    let compiled = table_rule_circuit();
    let straight = BackpropOptions { boolean_gradient: BooleanGradient::StraightThrough, ..Default::default() };
    assert!(matches!(GpuGradients::new(&gpu, &compiled, 0, straight), Err(GpuError::Gradient(_))));
}
