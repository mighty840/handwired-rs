//! Gradients on the GPU: the backward pass of one output, one record per invocation, summed over the records on the
//! CPU.
//!
//! A circuit's [`GradientTape`] is data for one fixed shader, which runs each record forward through the tape's
//! operations, keeping every value it keeps, then back through them, as [`GradientTape::backpropagate`] does. So
//! there is no shader per program to compile, whatever its length, and nothing to split into stages; each record
//! needs two f32 per tape wire of GPU memory while it runs, and records go in dispatches that fit.
//!
//! The values and gradients are the CPU reference's, bit for bit, with the forward shaders' guards against fused
//! multiply-adds (see [`wgsl`]) and their limits: subnormal values may flush to zero, and NaN and
//! infinities aren't handled. Two rules are computed in f32 instead of the reference's f64, so they differ in the last
//! bits: a comparison's surrogate slope (an exponential), and an index read's backward difference. Sums over records
//! are f64, in record order; [`Compiled::gradients`] sums each thread's records first, so totals agree to f64
//! rounding, not bit for bit.

use std::marker::PhantomData;

use handwired_rs::dsl::{Compiled, EncodedRecords, Params};
use handwired_rs::tape::OPERATION_WORDS;
use handwired_rs::{BackpropOptions, GradientTape, Range};
use wgpu::util::DeviceExt;

use crate::{Gpu, GpuError, encode_checked, wgsl};

/// The interpreter. Tape words are as [`GradientTape`] documents them; `work` holds each tape wire's value for every
/// record of the dispatch, then each one's gradient, wire by wire, so neighbouring invocations read neighbouring
/// words.
const SHADER: &str = r#"
struct Params {
    lowest: f32,
    records: u32,
    row: u32,
    operation_count: u32,
    width: u32,
    output: u32,
    source_size: u32,
    terms_at: u32,
    candidates_at: u32,
    seeds_at: u32,
    unused0: u32,
    unused1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tape: array<u32>;
@group(0) @binding(2) var<storage, read> inputs: array<f32>;
@group(0) @binding(3) var<storage, read_write> work: array<f32>;
@group(0) @binding(4) var<storage, read_write> results: array<f32>;

var<private> record: u32;

// max(x, lowest) is x for every finite x, but the compiler can't prove it, so no multiply-add fuses across it.
fn g(x: f32) -> f32 { return max(x, params.lowest); }

fn value(wire: u32) -> f32 { return work[wire * params.records + record]; }
fn set_value(wire: u32, x: f32) { work[wire * params.records + record] = x; }
fn gradient(wire: u32) -> f32 { return work[(params.width + wire) * params.records + record]; }
fn set_gradient(wire: u32, x: f32) { work[(params.width + wire) * params.records + record] = x; }
fn add(wire: u32, x: f32) { set_gradient(wire, g(gradient(wire) + x)); }

fn term_wire(term: u32) -> u32 { return tape[params.terms_at + 2u * term]; }
fn coefficient(term: u32) -> f32 { return bitcast<f32>(tape[params.terms_at + 2u * term + 1u]); }

fn sum(start: u32, count: u32, bias: u32) -> f32 {
    var x = bitcast<f32>(bias);
    for (var term = start; term < start + count; term++) {
        x = g(x + g(coefficient(term) * value(term_wire(term))));
    }
    return x;
}

fn relu(x: f32) -> f32 { return select(0.0, x, x > 0.0); }

// An index read's candidate position, or -1 where it reads +0.
fn position(op: u32, index: f32) -> i32 {
    let at = floor(index);
    let count = tape[op + 13u];
    if (at < 0.0) {
        if ((tape[op + 14u] & 8u) != 0u) { return -1; }
        return 0;
    }
    if (at > f32(count) - 1.0) {
        if ((tape[op + 14u] & 16u) != 0u) { return -1; }
        return i32(count) - 1;
    }
    return i32(at);
}

fn candidate(op: u32, at: i32) -> u32 { return tape[params.candidates_at + tape[op + 12u] + u32(at)]; }

fn read(op: u32, index: f32) -> f32 {
    let at = position(op, index);
    if (at < 0) { return 0.0; }
    return value(candidate(op, at));
}

fn forward(op: u32) -> f32 {
    let kind = tape[op];
    if (kind == 0u) { return sum(tape[op + 2u], tape[op + 3u], tape[op + 6u]); }
    if (kind == 1u) {
        return g(sum(tape[op + 2u], tape[op + 3u], tape[op + 6u]) * relu(sum(tape[op + 4u], tape[op + 5u], tape[op + 7u])));
    }
    if (kind == 2u) { return select(0.0, 1.0, sum(tape[op + 2u], tape[op + 3u], tape[op + 6u]) >= 0.5); }
    return read(op, sum(tape[op + 4u], tape[op + 5u], tape[op + 7u]));
}

fn backward(op: u32, grad: f32) {
    let rule = tape[op + 8u];
    if (rule == 0u) {
        for (var term = tape[op + 2u]; term < tape[op + 2u] + tape[op + 3u]; term++) {
            add(term_wire(term), g(grad * coefficient(term)));
        }
    } else if (rule == 1u) {
        let left = sum(tape[op + 2u], tape[op + 3u], tape[op + 6u]);
        let gate = sum(tape[op + 4u], tape[op + 5u], tape[op + 7u]);
        let slope = g(grad * relu(gate));
        for (var term = tape[op + 2u]; term < tape[op + 2u] + tape[op + 3u]; term++) {
            add(term_wire(term), g(slope * coefficient(term)));
        }
        if (gate > 0.0) {
            let scaled = g(grad * left);
            for (var term = tape[op + 4u]; term < tape[op + 4u] + tape[op + 5u]; term++) {
                add(term_wire(term), g(scaled * coefficient(term)));
            }
        }
    } else if (rule == 2u) {
        add(tape[op + 9u], g(grad * value(tape[op + 10u])));
        add(tape[op + 10u], g(grad * value(tape[op + 9u])));
    } else if (rule == 3u) {
        let gate = value(tape[op + 9u]);
        add(tape[op + 10u], g(grad * g(1.0 - gate)));
        add(tape[op + 11u], g(grad * gate));
        add(tape[op + 9u], g(grad * g(value(tape[op + 11u]) - value(tape[op + 10u]))));
    } else if (rule == 4u) {
        let width = bitcast<f32>(tape[op + 12u]);
        var difference = bitcast<f32>(tape[op + 13u]);
        for (var term = tape[op + 9u]; term < tape[op + 9u] + tape[op + 10u]; term++) {
            difference = difference + coefficient(term) * value(term_wire(term));
        }
        let sigmoid = 1.0 / (1.0 + exp(-difference / width));
        var slope = sigmoid * (1.0 - sigmoid) / width;
        if ((tape[op + 14u] & 4u) != 0u) { slope = 4.0 * slope * (1.0 - 2.0 * sigmoid); }
        for (var term = tape[op + 9u]; term < tape[op + 9u] + tape[op + 10u]; term++) {
            add(term_wire(term), g(grad * g(coefficient(term) * slope)));
        }
    } else if (rule == 5u) {
        let index = sum(tape[op + 4u], tape[op + 5u], tape[op + 7u]);
        let at = position(op, index);
        if (at >= 0) { add(candidate(op, at), grad); }
        if ((tape[op + 14u] & 32u) != 0u) {
            let below = floor(index);
            let scaled = g(grad * g(read(op, below) - read(op, below - 1.0)));
            for (var term = tape[op + 4u]; term < tape[op + 4u] + tape[op + 5u]; term++) {
                add(term_wire(term), g(scaled * coefficient(term)));
            }
        }
    }
}

@compute @workgroup_size(WORKGROUP)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    record = id.x + id.y * params.row;
    if (record >= params.records) { return; }
    let cells = params.source_size;
    let seed = inputs[params.seeds_at + record];
    if (seed == 0.0) {
        for (var cell = 0u; cell < cells; cell++) { results[record * cells + cell] = 0.0; }
        return;
    }
    for (var cell = 0u; cell < cells; cell++) { set_value(cell, inputs[record * cells + cell]); }
    for (var index = 0u; index < params.operation_count; index++) {
        let op = index * 16u;
        if ((tape[op + 14u] & 1u) != 0u) { set_value(tape[op + 1u], forward(op)); }
    }
    for (var wire = 0u; wire < params.width; wire++) { set_gradient(wire, 0.0); }
    set_gradient(params.output, seed);
    for (var index = params.operation_count; index > 0u; index--) {
        let op = (index - 1u) * 16u;
        if ((tape[op + 14u] & 2u) != 0u) {
            let grad = gradient(tape[op + 1u]);
            if (grad != 0.0) { backward(op, grad); }
        }
    }
    for (var cell = 0u; cell < cells; cell++) { results[record * cells + cell] = gradient(cell); }
}
"#;

/// The gradient of one output of a `#[handwired]` function's circuit with respect to every source cell, computed on
/// the GPU: what [`Compiled::gradients`] computes on the CPU.
pub struct GpuGradients<Args> {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tape_buffer: wgpu::Buffer,
    pub tape: GradientTape,
    source_ranges: Vec<Range>,
    /// At most this many records per dispatch; `None` fits each dispatch's buffers in the device's storage binding
    /// limit.
    pub dispatch_records: Option<usize>,
    marker: PhantomData<fn(Args)>,
}

impl<Args: Params> GpuGradients<Args> {
    /// The backward pass of output `output` (an index into the flattened outputs) under `options`. A rule that
    /// rejects an operation the gradient can reach is an error here, as [`GradientTape`] describes.
    pub fn new<R>(
        gpu: &Gpu,
        compiled: &Compiled<Args, R>,
        output: usize,
        options: BackpropOptions,
    ) -> Result<GpuGradients<Args>, GpuError> {
        assert!(
            output < compiled.outputs.len(),
            "output {output} is out of range: there are {}",
            compiled.outputs.len()
        );
        let circuit = &compiled.circuit;
        let tape = circuit.gradient_tape(compiled.outputs[output], options).map_err(GpuError::Gradient)?;
        let device = &gpu.device;
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("handwired gradients"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("handwired gradients"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("handwired gradients"),
            source: wgpu::ShaderSource::Wgsl(SHADER.replace("WORKGROUP", &wgsl::WORKGROUP_SIZE.to_string()).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("handwired gradients"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            return Err(GpuError::Shader(error.to_string()));
        }
        // Operations, then terms, then candidates; a storage binding can't be empty.
        let mut words = tape.operations.clone();
        words.extend(&tape.terms);
        words.extend(&tape.candidates);
        words.push(0);
        let tape_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired tape"),
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let source_ranges = (0..circuit.source_size()).map(|cell| circuit.range(circuit.source(cell))).collect();
        Ok(GpuGradients {
            pipeline,
            bind_group_layout,
            tape_buffer,
            tape,
            source_ranges,
            dispatch_records: None,
            marker: PhantomData,
        })
    }

    /// The gradients summed over `records` with per-record `weights` (all 1 when `None`), e.g. dLoss/dOutput for
    /// each record, as [`Compiled::gradients`] gives them. Cells are in [`Params::encode`] order.
    pub fn gradients(&self, gpu: &Gpu, records: &[Args], weights: Option<&[f32]>) -> Result<Vec<f64>, GpuError> {
        let sources = encode_checked(records, &self.source_ranges)?;
        self.sum(gpu, &sources, records.len(), weights)
    }

    /// [`gradients`](Self::gradients) of records encoded by [`Compiled::encode_records`].
    pub fn gradients_encoded(
        &self,
        gpu: &Gpu,
        records: &EncodedRecords,
        weights: Option<&[f32]>,
    ) -> Result<Vec<f64>, GpuError> {
        self.sum(gpu, records.sources(), records.len(), weights)
    }

    fn sum(&self, gpu: &Gpu, sources: &[f32], count: usize, weights: Option<&[f32]>) -> Result<Vec<f64>, GpuError> {
        let seeds = match weights {
            Some(weights) => {
                assert_eq!(weights.len(), count, "one weight per record");
                weights.to_vec()
            }
            None => vec![1.0; count],
        };
        let cells = self.tape.source_size;
        let mut total = vec![0f64; cells];
        for record in self.per_record(gpu, sources, &seeds)?.chunks(cells.max(1)) {
            for (sum, &gradient) in total.iter_mut().zip(record) {
                *sum += gradient as f64;
            }
        }
        Ok(total)
    }

    /// Each record's gradients, seeded with its `seeds` entry: `sources` holds one source table after another, and
    /// the result one row of source-cell gradients per record.
    pub fn per_record(&self, gpu: &Gpu, sources: &[f32], seeds: &[f32]) -> Result<Vec<f32>, GpuError> {
        let cells = self.tape.source_size;
        if cells == 0 || sources.len() != seeds.len() * cells {
            return Err(GpuError::Input(format!(
                "{} values isn't {} records of {cells} cells",
                sources.len(),
                seeds.len()
            )));
        }
        // Each dispatch's work buffer, two f32 per tape wire per record, must fit in one storage binding.
        let limits = gpu.device.limits();
        let binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        let fits = (binding / (self.tape.width as u64 * 2 * 4)).max(1) as usize;
        let per_dispatch = self.dispatch_records.map_or(fits, |records| records.clamp(1, fits));
        let mut gradients = Vec::with_capacity(sources.len());
        for (part, part_seeds) in sources.chunks(per_dispatch * cells).zip(seeds.chunks(per_dispatch)) {
            gradients.extend(self.dispatch(gpu, part, part_seeds)?);
        }
        Ok(gradients)
    }

    fn dispatch(&self, gpu: &Gpu, sources: &[f32], seeds: &[f32]) -> Result<Vec<f32>, GpuError> {
        let device = &gpu.device;
        let records = seeds.len();
        let cells = self.tape.source_size;
        let workgroups = (records as u32).div_ceil(wgsl::WORKGROUP_SIZE);
        let limit = device.limits().max_compute_workgroups_per_dimension;
        let (width, height) =
            if workgroups <= limit { (workgroups.max(1), 1) } else { (limit, workgroups.div_ceil(limit)) };
        let operations = self.tape.operations.len();
        let params: [u32; 12] = [
            f32::MIN.to_bits(),
            records as u32,
            width * wgsl::WORKGROUP_SIZE,
            (operations / OPERATION_WORDS) as u32,
            self.tape.width as u32,
            self.tape.output,
            cells as u32,
            operations as u32,
            (operations + self.tape.terms.len()) as u32,
            sources.len() as u32,
            0,
            0,
        ];
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired gradient params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut inputs = sources.to_vec();
        inputs.extend(seeds);
        let inputs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired gradient inputs"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let work = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("handwired gradient work"),
            size: (records * self.tape.width * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let result_size = (records * cells * 4) as u64;
        let results = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("handwired gradients"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("handwired gradient readback"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("handwired gradients"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.tape_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: inputs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: work.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: results.as_entire_binding() },
            ],
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("handwired gradients") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("handwired gradients"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.dispatch_workgroups(width, height, 1);
        }
        encoder.copy_buffer_to_buffer(&results, 0, &readback, 0, result_size);
        gpu.queue.submit([encoder.finish()]);
        crate::read_back(gpu, &readback)
    }
}
