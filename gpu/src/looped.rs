//! Loop programs: compiled steps that each run once per position of a document, e.g. once per character, and carry
//! values from one position to the next, as handwired's loop programs do (src/json-search/loop-program.ts).
//!
//! Written out once per position, a program grows with its documents, and so does its shader: a step of 400
//! operations over 256 characters is 100,000 operations, which take seconds to compile and run as stages. A loop
//! program compiles the step once, and its shader runs the step's statements inside a `for` loop, so the shader is
//! as long as its steps whatever the document's length.
//!
//! Each step's source cells are bound to the document's cell at the loop's position, a register (one value carried
//! from position to position, with an initial value), a field (one value per position, which one step writes and a
//! later one reads), or a shared cell (the same for every document, e.g. a table that a pattern writes). Each root
//! writes a register or a field. A program's outputs are registers' values after the last step.

use std::fmt::Write;

use handwired_rs::{TensorExecutor, TensorProgram};
use wgpu::util::DeviceExt;

use crate::{Gpu, GpuError, read_back, wgsl};

/// What a step's source cell reads.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    /// The document's cell at the loop's position.
    Position,
    Register(usize),
    Field(usize),
    Shared(usize),
}

/// What a step's root writes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Sink {
    Register(usize),
    Field(usize),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Direction {
    /// From position 0 up.
    Forward,
    /// From the last position down.
    Backward,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FieldKind {
    /// Any value: one f32 per position.
    Float,
    /// Only 0 or 1: one bit per position, 32 bit fields to a word on the GPU.
    Bit,
}

/// A compiled step and where its sources and roots go.
pub struct Step {
    pub program: TensorProgram,
    pub direction: Direction,
    /// One binding per source cell of `program`.
    pub sources: Vec<Binding>,
    /// One sink per root of `program`.
    pub sinks: Vec<Sink>,
}

pub struct LoopProgram {
    /// Positions per document: a document is that many cells.
    pub positions: usize,
    /// Each register's value before the first step.
    pub registers: Vec<f32>,
    pub fields: Vec<FieldKind>,
    pub shared_size: usize,
    /// Run one after another, each over every position.
    pub steps: Vec<Step>,
    /// The registers whose values after the last step are each document's outputs.
    pub outputs: Vec<usize>,
}

impl LoopProgram {
    fn check(&self) {
        for (index, step) in self.steps.iter().enumerate() {
            assert_eq!(step.sources.len(), step.program.source_size, "step {index}: one binding per source cell");
            assert_eq!(step.sinks.len(), step.program.roots.len(), "step {index}: one sink per root");
            for binding in &step.sources {
                match *binding {
                    Binding::Register(register) => {
                        assert!(register < self.registers.len(), "step {index}: no register {register}")
                    }
                    Binding::Field(field) => assert!(field < self.fields.len(), "step {index}: no field {field}"),
                    Binding::Shared(cell) => assert!(cell < self.shared_size, "step {index}: no shared cell {cell}"),
                    Binding::Position => {}
                }
            }
            for sink in &step.sinks {
                match *sink {
                    Sink::Register(register) => {
                        assert!(register < self.registers.len(), "step {index}: no register {register}")
                    }
                    Sink::Field(field) => assert!(field < self.fields.len(), "step {index}: no field {field}"),
                }
            }
        }
        for &register in &self.outputs {
            assert!(register < self.registers.len(), "no register {register} for an output");
        }
    }

    /// A runner on the CPU, with one [`TensorExecutor`] per step: the reference for [`GpuLoop`].
    pub fn cpu(&self) -> CpuLoop<'_> {
        self.check();
        CpuLoop {
            program: self,
            executors: self.steps.iter().map(|step| TensorExecutor::new(&step.program)).collect(),
            source: Vec::new(),
        }
    }
}

pub struct CpuLoop<'program> {
    program: &'program LoopProgram,
    executors: Vec<TensorExecutor<'program>>,
    source: Vec<f32>,
}

impl CpuLoop<'_> {
    /// One document's outputs.
    pub fn run(&mut self, document: &[f32], shared: &[f32]) -> Vec<f32> {
        let program = self.program;
        assert_eq!(document.len(), program.positions, "a document is one cell per position");
        assert_eq!(shared.len(), program.shared_size, "one value per shared cell");
        let mut registers = program.registers.clone();
        let mut fields = vec![vec![0f32; program.positions]; program.fields.len()];
        for (step, executor) in program.steps.iter().zip(&mut self.executors) {
            for index in 0..program.positions {
                let position = match step.direction {
                    Direction::Forward => index,
                    Direction::Backward => program.positions - 1 - index,
                };
                self.source.clear();
                self.source.extend(step.sources.iter().map(|binding| match *binding {
                    Binding::Position => document[position],
                    Binding::Register(register) => registers[register],
                    Binding::Field(field) => fields[field][position],
                    Binding::Shared(cell) => shared[cell],
                }));
                let roots = executor.run(&self.source);
                for (&value, sink) in roots.iter().zip(&step.sinks) {
                    match *sink {
                        Sink::Register(register) => registers[register] = value,
                        Sink::Field(field) => fields[field][position] = value,
                    }
                }
            }
        }
        program.outputs.iter().map(|&register| registers[register]).collect()
    }
}

/// A loop program on the GPU: one shader, one invocation per document.
pub struct GpuLoop {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tables: wgpu::Buffer,
    /// The shader's WGSL.
    pub shader: String,
    /// Statements across the steps, each run once per position.
    pub statements: usize,
    pub positions: usize,
    pub shared_size: usize,
    pub output_count: usize,
    /// At most this many documents per dispatch; `None` fits each dispatch's buffers in the device's storage binding
    /// limit.
    pub dispatch_documents: Option<usize>,
}

impl GpuLoop {
    pub fn new(gpu: &Gpu, program: &LoopProgram) -> Result<GpuLoop, GpuError> {
        program.check();
        let positions = program.positions;
        // Bit fields go 32 to a word: field `f` is bit `bit` of word array `word`.
        let mut bit_places = vec![None; program.fields.len()];
        let mut words = 0;
        let mut bits = 0;
        for (field, &kind) in program.fields.iter().enumerate() {
            if kind == FieldKind::Bit {
                if bits == 32 {
                    (words, bits) = (words + 1, 0);
                }
                bit_places[field] = Some((words, bits));
                bits += 1;
            }
        }
        let word_arrays = if bits > 0 { words + 1 } else { 0 };
        let read_field = |field: usize| match bit_places[field] {
            Some((word, bit)) => format!("f32((fw{word}[p] >> {bit}u) & 1u)"),
            None => format!("ff{field}[p]"),
        };

        let mut tables = Vec::new();
        let mut loops = String::new();
        let mut statements = 0;
        for step in &program.steps {
            let shared = step
                .sources
                .iter()
                .map(|binding| match *binding {
                    Binding::Shared(cell) => Some(cell as u32),
                    _ => None,
                })
                .collect();
            let code = wgsl::emit_step(&step.program, shared, tables);
            tables = code.tables;
            statements += code.statements;
            let header = match step.direction {
                Direction::Forward => format!("    for (var p: u32 = 0u; p < {positions}u; p++) {{\n"),
                Direction::Backward => format!(
                    "    for (var index: u32 = 0u; index < {positions}u; index++) {{\n        let p = {}u - index;\n",
                    positions - 1
                ),
            };
            loops.push_str(&header);
            for &position in &code.sources {
                let value = match step.sources[position] {
                    Binding::Position => "documents[base + p]".to_string(),
                    Binding::Register(register) => format!("r{register}"),
                    Binding::Field(field) => read_field(field),
                    Binding::Shared(cell) => format!("shared_cells[{cell}u]"),
                };
                writeln!(loops, "        let s{position}: f32 = {value};").unwrap();
            }
            for line in code.body.lines() {
                writeln!(loops, "    {line}").unwrap();
            }
            // Every source was read above, so a register can take its next value here.
            for (root, value) in code.roots.iter().enumerate() {
                writeln!(loops, "        let o{root}: f32 = {value};").unwrap();
            }
            for (root, sink) in step.sinks.iter().enumerate() {
                match *sink {
                    Sink::Register(register) => writeln!(loops, "        r{register} = o{root};").unwrap(),
                    Sink::Field(field) => match bit_places[field] {
                        Some((word, bit)) => {
                            writeln!(loops, "        fw{word}[p] = fw{word}[p] | (u32(o{root}) << {bit}u);").unwrap()
                        }
                        None => writeln!(loops, "        ff{field}[p] = o{root};").unwrap(),
                    },
                }
            }
            loops.push_str("    }\n");
        }

        let mut state = String::new();
        for (register, &initial) in program.registers.iter().enumerate() {
            writeln!(state, "    var r{register}: f32 = {};", wgsl::wgsl_float(initial)).unwrap();
        }
        for (field, &kind) in program.fields.iter().enumerate() {
            if kind == FieldKind::Float {
                writeln!(state, "    var ff{field}: array<f32, {positions}>;").unwrap();
            }
        }
        for word in 0..word_arrays {
            writeln!(state, "    var fw{word}: array<u32, {positions}>;").unwrap();
        }
        let output_count = program.outputs.len();
        let mut results = String::new();
        for (index, &register) in program.outputs.iter().enumerate() {
            writeln!(results, "    results[record * {output_count}u + {index}u] = r{register};").unwrap();
        }
        let shader = format!(
            "struct Params {{
    lowest: f32,
    records: u32,
    stride: u32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> documents: array<f32>;
@group(0) @binding(2) var<storage, read> shared_cells: array<f32>;
@group(0) @binding(3) var<storage, read> tables: array<f32>;
@group(0) @binding(4) var<storage, read_write> results: array<f32>;

// max(x, lowest) is x for every finite x, but the compiler can't prove it,
// so it keeps products and sums from being fused or regrouped across it.
fn g(x: f32) -> f32 {{
    return max(x, params.lowest);
}}

@compute @workgroup_size({workgroup})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let record = id.x + id.y * params.stride;
    if (record >= params.records) {{
        return;
    }}
    let base = record * {positions}u;
{state}{loops}{results}}}
",
            workgroup = wgsl::WORKGROUP_SIZE,
        );

        let device = &gpu.device;
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("handwired loop"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("handwired loop"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("handwired loop"),
            source: wgpu::ShaderSource::Wgsl(shader.as_str().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("handwired loop"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            return Err(GpuError::Shader(error.to_string()));
        }
        // A storage binding can't be empty.
        let table_values = if tables.is_empty() { vec![0.0f32] } else { tables };
        let tables = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired loop tables"),
            contents: bytemuck::cast_slice(&table_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(GpuLoop {
            pipeline,
            bind_group_layout,
            tables,
            shader,
            statements,
            positions,
            shared_size: program.shared_size,
            output_count,
            dispatch_documents: None,
        })
    }

    /// Upload documents once, to search them again and again: `documents` holds one document's cells after
    /// another, in dispatches that fit the device's storage binding limit.
    pub fn upload(&self, gpu: &Gpu, documents: &[f32]) -> Result<GpuDocuments, GpuError> {
        if !documents.len().is_multiple_of(self.positions) {
            return Err(GpuError::Input(format!(
                "{} cells isn't a whole number of {}-cell documents",
                documents.len(),
                self.positions
            )));
        }
        let limits = gpu.device.limits();
        let binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        let fits = (binding / (self.positions as u64 * 4)).max(1) as usize;
        let per_dispatch = self.dispatch_documents.map_or(fits, |documents| documents.clamp(1, fits));
        let parts = documents
            .chunks(per_dispatch * self.positions)
            .map(|part| {
                let buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("handwired loop documents"),
                    contents: bytemuck::cast_slice(part),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                (buffer, part.len() / self.positions)
            })
            .collect();
        Ok(GpuDocuments { parts, positions: self.positions })
    }

    /// Every document's outputs, one row per document, with these shared cells, e.g. a pattern's tables.
    pub fn run(&self, gpu: &Gpu, documents: &GpuDocuments, shared: &[f32]) -> Result<Vec<f32>, GpuError> {
        if documents.positions != self.positions || shared.len() != self.shared_size {
            return Err(GpuError::Input(format!(
                "documents of {} cells for a loop of {}, or {} shared values for {}",
                documents.positions,
                self.positions,
                shared.len(),
                self.shared_size
            )));
        }
        // A storage binding can't be empty.
        let shared_values = if shared.is_empty() { vec![0.0f32] } else { shared.to_vec() };
        let shared = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired loop shared cells"),
            contents: bytemuck::cast_slice(&shared_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut outputs = Vec::new();
        for (buffer, records) in &documents.parts {
            outputs.extend(self.dispatch(gpu, buffer, *records, &shared)?);
        }
        Ok(outputs)
    }

    fn dispatch(
        &self,
        gpu: &Gpu,
        documents: &wgpu::Buffer,
        records: usize,
        shared: &wgpu::Buffer,
    ) -> Result<Vec<f32>, GpuError> {
        let device = &gpu.device;
        let workgroups = (records as u32).div_ceil(wgsl::WORKGROUP_SIZE);
        let limit = device.limits().max_compute_workgroups_per_dimension;
        let (width, height) =
            if workgroups <= limit { (workgroups.max(1), 1) } else { (limit, workgroups.div_ceil(limit)) };
        let params: [u32; 4] = [f32::MIN.to_bits(), records as u32, width * wgsl::WORKGROUP_SIZE, 0];
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired loop params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let result_size = (records * self.output_count.max(1) * 4) as u64;
        let results = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("handwired loop results"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("handwired loop readback"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("handwired loop"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: documents.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: shared.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.tables.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: results.as_entire_binding() },
            ],
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("handwired loop") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("handwired loop"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(&self.pipeline);
            pass.dispatch_workgroups(width, height, 1);
        }
        encoder.copy_buffer_to_buffer(&results, 0, &readback, 0, result_size);
        gpu.queue.submit([encoder.finish()]);
        read_back(gpu, &readback)
    }
}

/// Documents on the GPU, uploaded once by [`GpuLoop::upload`].
pub struct GpuDocuments {
    parts: Vec<(wgpu::Buffer, usize)>,
    positions: usize,
}

impl GpuDocuments {
    pub fn len(&self) -> usize {
        self.parts.iter().map(|(_, records)| records).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
