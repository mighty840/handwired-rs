//! Run handwired circuits as WebGPU compute shaders.
//!
//! A program becomes one WGSL compute shader that evaluates one record per
//! invocation, so a batch of records, e.g. 100,000 documents to score, runs
//! in parallel. The shader reproduces the CPU reference evaluator's values
//! bit for bit, within the limits in [`wgsl`]'s documentation.
//!
//! ```no_run
//! # use handwired_rs::handwired;
//! # #[handwired] fn pick(a: u8) -> u8 { if a > 4 { 10 } else { 12 } }
//! let gpu = handwired_rs_gpu::Gpu::new().unwrap();
//! let batch = handwired_rs_gpu::GpuBatch::new(&gpu, &pick_circuit()).unwrap();
//! let results = batch.run(&gpu, &[(1,), (5,), (200,)]).unwrap();
//! assert_eq!(results, vec![12, 10, 10]);
//! ```

pub mod wgsl;

use std::fmt;
use std::marker::PhantomData;

use handwired_rs::dsl::{Compiled, HwType, Params};
use handwired_rs::{Circuit, Range, TensorProgram, Wire};
use wgpu::util::DeviceExt;

#[derive(Debug)]
pub enum GpuError {
    NoAdapter(String),
    Device(String),
    Shader(String),
    Input(String),
    Map(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoAdapter(message) => write!(formatter, "no GPU adapter: {message}"),
            GpuError::Device(message) => write!(formatter, "GPU device: {message}"),
            GpuError::Shader(message) => write!(formatter, "shader: {message}"),
            GpuError::Input(message) => write!(formatter, "input: {message}"),
            GpuError::Map(message) => write!(formatter, "reading results: {message}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// A GPU device and its queue.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
}

impl Gpu {
    /// The default adapter, preferring a discrete or integrated GPU. Set
    /// `WGPU_ADAPTER_NAME` to pick one by name, e.g. `llvmpipe` for Mesa's
    /// software Vulkan.
    pub fn new() -> Result<Gpu, GpuError> {
        pollster::block_on(Gpu::create())
    }

    async fn create() -> Result<Gpu, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = match std::env::var("WGPU_ADAPTER_NAME") {
            Ok(name) => {
                let name = name.to_lowercase();
                instance
                    .enumerate_adapters(wgpu::Backends::all())
                    .into_iter()
                    .find(|adapter| adapter.get_info().name.to_lowercase().contains(&name))
                    .ok_or_else(|| GpuError::NoAdapter(format!("no adapter named like {name:?}")))?
            }
            Err(_) => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .map_err(|error| GpuError::NoAdapter(error.to_string()))?,
        };
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            eprintln!("warning: using the software adapter {:?}; set WGPU_ADAPTER_NAME to choose another", info.name);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("handwired"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| GpuError::Device(error.to_string()))?;
        Ok(Gpu { device, queue, info })
    }
}

/// A compiled program on the GPU: the shader pipeline and its constant tables.
pub struct GpuProgram {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    tables: wgpu::Buffer,
    pub source_size: usize,
    pub output_count: usize,
    /// Operations the shader evaluates per record.
    pub operation_count: usize,
    pub wgsl: String,
}

impl GpuProgram {
    /// Compile the operations that `roots` need into a shader.
    pub fn new(gpu: &Gpu, circuit: &Circuit, roots: &[Wire]) -> Result<GpuProgram, GpuError> {
        let program = circuit.compile(Some(roots));
        GpuProgram::from_program(gpu, &program)
    }

    pub fn from_program(gpu: &Gpu, program: &TensorProgram) -> Result<GpuProgram, GpuError> {
        let shader = wgsl::emit(program);
        let device = &gpu.device;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("handwired program"),
            source: wgpu::ShaderSource::Wgsl(shader.code.as_str().into()),
        });
        // An explicit layout, since an automatic one drops bindings the shader doesn't read, e.g. empty tables.
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("handwired bindings"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("handwired program"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("handwired program"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            return Err(GpuError::Shader(error.to_string()));
        }
        // A storage binding can't be empty.
        let tables = if shader.tables.is_empty() { vec![0.0f32] } else { shader.tables.clone() };
        let tables = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired tables"),
            contents: bytemuck::cast_slice(&tables),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Ok(GpuProgram {
            pipeline,
            bind_group_layout,
            tables,
            source_size: shader.source_size,
            output_count: shader.output_count,
            operation_count: shader.operation_count,
            wgsl: shader.code,
        })
    }

    /// Evaluate a batch: `sources` holds one source table after another, and
    /// the result holds each record's outputs in the order of the roots.
    pub fn run(&self, gpu: &Gpu, sources: &[f32]) -> Result<Vec<f32>, GpuError> {
        if self.source_size == 0 || !sources.len().is_multiple_of(self.source_size) {
            return Err(GpuError::Input(format!(
                "{} values isn't a whole number of {}-cell records",
                sources.len(),
                self.source_size
            )));
        }
        let records = sources.len() / self.source_size;
        let device = &gpu.device;
        let workgroups = (records as u32).div_ceil(wgsl::WORKGROUP_SIZE);
        let limit = device.limits().max_compute_workgroups_per_dimension;
        let (width, height) =
            if workgroups <= limit { (workgroups.max(1), 1) } else { (limit, workgroups.div_ceil(limit)) };
        let params: [u32; 4] = [f32::MIN.to_bits(), records as u32, width * wgsl::WORKGROUP_SIZE, 0];
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("handwired sources"),
            contents: bytemuck::cast_slice(sources),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_size = (records * self.output_count.max(1) * 4) as u64;
        let outputs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("handwired outputs"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("handwired readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("handwired bindings"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: source.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tables.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: outputs.as_entire_binding() },
            ],
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("handwired run") });
        {
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("handwired"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width, height, 1);
        }
        encoder.copy_buffer_to_buffer(&outputs, 0, &readback, 0, output_size);
        gpu.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::Wait).map_err(|error| GpuError::Map(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| GpuError::Map(error.to_string()))?
            .map_err(|error| GpuError::Map(error.to_string()))?;
        let values = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        readback.unmap();
        Ok(values)
    }
}

/// A `#[handwired]` function's circuit on the GPU, run on typed arguments.
pub struct GpuBatch<Args, R> {
    pub program: GpuProgram,
    /// The declared range of each source cell.
    source_ranges: Vec<Range>,
    marker: PhantomData<fn(Args) -> R>,
}

impl<Args: Params, R: HwType> GpuBatch<Args, R> {
    pub fn new(gpu: &Gpu, compiled: &Compiled<Args, R>) -> Result<GpuBatch<Args, R>, GpuError> {
        let program = GpuProgram::from_program(gpu, &compiled.program)?;
        let circuit = &compiled.circuit;
        let source_ranges = (0..circuit.source_size()).map(|cell| circuit.range(circuit.source(cell))).collect();
        Ok(GpuBatch { program, source_ranges, marker: PhantomData })
    }

    /// The source tables for `records`, checked against the declared ranges.
    pub fn encode(&self, records: &[Args]) -> Result<Vec<f32>, GpuError> {
        let mut sources = Vec::with_capacity(records.len() * Args::CELLS);
        for (record_index, record) in records.iter().enumerate() {
            record.encode(&mut sources);
            let start = record_index * Args::CELLS;
            for (cell, (&value, range)) in sources[start..].iter().zip(&self.source_ranges).enumerate() {
                if !(value as f64 >= range.min && value as f64 <= range.max) {
                    return Err(GpuError::Input(format!(
                        "record {record_index}: source cell {cell} is {value}, outside its declared range {}..{}",
                        range.min, range.max
                    )));
                }
            }
        }
        Ok(sources)
    }

    pub fn run(&self, gpu: &Gpu, records: &[Args]) -> Result<Vec<R>, GpuError> {
        let outputs = self.program.run(gpu, &self.encode(records)?)?;
        Ok(outputs
            .chunks(self.program.output_count.max(1))
            .take(records.len())
            .map(|record| R::decode(&mut record.iter().copied()))
            .collect())
    }
}
