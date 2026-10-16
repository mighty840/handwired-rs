//! WGSL for a compiled tensor program: one compute invocation per record, one
//! `let` per operation, in the executor's order.
//!
//! What a GPU shader compiler is allowed to break, as handwired measured it on
//! Metal through wgpu and Dawn (lab/ui-exploration/codegen/wgsl.ts), and what
//! this writer does about it:
//! - It fuses `a + w * x` into a multiply-add and regroups sums, depending on
//!   the code around them. Every inexact product and every addition is wrapped
//!   in `g(x) = max(x, lowest)`, where `lowest` is the smallest finite f32,
//!   read from a uniform buffer: the compiler can't prove the max is the
//!   identity, so no fusion or regrouping reaches across it, and for every
//!   finite x it returns x.
//! - It folds the boolean atom `(score + 8388609) - 8388609` to `score`, so a
//!   boolean is written `select(0.0, 1.0, score >= 0.5)`, the same for every
//!   score within 0..1, the only range the builder accepts.
//! - It folds constant expressions in 64 bits, so every literal is one f32 with
//!   exact bits, and no expression combines two literals.
//!
//! Not guarded: a GPU may flush subnormal values to zero (RADV and lavapipe
//! do), which reads a subnormal operand as 0 and makes values closer than
//! 2^-126 compare equal, and WGSL lets an implementation assume no NaN or
//! infinity occurs. Results match the CPU reference where neither happens, up
//! to the sign of zero.

use std::fmt::Write;

use handwired_rs::{TensorProgram, Wire};

pub const WORKGROUP_SIZE: u32 = 64;

/// One f32 as a WGSL literal with the exact bits.
pub fn wgsl_float(value: f32) -> String {
    assert!(value.is_finite(), "a WGSL literal must be finite");
    if value == 0.0 {
        return if value.is_sign_negative() { "(-0.0f)".into() } else { "0.0f".into() };
    }
    if value.fract() == 0.0 && value.abs() < 16_777_216.0 {
        return if value < 0.0 { format!("({value:.1}f)") } else { format!("{value:.1}f") };
    }
    let bits = value.abs().to_bits();
    let exponent = (bits >> 23) & 0xff;
    let fraction = format!("{:06x}", (bits & 0x7f_ffff) << 1);
    let magnitude = if exponent == 0 {
        format!("0x0.{fraction}p-126f")
    } else {
        format!("0x1.{fraction}p{}f", exponent as i32 - 127)
    };
    if value < 0.0 { format!("(-{magnitude})") } else { magnitude }
}

/// The generated shader, plus the constant tables its index reads use.
pub struct Shader {
    pub code: String,
    /// Index-read tables whose candidates are all constants, concatenated.
    pub tables: Vec<f32>,
    pub source_size: usize,
    pub output_count: usize,
    pub operation_count: usize,
}

enum Value {
    Source(usize),
    Computed(usize),
    Constant(f32),
}

struct Writer<'a> {
    body: String,
    program: &'a TensorProgram,
    /// What each packed position holds, filled in as rows are written.
    values: Vec<Option<Value>>,
    tables: Vec<f32>,
    temporaries: usize,
}

impl Writer<'_> {
    fn name(&self, position: usize) -> String {
        match self.values[position].as_ref().expect("a row reads a value that isn't written yet") {
            Value::Source(position) => format!("s{position}"),
            Value::Computed(position) => format!("v{position}"),
            Value::Constant(value) => wgsl_float(*value),
        }
    }

    fn temporary(&mut self, expression: String) -> String {
        self.temporaries += 1;
        let name = format!("t{}", self.temporaries);
        writeln!(self.body, "    let {name}: f32 = {expression};").unwrap();
        name
    }

    /// The executor's left fold, `value = f(value + f(weight × input))` from
    /// the bias, with a guard around every inexact product and every addition.
    fn sum(&mut self, bias: f32, terms: &[(f32, usize)]) -> String {
        let mut accumulated: Option<String> =
            if bias == 0.0 && !terms.is_empty() { None } else { Some(wgsl_float(bias)) };
        for &(weight, position) in terms {
            let operand = self.name(position);
            let magnitude = weight.abs();
            // A power-of-two weight only changes the exponent, so its product is exact.
            let exact = magnitude.to_bits() & 0x7f_ffff == 0 && magnitude.is_normal();
            let product = if magnitude == 1.0 {
                operand
            } else if exact {
                format!("({} * {operand})", wgsl_float(magnitude))
            } else {
                self.temporary(format!("g({} * {operand})", wgsl_float(magnitude)))
            };
            accumulated = Some(match accumulated {
                // A bias of 0 is skipped: 0 + x is x, up to the sign of zero.
                None if weight < 0.0 => format!("(-{product})"),
                None => product,
                Some(previous) => {
                    let operator = if weight < 0.0 { '-' } else { '+' };
                    self.temporary(format!("g({previous} {operator} {product})"))
                }
            });
        }
        accumulated.unwrap()
    }
}

/// Write the shader for a compiled program.
pub fn emit(program: &TensorProgram) -> Shader {
    let source_size = program.source_size;
    let mut writer = Writer {
        body: String::new(),
        program,
        values: (0..program.value_count).map(|_| None).collect(),
        tables: Vec::new(),
        temporaries: 0,
    };
    for position in 0..source_size {
        writer.values[position] = Some(Value::Source(position));
        writeln!(writer.body, "    let s{position}: f32 = source[base + {position}u];").unwrap();
    }
    let mut available = source_size;
    let mut operation_count = 0;
    for layer in &writer.program.layers {
        let terms_of = |row: usize| -> Vec<(f32, usize)> {
            (layer.row_offsets[row] as usize..layer.row_offsets[row + 1] as usize)
                .map(|entry| (layer.weights[entry], layer.input_indices[layer.columns[entry] as usize] as usize))
                .collect()
        };
        for compute in &layer.gather_computes {
            // Σ position × bit and Σ bit: exact small integers for 0/1 bits, in any order.
            let bits: Vec<String> = compute.bit_positions.iter().map(|&bit| writer.name(bit as usize)).collect();
            let index: Vec<String> = bits
                .iter()
                .enumerate()
                .skip(1)
                .map(|(position, bit)| format!("{} * {bit}", wgsl_float(position as f32)))
                .collect();
            let index = if index.is_empty() { "0.0f".to_string() } else { index.join(" + ") };
            writeln!(writer.body, "    let gi{}: i32 = i32({index});", compute.slot).unwrap();
            writeln!(writer.body, "    let gm{}: f32 = {};", compute.slot, bits.join(" + ")).unwrap();
        }
        for row in 0..layer.output_size {
            let position = available + row;
            let bias = layer.bias[row];
            let terms = terms_of(row);
            let gate = layer.gate_rows.iter().position(|&gate_row| gate_row as usize == row);
            let boolean = layer.boolean_rows.iter().any(|&boolean_row| boolean_row as usize == row);
            let read = layer.index_reads.iter().find(|read| read.row as usize == row);
            operation_count += 1;
            if let Some(read) = layer.gather_reads.iter().find(|read| read.row as usize == row) {
                let count = read.candidate_positions.len();
                let names: Vec<String> =
                    read.candidate_positions.iter().map(|&candidate| writer.name(candidate as usize)).collect();
                writeln!(writer.body, "    var c{position} = array<f32, {count}>({});", names.join(", ")).unwrap();
                let mut contribution = format!("c{position}[clamp(gi{}, 0i, {}i)]", read.slot, count - 1);
                if read.masked {
                    // The mass is 0 or 1, so the product is exact and fusing it into the sum changes nothing.
                    contribution = format!("(gm{} * {contribution})", read.slot);
                }
                writeln!(writer.body, "    let v{position}: f32 = g({} + {contribution});", wgsl_float(bias)).unwrap();
                writer.values[position] = Some(Value::Computed(position));
                continue;
            }
            if let Some(read) = read {
                let index_terms = terms_of(read.index_row as usize);
                let index = writer.sum(layer.bias[read.index_row as usize], &index_terms);
                let count = read.candidate_positions.len();
                let slot = format!("p{position}");
                writeln!(writer.body, "    let {slot}: i32 = i32(floor({index}));").unwrap();
                let clamped = format!("clamp({slot}, 0i, {}i)", count - 1);
                let constants: Option<Vec<f32>> = read
                    .candidate_positions
                    .iter()
                    .map(|&candidate| match writer.values[candidate as usize] {
                        Some(Value::Constant(value)) => Some(value),
                        _ => None,
                    })
                    .collect();
                let mut value = match constants {
                    Some(constants) => {
                        let offset = writer.tables.len();
                        writer.tables.extend(constants);
                        format!("tables[{offset}u + u32({clamped})]")
                    }
                    None => {
                        let names: Vec<String> =
                            read.candidate_positions.iter().map(|&candidate| writer.name(candidate as usize)).collect();
                        writeln!(writer.body, "    var c{position} = array<f32, {count}>({});", names.join(", "))
                            .unwrap();
                        format!("c{position}[{clamped}]")
                    }
                };
                if read.below_zero {
                    value = format!("select({value}, 0.0f, {slot} < 0i)");
                }
                if read.above_zero {
                    value = format!("select({value}, 0.0f, {slot} > {}i)", count - 1);
                }
                writeln!(writer.body, "    let v{position}: f32 = {value};").unwrap();
                writer.values[position] = Some(Value::Computed(position));
                continue;
            }
            if let Some(gate) = gate {
                let gate_row = layer.output_size + gate;
                let gate_value = writer.sum(layer.bias[gate_row], &terms_of(gate_row));
                let relu = format!("max(0.0f, {gate_value})");
                let expression = if terms.is_empty() && bias == 1.0 {
                    // f(1 × max(0, gate)) is max(0, gate).
                    relu
                } else {
                    let left = writer.sum(bias, &terms);
                    format!("g({left} * {relu})")
                };
                writeln!(writer.body, "    let v{position}: f32 = {expression};").unwrap();
                writer.values[position] = Some(Value::Computed(position));
                continue;
            }
            if terms.is_empty() && !boolean {
                // A constant: readers use the literal.
                writer.values[position] = Some(Value::Constant(bias));
                continue;
            }
            let score = writer.sum(bias, &terms);
            let expression = if boolean { format!("select(0.0f, 1.0f, {score} >= 0.5f)") } else { score };
            writeln!(writer.body, "    let v{position}: f32 = {expression};").unwrap();
            writer.values[position] = Some(Value::Computed(position));
        }
        available += layer.output_size;
    }

    let mut outputs = String::new();
    for (index, root) in program.roots.iter().enumerate() {
        let position = packed_position(program, *root);
        writeln!(outputs, "    outputs[record * {}u + {index}u] = {};", program.roots.len(), writer.name(position))
            .unwrap();
    }
    let code = format!(
        "struct Params {{
    lowest: f32,
    records: u32,
    stride: u32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<storage, read> tables: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;

// max(x, lowest) is x for every finite x, but the compiler can't prove it,
// so it keeps products and sums from being fused or regrouped across it.
fn g(x: f32) -> f32 {{
    return max(x, params.lowest);
}}

@compute @workgroup_size({WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let record = id.x + id.y * params.stride;
    if (record >= params.records) {{
        return;
    }}
    let base = record * {source_size}u;
{body}{outputs}}}
",
        body = writer.body,
    );
    Shader { code, tables: writer.tables, source_size, output_count: program.roots.len(), operation_count }
}

fn packed_position(program: &TensorProgram, wire: Wire) -> usize {
    program.wire_order.iter().position(|&packed| packed == wire).expect("a root was removed from the program")
}
