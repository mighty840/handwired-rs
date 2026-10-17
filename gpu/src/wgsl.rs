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
//!
//! Drivers take time that grows much faster than the shader to compile one
//! long straight-line function (Mesa on RADV and lavapipe: 0.1 s for 1,200
//! statements, 6 s for 4,900, 17 s for 8,400), and inline every function call,
//! so a program longer than [`WHOLE_STATEMENTS`] is split into stages:
//! separate shaders run one after another, each about [`STAGE_STATEMENTS`]
//! statements, that pass the values later stages read through a storage
//! buffer. An f32 round-trips through memory unchanged, so the split doesn't
//! change any result.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashSet};
use std::fmt::Write;

use handwired_rs::{TensorProgram, Wire};

pub const WORKGROUP_SIZE: u32 = 64;

/// Statements up to which [`emit`] writes a program as one shader: about 0.6 s to compile on RADV, cached after.
pub const WHOLE_STATEMENTS: usize = 2000;

/// Statements per stage for a longer program, past which [`emit`] starts a new stage at the next row. Smaller
/// stages compile faster in total; each stage also adds a pass over the records, so larger ones run faster.
pub const STAGE_STATEMENTS: usize = 250;

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

/// The generated shaders, plus the constant tables their index reads use.
pub struct Shader {
    /// One compute shader per stage, run in order on the same records.
    pub stages: Vec<String>,
    /// Index-read tables whose candidates are all constants, concatenated.
    pub tables: Vec<f32>,
    /// Cells per record that stages pass on to later stages.
    pub carried: usize,
    /// Statements across the stages, not counting loads and stores between them.
    pub statements: usize,
    pub source_size: usize,
    pub output_count: usize,
    pub operation_count: usize,
}

enum Value {
    Source(usize),
    Computed(usize),
    Constant(f32),
}

/// One shader's statements, and what it reads from the sources and from earlier stages.
#[derive(Default)]
struct Stage {
    body: String,
    statements: usize,
    sources: BTreeSet<usize>,
    loads: BTreeSet<usize>,
    /// Gather slots whose index this stage has computed.
    gathers: HashSet<u32>,
    outputs: String,
}

struct Writer {
    stages: Vec<Stage>,
    /// What each packed position holds, filled in as rows are written.
    values: Vec<Option<Value>>,
    /// The stage that computes each position, and the last stage that reads it.
    made_in: Vec<usize>,
    last_read: Vec<usize>,
    /// Each gather slot's bit positions, from the layer that first reads it.
    gather_bits: BTreeMap<u32, Vec<u32>>,
    tables: Vec<f32>,
    temporaries: usize,
    /// The row being written, counted across layers, and the statements written before each row.
    row: usize,
    statements: usize,
    row_statements: Vec<usize>,
    /// The row that computes each position, and the last row that reads it (the row count for an output).
    defined_row: Vec<usize>,
    read_row: Vec<usize>,
}

impl Writer {
    fn stage(&mut self) -> &mut Stage {
        self.stages.last_mut().unwrap()
    }

    fn line(&mut self, statement: String) {
        let stage = self.stage();
        stage.body.push_str("    ");
        stage.body.push_str(&statement);
        stage.body.push('\n');
        stage.statements += 1;
        self.statements += 1;
    }

    fn name(&mut self, position: usize) -> String {
        let current = self.stages.len() - 1;
        match *self.values[position].as_ref().expect("a row reads a value that isn't written yet") {
            Value::Source(position) => {
                self.stage().sources.insert(position);
                format!("s{position}")
            }
            Value::Computed(position) => {
                self.read_row[position] = self.row;
                if self.made_in[position] != current {
                    self.stage().loads.insert(position);
                    self.last_read[position] = current;
                }
                format!("v{position}")
            }
            Value::Constant(value) => wgsl_float(value),
        }
    }

    fn computed(&mut self, position: usize, expression: String) {
        self.line(format!("let v{position}: f32 = {expression};"));
        self.values[position] = Some(Value::Computed(position));
        self.made_in[position] = self.stages.len() - 1;
        self.defined_row[position] = self.row;
    }

    fn temporary(&mut self, expression: String) -> String {
        self.temporaries += 1;
        let name = format!("t{}", self.temporaries);
        self.line(format!("let {name}: f32 = {expression};"));
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

    /// A gather slot's index and mass, written once in each stage that reads the slot.
    fn gather(&mut self, slot: u32) {
        if self.stage().gathers.contains(&slot) {
            return;
        }
        self.stage().gathers.insert(slot);
        // Σ position × bit and Σ bit: exact small integers for 0/1 bits, in any order.
        let bit_positions = self.gather_bits[&slot].clone();
        let bits: Vec<String> = bit_positions.iter().map(|&bit| self.name(bit as usize)).collect();
        let index: Vec<String> = bits
            .iter()
            .enumerate()
            .skip(1)
            .map(|(position, bit)| format!("{} * {bit}", wgsl_float(position as f32)))
            .collect();
        let index = if index.is_empty() { "0.0f".to_string() } else { index.join(" + ") };
        self.line(format!("let gi{slot}: i32 = i32({index});"));
        self.line(format!("let gm{slot}: f32 = {};", bits.join(" + ")));
    }
}

/// Write the shaders for a compiled program: one shader up to [`WHOLE_STATEMENTS`] statements, else stages of about
/// [`STAGE_STATEMENTS`].
pub fn emit(program: &TensorProgram) -> Shader {
    emit_with(program, WHOLE_STATEMENTS, STAGE_STATEMENTS)
}

/// Write the shaders for a compiled program in stages of about `statements`
/// statements (`usize::MAX` for one shader).
pub fn emit_staged(program: &TensorProgram, statements: usize) -> Shader {
    emit_with(program, statements, statements)
}

fn emit_with(program: &TensorProgram, whole: usize, stage: usize) -> Shader {
    let (writer, operation_count) = write(program, &BTreeSet::new());
    if writer.statements <= whole {
        return assemble(program, writer, operation_count);
    }
    let cuts = cuts(&writer, stage);
    let (writer, operation_count) = write(program, &cuts);
    assemble(program, writer, operation_count)
}

/// The rows to start stages at: each stage ends within half a stage of `budget` statements, at the row where the
/// fewest values computed before it are read after it, since each of those is stored and loaded again.
fn cuts(writer: &Writer, budget: usize) -> BTreeSet<usize> {
    // Every stage holds at least one statement.
    let budget = budget.max(2);
    let rows = writer.row_statements.len();
    let mut change = vec![0i64; rows + 2];
    for (position, &defined) in writer.defined_row.iter().enumerate() {
        let read = writer.read_row[position];
        if defined != usize::MAX && read > defined {
            change[defined + 1] += 1;
            change[read + 1] -= 1;
        }
    }
    let live: Vec<i64> = change
        .iter()
        .scan(0, |running, &change| {
            *running += change;
            Some(*running)
        })
        .collect();
    let mut cuts = BTreeSet::new();
    let (mut start, mut first) = (0, 1);
    while writer.statements - start > budget.saturating_add(budget / 2) {
        let low = start + budget / 2;
        let high = start.saturating_add(budget).saturating_add(budget / 2);
        let candidates = (first..rows).filter(|&row| (low..=high).contains(&writer.row_statements[row]));
        // Values passed on per statement of the stage: fewer crossings, spread over longer stages.
        let cost = |row: usize| live[row] as f64 / (writer.row_statements[row] - start) as f64;
        let Some(row) = candidates.min_by(|&a, &b| cost(a).total_cmp(&cost(b)).then(b.cmp(&a))).or_else(|| {
            // A row longer than the window: cut at the first row past it.
            (first..rows).find(|&row| writer.row_statements[row] > high)
        }) else {
            break;
        };
        cuts.insert(row);
        (start, first) = (writer.row_statements[row], row + 1);
    }
    cuts
}

/// Write every row's statements, starting a new stage at each row in `cuts`.
fn write(program: &TensorProgram, cuts: &BTreeSet<usize>) -> (Writer, usize) {
    let source_size = program.source_size;
    let mut writer = Writer {
        stages: vec![Stage::default()],
        values: (0..program.value_count).map(|_| None).collect(),
        made_in: vec![usize::MAX; program.value_count],
        last_read: vec![0; program.value_count],
        gather_bits: BTreeMap::new(),
        tables: Vec::new(),
        temporaries: 0,
        row: 0,
        statements: 0,
        row_statements: Vec::new(),
        defined_row: vec![usize::MAX; program.value_count],
        read_row: vec![0; program.value_count],
    };
    for position in 0..source_size {
        writer.values[position] = Some(Value::Source(position));
    }
    let mut available = source_size;
    let mut operation_count = 0;
    for layer in &program.layers {
        let terms_of = |row: usize| -> Vec<(f32, usize)> {
            (layer.row_offsets[row] as usize..layer.row_offsets[row + 1] as usize)
                .map(|entry| (layer.weights[entry], layer.input_indices[layer.columns[entry] as usize] as usize))
                .collect()
        };
        for compute in &layer.gather_computes {
            writer.gather_bits.insert(compute.slot, compute.bit_positions.clone());
        }
        for row in 0..layer.output_size {
            writer.row = writer.row_statements.len();
            writer.row_statements.push(writer.statements);
            if cuts.contains(&writer.row) {
                writer.stages.push(Stage::default());
            }
            let position = available + row;
            let bias = layer.bias[row];
            let terms = terms_of(row);
            let gate = layer.gate_rows.iter().position(|&gate_row| gate_row as usize == row);
            let boolean = layer.boolean_rows.iter().any(|&boolean_row| boolean_row as usize == row);
            let read = layer.index_reads.iter().find(|read| read.row as usize == row);
            operation_count += 1;
            if let Some(read) = layer.gather_reads.iter().find(|read| read.row as usize == row) {
                writer.gather(read.slot);
                let count = read.candidate_positions.len();
                let names: Vec<String> =
                    read.candidate_positions.iter().map(|&candidate| writer.name(candidate as usize)).collect();
                writer.line(format!("var c{position} = array<f32, {count}>({});", names.join(", ")));
                let mut contribution = format!("c{position}[clamp(gi{}, 0i, {}i)]", read.slot, count - 1);
                if read.masked {
                    // The mass is 0 or 1, so the product is exact and fusing it into the sum changes nothing.
                    contribution = format!("(gm{} * {contribution})", read.slot);
                }
                writer.computed(position, format!("g({} + {contribution})", wgsl_float(bias)));
                continue;
            }
            if let Some(read) = read {
                let index_terms = terms_of(read.index_row as usize);
                let index = writer.sum(layer.bias[read.index_row as usize], &index_terms);
                let count = read.candidate_positions.len();
                let slot = format!("p{position}");
                writer.line(format!("let {slot}: i32 = i32(floor({index}));"));
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
                        writer.line(format!("var c{position} = array<f32, {count}>({});", names.join(", ")));
                        format!("c{position}[{clamped}]")
                    }
                };
                if read.below_zero {
                    value = format!("select({value}, 0.0f, {slot} < 0i)");
                }
                if read.above_zero {
                    value = format!("select({value}, 0.0f, {slot} > {}i)", count - 1);
                }
                writer.computed(position, value);
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
                writer.computed(position, expression);
                continue;
            }
            if terms.is_empty() && !boolean {
                // A constant: readers use the literal.
                writer.values[position] = Some(Value::Constant(bias));
                continue;
            }
            let score = writer.sum(bias, &terms);
            let expression = if boolean { format!("select(0.0f, 1.0f, {score} >= 0.5f)") } else { score };
            writer.computed(position, expression);
        }
        available += layer.output_size;
    }
    let rows = writer.row_statements.len();
    for root in &program.roots {
        let position = packed_position(program, *root);
        if writer.defined_row[position] != usize::MAX {
            writer.read_row[position] = rows;
        }
    }
    (writer, operation_count)
}

/// The stages' shaders, with loads and stores for the values they pass on.
fn assemble(program: &TensorProgram, mut writer: Writer, operation_count: usize) -> Shader {
    let source_size = program.source_size;
    // Each output is written by the stage that computes it; sources and constants by the last.
    let output_count = program.roots.len();
    for (index, root) in program.roots.iter().enumerate() {
        let position = packed_position(program, *root);
        let stage = match writer.values[position] {
            Some(Value::Computed(position)) => writer.made_in[position],
            _ => writer.stages.len() - 1,
        };
        let value = if stage == writer.stages.len() - 1 { writer.name(position) } else { format!("v{position}") };
        writeln!(writer.stages[stage].outputs, "    outputs[record * {output_count}u + {index}u] = {value};").unwrap();
    }

    // A carried value holds a slot from the end of the stage that computes it to the start of the last stage that
    // reads it. A stage loads before it stores, so a slot freed by a stage's reads can take that stage's results.
    let mut slots: BTreeMap<usize, usize> = BTreeMap::new();
    let mut busy: BinaryHeap<Reverse<(usize, usize)>> = BinaryHeap::new();
    let mut free: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    let mut carried = 0;
    for position in 0..program.value_count {
        let made_in = writer.made_in[position];
        if made_in == usize::MAX || writer.last_read[position] <= made_in {
            continue;
        }
        while let Some(&Reverse((released, slot))) = busy.peek() {
            if released > made_in {
                break;
            }
            busy.pop();
            free.push(Reverse(slot));
        }
        let slot = match free.pop() {
            Some(Reverse(slot)) => slot,
            None => {
                carried += 1;
                carried - 1
            }
        };
        slots.insert(position, slot);
        busy.push(Reverse((writer.last_read[position], slot)));
    }

    let total = writer.stages.iter().map(|stage| stage.statements).sum();
    let mut stores: Vec<String> = vec![String::new(); writer.stages.len()];
    for (&position, slot) in &slots {
        writeln!(stores[writer.made_in[position]], "    carry[{slot}u * params.records + record] = v{position};")
            .unwrap();
    }
    let stages = writer
        .stages
        .iter()
        .zip(&stores)
        .map(|(stage, stores)| {
            let mut prologue = String::new();
            for position in &stage.sources {
                writeln!(prologue, "    let s{position}: f32 = source[base + {position}u];").unwrap();
            }
            for position in &stage.loads {
                writeln!(prologue, "    let v{position}: f32 = carry[{}u * params.records + record];", slots[position])
                    .unwrap();
            }
            format!(
                "struct Params {{
    lowest: f32,
    records: u32,
    stride: u32,
}}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<storage, read> tables: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(4) var<storage, read_write> carry: array<f32>;

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
{prologue}{body}{stores}{outputs}}}
",
                body = stage.body,
                outputs = stage.outputs,
            )
        })
        .collect();
    Shader { stages, tables: writer.tables, carried, statements: total, source_size, output_count, operation_count }
}

fn packed_position(program: &TensorProgram, wire: Wire) -> usize {
    program.wire_order.iter().position(|&packed| packed == wire).expect("a root was removed from the program")
}
