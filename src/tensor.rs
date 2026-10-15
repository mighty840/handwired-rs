//! Compilation into layers of sparse matrix multiplications, and a CPU executor.
//!
//! Independent operations at the same dependency depth share a layer. A layer
//! gathers its earlier inputs, evaluates one sparse affine projection (one row
//! per output, then one gate row per ReGLU, then one index row per index read),
//! applies index reads, ReGLU gates and boolean rounding, and appends its
//! outputs. Operations that no requested root reads are removed.
//!
//! Not yet ported from handwired: lowering one-hot masked sums to gathers, and
//! recomputing depths after lowering.

use crate::Error;
use crate::circuit::{Circuit, OpKind, OutOfRange, resolve_index_position};
use crate::numeric::{evaluate_boolean, relu32};
use crate::wire::Wire;

#[derive(Clone, Debug)]
pub struct IndexReadRow {
    pub row: u32,
    pub index_row: u32,
    pub below_zero: bool,
    pub above_zero: bool,
    /// Packed positions of the candidates.
    pub candidate_positions: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct TensorLayer {
    /// Packed positions this layer reads, in column order.
    pub input_indices: Vec<u32>,
    pub output_size: usize,
    /// One bias per projection row, already rounded to f32.
    pub bias: Vec<f32>,
    /// CSR sparse matrix over the projection rows.
    pub row_offsets: Vec<u32>,
    pub columns: Vec<u32>,
    pub weights: Vec<f32>,
    /// Output rows whose value is multiplied by relu of gate row `output_size + k`.
    pub gate_rows: Vec<u32>,
    pub boolean_rows: Vec<u32>,
    pub index_reads: Vec<IndexReadRow>,
}

#[derive(Clone, Debug)]
pub struct TensorProgram {
    pub layers: Vec<TensorLayer>,
    pub roots: Vec<Wire>,
    pub source_size: usize,
    /// Source cells plus live operations.
    pub value_count: usize,
    /// `wire_order[packed]` is the wire held at that packed position.
    pub wire_order: Vec<Wire>,
    /// The circuit's width, for full-width results.
    pub width: usize,
}

impl TensorProgram {
    pub fn stored_weight_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.weights.len()).sum()
    }

    pub fn live_operation_count(&self) -> usize {
        self.value_count - self.source_size
    }
}

impl Circuit {
    /// Compile the operations that `roots` need, or every operation when
    /// `roots` is `None`.
    pub fn compile(&self, roots: Option<&[Wire]>) -> TensorProgram {
        let width = self.width();
        let source_size = self.source_size();
        let roots: Vec<Wire> = match roots {
            Some(roots) => roots.to_vec(),
            None => (source_size..width).map(Wire::from_index).collect(),
        };
        let live = self.find_live_operations(&roots);
        let mut operations_by_layer: Vec<Vec<usize>> = Vec::new();
        for (operation_index, operation) in self.operations.iter().enumerate() {
            if !live[operation_index] {
                continue;
            }
            let layer = operation.depth as usize - 1;
            if operations_by_layer.len() <= layer {
                operations_by_layer.resize(layer + 1, Vec::new());
            }
            operations_by_layer[layer].push(operation_index);
        }

        let live_count = live.iter().filter(|&&is_live| is_live).count();
        let value_count = source_size + live_count;
        let mut wire_order: Vec<Wire> = (0..source_size).map(Wire::from_index).collect();
        let mut packed_by_wire = vec![u32::MAX; width];
        for (wire, packed) in packed_by_wire.iter_mut().enumerate().take(source_size) {
            *packed = wire as u32;
        }
        // The column of each wire in the layer being compiled, reset after each layer.
        let mut column_by_wire = vec![u32::MAX; width];
        let mut available = source_size;
        let mut layers = Vec::new();
        for operation_indices in operations_by_layer.iter().filter(|indices| !indices.is_empty()) {
            let output_size = operation_indices.len();
            let mut input_wires: Vec<Wire> = Vec::new();
            for &operation_index in operation_indices {
                let operation = &self.operations[operation_index];
                for &(wire, _) in operation.left_terms.iter().chain(&operation.gate_terms) {
                    if column_by_wire[wire.index()] == u32::MAX {
                        column_by_wire[wire.index()] = 0;
                        input_wires.push(wire);
                    }
                }
            }
            input_wires.sort_by_key(|wire| packed_by_wire[wire.index()]);
            for (column, wire) in input_wires.iter().enumerate() {
                column_by_wire[wire.index()] = column as u32;
            }
            let gate_count =
                operation_indices.iter().filter(|&&index| self.operations[index].kind == OpKind::Reglu).count();
            let index_count =
                operation_indices.iter().filter(|&&index| self.operations[index].kind == OpKind::Index).count();
            let projection_size = output_size + gate_count + index_count;
            let mut bias = vec![0f32; projection_size];
            let mut projection_terms: Vec<&[(Wire, f64)]> = vec![&[]; projection_size];
            let mut gate_rows = Vec::new();
            let mut boolean_rows = Vec::new();
            let mut pending_index_reads = Vec::new();
            let (mut gate_row, mut index_row) = (0, 0);
            for (row, &operation_index) in operation_indices.iter().enumerate() {
                let operation = &self.operations[operation_index];
                let output_wire = source_size + operation_index;
                wire_order.push(Wire::from_index(output_wire));
                packed_by_wire[output_wire] = (available + row) as u32;
                bias[row] = operation.left_bias as f32;
                projection_terms[row] = &operation.left_terms;
                match operation.kind {
                    OpKind::Reglu => {
                        let projection_row = output_size + gate_row;
                        bias[projection_row] = operation.gate_bias as f32;
                        projection_terms[projection_row] = &operation.gate_terms;
                        gate_rows.push(row as u32);
                        gate_row += 1;
                    }
                    OpKind::Boolean => boolean_rows.push(row as u32),
                    OpKind::Index => {
                        let projection_row = output_size + gate_count + index_row;
                        bias[projection_row] = operation.gate_bias as f32;
                        projection_terms[projection_row] = &operation.gate_terms;
                        pending_index_reads.push((row, projection_row, operation.index_read.as_ref().unwrap()));
                        index_row += 1;
                    }
                    OpKind::Affine => {}
                }
            }
            let index_reads = pending_index_reads
                .into_iter()
                .map(|(row, index_row, read)| IndexReadRow {
                    row: row as u32,
                    index_row: index_row as u32,
                    below_zero: read.below == OutOfRange::Zero,
                    above_zero: read.above == OutOfRange::Zero,
                    candidate_positions: self.tables[read.table]
                        .candidates
                        .iter()
                        .map(|candidate| {
                            let position = packed_by_wire[candidate.index()];
                            assert!((position as usize) < available, "index read candidate is not available");
                            position
                        })
                        .collect(),
                })
                .collect();
            let mut row_offsets = Vec::with_capacity(projection_size + 1);
            let mut columns = Vec::new();
            let mut weights = Vec::new();
            row_offsets.push(0);
            for terms in &projection_terms {
                for &(wire, coefficient) in terms.iter() {
                    let column = column_by_wire[wire.index()];
                    assert!(column != u32::MAX, "tensor dependency is not available");
                    columns.push(column);
                    weights.push(coefficient as f32);
                }
                row_offsets.push(columns.len() as u32);
            }
            for wire in &input_wires {
                column_by_wire[wire.index()] = u32::MAX;
            }
            layers.push(TensorLayer {
                input_indices: input_wires.iter().map(|wire| packed_by_wire[wire.index()]).collect(),
                output_size,
                bias,
                row_offsets,
                columns,
                weights,
                gate_rows,
                boolean_rows,
                index_reads,
            });
            available += output_size;
        }
        TensorProgram { layers, roots, source_size, value_count, wire_order, width }
    }

    fn find_live_operations(&self, roots: &[Wire]) -> Vec<bool> {
        let source_size = self.source_size();
        let mut live = vec![false; self.operations.len()];
        let mut visited_tables = vec![false; self.tables.len()];
        let mut pending = roots.to_vec();
        while let Some(wire) = pending.pop() {
            self.assert_wire(wire);
            let Some(operation_index) = wire.index().checked_sub(source_size) else { continue };
            if live[operation_index] {
                continue;
            }
            live[operation_index] = true;
            let operation = &self.operations[operation_index];
            pending.extend(operation.left_terms.iter().chain(&operation.gate_terms).map(|&(wire, _)| wire));
            if let Some(read) = &operation.index_read
                && !visited_tables[read.table]
            {
                visited_tables[read.table] = true;
                pending.extend(&self.tables[read.table].candidates);
            }
        }
        live
    }

    /// Every wire's value through the compiled program, with 0 for removed wires.
    pub fn evaluate_tensor(&self, source: &[f32], program: &TensorProgram) -> Result<Vec<f32>, Error> {
        self.validate_source_ranges(source)?;
        let mut executor = TensorExecutor::new(program);
        executor.run(source);
        let mut values = vec![0f32; program.width];
        for (packed, wire) in program.wire_order.iter().enumerate() {
            values[wire.index()] = executor.packed_values[packed];
        }
        Ok(values)
    }

    /// Check that a compiled evaluation reproduces every live value of the
    /// reference evaluator, compared with `==` like handwired's `===`.
    pub fn validate_tensor_evaluation(
        &self,
        source: &[f32],
        program: &TensorProgram,
        tensor_values: &[f32],
    ) -> Result<(), String> {
        let expected = self.evaluate_unchecked(source);
        for wire in &program.wire_order {
            let (got, want) = (tensor_values[wire.index()], expected[wire.index()]);
            if got != want {
                return Err(format!("tensor wire {} is {got}, the reference evaluator gives {want}", wire.index()));
            }
        }
        Ok(())
    }
}

/// Reusable buffers for running one program many times.
pub struct TensorExecutor<'program> {
    program: &'program TensorProgram,
    /// After a run, `packed_values[k]` is the value of `program.wire_order[k]`.
    pub packed_values: Vec<f32>,
    input: Vec<f32>,
    projection: Vec<f32>,
    root_positions: Vec<usize>,
    outputs: Vec<f32>,
}

impl<'program> TensorExecutor<'program> {
    pub fn new(program: &'program TensorProgram) -> TensorExecutor<'program> {
        let max_input = program.layers.iter().map(|layer| layer.input_indices.len()).max().unwrap_or(0);
        let max_projection = program.layers.iter().map(|layer| layer.bias.len()).max().unwrap_or(0);
        let root_positions = program
            .roots
            .iter()
            .map(|root| program.wire_order.iter().position(|wire| wire == root).expect("root was removed"))
            .collect();
        TensorExecutor {
            program,
            packed_values: vec![0.0; program.value_count],
            input: vec![0.0; max_input],
            projection: vec![0.0; max_projection],
            root_positions,
            outputs: vec![0.0; program.roots.len()],
        }
    }

    /// Evaluate the program and return the roots' values. The source is not
    /// range-checked; use [`Circuit::validate_source_ranges`] for that.
    pub fn run(&mut self, source: &[f32]) -> &[f32] {
        let program = self.program;
        assert_eq!(source.len(), program.source_size, "source table has the wrong length");
        let values = &mut self.packed_values;
        values[..source.len()].copy_from_slice(source);
        let mut available = source.len();
        for layer in &program.layers {
            for (column, &packed) in layer.input_indices.iter().enumerate() {
                self.input[column] = values[packed as usize];
            }
            let projection = &mut self.projection[..layer.bias.len()];
            for (row, output) in projection.iter_mut().enumerate() {
                let mut value = layer.bias[row];
                for entry in layer.row_offsets[row] as usize..layer.row_offsets[row + 1] as usize {
                    value += layer.weights[entry] * self.input[layer.columns[entry] as usize];
                }
                *output = value;
            }
            for read in &layer.index_reads {
                let index = projection[read.index_row as usize] as f64;
                let below = if read.below_zero { OutOfRange::Zero } else { OutOfRange::Clamp };
                let above = if read.above_zero { OutOfRange::Zero } else { OutOfRange::Clamp };
                projection[read.row as usize] =
                    match resolve_index_position(index, read.candidate_positions.len(), below, above) {
                        -2 => f32::NAN,
                        -1 => 0.0,
                        position => values[read.candidate_positions[position as usize] as usize],
                    };
            }
            for (gate, &row) in layer.gate_rows.iter().enumerate() {
                let gate_value = relu32(projection[layer.output_size + gate]);
                projection[row as usize] *= gate_value;
            }
            for &row in &layer.boolean_rows {
                projection[row as usize] = evaluate_boolean(projection[row as usize]);
            }
            values[available..available + layer.output_size].copy_from_slice(&projection[..layer.output_size]);
            available += layer.output_size;
        }
        for (output, &position) in self.outputs.iter_mut().zip(&self.root_positions) {
            *output = values[position];
        }
        &self.outputs
    }
}
