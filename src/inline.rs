//! Inlining one circuit into another, ported from handwired's `inlineCircuit`.

use std::collections::HashMap;

use crate::circuit::{Circuit, Comparison, Macro, OpKind};
use crate::wire::{Lin, Term, Wire};

impl Circuit {
    /// Rebuild every operation of `source` in this circuit, with source cell `k` read from `source_wires[k]`, and
    /// return where each of `source`'s wires landed, indexed by its wire. The operations go through this circuit's
    /// builder, so an expression that already exists here is shared, e.g. a sum both circuits compute. Declared
    /// ranges, grids and exact bits of `source`'s cells land on the mapped wires, and rounding points, comparison
    /// bits and one-hot groups carry over. With `allow_existing_readers`, those declarations may narrow wires that
    /// operations here already read (see [`assume_range_allowing_readers`](Self::assume_range_allowing_readers)).
    pub fn inline_circuit(
        &mut self,
        source: &Circuit,
        source_wires: &[Wire],
        allow_existing_readers: bool,
    ) -> Vec<Wire> {
        assert_eq!(source_wires.len(), source.source_size(), "inlined source mapping has the wrong length");
        let mut map: Vec<Wire> = source_wires.to_vec();
        let declare_range = |circuit: &mut Circuit, wire: Wire, min: f64, max: f64| {
            if allow_existing_readers {
                circuit.assume_range_allowing_readers(wire, min, max);
            } else {
                circuit.assume_range(wire, min, max);
            }
        };
        let declare_grid = |circuit: &mut Circuit, wire: Wire, step: f64| {
            if allow_existing_readers {
                circuit.assume_grid_allowing_readers(wire, step);
            } else {
                circuit.assume_grid(wire, step);
            }
        };
        for (cell, &to) in source_wires.iter().enumerate() {
            let from = Wire::from_index(cell);
            let range = source.range(from);
            declare_range(self, to, range.min, range.max);
            if let Some(grid) = source.grid(from) {
                declare_grid(self, to, grid);
            }
            if source.is_exact_bit(from) {
                declare_range(self, to, 0.0, 1.0);
                self.mark_exact_bit(to);
            }
        }
        // Each distinct table is mapped once, so reads of one large table share one mapped array.
        let mut mapped_tables: HashMap<usize, Vec<Wire>> = HashMap::new();
        for (operation_index, operation) in source.operations().iter().enumerate() {
            let map_terms = |terms: &[Term]| -> Vec<Term> {
                terms.iter().map(|&(wire, coefficient)| (map[wire.index()], coefficient)).collect()
            };
            let wire = match (operation.macro_op, operation.kind) {
                (Some(Macro::Multiply { left, right }), _) => self.multiply(map[left.index()], map[right.index()]),
                (Some(Macro::Select { gate, off, on }), _) => {
                    self.select(map[gate.index()], map[off.index()], map[on.index()])
                }
                (None, OpKind::Reglu) => self.append(
                    OpKind::Reglu,
                    Lin::new(map_terms(&operation.left_terms), operation.left_bias),
                    Lin::new(map_terms(&operation.gate_terms), operation.gate_bias),
                    operation.preserve_rounding,
                    None,
                ),
                (None, OpKind::Index) => {
                    let read = operation.index_read.as_ref().unwrap();
                    let candidates = mapped_tables
                        .entry(read.table)
                        .or_insert_with(|| {
                            source.tables[read.table].candidates.iter().map(|c| map[c.index()]).collect()
                        })
                        .clone();
                    // The accepted index range carries over, e.g. one declared on a small affine index.
                    self.index_read_affine(
                        Lin::new(map_terms(&operation.gate_terms), operation.gate_bias),
                        &candidates,
                        read.below,
                        read.above,
                        Some(read.index_range),
                    )
                }
                // emit_compare's atom on grid operands reads its score terms directly.
                (None, OpKind::Boolean) if operation.preserve_rounding => self.append(
                    OpKind::Boolean,
                    Lin::new(map_terms(&operation.left_terms), operation.left_bias),
                    Lin::default(),
                    true,
                    None,
                ),
                (None, kind) => {
                    let form = Lin::new(map_terms(&operation.left_terms), operation.left_bias);
                    let value = if operation.preserve_rounding { self.round_affine(form) } else { self.affine(form) };
                    if kind == OpKind::Boolean { self.boolean(value) } else { value }
                }
            };
            let from = Wire::from_index(source.source_size() + operation_index);
            let range = source.range(from);
            declare_range(self, wire, range.min, range.max);
            if let Some(grid) = source.grid(from) {
                declare_grid(self, wire, grid);
            }
            // A marked operation can fold into a source cell here, which has no operation to record it on.
            if let Some(index) = wire.index().checked_sub(self.source_size()) {
                if let Some(kind) = operation.rounding_point {
                    self.record_rounding_point(wire, kind);
                }
                // The mapped operation computes the same values from the mapped operands, so it still holds 0 or 1.
                if operation.comparison_bit {
                    self.operations[index].comparison_bit = true;
                }
                if let Some(comparison) = &operation.comparison {
                    self.operations[index].comparison = Some(Comparison {
                        terms: comparison
                            .terms
                            .iter()
                            .map(|&(wire, coefficient)| (map[wire.index()], coefficient))
                            .collect(),
                        bias: comparison.bias,
                        equal: comparison.equal,
                    });
                }
            }
            map.push(wire);
        }
        // One-hot groups are about runtime values, which the mapped wires still carry.
        for group in source.one_hot_groups() {
            let members: Vec<Wire> = group.members.iter().map(|member| map[member.index()]).collect();
            if allow_existing_readers {
                self.assume_one_hot_allowing_readers(&members, group.allow_all_zero);
            } else {
                self.assume_one_hot(&members, group.allow_all_zero);
            }
        }
        map
    }
}
