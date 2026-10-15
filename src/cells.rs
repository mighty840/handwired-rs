//! Cell declarations for a program's source table.
//!
//! A simplified port of handwired's `cells.ts`. Each cell is declared once,
//! with what happens to it around an evaluation, its range, and optionally a
//! grid. The table gives the circuit its declarations, the initial table, a
//! validator, and the next table after an evaluation.
//!
//! Building has two phases, since a circuit needs its source size first:
//! declare every cell and call [`SourceTable::create_circuit`], then build,
//! pair each state cell with its next value through [`SourceTable::next`],
//! and call [`SourceTable::finish`].

use crate::circuit::Circuit;
use crate::numeric::{Range, is_power_of_two_grid};
use crate::wire::Wire;

/// What happens to a cell around an evaluation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Set once in the initial table. The program keeps it.
    Constant,
    /// The host writes it before an evaluation. The program keeps it.
    Input,
    /// An exact bit the host sets to 1 when an event arrives. The program resets it to 0.
    Event,
    /// The program computes its next value.
    State,
}

#[derive(Copy, Clone, Debug)]
pub struct CellSpec {
    pub policy: Policy,
    pub range: Range,
    pub initial: f64,
    pub exact_bit: bool,
    pub grid: Option<f64>,
}

impl CellSpec {
    fn new(policy: Policy, min: f64, max: f64, exact_bit: bool) -> CellSpec {
        assert!(min <= max, "a cell's range is inverted: {min}..{max}");
        CellSpec { policy, range: Range::new(min, max), initial: 0.0, exact_bit, grid: None }
    }

    pub fn initial(mut self, value: f64) -> CellSpec {
        self.initial = value;
        self
    }

    /// Every value is a multiple of `step`, a power of two, e.g. 1/64 for pointer coordinates.
    pub fn grid(mut self, step: f64) -> CellSpec {
        assert!(is_power_of_two_grid(step), "a grid step must be a power of two");
        self.grid = Some(step);
        self
    }
}

pub fn constant(value: f64, min: f64, max: f64) -> CellSpec {
    CellSpec::new(Policy::Constant, min, max, false).initial(value)
}
pub fn input(min: f64, max: f64) -> CellSpec {
    CellSpec::new(Policy::Input, min, max, false)
}
pub fn input_bit() -> CellSpec {
    CellSpec::new(Policy::Input, 0.0, 1.0, true)
}
pub fn event() -> CellSpec {
    CellSpec::new(Policy::Event, 0.0, 1.0, true)
}
pub fn state(min: f64, max: f64) -> CellSpec {
    CellSpec::new(Policy::State, min, max, false)
}
pub fn state_bit() -> CellSpec {
    CellSpec::new(Policy::State, 0.0, 1.0, true)
}

#[derive(Clone, Debug)]
struct DeclaredCell {
    name: String,
    spec: CellSpec,
    next: Option<Wire>,
}

#[derive(Clone, Debug, Default)]
pub struct SourceTable {
    cells: Vec<DeclaredCell>,
    circuit_created: bool,
}

impl SourceTable {
    pub fn new() -> SourceTable {
        SourceTable::default()
    }

    /// Declare one cell, returning its wire.
    pub fn declare(&mut self, name: impl Into<String>, spec: CellSpec) -> Wire {
        assert!(!self.circuit_created, "declare every cell before create_circuit");
        let name = name.into();
        assert!(self.cells.iter().all(|cell| cell.name != name), "cell {name} is declared twice");
        self.cells.push(DeclaredCell { name, spec, next: None });
        Wire::from_index(self.cells.len() - 1)
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// A circuit with every cell's range, grid and exact-bit declaration.
    pub fn create_circuit(&mut self) -> Circuit {
        self.circuit_created = true;
        let mut circuit = Circuit::new(self.cells.len());
        for (index, cell) in self.cells.iter().enumerate() {
            let wire = Wire::from_index(index);
            if cell.spec.exact_bit {
                circuit.assume_bit(wire);
                circuit.assume_grid(wire, 1.0);
            } else {
                circuit.assume_range(wire, cell.spec.range.min, cell.spec.range.max);
            }
            if let Some(grid) = cell.spec.grid {
                circuit.assume_grid(wire, grid);
            }
        }
        circuit
    }

    /// Pair a state cell with the wire that holds its next value.
    pub fn next(&mut self, cell: Wire, value: Wire) {
        let declared = &mut self.cells[cell.index()];
        assert_eq!(declared.spec.policy, Policy::State, "{} is not a state cell", declared.name);
        assert!(declared.next.is_none(), "{} already has a next value", declared.name);
        declared.next = Some(value);
    }

    /// Check that every state cell has a next value.
    pub fn finish(&self) {
        for cell in &self.cells {
            if cell.spec.policy == Policy::State {
                assert!(cell.next.is_some(), "state cell {} has no next value", cell.name);
            }
        }
    }

    /// The wires the host needs after an evaluation: every state cell's next value.
    pub fn next_wires(&self) -> Vec<Wire> {
        self.cells.iter().filter_map(|cell| cell.next).collect()
    }

    pub fn initial(&self) -> Vec<f32> {
        self.cells.iter().map(|cell| cell.spec.initial as f32).collect()
    }

    /// Check every cell's range, exact bit and grid.
    pub fn validate(&self, source: &[f32]) -> Result<(), String> {
        if source.len() != self.cells.len() {
            return Err(format!("source table has {} cells, expected {}", source.len(), self.cells.len()));
        }
        for (cell, &value) in self.cells.iter().zip(source) {
            let value = value as f64;
            let range = cell.spec.range;
            if !(value >= range.min && value <= range.max) {
                return Err(format!("{} is {value}, outside {}..{}", cell.name, range.min, range.max));
            }
            if cell.spec.exact_bit && value != 0.0 && value != 1.0 {
                return Err(format!("{} must be exactly 0 or 1, got {value}", cell.name));
            }
            if let Some(grid) = cell.spec.grid
                && (value / grid).fract() != 0.0
            {
                return Err(format!("{} must be a multiple of {grid}, got {value}", cell.name));
            }
        }
        Ok(())
    }

    /// The source table for the next evaluation: state cells take their next
    /// value, event cells reset to 0, and the rest keep their value.
    pub fn advance(&self, source: &[f32], values: &[f32]) -> Vec<f32> {
        self.cells
            .iter()
            .zip(source)
            .map(|(cell, &value)| match (cell.spec.policy, cell.next) {
                (Policy::State, Some(next)) => values[next.index()],
                (Policy::Event, _) => 0.0,
                _ => value,
            })
            .collect()
    }
}
