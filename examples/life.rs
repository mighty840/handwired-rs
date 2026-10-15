//! Conway's Game of Life on a 24 × 24 torus, where each generation is one pass
//! through a circuit compiled from an ordinary `life_step` function. Every
//! generation is checked against the function, then printed as one line of
//! `.` and `#`, row after row.
//!
//! cargo run --release --example life > generations.txt

use handwired_rs::TensorExecutor;
use handwired_rs::dsl::Params;
use handwired_rs::handwired;

const N: usize = 24;

#[handwired]
fn life_step(grid: [[bool; 24]; 24]) -> [[bool; 24]; 24] {
    let mut next: [[bool; 24]; 24] = [[false; 24]; 24];
    for row in 0..24 {
        for column in 0..24 {
            let mut neighbors: u8 = 0;
            for dr in 0..3 {
                for dc in 0..3 {
                    if dr != 1 || dc != 1 {
                        // Wrap around the edges: the grid is a torus.
                        neighbors += grid[(row + dr + 23) % 24][(column + dc + 23) % 24] as u8;
                    }
                }
            }
            next[row][column] = neighbors == 3 || (grid[row][column] && neighbors == 2);
        }
    }
    next
}

fn main() {
    let compiled = life_step_circuit();
    let report = compiled.verify(200).unwrap();
    eprintln!(
        "life_step: {} live operations in {} layers; verified on {} random grids",
        compiled.program.live_operation_count(),
        compiled.program.layers.len(),
        report.checked
    );
    // Two gliders, a lightweight spaceship and an R-pentomino.
    let mut grid = [[false; N]; N];
    for (row, column) in [(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)] {
        grid[row][column] = true;
    }
    for (row, column) in [(14, 4), (15, 5), (16, 3), (16, 4), (16, 5)] {
        grid[row][column] = true;
    }
    for (row, column) in [(4, 14), (4, 17), (5, 18), (6, 14), (6, 18), (7, 15), (7, 16), (7, 17), (7, 18)] {
        grid[row][column] = true;
    }
    for (row, column) in [(16, 16), (16, 17), (17, 15), (17, 16), (18, 16)] {
        grid[row][column] = true;
    }
    let mut executor = TensorExecutor::new(&compiled.program);
    let mut source = Vec::new();
    for generation in 0..120 {
        println!(
            "{}",
            grid.iter().flat_map(|row| row.iter().map(|&alive| if alive { '#' } else { '.' })).collect::<String>()
        );
        source.clear();
        (grid,).encode(&mut source);
        let outputs = executor.run(&source);
        let next: [[bool; N]; N] =
            std::array::from_fn(|row| std::array::from_fn(|column| outputs[row * N + column] == 1.0));
        assert_eq!(next, life_step(grid), "generation {generation}: the circuit disagrees with life_step()");
        grid = next;
    }
}
