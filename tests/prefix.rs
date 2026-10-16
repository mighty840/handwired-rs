//! Prefix networks, ported from handwired's tests/runtime/prefix-networks.test.ts,
//! with the running-totals cost its docs/patterns.md states.

use handwired_rs::{BatchExecutor, Circuit, Lin, OutOfRange, Wire, brent_kung_prefix, sklansky_prefix};

type TableCombine<'a> = &'a mut dyn FnMut(&Vec<Wire>, &Vec<Wire>) -> Vec<Wire>;
type TableNetwork = fn(&[Vec<Wire>], TableCombine) -> Vec<Vec<Wire>>;
type Network = fn(&[Tracked], &mut dyn FnMut(&Tracked, &Tracked) -> Tracked) -> Vec<Tracked>;

/// Combines strings, so any wrong grouping or order shows in the text, with the longest chain behind each item.
#[derive(Clone)]
struct Tracked {
    text: String,
    depth: usize,
}

fn sklansky(items: &[Tracked], combine: &mut dyn FnMut(&Tracked, &Tracked) -> Tracked) -> Vec<Tracked> {
    sklansky_prefix(items, combine)
}

fn brent_kung(items: &[Tracked], combine: &mut dyn FnMut(&Tracked, &Tracked) -> Tracked) -> Vec<Tracked> {
    brent_kung_prefix(items, combine)
}

fn run_strings(network: Network, count: usize) -> (Vec<String>, usize, usize) {
    let items: Vec<Tracked> = (0..count).map(|index| Tracked { text: format!("{index},"), depth: 0 }).collect();
    let mut combines = 0;
    let prefixes = network(&items, &mut |earlier, later| {
        combines += 1;
        Tracked { text: earlier.text.clone() + &later.text, depth: earlier.depth.max(later.depth) + 1 }
    });
    let depth = prefixes.iter().map(|prefix| prefix.depth).max().unwrap_or(0);
    (prefixes.into_iter().map(|prefix| prefix.text).collect(), combines, depth)
}

#[test]
fn every_prefix_is_the_ordered_combination_of_its_items() {
    for count in 1..=70 {
        let expected: Vec<String> =
            (0..count).map(|index| (0..=index).map(|item| format!("{item},")).collect()).collect();
        assert_eq!(run_strings(sklansky, count).0, expected, "sklansky, {count} items");
        assert_eq!(run_strings(brent_kung, count).0, expected, "brent-kung, {count} items");
    }
}

#[test]
fn combines_and_longest_chains_follow_the_formulas() {
    for count in [4usize, 16, 64, 256] {
        let log = count.ilog2() as usize;
        let (_, combines, depth) = run_strings(sklansky, count);
        assert_eq!((combines, depth), (count / 2 * log, log));
        let (_, combines, depth) = run_strings(brent_kung, count);
        assert_eq!((combines, depth), (2 * count - 2 - log, 2 * log - 2));
    }
}

/// A state machine over queued events as a prefix scan: each event is a table of S index reads from state to next
/// state, and two tables combine as later[earlier[state]]. Returns the circuit, the state after each event, and the
/// number of combines.
fn state_machine_scan(
    network: TableNetwork,
    states: usize,
    events: usize,
    symbols: usize,
    initial_cell: bool,
) -> (Circuit, Vec<Wire>, usize) {
    let table_size = states * symbols;
    let mut circuit = Circuit::new(1 + table_size + events);
    let cell = |circuit: &mut Circuit, index: usize, maximum: f64| {
        let wire = circuit.source(index);
        circuit.assume_range(wire, 0.0, maximum);
        circuit.assume_grid(wire, 1.0);
        wire
    };
    let initial = cell(&mut circuit, 0, (states - 1) as f64);
    let next: Vec<Wire> = (0..table_size).map(|entry| cell(&mut circuit, 1 + entry, (states - 1) as f64)).collect();
    let event_symbols: Vec<Wire> =
        (0..events).map(|event| cell(&mut circuit, 1 + table_size + event, (symbols - 1) as f64)).collect();
    let tables: Vec<Vec<Wire>> = event_symbols
        .iter()
        .map(|&symbol| {
            (0..states)
                .map(|state| {
                    let index = Lin::new(vec![(symbol, 1.0)], (state * symbols) as f64);
                    circuit.index_read_affine(index, &next, OutOfRange::Clamp, OutOfRange::Clamp, None)
                })
                .collect()
        })
        .collect();
    let mut combines = 0;
    let prefixes = network(&tables, &mut |earlier, later| {
        combines += 1;
        earlier.iter().map(|&middle| circuit.index_read(middle, later, OutOfRange::Clamp)).collect()
    });
    let after: Vec<Wire> = prefixes
        .iter()
        .map(|prefix| if initial_cell { circuit.index_read(initial, prefix, OutOfRange::Clamp) } else { prefix[0] })
        .collect();
    (circuit, after, combines)
}

#[test]
fn a_state_machine_scan_gives_the_loops_states() {
    let (states, events, symbols) = (4, 32, 3);
    let mut seed: u32 = 7;
    let mut random = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        seed as f64 / 4_294_967_296.0
    };
    fn sklansky_tables(items: &[Vec<Wire>], combine: TableCombine) -> Vec<Vec<Wire>> {
        sklansky_prefix(items, combine)
    }
    fn brent_kung_tables(items: &[Vec<Wire>], combine: TableCombine) -> Vec<Vec<Wire>> {
        brent_kung_prefix(items, combine)
    }
    let networks: [TableNetwork; 2] = [sklansky_tables, brent_kung_tables];
    for network in networks {
        for initial_cell in [true, false] {
            let (circuit, after, combines) = state_machine_scan(network, states, events, symbols, initial_cell);
            let program = circuit.compile(Some(&after));
            // S × (k + combines) + k live operations from a state in a cell, or k + S × combines from state 0.
            let expected =
                if initial_cell { states * (events + combines) + events } else { events + states * combines };
            assert_eq!(program.live_operation_count(), expected);
            for _ in 0..10 {
                let mut source = vec![0.0f32; circuit.source_size()];
                source[0] = if initial_cell { (random() * states as f64).floor() as f32 } else { 0.0 };
                for entry in 0..states * symbols {
                    source[1 + entry] = (random() * states as f64).floor() as f32;
                }
                for event in 0..events {
                    source[1 + states * symbols + event] = (random() * symbols as f64).floor() as f32;
                }
                let values = circuit.evaluate_tensor(&source, &program).unwrap();
                let mut state = source[0] as usize;
                for (event, wire) in after.iter().enumerate() {
                    let symbol = source[1 + states * symbols + event] as usize;
                    state = source[1 + state * symbols + symbol] as usize;
                    assert_eq!(values[wire.index()], state as f32);
                }
            }
        }
    }
}

#[test]
fn running_totals_cost_what_handwireds_docs_state() {
    // docs/patterns.md, "Running totals": at 1,024 rows, Brent-Kung is 2,036 operations in 18 layers, against
    // 1,023 in 1,023 for the chain. Whole-pixel heights keep every partial sum exact, so both match the loop.
    let rows = 1024;
    let mut circuit = Circuit::new(rows);
    let heights: Vec<Wire> = (0..rows).map(|row| circuit.source(row)).collect();
    for &height in &heights {
        circuit.assume_range(height, 0.0, 300.0);
        circuit.assume_grid(height, 1.0);
    }
    let totals = brent_kung_prefix(&heights, |&earlier, &later| circuit.round_affine(earlier + later));
    let program = circuit.compile(Some(&totals[1..]));
    assert_eq!((program.live_operation_count(), program.layers.len()), (2036, 18));
    let mut chain = vec![heights[0]];
    for &height in &heights[1..] {
        let previous = *chain.last().unwrap();
        chain.push(circuit.round_affine(previous + height));
    }
    let chain_program = circuit.compile(Some(&chain[1..]));
    assert_eq!((chain_program.live_operation_count(), chain_program.layers.len()), (1023, 1023));

    let source: Vec<f32> = (0..rows).map(|row| (20 + row * 37 % 281) as f32).collect();
    let mut running = 0.0f32;
    let loop_totals: Vec<f32> = source
        .iter()
        .map(|&height| {
            running += height;
            running
        })
        .collect();
    assert_eq!(BatchExecutor::new(&program).run(&source), loop_totals[1..]);
    assert_eq!(BatchExecutor::new(&chain_program).run(&source), loop_totals[1..]);
}

#[handwired_rs::handwired]
fn row_bottoms(#[range(0, 300)] heights: [u16; 32]) -> [u16; 32] {
    handwired_rs::dsl::running_sums(heights)
}

/// For each item, how many of the items up to it are at least `threshold`: a running count of hits.
#[handwired_rs::handwired]
fn hits_so_far(values: [u8; 24], threshold: u8) -> [u16; 24] {
    let mut hits = [0u16; 24];
    for item in 0..24 {
        if values[item] >= threshold {
            hits[item] = 1;
        }
    }
    handwired_rs::dsl::running_sums(hits)
}

#[test]
fn handwired_functions_get_running_sums_in_logarithmic_depth() {
    let compiled = row_bottoms_circuit();
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    // 32 items: 2 × 32 - 2 - 5 = 57 additions in 2 × 5 - 2 = 8 layers, against 31 layers for a loop.
    assert_eq!((compiled.program.live_operation_count(), compiled.program.layers.len()), (57, 8));

    let compiled = hits_so_far_circuit();
    assert_eq!(compiled.verify(2_000).unwrap().skipped, 0);
    let records: Vec<([u8; 24], u8)> =
        (0..500u32).map(|n| (std::array::from_fn(|item| (n * 31 + item as u32 * 17) as u8), (n * 7) as u8)).collect();
    for (record, result) in records.iter().zip(compiled.run_batch(&records).unwrap()) {
        assert_eq!(result, hits_so_far(record.0, record.1));
    }
}
