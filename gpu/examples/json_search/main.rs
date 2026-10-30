//! Search 100,000 JSON documents for regular expressions on the GPU, after handwired's JSON search demo: one fixed
//! loop program, compiled from three ordinary Rust functions, with the pattern as data, and every document's count
//! checked against a reference written from the definition of the search.
//!
//! cargo run --release -p handwired-rs-gpu --example json_search

mod search;

use std::time::{Duration, Instant};

use handwired_rs_gpu::{Gpu, GpuLoop};
use search::{
    POSITIONS, compile, document, document_cells, loop_program, native_count, reference_matches, shared_cells,
};

const DOCUMENTS: usize = 100_000;
const SEED: u64 = 0x150_5EA6C;

const SEARCHES: [(&str, bool); 6] = [
    ("error", true),
    (r"\d\d\d\d-\d\d", true),
    ("(the|an?) [a-z]+", false),
    ("[A-Z][a-z]+ [A-Z]", false),
    (r#"": true"#, true),
    ("re[a-z]+y", false),
];

fn main() {
    let texts: Vec<String> = (0..DOCUMENTS as u64).map(|index| document(SEED, index)).collect();
    let mut documents = Vec::with_capacity(DOCUMENTS * POSITIONS);
    for text in &texts {
        document_cells(text.as_bytes(), &mut documents);
    }
    let average = texts.iter().map(String::len).sum::<usize>() as f64 / DOCUMENTS as f64;
    println!("{DOCUMENTS} JSON documents of up to {POSITIONS} bytes, {average:.0} on average, e.g.\n  {}\n", texts[1]);

    let start = Instant::now();
    let program = loop_program();
    let sizes: Vec<String> =
        program.steps.iter().map(|step| format!("{} operations", step.program.live_operation_count())).collect();
    println!("steps: tokenizer, reverse and forward, {}; built in {:.1?}", sizes.join(", "), start.elapsed());

    let gpu = Gpu::new().expect("a GPU adapter");
    let start = Instant::now();
    let searcher = GpuLoop::new(&gpu, &program).expect("the loop's shader");
    println!(
        "GPU: {}; one shader of {} statements, run at each of {POSITIONS} positions, set up in {:.1?}",
        gpu.info.name,
        searcher.statements,
        start.elapsed()
    );

    let start = Instant::now();
    let uploaded = searcher.upload(&gpu, &documents).unwrap();
    println!("documents uploaded to the GPU once, in {:.1?}\n", start.elapsed());

    let mut cpu = program.cpu();
    for (source, everything) in SEARCHES {
        let pattern = compile(source).expect("an example pattern compiles");
        let shared = shared_cells(&pattern, everything);
        let scope = if everything { "everything" } else { "strings" };

        // The first run of a pattern also uploads it; the second is the steady state.
        searcher.run(&gpu, &uploaded, &shared).unwrap();
        let start = Instant::now();
        let counts = searcher.run(&gpu, &uploaded, &shared).unwrap();
        let gpu_time = start.elapsed();

        let start = Instant::now();
        let expected: Vec<usize> =
            texts.iter().map(|text| reference_matches(&pattern, text.as_bytes(), everything).len()).collect();
        let reference_time = start.elapsed();
        let mismatches =
            counts.iter().zip(&expected).filter(|(count, expected)| **count as usize != **expected).count();
        assert_eq!(mismatches, 0, "{source}: the GPU's counts differ from the reference's");

        // The same loop program on the CPU, and the steps' own Rust functions, on the first documents.
        let sample = 300;
        let start = Instant::now();
        for index in 0..sample {
            let cells = &documents[index * POSITIONS..(index + 1) * POSITIONS];
            assert_eq!(cpu.run(cells, &shared)[0], counts[index], "{source}: CPU loop, document {index}");
        }
        let cpu_time = start.elapsed();
        let start = Instant::now();
        for (index, text) in texts.iter().take(sample).enumerate() {
            assert_eq!(native_count(&pattern, text.as_bytes(), everything) as f32, counts[index], "{source}: steps");
        }
        let native_time = start.elapsed();

        let matching = counts.iter().filter(|&&count| count > 0.0).count();
        let total: f32 = counts.iter().sum();
        println!(
            "{source:<20} in {scope:<10} {} states, {} classes: {matching:>6} documents match, {total:>7} matches",
            pattern.states, pattern.classes
        );
        println!(
            "{:6}GPU {gpu_time:.1?}; per 100,000 documents: the reference {reference_time:.1?}, the steps' Rust functions {:.1?}, the CPU loop {:.1?}",
            "",
            per_hundred_thousand(native_time, sample),
            per_hundred_thousand(cpu_time, sample),
        );
    }

    // One document with its matches marked, from the reference.
    let pattern = compile("(the|an?) [a-z]+").unwrap();
    let text = texts.iter().find(|text| !reference_matches(&pattern, text.as_bytes(), false).is_empty()).unwrap();
    let mut marked = String::new();
    let mut last = 0;
    for (start, end) in reference_matches(&pattern, text.as_bytes(), false) {
        marked.push_str(&text[last..start]);
        marked.push_str(&format!("[{}]", &text[start..end]));
        last = end;
    }
    marked.push_str(&text[last..]);
    println!("\n(the|an?) [a-z]+ in strings, marked:\n  {marked}");
}

fn per_hundred_thousand(time: Duration, documents: usize) -> Duration {
    time * (DOCUMENTS / documents) as u32
}
