//! The JSON search example's loop program against the reference, on random patterns and texts: on the GPU, as the
//! same loop on the CPU, and as the steps' own Rust functions.

#[path = "../examples/json_search/search.rs"]
mod search;

use handwired_rs_gpu::{Gpu, GpuLoop};
use search::{
    POSITIONS, Pattern, compile, document_cells, loop_program, native_count, reference_matches, shared_cells,
};

fn gpu() -> Option<Gpu> {
    match Gpu::new() {
        Ok(gpu) => {
            eprintln!("GPU: {} ({:?}, {:?})", gpu.info.name, gpu.info.device_type, gpu.info.backend);
            Some(gpu)
        }
        // CI sets HANDWIRED_REQUIRE_GPU, so a missing adapter fails instead of passing untested.
        Err(error) if std::env::var_os("HANDWIRED_REQUIRE_GPU").is_some() => panic!("{error}"),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// A random pattern over a few characters, quotes and a backslash among them, so that matches meet strings'
/// edges and escapes.
fn random_pattern(rng: &mut Rng, depth: u32) -> String {
    const ATOMS: [&str; 9] = ["a", "b", "c", r#"""#, r"\\", "[ab]", "[^a]", ".", r"\d"];
    match if depth == 0 { 0 } else { rng.below(5) } {
        0 | 1 => ATOMS[rng.below(ATOMS.len() as u64) as usize].to_string(),
        2 => format!("{}{}", random_pattern(rng, depth - 1), random_pattern(rng, depth - 1)),
        3 => format!("({}|{})", random_pattern(rng, depth - 1), random_pattern(rng, depth - 1)),
        _ => format!("({}){}", random_pattern(rng, depth - 1), ["?", "*", "+"][rng.below(3) as usize]),
    }
}

fn random_text(rng: &mut Rng) -> Vec<u8> {
    const ALPHABET: &[u8] = br#"abc"\ 1:{}"#;
    let length = match rng.below(8) {
        0 => POSITIONS as u64,
        1 => 0,
        _ => rng.below(40),
    };
    (0..length).map(|_| ALPHABET[rng.below(ALPHABET.len() as u64) as usize]).collect()
}

#[test]
fn the_steps_are_their_functions() {
    // Each step's circuit against its Rust function, on sampled inputs.
    assert_eq!(search::tokenize_step_circuit().verify(2_000).unwrap().skipped, 0);
    assert_eq!(search::reverse_step_circuit().verify(2_000).unwrap().skipped, 0);
    assert_eq!(search::forward_step_circuit().verify(2_000).unwrap().skipped, 0);
}

#[test]
fn hand_picked_searches() {
    let count = |pattern: &str, text: &str, everything: bool| {
        let pattern = compile(pattern).unwrap();
        reference_matches(&pattern, text.as_bytes(), everything).len() as u16
    };
    // Leftmost, then longest, then on from its end: `a|ab` finds `ab`, and `a*` never an empty match.
    assert_eq!(count("a|ab", "xabx", true), 1);
    assert_eq!(count("a*", "baab", true), 1);
    assert_eq!(count("a|abab", "xabab", true), 1);
    // In strings: quotes count, escapes and keys' outsides don't.
    assert_eq!(count("a", r#"a "a" a"#, false), 1);
    assert_eq!(count("bc", r#""b\nc" "bc""#, false), 1);
    assert_eq!(count(r#""a"#, r#"{"ab": "a"}"#, false), 2);
    for (pattern, text, everything) in [("a|ab", "xabx", true), ("bc", r#""b\nc" "bc""#, false), ("a*", "baab", true)] {
        let compiled = compile(pattern).unwrap();
        assert_eq!(native_count(&compiled, text.as_bytes(), everything), count(pattern, text, everything));
    }
    assert!(compile("(abcdefghi)").unwrap_err().message.contains("needs 10 states"));
    assert!(compile("a{2}").is_err());
}

#[test]
fn random_searches_agree_on_the_gpu_on_the_cpu_and_with_the_reference() {
    let program = loop_program();
    let mut cpu = program.cpu();
    let gpu = gpu();
    let searcher = gpu.as_ref().map(|gpu| GpuLoop::new(gpu, &program).unwrap());
    let mut rng = Rng(0x15_0B5E);
    let mut searched = 0;
    let mut matches = 0;
    while searched < 160 {
        let Ok(pattern) = compile(&random_pattern(&mut rng, 3)) else { continue };
        searched += 1;
        let texts: Vec<Vec<u8>> = (0..8).map(|_| random_text(&mut rng)).collect();
        let mut cells = Vec::new();
        for text in &texts {
            document_cells(text, &mut cells);
        }
        for everything in [true, false] {
            let shared = shared_cells(&pattern, everything);
            let expected: Vec<f32> =
                texts.iter().map(|text| reference_matches(&pattern, text, everything).len() as f32).collect();
            matches += expected.iter().sum::<f32>() as usize;
            for (index, text) in texts.iter().enumerate() {
                let label =
                    format!("{:?} on {:?}, everything {everything}", pattern.source, String::from_utf8_lossy(text));
                assert_eq!(native_count(&pattern, text, everything) as f32, expected[index], "steps: {label}");
                let document = &cells[index * POSITIONS..(index + 1) * POSITIONS];
                assert_eq!(cpu.run(document, &shared), vec![expected[index]], "CPU loop: {label}");
            }
            if let (Some(gpu), Some(searcher)) = (&gpu, &searcher) {
                let uploaded = searcher.upload(gpu, &cells).unwrap();
                assert_eq!(searcher.run(gpu, &uploaded, &shared).unwrap(), expected, "GPU: {:?}", pattern.source);
            }
        }
    }
    // The random texts do have matches to find.
    assert!(matches > 1_000, "{matches}");
}

#[test]
fn documents_split_across_dispatches_give_the_same_counts() {
    let Some(gpu) = gpu() else { return };
    let program = loop_program();
    let mut searcher = GpuLoop::new(&gpu, &program).unwrap();
    let pattern: Pattern = compile("(the|an?) [a-z]+").unwrap();
    let texts: Vec<String> = (0..1000).map(|index| search::document(7, index)).collect();
    let mut cells = Vec::new();
    for text in &texts {
        document_cells(text.as_bytes(), &mut cells);
    }
    let shared = shared_cells(&pattern, false);
    let whole = searcher.run(&gpu, &searcher.upload(&gpu, &cells).unwrap(), &shared).unwrap();
    searcher.dispatch_documents = Some(70);
    let split = searcher.run(&gpu, &searcher.upload(&gpu, &cells).unwrap(), &shared).unwrap();
    assert_eq!(whole, split);
    let expected: Vec<f32> =
        texts.iter().map(|text| reference_matches(&pattern, text.as_bytes(), false).len() as f32).collect();
    assert_eq!(whole, expected);
}
