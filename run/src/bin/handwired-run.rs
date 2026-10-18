//! `handwired-run PROGRAM`: run an exported program on each line of standard input, a record of input cells
//! separated by spaces or commas, printing its output cells. `handwired-run --info PROGRAM` describes it.

use std::io::{self, BufRead, BufWriter, Write};
use std::process::ExitCode;

use handwired_rs_run::Program;

const USAGE: &str = "usage: handwired-run [--info] PROGRAM\n\
    Runs PROGRAM, exported by handwired-rs, on each line of standard input (input cells separated by spaces or\n\
    commas) and prints its output cells, one line per record. --info describes the program instead.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (info, path) = match args.as_slice() {
        [flag, path] if flag == "--info" => (true, path),
        [path] if !path.starts_with('-') => (false, path),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("handwired-run: {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let program = match Program::parse(&bytes) {
        Ok(program) => program,
        Err(error) => {
            eprintln!("handwired-run: {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    if info {
        describe(&program);
        return ExitCode::SUCCESS;
    }
    match run(&program) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("handwired-run: {error}");
            ExitCode::FAILURE
        }
    }
}

fn describe(program: &Program) {
    println!("{}", program.description());
    println!(
        "{} input cells, {} output cells, {} layers, {} values, {} scratch cells",
        program.input_count(),
        program.output_count(),
        program.layer_count(),
        program.value_count(),
        program.scratch_len()
    );
    for index in 0..program.input_count() {
        if let Some((low, high)) = program.input_range(index) {
            match program.input_grid(index) {
                Some(step) => println!("input {index}: {low}..={high} in steps of {step}"),
                None => println!("input {index}: {low}..={high}"),
            }
        }
    }
}

/// Runs every record; false if any was refused.
fn run(program: &Program) -> io::Result<bool> {
    let mut scratch = vec![0.0f32; program.scratch_len()];
    let mut outputs = vec![0.0f32; program.output_count()];
    let mut all_ran = true;
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    for (number, line) in io::stdin().lock().lines().enumerate() {
        let line = line?;
        let fields: Result<Vec<f32>, _> = line
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|field| !field.is_empty())
            .map(str::parse)
            .collect();
        let result = match fields {
            Ok(inputs) => program.run(&inputs, &mut scratch, &mut outputs).map_err(|error| error.to_string()),
            Err(error) => Err(format!("not a number: {error}")),
        };
        match result {
            Ok(()) => {
                let cells: Vec<String> = outputs.iter().map(f32::to_string).collect();
                writeln!(out, "{}", cells.join(" "))?;
            }
            Err(error) => {
                out.flush()?;
                eprintln!("handwired-run: line {}: {error}", number + 1);
                all_ran = false;
            }
        }
    }
    out.flush()?;
    Ok(all_ran)
}
