//! Writes `heater.hwc.proof.json`, the proof report the Yocto layer installs beside its example program: every
//! input the runner accepts for `heater.hwc` is run through handwired-rs-run and compared, bit for bit, with what
//! the Rust function returns. `cargo run --release -p handwired-rs-sign --features log --example heater_report --
//! PROGRAM [REPORT]` (REPORT defaults to PROGRAM.proof.json). The report has no dates, so the same program always
//! gives the same report.

use handwired_rs::dsl::HwType;
use handwired_rs::handwired;
use handwired_rs_run::Program;

/// The function run/examples/export_heater.rs exports; the report is refused unless PROGRAM is exactly its export.
#[handwired]
fn heater(#[range(-400, 1250)] tenths: i16, #[range(5, 30)] setpoint: i16, boost: bool) -> (u8, bool) {
    let error = setpoint * 10 - tenths;
    let duty = if error <= 0 {
        0
    } else if error > 100 {
        100
    } else {
        error as u8
    };
    (if boost { duty.saturating_add(20).min(100) } else { duty }, error < -50)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().cloned().unwrap_or_else(|| "heater.hwc".into());
    let out = args.get(1).cloned().unwrap_or_else(|| format!("{path}.proof.json"));
    let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
    assert!(bytes == heater_circuit().export("heater"), "{path} isn't what export_heater writes for this function");

    let program = Program::parse(&bytes).unwrap();
    let mut scratch = vec![0f32; program.scratch_len()];
    let mut outputs = vec![0f32; program.output_count()];
    let (mut checked, mut mismatches) = (0u64, 0u64);
    for tenths in -400..=1250i16 {
        for setpoint in 5..=30i16 {
            for boost in [false, true] {
                let (mut cells, mut native) = (Vec::new(), Vec::new());
                (tenths, setpoint, boost).encode(&mut cells);
                heater(tenths, setpoint, boost).encode(&mut native);
                program.run(&cells, &mut scratch, &mut outputs).unwrap();
                checked += 1;
                if outputs.iter().map(|value| value.to_bits()).ne(native.iter().map(|value| value.to_bits())) {
                    mismatches += 1;
                }
            }
        }
    }
    assert_eq!(checked, 1651 * 26 * 2);
    let result = if mismatches == 0 { "holds" } else { "violated" };
    let name = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().into_owned();
    let report = format!(
        r#"{{
  "format": "handwired-proof-report 1",
  "program": {{
    "file": "{name}",
    "sha256": "{sha256}",
    "bytes": {length},
    "format_version": 3
  }},
  "domain": "every input handwired-rs-run accepts for this file: tenths in -400..=1250, setpoint in 5..=30, boost 0 or 1 ({checked} inputs)",
  "claims": [
    {{
      "kind": "exhaustive",
      "statement": "on every input of the domain, the output cells handwired-rs-run computes equal, bit for bit, the encoding of what the Rust function heater returns",
      "method": "enumeration of the domain",
      "tool": "handwired-rs-sign example heater_report, with handwired-rs-run from the same repository",
      "inputs": {checked},
      "mismatches": {mismatches},
      "result": "{result}"
    }}
  ],
  "assumes": [
    "IEEE 754 binary32 addition and multiplication, round to nearest, subnormals not flushed, on the device",
    "the runner as built for the device computes as the runner that ran the check"
  ],
  "not_covered": [
    "the sign of a zero output",
    "whether the Rust function is the right function: the claim is equality with it"
  ],
  "generated_by": "handwired-rs-sign example heater_report"
}}
"#,
        sha256 = hex(&handwired_rs_sign::log::sha256(&bytes)),
        length = bytes.len(),
    );
    std::fs::write(&out, report).unwrap_or_else(|error| panic!("{out}: {error}"));
    println!("{out}: {checked} inputs, {mismatches} mismatches");
    assert_eq!(mismatches, 0);
}
