//! Writes `heater.hwc`, the example program the Yocto layer (`meta-handwired`) installs: verified against the
//! function first, then exported. `cargo run -p handwired-rs-run --example export_heater -- OUT`, or `-- --check
//! FILE` to check that FILE is this function's export, whichever handwired-rs version wrote it.

use handwired_rs::handwired;

/// A heater's duty from a temperature in tenths of a degree and a setpoint in degrees, and whether to cool instead.
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

/// Whether `bytes` is this function's export. Only the description's last line, "built by handwired-rs <version>",
/// may differ from a fresh export, so a new release doesn't make the committed, signed example stale.
fn is_this_export(bytes: &[u8]) -> bool {
    let compiled = heater_circuit();
    let fresh_bytes = compiled.export("heater");
    let (Ok(given), Ok(fresh)) =
        (handwired_rs_run::Program::parse(bytes), handwired_rs_run::Program::parse(&fresh_bytes))
    else {
        return false;
    };
    let head = |description: &str| match description.rsplit_once('\n') {
        Some((head, last)) if last.starts_with("built by handwired-rs ") => Some(head.to_string()),
        _ => None,
    };
    head(given.description()).is_some()
        && head(given.description()) == head(fresh.description())
        && compiled.program.to_bytes(given.description()) == bytes
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--check") {
        let path = args.get(1).expect("--check needs a file");
        let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{path}: {error}"));
        assert!(is_this_export(&bytes), "{path} isn't what this example exports");
        println!("{path}: this function's export");
        return;
    }
    let path = args.first().cloned().unwrap_or_else(|| "heater.hwc".into());
    let compiled = heater_circuit();
    let report = compiled.verify(100_000).expect("the circuit matches the function");
    std::fs::write(&path, compiled.export("heater")).unwrap();
    println!("{path}: verified on {} inputs", report.checked);
}
