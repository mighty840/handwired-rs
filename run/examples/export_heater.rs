//! Writes `heater.hwc`, the example program the Yocto layer (`meta-handwired`) installs: verified against the
//! function first, then exported. `cargo run -p handwired-rs-run --example export_heater -- OUT`.

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

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "heater.hwc".into());
    let compiled = heater_circuit();
    let report = compiled.verify(100_000).expect("the circuit matches the function");
    std::fs::write(&path, compiled.export("heater")).unwrap();
    println!("{path}: verified on {} inputs", report.checked);
}
