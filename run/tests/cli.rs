//! The `handwired-run` command on an exported program.
#![cfg(feature = "std")]

use std::io::Write;
use std::process::{Command, Stdio};

use handwired_rs::handwired;

/// A heater's duty from a temperature and a setpoint, as firmware logic.
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

#[test]
fn the_command_runs_records_and_describes_programs() {
    let directory = std::env::temp_dir().join(format!("handwired-run-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("heater.hwc");
    std::fs::write(&path, heater_circuit().export("heater")).unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_handwired-run"))
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"200 21 0\n215, 20, 1\n9999 20 0\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let expected = [heater(200, 21, false), heater(215, 20, true)]
        .map(|(duty, cooling)| format!("{duty} {}", cooling as u8))
        .join("\n");
    assert_eq!(stdout.trim_end(), expected);
    assert!(stderr.contains("line 3: input cell 0 is 9999"), "{stderr}");
    assert!(!output.status.success(), "a refused record fails the run");

    let info = Command::new(env!("CARGO_BIN_EXE_handwired-run")).arg("--info").arg(&path).output().unwrap();
    let info = String::from_utf8(info.stdout).unwrap();
    assert!(info.starts_with("heater\nparameters: (i16, i16, bool)\nresult: (u8, bool)"), "{info}");
    assert!(info.contains("3 input cells, 2 output cells"), "{info}");
    assert!(info.contains("input 1: 5..=30"), "{info}");

    std::fs::remove_dir_all(&directory).unwrap();
}
