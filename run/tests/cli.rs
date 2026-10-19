//! The `handwired-run` command on an exported program.
#![cfg(feature = "std")]

use std::io::Write;
use std::process::{Command, Stdio};

use handwired_rs::dsl::HwType;
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
    child.stdin.take().unwrap().write_all(b"200 21 0\n215, 20, 1\n9999 20 0\n200.5 21 0\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let expected = [heater(200, 21, false), heater(215, 20, true)]
        .map(|(duty, cooling)| format!("{duty} {}", cooling as u8))
        .join("\n");
    assert_eq!(stdout.trim_end(), expected);
    assert!(stderr.contains("line 3: input cell 0 is 9999"), "{stderr}");
    assert!(stderr.contains("line 4: input cell 0 is 200.5, not a multiple of 1"), "{stderr}");
    assert!(!output.status.success(), "a refused record fails the run");

    let info = Command::new(env!("CARGO_BIN_EXE_handwired-run")).arg("--info").arg(&path).output().unwrap();
    let info = String::from_utf8(info.stdout).unwrap();
    assert!(info.starts_with("heater\nparameters: (i16, i16, bool)\nresult: (u8, bool)"), "{info}");
    assert!(info.contains("3 input cells, 2 output cells"), "{info}");
    assert!(info.contains("input 1: 5..=30 in steps of 1"), "{info}");

    std::fs::remove_dir_all(&directory).unwrap();
}

/// A fieldless enum with gaps between its discriminants.
#[derive(Clone, Copy, Debug, PartialEq, HwType)]
enum Speed {
    Slow = 1,
    Fast = 4,
}

/// A fan's duty from an optional speed.
#[handwired]
fn fan(speed: Option<Speed>) -> u8 {
    match speed {
        Some(speed) => speed as u8 * 20,
        None => 0,
    }
}

#[test]
fn the_command_refuses_and_describes_enum_encodings() {
    let directory = std::env::temp_dir().join(format!("handwired-run-enum-test-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("fan.hwc");
    std::fs::write(&path, fan_circuit().export("fan")).unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_handwired-run"))
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"1 4\n0 0\n0 4\n1 2\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(stdout.trim_end(), "80\n0");
    assert!(stderr.contains("line 3: input cell 1 is 4, but it must be 0 unless input cell 0 is 1"), "{stderr}");
    assert!(stderr.contains("line 4: input cell 1 is 2, not one of the values its function declared"), "{stderr}");

    let info = Command::new(env!("CARGO_BIN_EXE_handwired-run")).arg("--info").arg(&path).output().unwrap();
    let info = String::from_utf8(info.stdout).unwrap();
    assert!(info.contains("input 1: 1..=4 in steps of 1, one of 1, 4, where input 0 is 1; else 0"), "{info}");

    std::fs::remove_dir_all(&directory).unwrap();
}
