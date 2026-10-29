//! The `handwired-sign` command: keys, a signature, and the checks a build or a device runs.
#![cfg(feature = "std")]

use std::path::Path;
use std::process::{Command, Output};

fn handwired_sign(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_handwired-sign")).args(args).output().unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn the_command_makes_keys_signs_and_verifies() {
    let directory = std::env::temp_dir().join(format!("handwired-sign-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let path = |name: &str| directory.join(name);
    let arg = |text: &str| Path::new(text).to_path_buf();
    std::fs::write(path("pump.hwc"), b"HWCIRC\0\0not run here, only signed").unwrap();

    let made = handwired_sign(&[&arg("keygen"), &path("release.key"), &path("release.pub")]);
    assert!(made.status.success(), "{}", text(&made.stderr));
    assert_eq!(std::fs::read(path("release.key")).unwrap().len(), 64);
    assert_eq!(std::fs::read(path("release.pub")).unwrap().len(), 32);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(path("release.key")).unwrap().permissions().mode() & 0o777, 0o600);
    }
    // Never replaces a key.
    let again = handwired_sign(&[&arg("keygen"), &path("release.key"), &path("other.pub")]);
    assert_eq!(again.status.code(), Some(1));
    assert!(handwired_sign(&[&arg("keygen"), &path("other.key"), &path("other.pub")]).status.success());

    let signed = handwired_sign(&[
        &arg("sign"),
        &arg("--key"),
        &path("release.key"),
        &arg("--slot"),
        &arg("pump"),
        &arg("--sequence"),
        &arg("12"),
        &path("pump.hwc"),
    ]);
    assert!(signed.status.success(), "{}", text(&signed.stderr));
    let verified = handwired_sign(&[&arg("verify"), &arg("--key"), &path("release.pub"), &path("pump.hwc")]);
    assert!(verified.status.success(), "{}", text(&verified.stderr));
    assert!(text(&verified.stdout).contains("slot pump, sequence 12, signed by "));
    let shown = handwired_sign(&[&arg("show"), &path("pump.hwc.sig")]);
    assert!(text(&shown.stdout).starts_with("slot pump, sequence 12, key "));

    // Another key, or a changed program, fails with status 1.
    let untrusted = handwired_sign(&[&arg("verify"), &arg("--key"), &path("other.pub"), &path("pump.hwc")]);
    assert_eq!(untrusted.status.code(), Some(1));
    assert!(text(&untrusted.stderr).contains("isn't trusted"));
    let both = handwired_sign(&[
        &arg("verify"),
        &arg("--key"),
        &path("other.pub"),
        &arg("--key"),
        &path("release.pub"),
        &path("pump.hwc"),
    ]);
    assert!(both.status.success());
    std::fs::write(path("pump.hwc"), b"HWCIRC\0\0not run here, only signed!").unwrap();
    let changed = handwired_sign(&[&arg("verify"), &arg("--key"), &path("release.pub"), &path("pump.hwc")]);
    assert_eq!(changed.status.code(), Some(1));
    assert!(text(&changed.stderr).contains("doesn't match"));

    // Only programs are signed, and a wrong command line is status 2.
    std::fs::write(path("notes.txt"), b"hello").unwrap();
    let not_program = handwired_sign(&[
        &arg("sign"),
        &arg("--key"),
        &path("release.key"),
        &arg("--slot"),
        &arg("pump"),
        &arg("--sequence"),
        &arg("1"),
        &path("notes.txt"),
    ]);
    assert_eq!(not_program.status.code(), Some(1));
    assert_eq!(handwired_sign(&[&arg("sign"), &path("pump.hwc")]).status.code(), Some(2));
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn the_command_admits_programs_into_a_log() {
    let directory = std::env::temp_dir().join(format!("handwired-sign-log-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let path = |name: &str| directory.join(name);
    let arg = |text: &str| Path::new(text).to_path_buf();
    assert!(handwired_sign(&[&arg("keygen"), &path("release.key"), &path("release.pub")]).status.success());
    let sign_as = |name: &str, sequence: &str| {
        let signed = handwired_sign(&[
            &arg("sign"),
            &arg("--key"),
            &path("release.key"),
            &arg("--slot"),
            &arg("pump"),
            &arg("--sequence"),
            &arg(sequence),
            &path(name),
        ]);
        assert!(signed.status.success(), "{}", text(&signed.stderr));
    };
    std::fs::write(path("v1.hwc"), b"HWCIRC\0\0pump 1").unwrap();
    std::fs::write(path("v2.hwc"), b"HWCIRC\0\0pump 2").unwrap();
    sign_as("v1.hwc", "1");
    sign_as("v2.hwc", "2");

    let made = handwired_sign(&[&arg("log"), &arg("new"), &path("pump.log"), &arg("--id"), &arg(&"ab".repeat(32))]);
    assert!(made.status.success(), "{}", text(&made.stderr));
    let admit = |program: &str, time: &str| {
        handwired_sign(&[
            &arg("admit"),
            &arg("--log"),
            &path("pump.log"),
            &arg("--key"),
            &path("release.pub"),
            &arg("--slot"),
            &arg("pump"),
            &arg("--time"),
            &arg(time),
            &path(program),
        ])
    };
    let first = admit("v2.hwc", "100");
    assert!(first.status.success(), "{}", text(&first.stderr));
    assert!(text(&first.stdout).contains("installed in slot pump, sequence 2"));
    let older = admit("v1.hwc", "101");
    assert_eq!(older.status.code(), Some(1));
    assert!(text(&older.stderr).contains("older than the installed release"));
    std::fs::remove_file(path("v1.hwc.sig")).unwrap();
    assert_eq!(admit("v1.hwc", "102").status.code(), Some(1));

    let checked = handwired_sign(&[&arg("log"), &arg("check"), &path("pump.log")]);
    assert!(text(&checked.stdout).contains("3 entries, head "), "{}", text(&checked.stderr));
    let shown = text(&handwired_sign(&[&arg("log"), &arg("show"), &path("pump.log")]).stdout);
    let lines: Vec<&str> = shown.lines().collect();
    assert!(lines[0].starts_with("0: time 100, slot pump, sequence 2, program ") && lines[0].ends_with(": installed"));
    assert!(lines[1].ends_with("refused: older than the installed release"));
    assert!(lines[2].contains("sequence 0") && lines[2].ends_with("refused: the signature can't be read"));

    let mut bytes = std::fs::read(path("pump.log")).unwrap();
    bytes[44] ^= 1;
    std::fs::write(path("pump.log"), bytes).unwrap();
    let broken = handwired_sign(&[&arg("log"), &arg("check"), &path("pump.log")]);
    assert_eq!(broken.status.code(), Some(1));
    assert!(text(&broken.stderr).contains("entry 1 doesn't follow the one before"));
    assert_eq!(admit("v2.hwc", "103").status.code(), Some(1), "a broken log decides nothing");
    std::fs::remove_dir_all(&directory).unwrap();
}
