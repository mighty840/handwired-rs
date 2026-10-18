//! Picks the memory layout for the board, and checks that the suite to embed exists.
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let memory = if std::env::var("CARGO_FEATURE_WL55").is_ok() {
        // STM32WL55JC: 256 KB flash, 64 KB SRAM.
        "MEMORY {\n  FLASH : ORIGIN = 0x08000000, LENGTH = 256K\n  RAM : ORIGIN = 0x20000000, LENGTH = 64K\n}\n"
    } else if std::env::var("TARGET").unwrap().starts_with("thumbv6m") {
        // QEMU's microbit, an nRF51 (Cortex-M0): 256 KB flash, 16 KB SRAM.
        "MEMORY {\n  FLASH : ORIGIN = 0x00000000, LENGTH = 256K\n  RAM : ORIGIN = 0x20000000, LENGTH = 16K\n}\n"
    } else {
        // QEMU's lm3s6965evb (Cortex-M3): 256 KB flash, 64 KB SRAM.
        "MEMORY {\n  FLASH : ORIGIN = 0x00000000, LENGTH = 256K\n  RAM : ORIGIN = 0x20000000, LENGTH = 64K\n}\n"
    };
    std::fs::write(out.join("memory.x"), memory).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg=-Tlink.x");
    println!("cargo:rerun-if-changed=build.rs");
    // Cortex-M0 has no cycle counter.
    println!("cargo::rustc-check-cfg=cfg(no_cycle_counter)");
    if std::env::var("TARGET").unwrap().starts_with("thumbv6m") {
        println!("cargo:rustc-cfg=no_cycle_counter");
    }
    println!("cargo:rerun-if-changed=suite.bin");
    assert!(
        std::path::Path::new("suite.bin").exists(),
        "generate suite.bin first: cargo run -p handwired-rs-run --example hardware_suite -- run/hardware/firmware/suite.bin 60"
    );
}
