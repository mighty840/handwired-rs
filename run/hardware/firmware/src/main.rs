//! Runs the suite embedded from `suite.bin` (see `../suite`) and prints, per case, its records, mismatched output
//! bits and cycles per run, then `=== N passed, M failed ===`. On the NUCLEO-WL55JC1 (feature `wl55`) it prints over
//! LPUART1, the ST-LINK's virtual COM port, at 115200 baud and then idles; on QEMU (feature `qemu`) it prints over
//! semihosting and exits.

#![no_std]
#![no_main]

use core::fmt::Write;

use cortex_m_rt::entry;
use handwired_rs_run_suite::{CaseResult, run_suite};

static SUITE: &[u8] = include_bytes!("../suite.bin");

/// Cells for the largest program's scratch and outputs.
const SCRATCH: usize = 2048;
const OUTPUTS: usize = 32;

#[cfg(feature = "wl55")]
mod board {
    use stm32wl::stm32wl5x_cm4 as pac;

    /// LPUART1 on PA2 (TX) and PA3 (RX), AF8, from the 32 MHz TCXO, as the ferrite fleet firmware sets it up.
    pub struct Out(pac::LPUART);

    impl Out {
        pub fn take() -> Out {
            let dp = pac::Peripherals::take().unwrap();
            // 32 MHz needs one flash wait state.
            dp.FLASH.acr.modify(|_, w| unsafe { w.latency().bits(1) });
            while dp.FLASH.acr.read().latency().bits() != 1 {}
            dp.RCC.cr.modify(|_, w| w.hsebyppwr().set_bit().hseon().set_bit());
            while dp.RCC.cr.read().hserdy().bit_is_clear() {}
            dp.RCC.cfgr.modify(|_, w| w.sw().bits(0b10));
            while dp.RCC.cfgr.read().sws().bits() != 0b10 {}
            dp.RCC.ahb2enr.modify(|_, w| w.gpioaen().set_bit());
            dp.RCC.apb1enr2.modify(|_, w| w.lpuart1en().set_bit());
            dp.GPIOA.moder.modify(|_, w| w.moder2().bits(0b10).moder3().bits(0b10));
            dp.GPIOA.afrl.modify(|_, w| w.afrl2().af8().afrl3().af8());
            // BRR = 256 × 32 MHz / 115200.
            dp.LPUART.brr.write(|w| unsafe { w.bits(71111) });
            dp.LPUART.cr1.write(|w| w.te().set_bit().ue().set_bit());
            Out(dp.LPUART)
        }

        pub fn finish(&mut self) -> ! {
            while self.0.isr.read().tc().bit_is_clear() {}
            loop {
                cortex_m::asm::wfi();
            }
        }
    }

    impl core::fmt::Write for Out {
        fn write_str(&mut self, text: &str) -> core::fmt::Result {
            for &byte in text.as_bytes() {
                if byte == b'\n' {
                    self.byte(b'\r');
                }
                self.byte(byte);
            }
            Ok(())
        }
    }

    impl Out {
        fn byte(&mut self, byte: u8) {
            while self.0.isr.read().txfnf().bit_is_clear() {}
            self.0.tdr.write(|w| w.tdr().bits(byte as u16));
        }
    }
}

#[cfg(feature = "qemu")]
mod board {
    pub struct Out(cortex_m_semihosting::hio::HostStream);

    impl Out {
        pub fn take() -> Out {
            Out(cortex_m_semihosting::hio::hstdout().unwrap())
        }

        pub fn finish(&mut self) -> ! {
            cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
            loop {}
        }
    }

    impl core::fmt::Write for Out {
        fn write_str(&mut self, text: &str) -> core::fmt::Result {
            self.0.write_all(text.as_bytes()).map_err(|_| core::fmt::Error)
        }
    }
}

#[entry]
fn main() -> ! {
    #[cfg(not(no_cycle_counter))]
    {
        let mut core = cortex_m::Peripherals::take().unwrap();
        core.DCB.enable_trace();
        core.DWT.enable_cycle_counter();
    }
    let mut out = board::Out::take();
    let _ = writeln!(out, "\nhandwired-rs-run check: {} suite bytes", SUITE.len());

    let mut scratch = [0f32; SCRATCH];
    let mut outputs = [0f32; OUTPUTS];
    let (mut passed, mut failed) = (0, 0);
    let mut report = |case: &CaseResult| {
        let ok = case.error.is_none() && case.mismatches == 0;
        if ok {
            passed += 1;
        } else {
            failed += 1;
        }
        let (least, most, total) = case.cycles;
        let mean = if case.records == 0 { 0 } else { total / case.records as u64 };
        let (bytes, layers, values, cells) = case.program;
        let _ = writeln!(
            out,
            "{} {}: {} records, {} mismatched outputs, cycles per run min {} max {} mean {} ({} bytes, {} layers, {} values, {} scratch cells)",
            if ok { "PASS" } else { "FAIL" },
            case.name,
            case.records,
            case.mismatches,
            if least == u32::MAX { 0 } else { least },
            most,
            mean,
            bytes,
            layers,
            values,
            cells,
        );
        if let Some(error) = case.error {
            let _ = writeln!(out, "  error: {error}");
        }
        if let Some((record, cell, got, want)) = case.first_mismatch {
            let _ = writeln!(
                out,
                "  first mismatch: record {record}, output {cell}: {:e} ({got:#010x}) here, {:e} ({want:#010x}) on the host",
                f32::from_bits(got),
                f32::from_bits(want)
            );
        }
    };
    #[cfg(not(no_cycle_counter))]
    let cycles = cortex_m::peripheral::DWT::cycle_count;
    #[cfg(no_cycle_counter)]
    let cycles = || 0;
    let result = run_suite(SUITE, &mut scratch, &mut outputs, cycles, &mut report);
    if let Err(error) = result {
        failed += 1;
        let _ = writeln!(out, "FAIL suite: {error:?}");
    }
    let _ = writeln!(out, "=== {passed} passed, {failed} failed ===");
    out.finish()
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {
        cortex_m::asm::nop();
    }
}
