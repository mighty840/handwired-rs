//! The README's `#[handwired]` example, verbatim.

use handwired_rs::handwired;

#[handwired]
fn toggle_step(
    on: bool,
    armed: bool,
    down: bool,
    up: bool,
    cancel: bool,
    #[range(-16384, 16384)]
    #[grid(1.0 / 64.0)]
    x: f32,
    #[range(0, 16384)] left: i32,
) -> (bool, bool) {
    let inside = x >= left as f32 && x < left as f32 + 52.0;
    let mut on = on;
    let mut armed = armed;
    if down {
        armed = inside;
    }
    if up && armed && inside {
        on = !on;
    }
    if up || cancel {
        armed = false;
    }
    (on, armed)
}

#[test]
fn readme_example() {
    let compiled = toggle_step_circuit(); // the circuit, compiled to tensor layers
    compiled.verify(20_000).unwrap(); // vs. the original function, on sampled inputs
    let (on, armed) = compiled.eval(&(false, false, true, false, false, 10.0, 0)).unwrap();
    assert_eq!((on, armed), (false, true));
    println!("live operations: {}", compiled.program.live_operation_count());
}
