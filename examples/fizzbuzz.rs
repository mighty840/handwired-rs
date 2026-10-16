//! FizzBuzz as an exact neural network: no training, right on every number.
//!
//! cargo run --example fizzbuzz

use handwired_rs::handwired;

/// 0 = the number itself, 1 = Fizz, 2 = Buzz, 3 = FizzBuzz.
#[handwired]
#[allow(clippy::manual_is_multiple_of)]
fn fizzbuzz(#[range(1, 100)] n: u8) -> u8 {
    if n % 15 == 0 {
        3
    } else if n % 5 == 0 {
        2
    } else if n % 3 == 0 {
        1
    } else {
        0
    }
}

fn main() {
    let net = fizzbuzz_circuit();
    println!(
        "fizzbuzz as a ReLU network: {} operations in {} layers",
        net.program.live_operation_count(),
        net.program.layers.len()
    );
    let mut line = Vec::new();
    for n in 1..=100u8 {
        let answer = net.eval(&(n,)).unwrap();
        assert_eq!(answer, fizzbuzz(n), "the network got {n} wrong");
        line.push(match answer {
            3 => "FizzBuzz".to_string(),
            2 => "Buzz".to_string(),
            1 => "Fizz".to_string(),
            _ => n.to_string(),
        });
    }
    println!("{}", line.join(" "));
    println!("100 of 100 correct, with no training.");
}
