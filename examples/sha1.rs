//! SHA-1's block function as an exact circuit, ported from the `sha1_smol`
//! crate (1.0.1) and checked against it.
//!
//! The port follows `Sha1State::process` and the intrinsic emulations it calls
//! (`sha1msg1`, `sha1msg2`, `sha1rnds4c/p/m`, ...) function for function, on
//! plain `u32` words with `wrapping_add`, `rotate_left` and `^ & |`. A circuit
//! value is an f32 that's exact only within ±2^24, so inside the circuit each
//! word is held as its 4 bytes: `wrapping_add` carries from byte to byte,
//! bitwise operators split bytes into bits, and rotations move bits. The
//! function takes and returns bytes, which fit one cell each.
//!
//! cargo run --release --example sha1
//!
//! The ported code is from sha1_smol, under the BSD 3-Clause License:
//! Copyright (c) 2018, the respective contributors, as shown by the AUTHORS file
//! of https://github.com/mitsuhiko/sha1-smol. All rights reserved.
//! Redistribution and use in source and binary forms, with or without
//! modification, are permitted provided that the following conditions are met:
//! redistributions of source code must retain the above copyright notice, this
//! list of conditions and the following disclaimer; redistributions in binary
//! form must reproduce it in the documentation and/or other materials provided
//! with the distribution; neither the name of the copyright holder nor the
//! names of its contributors may be used to endorse or promote products derived
//! from this software without specific prior written permission. THIS SOFTWARE
//! IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
//! OR IMPLIED WARRANTIES ARE DISCLAIMED; IN NO EVENT SHALL THE COPYRIGHT HOLDER
//! OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES ARISING IN ANY WAY OUT OF THE USE
//! OF THIS SOFTWARE.

use std::time::Instant;

use handwired_rs::dsl::Params;
use handwired_rs::{BatchExecutor, handwired};

/// sha1_smol's `u32x4`.
type Quad = [u32; 4];

// Round key constants
const K0: u32 = 0x5A827999u32;
const K1: u32 = 0x6ED9EBA1u32;
const K2: u32 = 0x8F1BBCDCu32;
const K3: u32 = 0xCA62C1D6u32;

const DEFAULT_STATE: [u32; 5] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];

// ---- Ported from sha1_smol -------------------------------------------------------------------------------------

/// Adds a word to the first element of a vector.
#[handwired]
fn sha1_first_add(e: u32, w0: Quad) -> Quad {
    [e.wrapping_add(w0[0]), w0[1], w0[2], w0[3]]
}

/// Emulates `llvm.x86.sha1msg1` intrinsic.
#[handwired]
fn sha1msg1(a: Quad, b: Quad) -> Quad {
    [a[0] ^ a[2], a[1] ^ a[3], a[2] ^ b[0], a[3] ^ b[1]]
}

/// Emulates `llvm.x86.sha1msg2` intrinsic.
#[handwired]
fn sha1msg2(a: Quad, b: Quad) -> Quad {
    let w16 = (a[0] ^ b[1]).rotate_left(1);
    let w17 = (a[1] ^ b[2]).rotate_left(1);
    let w18 = (a[2] ^ b[3]).rotate_left(1);
    let w19 = (a[3] ^ w16).rotate_left(1);
    [w16, w17, w18, w19]
}

/// `sha1msg2(sha1msg1(v0, v1) ^ v2, v3)`, the `schedule!` macro.
#[handwired]
fn schedule(v0: Quad, v1: Quad, v2: Quad, v3: Quad) -> Quad {
    let first = sha1msg1(v0, v1);
    sha1msg2([first[0] ^ v2[0], first[1] ^ v2[1], first[2] ^ v2[2], first[3] ^ v2[3]], v3)
}

/// Emulates `llvm.x86.sha1nexte` intrinsic.
#[handwired]
fn sha1_first_half(abcd: Quad, msg: Quad) -> Quad {
    sha1_first_add(abcd[0].rotate_left(30), msg)
}

/// Choose, `bool3ary_202!`.
#[handwired]
fn choose(a: u32, b: u32, c: u32) -> u32 {
    c ^ (a & (b ^ c))
}

/// Parity, `bool3ary_150!`.
#[handwired]
fn parity(a: u32, b: u32, c: u32) -> u32 {
    a ^ b ^ c
}

/// Majority, `bool3ary_232!`.
#[handwired]
fn majority(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}

/// `work + KnV` in `sha1_digest_round_x4`.
#[handwired]
fn add_key(work: Quad, k: u32) -> Quad {
    [work[0].wrapping_add(k), work[1].wrapping_add(k), work[2].wrapping_add(k), work[3].wrapping_add(k)]
}

/// Emulates `llvm.x86.sha1rnds4` intrinsic: four rounds with Choose.
#[handwired]
fn sha1rnds4c(abcd: Quad, msg: Quad) -> Quad {
    let (mut a, mut b, mut c, mut d) = (abcd[0], abcd[1], abcd[2], abcd[3]);
    let (t, u, v, w) = (msg[0], msg[1], msg[2], msg[3]);
    let mut e = 0u32;

    e = e.wrapping_add(a.rotate_left(5)).wrapping_add(choose(b, c, d)).wrapping_add(t);
    b = b.rotate_left(30);

    d = d.wrapping_add(e.rotate_left(5)).wrapping_add(choose(a, b, c)).wrapping_add(u);
    a = a.rotate_left(30);

    c = c.wrapping_add(d.rotate_left(5)).wrapping_add(choose(e, a, b)).wrapping_add(v);
    e = e.rotate_left(30);

    b = b.wrapping_add(c.rotate_left(5)).wrapping_add(choose(d, e, a)).wrapping_add(w);
    d = d.rotate_left(30);

    [b, c, d, e]
}

/// Emulates `llvm.x86.sha1rnds4` intrinsic: four rounds with Parity.
#[handwired]
fn sha1rnds4p(abcd: Quad, msg: Quad) -> Quad {
    let (mut a, mut b, mut c, mut d) = (abcd[0], abcd[1], abcd[2], abcd[3]);
    let (t, u, v, w) = (msg[0], msg[1], msg[2], msg[3]);
    let mut e = 0u32;

    e = e.wrapping_add(a.rotate_left(5)).wrapping_add(parity(b, c, d)).wrapping_add(t);
    b = b.rotate_left(30);

    d = d.wrapping_add(e.rotate_left(5)).wrapping_add(parity(a, b, c)).wrapping_add(u);
    a = a.rotate_left(30);

    c = c.wrapping_add(d.rotate_left(5)).wrapping_add(parity(e, a, b)).wrapping_add(v);
    e = e.rotate_left(30);

    b = b.wrapping_add(c.rotate_left(5)).wrapping_add(parity(d, e, a)).wrapping_add(w);
    d = d.rotate_left(30);

    [b, c, d, e]
}

/// Emulates `llvm.x86.sha1rnds4` intrinsic: four rounds with Majority.
#[handwired]
fn sha1rnds4m(abcd: Quad, msg: Quad) -> Quad {
    let (mut a, mut b, mut c, mut d) = (abcd[0], abcd[1], abcd[2], abcd[3]);
    let (t, u, v, w) = (msg[0], msg[1], msg[2], msg[3]);
    let mut e = 0u32;

    e = e.wrapping_add(a.rotate_left(5)).wrapping_add(majority(b, c, d)).wrapping_add(t);
    b = b.rotate_left(30);

    d = d.wrapping_add(e.rotate_left(5)).wrapping_add(majority(a, b, c)).wrapping_add(u);
    a = a.rotate_left(30);

    c = c.wrapping_add(d.rotate_left(5)).wrapping_add(majority(e, a, b)).wrapping_add(v);
    e = e.rotate_left(30);

    b = b.wrapping_add(c.rotate_left(5)).wrapping_add(majority(d, e, a)).wrapping_add(w);
    d = d.rotate_left(30);

    [b, c, d, e]
}

/// Four big-endian bytes as a word.
#[handwired]
fn word_from_bytes(bytes: [u8; 4]) -> u32 {
    (bytes[0] as u32) << 24 | (bytes[1] as u32) << 16 | (bytes[2] as u32) << 8 | bytes[3] as u32
}

/// `Sha1State::process` on a state and a block given as bytes, the state as five big-endian words. Inside, the
/// words are `u32`s beyond ±2^24, held as bytes; only bytes go in and out.
#[handwired]
fn sha1_block(state: [u8; 20], block: [u8; 64]) -> [u8; 20] {
    let mut words = [0u32; 16];
    for i in 0..16 {
        words[i] = word_from_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
    }
    let mut s = [0u32; 5];
    for i in 0..5 {
        s[i] = word_from_bytes([state[4 * i], state[4 * i + 1], state[4 * i + 2], state[4 * i + 3]]);
    }

    // Rounds 0..20
    let mut h0 = [s[0], s[1], s[2], s[3]];
    let mut w0 = [words[0], words[1], words[2], words[3]];
    let mut h1 = sha1rnds4c(h0, add_key(sha1_first_add(s[4], w0), K0));
    let mut w1 = [words[4], words[5], words[6], words[7]];
    h0 = sha1rnds4c(h1, add_key(sha1_first_half(h0, w1), K0));
    let mut w2 = [words[8], words[9], words[10], words[11]];
    h1 = sha1rnds4c(h0, add_key(sha1_first_half(h1, w2), K0));
    let mut w3 = [words[12], words[13], words[14], words[15]];
    h0 = sha1rnds4c(h1, add_key(sha1_first_half(h0, w3), K0));
    let mut w4 = schedule(w0, w1, w2, w3);
    h1 = sha1rnds4c(h0, add_key(sha1_first_half(h1, w4), K0));

    // Rounds 20..40
    w0 = schedule(w1, w2, w3, w4);
    h0 = sha1rnds4p(h1, add_key(sha1_first_half(h0, w0), K1));
    w1 = schedule(w2, w3, w4, w0);
    h1 = sha1rnds4p(h0, add_key(sha1_first_half(h1, w1), K1));
    w2 = schedule(w3, w4, w0, w1);
    h0 = sha1rnds4p(h1, add_key(sha1_first_half(h0, w2), K1));
    w3 = schedule(w4, w0, w1, w2);
    h1 = sha1rnds4p(h0, add_key(sha1_first_half(h1, w3), K1));
    w4 = schedule(w0, w1, w2, w3);
    h0 = sha1rnds4p(h1, add_key(sha1_first_half(h0, w4), K1));

    // Rounds 40..60
    w0 = schedule(w1, w2, w3, w4);
    h1 = sha1rnds4m(h0, add_key(sha1_first_half(h1, w0), K2));
    w1 = schedule(w2, w3, w4, w0);
    h0 = sha1rnds4m(h1, add_key(sha1_first_half(h0, w1), K2));
    w2 = schedule(w3, w4, w0, w1);
    h1 = sha1rnds4m(h0, add_key(sha1_first_half(h1, w2), K2));
    w3 = schedule(w4, w0, w1, w2);
    h0 = sha1rnds4m(h1, add_key(sha1_first_half(h0, w3), K2));
    w4 = schedule(w0, w1, w2, w3);
    h1 = sha1rnds4m(h0, add_key(sha1_first_half(h1, w4), K2));

    // Rounds 60..80
    w0 = schedule(w1, w2, w3, w4);
    h0 = sha1rnds4p(h1, add_key(sha1_first_half(h0, w0), K3));
    w1 = schedule(w2, w3, w4, w0);
    h1 = sha1rnds4p(h0, add_key(sha1_first_half(h1, w1), K3));
    w2 = schedule(w3, w4, w0, w1);
    h0 = sha1rnds4p(h1, add_key(sha1_first_half(h0, w2), K3));
    w3 = schedule(w4, w0, w1, w2);
    h1 = sha1rnds4p(h0, add_key(sha1_first_half(h1, w3), K3));
    w4 = schedule(w0, w1, w2, w3);
    h0 = sha1rnds4p(h1, add_key(sha1_first_half(h0, w4), K3));

    let e = h1[0].rotate_left(30);
    let next = [
        s[0].wrapping_add(h0[0]),
        s[1].wrapping_add(h0[1]),
        s[2].wrapping_add(h0[2]),
        s[3].wrapping_add(h0[3]),
        s[4].wrapping_add(e),
    ];
    let mut result = [0u8; 20];
    for i in 0..5 {
        for byte in 0..4 {
            result[4 * i + byte] = (next[i] >> (24 - 8 * byte)) as u8;
        }
    }
    result
}

// ---- Driving it ------------------------------------------------------------------------------------------------

/// SHA-1 padding: the message, 0x80, zeros, and the bit length, in 64-byte blocks.
fn pad(message: &[u8]) -> Vec<[u8; 64]> {
    let mut bytes = message.to_vec();
    bytes.push(0x80);
    while bytes.len() % 64 != 56 {
        bytes.push(0);
    }
    bytes.extend_from_slice(&(message.len() as u64 * 8).to_be_bytes());
    bytes.chunks(64).map(|chunk| chunk.try_into().unwrap()).collect()
}

fn initial_state() -> [u8; 20] {
    let mut state = [0u8; 20];
    for (index, word) in DEFAULT_STATE.iter().enumerate() {
        state[4 * index..4 * index + 4].copy_from_slice(&word.to_be_bytes());
    }
    state
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Hash every message through the circuit, one batch per block position: block k of every message that has one
/// runs in the same batch, its state taken from block k - 1.
fn hash_all(program: &handwired_rs::TensorProgram, messages: &[Vec<u8>]) -> Vec<[u8; 20]> {
    let padded: Vec<Vec<[u8; 64]>> = messages.iter().map(|message| pad(message)).collect();
    let mut states = vec![initial_state(); messages.len()];
    let longest = padded.iter().map(Vec::len).max().unwrap_or(0);
    for position in 0..longest {
        let active: Vec<usize> = (0..messages.len()).filter(|&index| padded[index].len() > position).collect();
        let mut sources = Vec::new();
        for &index in &active {
            (states[index], padded[index][position]).encode(&mut sources);
        }
        let outputs = BatchExecutor::run_parallel(program, &sources, None);
        for (&index, digest) in active.iter().zip(outputs.chunks(20)) {
            states[index] = std::array::from_fn(|byte| digest[byte] as u8);
        }
    }
    states
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() {
    let start = Instant::now();
    let compiled = sha1_block_circuit();
    println!(
        "sha1_block: {} live operations in {} layers, {} stored weights, built in {:.1?}",
        compiled.program.live_operation_count(),
        compiled.program.layers.len(),
        compiled.program.stored_weight_count(),
        start.elapsed()
    );
    let start = Instant::now();
    let report = compiled.verify(64).unwrap();
    println!(
        "matches the ported Rust function on {} random states and blocks ({:.1?})",
        report.checked,
        start.elapsed()
    );

    let vectors: [(&[u8], &str); 4] = [
        (b"", "da39a3ee5e6b4b0d3255bfef95601890afd80709"),
        (b"abc", "a9993e364706816aba3e25717850c26c9cd0d89d"),
        (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "84983e441c3bd26ebaae4aa1f95129e5e54670f1"),
        (b"The quick brown fox jumps over the lazy dog", "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"),
    ];
    let messages: Vec<Vec<u8>> = vectors.iter().map(|(message, _)| message.to_vec()).collect();
    for ((message, expected), digest) in vectors.iter().zip(hash_all(&compiled.program, &messages)) {
        assert_eq!(hex(&digest), *expected);
        assert_eq!(digest, sha1_smol::Sha1::from(message).digest().bytes());
        println!("  {:<58} {}", format!("{:?}", String::from_utf8_lossy(message)), hex(&digest));
    }

    let mut rng = Rng(0x5EED_5EED_5EED_5EED);
    let messages: Vec<Vec<u8>> = (0..2_000)
        .map(|_| {
            let length = (rng.next() % 300) as usize;
            (0..length).map(|_| rng.next() as u8).collect()
        })
        .collect();
    let blocks: usize = messages.iter().map(|message| pad(message).len()).sum();
    let start = Instant::now();
    let digests = hash_all(&compiled.program, &messages);
    let circuit_time = start.elapsed();
    let start = Instant::now();
    let library: Vec<[u8; 20]> =
        messages.iter().map(|message| sha1_smol::Sha1::from(message).digest().bytes()).collect();
    let library_time = start.elapsed();
    assert_eq!(digests, library, "the circuit disagrees with sha1_smol");
    println!(
        "{} random messages of 0..300 bytes ({blocks} blocks): every digest matches sha1_smol \
         (circuit {circuit_time:.1?} on the batch executor, sha1_smol {library_time:.1?})",
        messages.len()
    );
}
