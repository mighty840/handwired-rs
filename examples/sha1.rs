//! SHA-1's block function as an exact circuit, ported from the `sha1_smol`
//! crate (1.0.1) and checked against it.
//!
//! The port follows `Sha1State::process` and the intrinsic emulations it calls
//! (`sha1msg1`, `sha1msg2`, `sha1rnds4c/p/m`, ...) function for function. A
//! circuit value is an f32 that's exact only within ±2^24, so a 32-bit word is
//! held as 32 bits: XOR, AND, OR and NOT work bit by bit, a rotation only
//! renames bits, and `wrapping_add` adds the words byte by byte, carrying from
//! byte to byte, with the bits of each byte sum read back by integer `%`
//! (a table read) and a comparison.
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

/// A 32-bit word, least significant bit first: `word[i]` is bit `i`.
type Word = [bool; 32];
/// sha1_smol's `u32x4`.
type Quad = [Word; 4];

const fn word(value: u32) -> Word {
    let mut bits = [false; 32];
    let mut bit = 0;
    while bit < 32 {
        bits[bit] = (value >> bit) & 1 == 1;
        bit += 1;
    }
    bits
}

// Round key constants
const K0: Word = word(0x5A827999);
const K1: Word = word(0x6ED9EBA1);
const K2: Word = word(0x8F1BBCDC);
const K3: Word = word(0xCA62C1D6);

/// Powers of two, since `#[handwired]` has no shifts.
const POW2: [u16; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const BYTE_WEIGHT: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

const DEFAULT_STATE: [u32; 5] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0];

// ---- Words ------------------------------------------------------------------------------------------------------

#[handwired]
fn rotl1(w: Word) -> Word {
    let mut rotated = w;
    for bit in 0..32 {
        rotated[(bit + 1) % 32] = w[bit];
    }
    rotated
}

#[handwired]
fn rotl5(w: Word) -> Word {
    let mut rotated = w;
    for bit in 0..32 {
        rotated[(bit + 5) % 32] = w[bit];
    }
    rotated
}

#[handwired]
fn rotl30(w: Word) -> Word {
    let mut rotated = w;
    for bit in 0..32 {
        rotated[(bit + 30) % 32] = w[bit];
    }
    rotated
}

#[handwired]
fn xor(a: Word, b: Word) -> Word {
    let mut result = a;
    for bit in 0..32 {
        result[bit] = a[bit] ^ b[bit];
    }
    result
}

/// `a.wrapping_add(b).wrapping_add(c)`: per byte, the column sum plus the carry
/// is at most 3 × 255 + 2, and its bits and carry are read back exactly.
#[handwired]
#[allow(clippy::needless_range_loop)]
fn add3(a: Word, b: Word, c: Word) -> Word {
    let mut sum = a;
    let mut carry: u16 = 0;
    for byte in 0..4 {
        let mut total: u16 = carry;
        for bit in 0..8 {
            let at = byte * 8 + bit;
            let weight: u16 = POW2[bit];
            total = total + (a[at] as u16) * weight + (b[at] as u16) * weight + (c[at] as u16) * weight;
        }
        for bit in 0..8 {
            sum[byte * 8 + bit] = total % POW2[bit + 1] >= POW2[bit];
        }
        carry = total / 256;
    }
    sum
}

#[handwired]
fn add2(a: Word, b: Word) -> Word {
    add3(a, b, [false; 32])
}

// ---- Ported from sha1_smol -------------------------------------------------------------------------------------

/// Adds a word to the first element of a vector.
#[handwired]
fn sha1_first_add(e: Word, w0: Quad) -> Quad {
    [add2(e, w0[0]), w0[1], w0[2], w0[3]]
}

/// Emulates `llvm.x86.sha1msg1` intrinsic.
#[handwired]
fn sha1msg1(a: Quad, b: Quad) -> Quad {
    [xor(a[0], a[2]), xor(a[1], a[3]), xor(a[2], b[0]), xor(a[3], b[1])]
}

/// Emulates `llvm.x86.sha1msg2` intrinsic.
#[handwired]
fn sha1msg2(a: Quad, b: Quad) -> Quad {
    let w16 = rotl1(xor(a[0], b[1]));
    let w17 = rotl1(xor(a[1], b[2]));
    let w18 = rotl1(xor(a[2], b[3]));
    let w19 = rotl1(xor(a[3], w16));
    [w16, w17, w18, w19]
}

/// `sha1msg2(sha1msg1(v0, v1) ^ v2, v3)`, the `schedule!` macro.
#[handwired]
fn schedule(v0: Quad, v1: Quad, v2: Quad, v3: Quad) -> Quad {
    let first = sha1msg1(v0, v1);
    let mixed = [xor(first[0], v2[0]), xor(first[1], v2[1]), xor(first[2], v2[2]), xor(first[3], v2[3])];
    sha1msg2(mixed, v3)
}

/// Emulates `llvm.x86.sha1nexte` intrinsic.
#[handwired]
fn sha1_first_half(abcd: Quad, msg: Quad) -> Quad {
    sha1_first_add(rotl30(abcd[0]), msg)
}

/// Choose, `c ^ (a & (b ^ c))`.
#[handwired]
fn choose(a: Word, b: Word, c: Word) -> Word {
    let mut result = a;
    for bit in 0..32 {
        result[bit] = c[bit] ^ (a[bit] & (b[bit] ^ c[bit]));
    }
    result
}

/// Parity, `a ^ b ^ c`.
#[handwired]
fn parity(a: Word, b: Word, c: Word) -> Word {
    let mut result = a;
    for bit in 0..32 {
        result[bit] = a[bit] ^ b[bit] ^ c[bit];
    }
    result
}

/// Majority, `(a & b) ^ (a & c) ^ (b & c)`.
#[handwired]
fn majority(a: Word, b: Word, c: Word) -> Word {
    let mut result = a;
    for bit in 0..32 {
        result[bit] = (a[bit] & b[bit]) ^ (a[bit] & c[bit]) ^ (b[bit] & c[bit]);
    }
    result
}

/// `work + KnV` in `sha1_digest_round_x4`.
#[handwired]
fn add_key(work: Quad, k: Word) -> Quad {
    [add2(work[0], k), add2(work[1], k), add2(work[2], k), add2(work[3], k)]
}

/// Emulates `llvm.x86.sha1rnds4` intrinsic: four rounds with Choose.
#[handwired]
fn sha1rnds4c(abcd: Quad, msg: Quad) -> Quad {
    let (mut a, mut b, mut c, mut d) = (abcd[0], abcd[1], abcd[2], abcd[3]);
    let (t, u, v, w) = (msg[0], msg[1], msg[2], msg[3]);
    let mut e = [false; 32];

    e = add3(add2(e, rotl5(a)), choose(b, c, d), t);
    b = rotl30(b);

    d = add3(add2(d, rotl5(e)), choose(a, b, c), u);
    a = rotl30(a);

    c = add3(add2(c, rotl5(d)), choose(e, a, b), v);
    e = rotl30(e);

    b = add3(add2(b, rotl5(c)), choose(d, e, a), w);
    d = rotl30(d);

    [b, c, d, e]
}

/// Emulates `llvm.x86.sha1rnds4` intrinsic: four rounds with Parity.
#[handwired]
fn sha1rnds4p(abcd: Quad, msg: Quad) -> Quad {
    let (mut a, mut b, mut c, mut d) = (abcd[0], abcd[1], abcd[2], abcd[3]);
    let (t, u, v, w) = (msg[0], msg[1], msg[2], msg[3]);
    let mut e = [false; 32];

    e = add3(add2(e, rotl5(a)), parity(b, c, d), t);
    b = rotl30(b);

    d = add3(add2(d, rotl5(e)), parity(a, b, c), u);
    a = rotl30(a);

    c = add3(add2(c, rotl5(d)), parity(e, a, b), v);
    e = rotl30(e);

    b = add3(add2(b, rotl5(c)), parity(d, e, a), w);
    d = rotl30(d);

    [b, c, d, e]
}

/// Emulates `llvm.x86.sha1rnds4` intrinsic: four rounds with Majority.
#[handwired]
fn sha1rnds4m(abcd: Quad, msg: Quad) -> Quad {
    let (mut a, mut b, mut c, mut d) = (abcd[0], abcd[1], abcd[2], abcd[3]);
    let (t, u, v, w) = (msg[0], msg[1], msg[2], msg[3]);
    let mut e = [false; 32];

    e = add3(add2(e, rotl5(a)), majority(b, c, d), t);
    b = rotl30(b);

    d = add3(add2(d, rotl5(e)), majority(a, b, c), u);
    a = rotl30(a);

    c = add3(add2(c, rotl5(d)), majority(e, a, b), v);
    e = rotl30(e);

    b = add3(add2(b, rotl5(c)), majority(d, e, a), w);
    d = rotl30(d);

    [b, c, d, e]
}

#[handwired]
fn byte_bits(value: u8) -> [bool; 8] {
    let wide = value as u16;
    let mut bits = [false; 8];
    for bit in 0..8 {
        bits[bit] = wide % POW2[bit + 1] >= POW2[bit];
    }
    bits
}

/// Four big-endian bytes as a word.
#[handwired]
fn word_from_bytes(bytes: [u8; 4]) -> Word {
    let mut result = [false; 32];
    for byte in 0..4 {
        let bits = byte_bits(bytes[byte]);
        for bit in 0..8 {
            result[(3 - byte) * 8 + bit] = bits[bit];
        }
    }
    result
}

#[handwired]
fn word_to_bytes(w: Word) -> [u8; 4] {
    let mut bytes = [0u8; 4];
    for byte in 0..4 {
        let mut value: u8 = 0;
        for bit in 0..8 {
            value += (w[(3 - byte) * 8 + bit] as u8) * BYTE_WEIGHT[bit];
        }
        bytes[byte] = value;
    }
    bytes
}

/// `Sha1State::process` on a state and a block given as bytes, the state as five big-endian words.
#[handwired]
fn sha1_block(state: [u8; 20], block: [u8; 64]) -> [u8; 20] {
    let mut words = [[false; 32]; 16];
    for i in 0..16 {
        words[i] = word_from_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
    }
    let mut s = [[false; 32]; 5];
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

    let e = rotl30(h1[0]);
    let next = [add2(s[0], h0[0]), add2(s[1], h0[1]), add2(s[2], h0[2]), add2(s[3], h0[3]), add2(s[4], e)];
    let mut result = [0u8; 20];
    for i in 0..5 {
        let bytes = word_to_bytes(next[i]);
        for byte in 0..4 {
            result[4 * i + byte] = bytes[byte];
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
