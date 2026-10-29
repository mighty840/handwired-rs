//! A signature is accepted only for the program, slot and sequence it was made for, and only under a trusted key;
//! anything else is refused, never a panic.
#![cfg(feature = "alloc")]

use handwired_rs::handwired;
use handwired_rs_sign::{public_key, read, secret_key_from_seed, sign, verify, Error, MAGIC};

/// Test keys from fixed seeds: never use these for anything real.
fn test_only_key(byte: u8) -> ([u8; 64], [u8; 32]) {
    let secret = secret_key_from_seed([byte; 32]).unwrap();
    (secret, public_key(&secret).unwrap())
}

/// A small program as handwired-rs exports it.
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

fn program() -> Vec<u8> {
    heater_circuit().export("heater")
}

#[test]
fn signed_program_verifies_then_runs() {
    let (secret, public) = test_only_key(1);
    let program = program();
    let signature = sign(&program, "heater", 7, &secret).unwrap();
    let signed = verify(&program, &signature, &[public]).unwrap();
    assert_eq!((signed.slot, signed.sequence, signed.key), ("heater", 7, 0));
    let parsed = handwired_rs_run::Program::parse(&program).unwrap();
    let mut scratch = vec![0.0; parsed.scratch_len()];
    let mut outputs = vec![0.0; parsed.output_count()];
    parsed.run(&[200.0, 21.0, 0.0], &mut scratch, &mut outputs).unwrap();
    assert_eq!(outputs, [10.0, 0.0]);
}

#[test]
fn reports_which_trusted_key_signed() {
    let (secret, public) = test_only_key(2);
    let others = [test_only_key(3).1, test_only_key(4).1];
    let program = program();
    let signature = sign(&program, "heater", 1, &secret).unwrap();
    assert_eq!(verify(&program, &signature, &[others[0], public, others[1]]).unwrap().key, 1);
}

/// The public key in the file only says who signed; a valid signature by a key that isn't trusted is refused.
#[test]
fn valid_signature_by_untrusted_key_is_refused() {
    let (secret, _) = test_only_key(5);
    let (_, trusted) = test_only_key(6);
    let program = program();
    let signature = sign(&program, "heater", 1, &secret).unwrap();
    assert_eq!(verify(&program, &signature, &[trusted]), Err(Error::UntrustedKey));
    assert_eq!(verify(&program, &signature, &[]), Err(Error::UntrustedKey));
}

/// Swapping in a trusted key's bytes doesn't help: the signature was made by another key.
#[test]
fn claiming_a_trusted_key_is_refused() {
    let (secret, _) = test_only_key(5);
    let (_, trusted) = test_only_key(6);
    let program = program();
    let mut signature = sign(&program, "heater", 1, &secret).unwrap();
    signature[12..44].copy_from_slice(&trusted);
    assert_eq!(verify(&program, &signature, &[trusted]), Err(Error::Mismatch));
}

#[test]
fn every_flipped_bit_is_refused() {
    let (secret, public) = test_only_key(7);
    let program = program();
    let signature = sign(&program, "heater", 3, &secret).unwrap();
    for at in 0..program.len() {
        let mut changed = program.clone();
        changed[at] ^= 1 << (at % 8);
        assert_eq!(verify(&changed, &signature, &[public]), Err(Error::Mismatch), "program byte {at}");
    }
    for at in 0..signature.len() {
        let mut changed = signature.clone();
        changed[at] ^= 1 << (at % 8);
        assert!(verify(&program, &changed, &[public]).is_err(), "signature byte {at}");
    }
}

#[test]
fn slot_and_sequence_are_signed() {
    let (secret, public) = test_only_key(8);
    let program = program();
    let signature = sign(&program, "heater", 3, &secret).unwrap();
    // Another slot name of the same length, and the next sequence, in place of the signed ones.
    let mut moved = signature.clone();
    moved[53..59].copy_from_slice(b"estop!");
    assert_eq!(verify(&program, &moved, &[public]), Err(Error::Mismatch));
    let mut newer = signature.clone();
    newer[44..52].copy_from_slice(&4u64.to_le_bytes());
    assert_eq!(verify(&program, &newer, &[public]), Err(Error::Mismatch));
    // Signed for another slot or sequence, it says so.
    let other = sign(&program, "estop", 9, &secret).unwrap();
    let signed = verify(&program, &other, &[public]).unwrap();
    assert_eq!((signed.slot, signed.sequence), ("estop", 9));
}

#[test]
fn damaged_files_are_refused_without_panicking() {
    let (secret, public) = test_only_key(9);
    let program = program();
    let signature = sign(&program, "heater", 1, &secret).unwrap();
    for length in 0..signature.len() {
        let error = verify(&program, &signature[..length], &[public]).unwrap_err();
        let expected = if length < MAGIC.len() && !MAGIC.starts_with(&signature[..length]) {
            Error::NotASignature
        } else {
            Error::Truncated
        };
        assert_eq!(error, expected, "{length} bytes");
    }
    let mut longer = signature.clone();
    longer.push(0);
    assert_eq!(verify(&program, &longer, &[public]), Err(Error::Malformed("bytes after the signature")));
    assert_eq!(verify(&program, b"HWCIRC\0\0", &[public]), Err(Error::NotASignature));
    let mut version = signature.clone();
    version[8] = 2;
    assert_eq!(verify(&program, &version, &[public]), Err(Error::UnsupportedVersion(2)));
    let mut long_slot = signature.clone();
    long_slot[52] = 65;
    assert!(matches!(verify(&program, &long_slot, &[public]), Err(Error::Malformed(_))));
    let mut not_utf8 = signature.clone();
    not_utf8[53] = 0xFF;
    assert_eq!(verify(&program, &not_utf8, &[public]), Err(Error::Malformed("the slot name isn't UTF-8")));
    assert_eq!(
        sign(&program, &"x".repeat(65), 1, &secret).unwrap_err(),
        Error::Malformed("the slot name is longer than 64 bytes")
    );
    let mut wrong_half = secret;
    wrong_half[63] ^= 1;
    assert_eq!(sign(&program, "heater", 1, &wrong_half), Err(Error::InvalidKey));
}

#[test]
fn read_shows_the_claim_unchecked() {
    let (secret, public) = test_only_key(10);
    let signature = sign(b"HWCIRC\0\0anything", "pump", 12, &secret).unwrap();
    let claim = read(&signature).unwrap();
    assert_eq!((claim.public_key, claim.slot, claim.sequence), (public, "pump", 12));
}

/// The format, pinned: this key, program, slot and sequence give exactly these bytes (signing is deterministic).
/// tests/independent_check.py checks the same bytes with another Ed25519 implementation.
#[test]
fn format_is_pinned() {
    let (secret, _) = test_only_key(0x42);
    let signature = sign(b"HWCIRC\0\0test program", "estop", 258, &secret).unwrap();
    let expected = include_str!("data/pinned.sig.hex").trim();
    assert_eq!(signature.iter().map(|byte| format!("{byte:02x}")).collect::<String>(), expected);
}
