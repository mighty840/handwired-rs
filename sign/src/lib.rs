//! Signatures for programs exported by [handwired-rs](https://crates.io/crates/handwired-rs): a device checks that a
//! program file comes from a key it trusts, for the slot it is loaded into, before it hands the bytes to
//! [handwired-rs-run](https://crates.io/crates/handwired-rs-run). `no_std`, no allocation, no `unsafe`; its one
//! dependency, `ed25519-compact`, has none of its own.
//!
//! A signature is a separate file next to the program (`heater.hwc.sig` beside `heater.hwc`), so the program's
//! bytes, and the SHA-256 that identifies them in manifests and logs, don't change. It is Ed25519 (RFC 8032) over
//!
//! ```text
//! "handwired-rs signed program v1\0" ‖ slot length (1 byte) ‖ slot ‖ sequence (u64, little-endian) ‖ program
//! ```
//!
//! The slot names the place the program is for, e.g. `estop` or `heater`, so a validly signed program can't be
//! loaded into a slot it wasn't signed for. The sequence numbers a slot's releases: [`verify`] returns it, and the
//! device must refuse one lower than the last it accepted for that slot, or an old, validly signed program could be
//! loaded again. Keeping that number, and a log of the programs loaded, is the device's job.
//!
//! The signature file is
//!
//! ```text
//! "HWCSIG\0\0" ‖ version (u32 LE, 1) ‖ signer's public key (32) ‖ sequence (u64 LE) ‖ slot length (1) ‖ slot ‖
//! signature (64)
//! ```
//!
//! The public key in it only says which key signed; [`verify`] accepts it only if it is one of the trusted keys.
//!
//! ```
//! # fn demo(program: &[u8], signature: &[u8], trusted: &[[u8; 32]]) -> Result<(), handwired_rs_sign::Error> {
//! let signed = handwired_rs_sign::verify(program, signature, trusted)?;
//! assert_eq!(signed.slot, "heater");
//! // Then: refuse signed.sequence if it is below the last one accepted for this slot, record it, and parse the
//! // program with handwired_rs_run::Program::parse(program).
//! # Ok(())
//! # }
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use core::fmt;
use ed25519_compact::{PublicKey, Signature};

/// The first 8 bytes of every signature file.
pub const MAGIC: [u8; 8] = *b"HWCSIG\0\0";
/// The signature format version this crate reads and writes.
pub const VERSION: u32 = 1;
/// What every signed message starts with, so a signature over a program can't be taken for one over anything else.
pub const DOMAIN: &[u8] = b"handwired-rs signed program v1\0";
/// The longest slot name, in bytes.
pub const MAX_SLOT: usize = 64;
/// Bytes in a public key.
pub const PUBLIC_KEY_BYTES: usize = 32;
/// Bytes in a secret key: the 32-byte seed, then the public key.
pub const SECRET_KEY_BYTES: usize = 64;

const HEADER: usize = MAGIC.len() + 4 + PUBLIC_KEY_BYTES + 8 + 1;

/// Why a signature isn't accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The signature file ends early.
    Truncated,
    /// The bytes don't start like a signature file.
    NotASignature,
    /// A signature format version this crate doesn't read.
    UnsupportedVersion(u32),
    /// The signature file's parts don't fit together, e.g. bytes after the signature.
    Malformed(&'static str),
    /// The key that signed isn't one of the trusted keys.
    UntrustedKey,
    /// The signature isn't valid for this program, slot and sequence under the key that claims it.
    Mismatch,
    /// A key's bytes aren't a valid key.
    InvalidKey,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => write!(f, "the signature's bytes end early"),
            Error::NotASignature => write!(f, "not a handwired program signature"),
            Error::UnsupportedVersion(version) => {
                write!(f, "signature format version {version}; this crate reads version {VERSION}")
            }
            Error::Malformed(what) => write!(f, "malformed signature: {what}"),
            Error::UntrustedKey => write!(f, "signed by a key that isn't trusted"),
            Error::Mismatch => write!(f, "the signature doesn't match the program, its slot and its sequence"),
            Error::InvalidKey => write!(f, "not a valid Ed25519 key"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// What a valid signature says about a program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Signed<'a> {
    /// The slot the program was signed for.
    pub slot: &'a str,
    /// Its release number in that slot. The device must refuse one below the last it accepted for the slot.
    pub sequence: u64,
    /// Which of the trusted keys signed it.
    pub key: usize,
}

/// A signature file's fields, before the signature is checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Claim<'a> {
    /// The public key of whoever signed it, by their own account.
    pub public_key: [u8; PUBLIC_KEY_BYTES],
    /// The slot.
    pub slot: &'a str,
    /// The sequence.
    pub sequence: u64,
    /// The Ed25519 signature.
    pub signature: [u8; 64],
}

/// Read a signature file without checking its signature: what it claims, for display and lookup only.
pub fn read(signature: &[u8]) -> Result<Claim<'_>, Error> {
    if signature.len() < MAGIC.len() {
        return Err(if MAGIC.starts_with(signature) { Error::Truncated } else { Error::NotASignature });
    }
    if signature[..MAGIC.len()] != MAGIC {
        return Err(Error::NotASignature);
    }
    if signature.len() < HEADER {
        return Err(Error::Truncated);
    }
    let version = u32::from_le_bytes(signature[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(Error::UnsupportedVersion(version));
    }
    let public_key: [u8; PUBLIC_KEY_BYTES] = signature[12..44].try_into().unwrap();
    let sequence = u64::from_le_bytes(signature[44..52].try_into().unwrap());
    let slot_length = signature[52] as usize;
    if slot_length > MAX_SLOT {
        return Err(Error::Malformed("the slot name is longer than 64 bytes"));
    }
    let length = HEADER + slot_length + 64;
    if signature.len() < length {
        return Err(Error::Truncated);
    }
    if signature.len() > length {
        return Err(Error::Malformed("bytes after the signature"));
    }
    let slot = core::str::from_utf8(&signature[HEADER..HEADER + slot_length])
        .map_err(|_| Error::Malformed("the slot name isn't UTF-8"))?;
    let signature = signature[HEADER + slot_length..].try_into().unwrap();
    Ok(Claim { public_key, slot, sequence, signature })
}

/// Check that `signature` is a valid signature of `program` by one of the `trusted` keys, and return the slot and
/// sequence it was signed for. Reads the program once and allocates nothing.
pub fn verify<'a>(
    program: &[u8],
    signature: &'a [u8],
    trusted: &[[u8; PUBLIC_KEY_BYTES]],
) -> Result<Signed<'a>, Error> {
    let claim = read(signature)?;
    let key = trusted.iter().position(|key| *key == claim.public_key).ok_or(Error::UntrustedKey)?;
    let public_key = PublicKey::new(claim.public_key);
    let mut state = public_key.verify_incremental(&Signature::new(claim.signature)).map_err(|error| match error {
        ed25519_compact::Error::InvalidPublicKey | ed25519_compact::Error::WeakPublicKey => Error::InvalidKey,
        _ => Error::Mismatch,
    })?;
    state.absorb(DOMAIN);
    state.absorb([claim.slot.len() as u8]);
    state.absorb(claim.slot.as_bytes());
    state.absorb(claim.sequence.to_le_bytes());
    state.absorb(program);
    state.verify().map_err(|_| Error::Mismatch)?;
    Ok(Signed { slot: claim.slot, sequence: claim.sequence, key })
}

/// The public key of a secret key (its last 32 bytes, checked against its seed).
pub fn public_key(secret_key: &[u8; SECRET_KEY_BYTES]) -> Result<[u8; PUBLIC_KEY_BYTES], Error> {
    let pair = key_pair(secret_key)?;
    Ok(*pair.pk)
}

/// A secret key from a 32-byte random seed: the seed, then its public key.
pub fn secret_key_from_seed(seed: [u8; 32]) -> Result<[u8; SECRET_KEY_BYTES], Error> {
    let pair =
        ed25519_compact::KeyPair::try_from_seed(ed25519_compact::Seed::new(seed)).map_err(|_| Error::InvalidKey)?;
    Ok(*pair.sk)
}

fn key_pair(secret_key: &[u8; SECRET_KEY_BYTES]) -> Result<ed25519_compact::KeyPair, Error> {
    let pair = ed25519_compact::KeyPair::from_slice(secret_key).map_err(|_| Error::InvalidKey)?;
    pair.validate().map_err(|_| Error::InvalidKey)?;
    Ok(pair)
}

/// Sign `program` for `slot` and `sequence`: the bytes of its signature file. Signing is deterministic (RFC 8032),
/// so the same program, slot, sequence and key give the same file.
#[cfg(feature = "alloc")]
pub fn sign(
    program: &[u8],
    slot: &str,
    sequence: u64,
    secret_key: &[u8; SECRET_KEY_BYTES],
) -> Result<alloc::vec::Vec<u8>, Error> {
    use alloc::vec::Vec;
    if slot.len() > MAX_SLOT {
        return Err(Error::Malformed("the slot name is longer than 64 bytes"));
    }
    let pair = key_pair(secret_key)?;
    let mut message = Vec::with_capacity(DOMAIN.len() + 1 + slot.len() + 8 + program.len());
    message.extend_from_slice(DOMAIN);
    message.push(slot.len() as u8);
    message.extend_from_slice(slot.as_bytes());
    message.extend_from_slice(&sequence.to_le_bytes());
    message.extend_from_slice(program);
    let signature = pair.sk.sign(&message, None);
    let mut file = Vec::with_capacity(HEADER + slot.len() + 64);
    file.extend_from_slice(&MAGIC);
    file.extend_from_slice(&VERSION.to_le_bytes());
    file.extend_from_slice(&pair.pk[..]);
    file.extend_from_slice(&sequence.to_le_bytes());
    file.push(slot.len() as u8);
    file.extend_from_slice(slot.as_bytes());
    file.extend_from_slice(&signature[..]);
    Ok(file)
}
