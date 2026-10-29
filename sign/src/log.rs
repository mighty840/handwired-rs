//! A version log: every program a device was asked to install into a slot, accepted or refused, in order, each
//! entry chained to the one before by SHA-256. A device keeps it to show which logic it ran and when it changed,
//! and [`admit`] uses it to refuse releases older than the slot's installed one.
//!
//! A log is a header, then fixed-size entries:
//!
//! ```text
//! header: "HWCLOG\0\0" ‖ version (u32 LE, 1) ‖ log id (32 bytes, the device's choice, e.g. a hash of its serial)
//! entry:  time (u64 LE) ‖ outcome (1) ‖ slot length (1) ‖ slot (64, zero-padded) ‖ sequence (u64 LE) ‖
//!         program SHA-256 (32) ‖ signer's public key (32) ‖ previous (32)
//! ```
//!
//! `previous` is the hash of the entry before, SHA-256("handwired-rs version log entry v1\0" ‖ entry), and for the
//! first entry SHA-256("handwired-rs version log v1\0" ‖ header). Time is whatever the device says (0 if it has no
//! clock); the order of entries doesn't depend on it.
//!
//! The chain makes a change to any entry, or a deleted or reordered one, show up in [`check`]. It can't show that
//! entries were cut off the end, or that the whole log was replaced by another consistent one: for that the device
//! must keep the [`Head`] (entry count and last hash) where an attacker can't write it, or report it elsewhere, and
//! compare. Refusals are logged too: they are the evidence of attempted changes.
//!
//! The policy [`admit`] applies for a slot: the signature must be valid under a trusted key and for this slot; a
//! sequence below the slot's last installed one is refused; the same sequence is accepted only for the same program
//! (a re-install), never for different bytes. The device appends the entry [`admit`] returns, whatever the outcome,
//! before it acts on an acceptance.

use crate::{Error, Signed, MAX_SLOT, PUBLIC_KEY_BYTES};
use hmac_sha256::Hash;

/// The first 8 bytes of every log.
pub const MAGIC: [u8; 8] = *b"HWCLOG\0\0";
/// The log format version this crate reads and writes.
pub const VERSION: u32 = 1;
/// Bytes in a log's header.
pub const HEADER_BYTES: usize = 8 + 4 + 32;
/// Bytes in each entry.
pub const ENTRY_BYTES: usize = 8 + 1 + 1 + MAX_SLOT + 8 + 32 + PUBLIC_KEY_BYTES + 32;

const HEADER_DOMAIN: &[u8] = b"handwired-rs version log v1\0";
const ENTRY_DOMAIN: &[u8] = b"handwired-rs version log entry v1\0";

/// What happened to a program presented for a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Accepted: signed by a trusted key, for this slot, and not older than the slot's installed release.
    Installed,
    /// Refused, for this reason.
    Refused(Refused),
}

/// Why a program was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Refused {
    /// Signed by a key that isn't trusted.
    UntrustedKey,
    /// The signature doesn't match the program, slot and sequence, or a trusted key isn't a valid key.
    Mismatch,
    /// Validly signed, but for another slot.
    OtherSlot,
    /// Validly signed, but its sequence is below the slot's installed release.
    OlderRelease,
    /// Validly signed with the slot's installed sequence, but not the installed program.
    ReleaseReused,
    /// The signature file can't be read.
    UnreadableSignature,
}

impl Outcome {
    fn code(self) -> u8 {
        match self {
            Outcome::Installed => 0,
            Outcome::Refused(Refused::UntrustedKey) => 1,
            Outcome::Refused(Refused::Mismatch) => 2,
            Outcome::Refused(Refused::OtherSlot) => 3,
            Outcome::Refused(Refused::OlderRelease) => 4,
            Outcome::Refused(Refused::ReleaseReused) => 5,
            Outcome::Refused(Refused::UnreadableSignature) => 6,
        }
    }

    fn from_code(code: u8) -> Option<Outcome> {
        Some(match code {
            0 => Outcome::Installed,
            1 => Outcome::Refused(Refused::UntrustedKey),
            2 => Outcome::Refused(Refused::Mismatch),
            3 => Outcome::Refused(Refused::OtherSlot),
            4 => Outcome::Refused(Refused::OlderRelease),
            5 => Outcome::Refused(Refused::ReleaseReused),
            6 => Outcome::Refused(Refused::UnreadableSignature),
            _ => return None,
        })
    }
}

impl core::fmt::Display for Outcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Outcome::Installed => "installed",
            Outcome::Refused(Refused::UntrustedKey) => "refused: signed by a key that isn't trusted",
            Outcome::Refused(Refused::Mismatch) => "refused: the signature doesn't match",
            Outcome::Refused(Refused::OtherSlot) => "refused: signed for another slot",
            Outcome::Refused(Refused::OlderRelease) => "refused: older than the installed release",
            Outcome::Refused(Refused::ReleaseReused) => "refused: the installed release number with other bytes",
            Outcome::Refused(Refused::UnreadableSignature) => "refused: the signature can't be read",
        })
    }
}

/// Why a log isn't accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogError {
    /// The bytes don't start like a log.
    NotALog,
    /// A log format version this crate doesn't read.
    UnsupportedVersion(u32),
    /// The log ends inside its header or an entry, e.g. after an interrupted write.
    Truncated,
    /// This entry's `previous` isn't the hash of the entry before: an entry was changed, removed or moved.
    Broken {
        /// The entry, counting from 0.
        entry: usize,
    },
    /// This entry isn't one this crate writes.
    Malformed {
        /// The entry, counting from 0.
        entry: usize,
        /// What's wrong.
        what: &'static str,
    },
}

impl core::fmt::Display for LogError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LogError::NotALog => write!(f, "not a handwired version log"),
            LogError::UnsupportedVersion(version) => {
                write!(f, "log format version {version}; this crate reads version {VERSION}")
            }
            LogError::Truncated => write!(f, "the log ends inside its header or an entry"),
            LogError::Broken { entry } => {
                write!(f, "entry {entry} doesn't follow the one before: an entry was changed, removed or moved")
            }
            LogError::Malformed { entry, what } => write!(f, "entry {entry}: {what}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LogError {}

/// Where a log ends: how many entries it has and the hash the next entry must name as `previous`. Kept where an
/// attacker can't write, it shows whether entries were cut off the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Head {
    /// Entries in the log.
    pub entries: usize,
    /// The last entry's hash, or the header's for an empty log.
    pub hash: [u8; 32],
}

/// One entry of a checked log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry<'a> {
    /// The device's time when the entry was made (0 if it has none).
    pub time: u64,
    /// Installed or refused, and why.
    pub outcome: Outcome,
    /// The slot the program was presented for.
    pub slot: &'a str,
    /// The sequence its signature claims (0 if the signature can't be read).
    pub sequence: u64,
    /// SHA-256 of the program's bytes.
    pub program: [u8; 32],
    /// The public key its signature claims (zeros if the signature can't be read).
    pub key: [u8; PUBLIC_KEY_BYTES],
}

/// A new log's header, which is the whole of an empty log.
pub fn header(id: [u8; 32]) -> [u8; HEADER_BYTES] {
    let mut header = [0u8; HEADER_BYTES];
    header[..8].copy_from_slice(&MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..].copy_from_slice(&id);
    header
}

/// SHA-256 of `bytes`: a program's identity in the log, as in a `sha256sum` manifest.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Hash::hash(bytes)
}

fn chained(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hash = Hash::new();
    hash.update(domain);
    hash.update(bytes);
    hash.finalize()
}

fn word<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
    bytes[at..at + N].try_into().unwrap()
}

/// The entries of a log, one by one; [`entries`] gives them only for a log that passes [`check`].
#[derive(Clone)]
pub struct Entries<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for Entries<'a> {
    type Item = Entry<'a>;

    fn next(&mut self) -> Option<Entry<'a>> {
        if self.rest.is_empty() {
            return None;
        }
        let (entry, rest) = self.rest.split_at(ENTRY_BYTES);
        self.rest = rest;
        Some(parse(entry, 0).expect("checked"))
    }
}

fn parse(entry: &[u8], index: usize) -> Result<Entry<'_>, LogError> {
    let malformed = |what| LogError::Malformed { entry: index, what };
    let outcome = Outcome::from_code(entry[8]).ok_or(malformed("an unknown outcome"))?;
    let length = entry[9] as usize;
    if length > MAX_SLOT {
        return Err(malformed("the slot name is longer than 64 bytes"));
    }
    if entry[10 + length..10 + MAX_SLOT].iter().any(|&byte| byte != 0) {
        return Err(malformed("bytes after the slot name"));
    }
    let slot = core::str::from_utf8(&entry[10..10 + length]).map_err(|_| malformed("the slot name isn't UTF-8"))?;
    let at = 10 + MAX_SLOT;
    Ok(Entry {
        time: u64::from_le_bytes(word(entry, 0)),
        outcome,
        slot,
        sequence: u64::from_le_bytes(word(entry, at)),
        program: word(entry, at + 8),
        key: word(entry, at + 40),
    })
}

/// Check a log: its header, and that every entry is well formed and names the one before. Returns its head.
pub fn check(log: &[u8]) -> Result<Head, LogError> {
    if log.len() < MAGIC.len() || log[..MAGIC.len()] != MAGIC {
        return Err(if MAGIC.starts_with(log) { LogError::Truncated } else { LogError::NotALog });
    }
    if log.len() < HEADER_BYTES {
        return Err(LogError::Truncated);
    }
    let version = u32::from_le_bytes(word(log, 8));
    if version != VERSION {
        return Err(LogError::UnsupportedVersion(version));
    }
    let body = &log[HEADER_BYTES..];
    if body.len() % ENTRY_BYTES != 0 {
        return Err(LogError::Truncated);
    }
    let mut hash = chained(HEADER_DOMAIN, &log[..HEADER_BYTES]);
    for (index, entry) in body.chunks_exact(ENTRY_BYTES).enumerate() {
        parse(entry, index)?;
        if entry[ENTRY_BYTES - 32..] != hash {
            return Err(LogError::Broken { entry: index });
        }
        hash = chained(ENTRY_DOMAIN, entry);
    }
    Ok(Head { entries: body.len() / ENTRY_BYTES, hash })
}

/// The entries of a log that passes [`check`], oldest first.
pub fn entries(log: &[u8]) -> Result<Entries<'_>, LogError> {
    check(log)?;
    Ok(Entries { rest: &log[HEADER_BYTES..] })
}

/// The slot's installed program: its last entry with the outcome [`Outcome::Installed`].
pub fn installed<'a>(log: &'a [u8], slot: &str) -> Result<Option<Entry<'a>>, LogError> {
    Ok(entries(log)?.filter(|entry| entry.outcome == Outcome::Installed && entry.slot == slot).last())
}

/// What [`admit`] decided, and the entry recording it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission<'a> {
    /// Installed or refused, and why.
    pub outcome: Outcome,
    /// What the signature says, if it was accepted.
    pub signed: Option<Signed<'a>>,
    /// The entry to append to the log, whatever the outcome, before acting on it.
    pub entry: [u8; ENTRY_BYTES],
}

/// Decide whether `program`, with its `signature`, may be installed into `slot`: the signature must be valid under
/// one of the `trusted` keys and for `slot`, and its sequence not below the slot's installed release (equal only for
/// the same program). Returns the decision and the entry to append to `log`; `log` itself isn't changed. Refuses to
/// decide on a log that fails [`check`], since its installed releases can't be trusted. `slot` must be at most 64
/// bytes.
pub fn admit<'a>(
    log: &[u8],
    program: &[u8],
    signature: &'a [u8],
    trusted: &[[u8; PUBLIC_KEY_BYTES]],
    slot: &str,
    time: u64,
) -> Result<Admission<'a>, LogError> {
    assert!(slot.len() <= MAX_SLOT, "a slot name is at most 64 bytes");
    let head = check(log)?;
    let digest = sha256(program);
    let claim = crate::read(signature).ok();
    let (outcome, signed) = match crate::verify(program, signature, trusted) {
        Err(Error::UntrustedKey) => (Outcome::Refused(Refused::UntrustedKey), None),
        Err(Error::Mismatch | Error::InvalidKey) => (Outcome::Refused(Refused::Mismatch), None),
        Err(_) => (Outcome::Refused(Refused::UnreadableSignature), None),
        Ok(signed) if signed.slot != slot => (Outcome::Refused(Refused::OtherSlot), None),
        Ok(signed) => match installed(log, slot)? {
            Some(last) if signed.sequence < last.sequence => (Outcome::Refused(Refused::OlderRelease), None),
            Some(last) if signed.sequence == last.sequence && last.program != digest => {
                (Outcome::Refused(Refused::ReleaseReused), None)
            }
            _ => (Outcome::Installed, Some(signed)),
        },
    };
    let mut entry = [0u8; ENTRY_BYTES];
    entry[..8].copy_from_slice(&time.to_le_bytes());
    entry[8] = outcome.code();
    entry[9] = slot.len() as u8;
    entry[10..10 + slot.len()].copy_from_slice(slot.as_bytes());
    let at = 10 + MAX_SLOT;
    if let Some(claim) = claim {
        entry[at..at + 8].copy_from_slice(&claim.sequence.to_le_bytes());
        entry[at + 40..at + 72].copy_from_slice(&claim.public_key);
    }
    entry[at + 8..at + 40].copy_from_slice(&digest);
    entry[at + 72..].copy_from_slice(&head.hash);
    Ok(Admission { outcome, signed, entry })
}
