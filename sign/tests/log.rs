//! The version log: what admit installs and refuses, and which changes to a log check finds.
#![cfg(all(feature = "alloc", feature = "log"))]

use handwired_rs_sign::log::{
    admit, check, entries, header, installed, sha256, Head, LogError, Outcome, Refused, ENTRY_BYTES, HEADER_BYTES,
};
use handwired_rs_sign::{public_key, secret_key_from_seed, sign};

/// Test keys from fixed seeds: never use these for anything real.
fn test_only_key(byte: u8) -> ([u8; 64], [u8; 32]) {
    let secret = secret_key_from_seed([byte; 32]).unwrap();
    (secret, public_key(&secret).unwrap())
}

/// Presents `program` for `slot` and appends the entry, as a device does; returns the outcome.
fn present(log: &mut Vec<u8>, program: &[u8], signature: &[u8], trusted: &[[u8; 32]], slot: &str) -> Outcome {
    let time = 1_800_000_000 + (log.len() - HEADER_BYTES) as u64 / ENTRY_BYTES as u64;
    let admission = admit(log, program, signature, trusted, slot, time).unwrap();
    assert_eq!(admission.outcome == Outcome::Installed, admission.signed.is_some());
    log.extend_from_slice(&admission.entry);
    admission.outcome
}

/// A log of one slot's releases, with each kind of refusal, as the pinned log has it.
fn history() -> Vec<u8> {
    let (release, trusted) = test_only_key(0x42);
    let (stranger, _) = test_only_key(0x43);
    let trusted = [trusted];
    let first = b"HWCIRC\0\0thermostat, release 1".as_slice();
    let second = b"HWCIRC\0\0thermostat, release 2".as_slice();
    let other = b"HWCIRC\0\0thermostat, other bytes".as_slice();
    let mut log = header([7; 32]).to_vec();
    let mut changed = second.to_vec();
    changed[8] ^= 1;
    let steps: [(&[u8], Vec<u8>, &str, Outcome); 9] = [
        (first, sign(first, "thermostat", 1, &release).unwrap(), "thermostat", Outcome::Installed),
        (second, sign(second, "thermostat", 2, &release).unwrap(), "thermostat", Outcome::Installed),
        (first, sign(first, "thermostat", 1, &release).unwrap(), "thermostat", Outcome::Refused(Refused::OlderRelease)),
        (
            other,
            sign(other, "thermostat", 2, &release).unwrap(),
            "thermostat",
            Outcome::Refused(Refused::ReleaseReused),
        ),
        (second, sign(second, "thermostat", 2, &release).unwrap(), "thermostat", Outcome::Installed),
        (second, sign(second, "pump", 3, &release).unwrap(), "thermostat", Outcome::Refused(Refused::OtherSlot)),
        (
            second,
            sign(second, "thermostat", 3, &stranger).unwrap(),
            "thermostat",
            Outcome::Refused(Refused::UntrustedKey),
        ),
        (&changed, sign(second, "thermostat", 3, &release).unwrap(), "thermostat", Outcome::Refused(Refused::Mismatch)),
        (second, Vec::new(), "thermostat", Outcome::Refused(Refused::UnreadableSignature)),
    ];
    for (step, (program, signature, slot, expected)) in steps.iter().enumerate() {
        assert_eq!(present(&mut log, program, signature, &trusted, slot), *expected, "step {step}");
    }
    log
}

#[test]
fn installs_and_refusals_are_all_logged() {
    let log = history();
    let head = check(&log).unwrap();
    assert_eq!(head.entries, 9);
    let all: Vec<_> = entries(&log).unwrap().collect();
    assert!(all.iter().all(|entry| entry.slot == "thermostat"));
    assert_eq!(all.iter().map(|entry| entry.sequence).collect::<Vec<_>>(), [1, 2, 1, 2, 2, 3, 3, 3, 0]);
    assert_eq!(all[0].program, sha256(b"HWCIRC\0\0thermostat, release 1"));
    assert_eq!(all[0].key, test_only_key(0x42).1);
    assert_eq!(all[6].key, test_only_key(0x43).1, "the key an untrusted signature claims is recorded");
    assert_eq!((all[8].key, all[8].sequence), ([0; 32], 0), "nothing is claimed by an unreadable signature");
    assert_eq!(
        all.iter().map(|entry| entry.time).collect::<Vec<_>>(),
        (1_800_000_000..1_800_000_009).collect::<Vec<_>>()
    );
    let last = installed(&log, "thermostat").unwrap().unwrap();
    assert_eq!((last.sequence, last.program), (2, sha256(b"HWCIRC\0\0thermostat, release 2")));
    assert_eq!(installed(&log, "pump").unwrap(), None);
}

#[test]
fn slots_are_independent() {
    let (release, public) = test_only_key(0x44);
    let mut log = header([0; 32]).to_vec();
    let pump = b"HWCIRC\0\0pump".as_slice();
    let estop = b"HWCIRC\0\0estop".as_slice();
    let installed_ = Outcome::Installed;
    assert_eq!(present(&mut log, pump, &sign(pump, "pump", 5, &release).unwrap(), &[public], "pump"), installed_);
    assert_eq!(present(&mut log, estop, &sign(estop, "estop", 1, &release).unwrap(), &[public], "estop"), installed_);
    assert_eq!(installed(&log, "pump").unwrap().unwrap().sequence, 5);
    assert_eq!(installed(&log, "estop").unwrap().unwrap().sequence, 1);
}

#[test]
fn every_changed_entry_is_found() {
    let log = history();
    let head = check(&log).unwrap();
    for at in HEADER_BYTES..log.len() {
        let mut changed = log.clone();
        changed[at] ^= 1 << (at % 8);
        let entry = (at - HEADER_BYTES) / ENTRY_BYTES;
        match check(&changed) {
            // Nothing names the last entry's hash but the head: as with a cut, only the head shows the change.
            Ok(changed_head) if entry == head.entries - 1 => assert_ne!(changed_head, head, "byte {at}"),
            Err(LogError::Broken { entry: found }) => assert!(found == entry || found == entry + 1, "byte {at}"),
            Err(LogError::Malformed { entry: found, .. }) => assert_eq!(found, entry, "byte {at}"),
            other => panic!("byte {at}: {other:?}"),
        }
    }
    // The header is chained into the first entry.
    for at in 12..HEADER_BYTES {
        let mut changed = log.clone();
        changed[at] ^= 1;
        assert_eq!(check(&changed), Err(LogError::Broken { entry: 0 }), "header byte {at}");
    }
}

#[test]
fn removed_and_reordered_entries_are_found() {
    let log = history();
    let entry = |index: usize| HEADER_BYTES + index * ENTRY_BYTES;
    for index in 0..8 {
        let mut removed = log.clone();
        removed.drain(entry(index)..entry(index + 1));
        assert_eq!(check(&removed), Err(LogError::Broken { entry: index }), "removing {index}");
        let mut swapped = log.clone();
        let (a, b) = (log[entry(index)..entry(index + 1)].to_vec(), log[entry(index + 1)..entry(index + 2)].to_vec());
        swapped[entry(index)..entry(index + 1)].copy_from_slice(&b);
        swapped[entry(index + 1)..entry(index + 2)].copy_from_slice(&a);
        assert_eq!(check(&swapped), Err(LogError::Broken { entry: index }), "swapping {index}");
    }
}

/// The limit of a chain: entries cut off the end leave a consistent log. Only the head, kept elsewhere, shows it.
#[test]
fn cutting_entries_off_the_end_changes_only_the_head() {
    let log = history();
    let head = check(&log).unwrap();
    let cut = &log[..log.len() - 4 * ENTRY_BYTES];
    let shorter = check(cut).unwrap();
    assert_eq!(shorter.entries, head.entries - 4);
    assert_ne!(shorter, head);
    assert_eq!(check(&log[..HEADER_BYTES]).unwrap().entries, 0);
    // An interrupted write leaves part of an entry, which isn't a log.
    assert_eq!(check(&log[..log.len() - 1]), Err(LogError::Truncated));
}

#[test]
fn a_damaged_log_decides_nothing() {
    let (release, public) = test_only_key(0x45);
    let program = b"HWCIRC\0\0x".as_slice();
    let signature = sign(program, "x", 1, &release).unwrap();
    let mut log = header([1; 32]).to_vec();
    present(&mut log, program, &signature, &[public], "x");
    log[HEADER_BYTES + 20] ^= 1;
    assert!(admit(&log, program, &signature, &[public], "x", 0).is_err());
    assert_eq!(check(b"HWCSIG\0\0"), Err(LogError::NotALog));
    assert_eq!(check(b"HWCLOG"), Err(LogError::Truncated));
    let mut version = header([0; 32]);
    version[8] = 2;
    assert_eq!(check(&version), Err(LogError::UnsupportedVersion(2)));
}

/// The format, pinned: tests/independent_check.py recomputes every hash of this log with Python's hashlib.
#[test]
fn format_is_pinned() {
    let log = history();
    let expected = include_str!("data/pinned.log.hex").trim();
    assert_eq!(log.iter().map(|byte| format!("{byte:02x}")).collect::<String>(), expected);
    let Head { hash, .. } = check(&log).unwrap();
    assert_eq!(
        hash.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
        include_str!("data/pinned.head.hex").trim()
    );
}
