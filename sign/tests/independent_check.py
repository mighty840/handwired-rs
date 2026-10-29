"""Builds the pinned signature file and the pinned version log from the formats documented in src/lib.rs and
src/log.rs, with other implementations of Ed25519 (Python's `cryptography`, OpenSSL underneath) and SHA-256
(hashlib), and checks they equal data/pinned.sig.hex, data/pinned.log.hex and data/pinned.head.hex, which the Rust
tests pin. Run: python3 -I tests/independent_check.py (from sign/); --write writes the log files instead."""
import hashlib
import os
import struct
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

seed, program, slot, sequence = bytes([0x42]) * 32, b'HWCIRC\0\0test program', b'estop', 258
key = Ed25519PrivateKey.from_private_bytes(seed)
public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
message = b'handwired-rs signed program v1\0' + bytes([len(slot)]) + slot + struct.pack('<Q', sequence) + program
signature = key.sign(message)  # RFC 8032 signing is deterministic
key.public_key().verify(signature, message)
file = b'HWCSIG\0\0' + struct.pack('<I', 1) + public + struct.pack('<Q', sequence) + bytes([len(slot)]) + slot + signature
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
pinned = open(os.path.join(DATA, 'pinned.sig.hex')).read().strip()
if file.hex() != pinned:
    sys.exit(f'differs:\n  python {file.hex()}\n  pinned {pinned}')
print('pinned signature matches an independent Ed25519 implementation')


# The version log of tests/log.rs `history()`: one slot, two releases, and each kind of refusal. An entry holds what
# the signature claims (sequence, key), not the signature, so it is built from the keys alone.
def public_of(byte):
    key = Ed25519PrivateKey.from_private_bytes(bytes([byte]) * 32)
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


release, stranger, nobody = public_of(0x42), public_of(0x43), bytes(32)
first, second = b'HWCIRC\0\0thermostat, release 1', b'HWCIRC\0\0thermostat, release 2'
other = b'HWCIRC\0\0thermostat, other bytes'
changed = bytearray(second)
changed[8] ^= 1
INSTALLED, OLDER, REUSED, OTHER_SLOT, UNTRUSTED, MISMATCH, UNREADABLE = 0, 4, 5, 3, 1, 2, 6
steps = [  # program, claimed sequence, claimed key, outcome
    (first, 1, release, INSTALLED), (second, 2, release, INSTALLED), (first, 1, release, OLDER),
    (other, 2, release, REUSED), (second, 2, release, INSTALLED), (second, 3, release, OTHER_SLOT),
    (second, 3, stranger, UNTRUSTED), (bytes(changed), 3, release, MISMATCH), (second, 0, nobody, UNREADABLE),
]
header = b'HWCLOG\0\0' + struct.pack('<I', 1) + bytes([7]) * 32
log = header
previous = hashlib.sha256(b'handwired-rs version log v1\0' + header).digest()
for index, (program, sequence, key, outcome) in enumerate(steps):
    slot = b'thermostat'
    entry = (struct.pack('<Q', 1_800_000_000 + index) + bytes([outcome, len(slot)]) + slot.ljust(64, b'\0')
             + struct.pack('<Q', sequence) + hashlib.sha256(program).digest() + key + previous)
    assert len(entry) == 178
    log += entry
    previous = hashlib.sha256(b'handwired-rs version log entry v1\0' + entry).digest()
if '--write' in sys.argv:
    open(os.path.join(DATA, 'pinned.log.hex'), 'w').write(log.hex() + '\n')
    open(os.path.join(DATA, 'pinned.head.hex'), 'w').write(previous.hex() + '\n')
for name, value in (('pinned.log.hex', log), ('pinned.head.hex', previous)):
    if open(os.path.join(DATA, name)).read().strip() != value.hex():
        sys.exit(f'{name} differs from the log built here')
print('pinned version log matches one built with hashlib from the documented layout')
