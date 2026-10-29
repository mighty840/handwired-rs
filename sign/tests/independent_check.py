"""Builds the pinned signature file from the format documented in src/lib.rs with another Ed25519 implementation
(Python's `cryptography`, OpenSSL underneath) and checks it equals data/pinned.sig.hex, which the Rust test pins.
Run: python3 -I tests/independent_check.py (from sign/)."""
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
pinned = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pinned.sig.hex')).read().strip()
if file.hex() != pinned:
    sys.exit(f'differs:\n  python {file.hex()}\n  pinned {pinned}')
print('pinned signature matches an independent Ed25519 implementation')
