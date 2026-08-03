import hashlib
from pathlib import Path


def test_decision_hash_changes_before_future_gate(tmp_path):
    path=tmp_path/"decision"; path.write_bytes(b"U-only")
    frozen=hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen==hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"changed")
    assert frozen!=hashlib.sha256(path.read_bytes()).hexdigest()
