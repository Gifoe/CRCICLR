from pathlib import Path

from hsc_tta.v3.provenance import sha256_file


def test_streaming_sha256(tmp_path: Path):
    path = tmp_path / "artifact.bin"; path.write_bytes(b"ProbeCert-V3")
    assert sha256_file(path) == "25ac5829754c1cde67fcff6c40f16162781e2af5a4ab88e14a8dbd01388c29c8"
