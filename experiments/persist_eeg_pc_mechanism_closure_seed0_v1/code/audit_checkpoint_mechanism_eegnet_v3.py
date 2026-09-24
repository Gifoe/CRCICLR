"""Versioned artifact-path adapter for the frozen EEGNet checkpoint audit.

The analytical source remains the hash-locked v2 implementation. This adapter
redirects only its input from the completed V2 final-P backtrace and its output
to a fresh v2 mechanism-evidence directory. No metric, data, fit, cap, or
projector definition is modified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE_NAME = "audit_checkpoint_mechanism_eegnet_v2.py"
BASE_SHA256 = "097f6c25e200392359d630add0bfa7c36431bc3813234ade48771e481096a520"


def transform_source(source: str) -> str:
    replacements = (
        ('cell / "checkpoint_mechanism_v1"', 'cell / "checkpoint_mechanism_v2"', 2),
        ('cell / "FINAL_P_BACKTRACE_V1.json"', 'cell / "FINAL_P_BACKTRACE_V2.json"', 1),
        ('CHECKPOINT_MECHANISM_V1_COMPLETE', 'CHECKPOINT_MECHANISM_V2_COMPLETE', 1),
    )
    for old, new, expected_count in replacements:
        if source.count(old) != expected_count:
            raise RuntimeError(f"locked source transformation mismatch for {old!r}")
        source = source.replace(old, new)
    return source


def main() -> None:
    base = Path(__file__).with_name(BASE_NAME)
    raw = base.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASE_SHA256:
        raise RuntimeError("locked checkpoint-mechanism v2 source SHA mismatch")
    source = transform_source(raw.decode("utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(Path(__file__).resolve())}
    exec(compile(source, str(Path(__file__).resolve()), "exec"), namespace)


if __name__ == "__main__":
    main()
