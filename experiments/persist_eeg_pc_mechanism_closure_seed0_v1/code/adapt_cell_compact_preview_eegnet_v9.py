"""Create compact-builder V9 from V8, preserving V7/V8 failure artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE_SHA256 = "8123c3db4309d24fc9f7d3e46aaa4e32141c464745932f9a6c8ddb34b5287108"
REPLACEMENTS = (
    ('eegnet_{TASK.lower()}_fold{FOLD}_seed0_v8', 'eegnet_{TASK.lower()}_fold{FOLD}_seed0_v9', 1),
    ('if len(mechanism) != 6 * 52:', 'if len(mechanism) != len(schedule_epochs) * 52:', 1),
)


def adapt_text(source: str) -> str:
    for old, new, count in REPLACEMENTS:
        actual = source.count(old)
        if actual != count:
            raise RuntimeError(f"locked V9 compact-builder replacement mismatch: {old!r}: {actual} != {count}")
        source = source.replace(old, new)
    if 'fold{FOLD}_seed0_v8' in source or 'len(mechanism) != 6 * 52' in source:
        raise RuntimeError("stale V8 output/coverage assumptions remain")
    compile(source, "<build_cell_compact_preview_eegnet_v9>", "exec")
    return source


def main() -> None:
    code = Path(__file__).resolve().parent
    base = code / "build_cell_compact_preview_eegnet_v8.py"
    output = code / "build_cell_compact_preview_eegnet_v9.py"
    raw = base.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BASE_SHA256:
        raise RuntimeError(f"pinned compact V8 SHA mismatch: {digest}")
    if output.exists():
        raise RuntimeError("V9 compact builder exists; preserve and inspect")
    transformed = adapt_text(raw.decode("utf-8"))
    output.write_text(transformed, encoding="utf-8", newline="\n")
    print(f"COMPACT_BUILDER_V9_CREATED base_sha256={digest} output_sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
