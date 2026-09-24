"""Create the deduplicated-schedule compact builder from the pinned V7 source."""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE_SHA256 = "0d498e3b9d3c05dd1fa1e98e45049a801cf627c011169467c5a64ea8f0c6b9a7"
REPLACEMENTS = (
    ('eegnet_{TASK.lower()}_fold{FOLD}_seed0_v7', 'eegnet_{TASK.lower()}_fold{FOLD}_seed0_v8', 1),
    ('FINAL_P_BACKTRACE_V1.json', 'FINAL_P_BACKTRACE_V2.json', 3),
    ('checkpoint_mechanism_v1', 'checkpoint_mechanism_v2', 1),
    ('if set(epochs) != set(backs) or len(schedule_epochs) != 6 or len(set(schedule_epochs)) != 6:',
     'if (set(epochs) != set(backs) or len(schedule_epochs) != len(set(schedule_epochs)) or\n'
     '                schedule_epochs != [int(value) for value in backtrace["checkpoint_schedule_epochs"]]):', 1),
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def adapt_text(source: str) -> str:
    for old, new, count in REPLACEMENTS:
        actual = source.count(old)
        if actual != count:
            raise RuntimeError(f"locked compact-builder replacement count mismatch: {old!r}: {actual} != {count}")
        source = source.replace(old, new)
    if "FINAL_P_BACKTRACE_V1.json" in source or "checkpoint_mechanism_v1" in source:
        raise RuntimeError("stale compact-builder artifact paths remain")
    compile(source, "<build_cell_compact_preview_eegnet_v8>", "exec")
    return source


def main() -> None:
    code = Path(__file__).resolve().parent
    base = code / "build_cell_compact_preview_eegnet_v7.py"
    output = code / "build_cell_compact_preview_eegnet_v8.py"
    raw = base.read_bytes()
    digest = sha256(raw)
    if digest != BASE_SHA256:
        raise RuntimeError(f"pinned V7 compact-builder SHA mismatch: {digest}")
    if output.exists():
        raise RuntimeError("V8 compact-builder already exists; preserve and inspect")
    transformed = adapt_text(raw.decode("utf-8"))
    output.write_text(transformed, encoding="utf-8", newline="\n")
    print(f"COMPACT_BUILDER_V8_CREATED base_sha256={digest} output_sha256={sha256(output.read_bytes())}")


if __name__ == "__main__":
    main()
