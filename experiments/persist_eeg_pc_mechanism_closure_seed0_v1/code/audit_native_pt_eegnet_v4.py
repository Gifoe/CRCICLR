"""V4 split-domain correction for the frozen EEGNet native-P_t audit.

V3 compared the SEARCH-fold split digest to the distinct task/coupling split
digest. V4 validates each against its own frozen source and the COMPLETE
Phase-1 cell, while preserving the V3 FAIL_CLOSED JSON and recording both
implementation hashes. The transformed V3 analysis body is hash-pinned below.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

BASE_SOURCE_SHA256 = "14ade22a3391d3461fffeb0a3da43d896633b03eedc4c4d89acec90c0af72d5a"
PATCH_ID = "V4_SPLIT_DOMAIN_IDENTITY_VALIDATION_V3_FAILURE_PRESERVED"
BASE = Path(__file__).with_name("audit_native_pt_eegnet_v3.py")
HERE = Path(__file__).resolve()


def transformed_source() -> str:
    raw = BASE.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != BASE_SOURCE_SHA256:
        raise RuntimeError(f"pinned V3 source SHA mismatch: {actual}")
    source = raw.decode("utf-8")
    substitutions = [
        (
            'failure = cell / "checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.FAIL_CLOSED.json"',
            'failure = cell / "checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.V4.FAIL_CLOSED.json"',
            1,
        ),
        (
            '        if frozen_recipe != record["recipe"] or data["split_sha256"] != record["split_sha256"]:\n'
            '            raise RuntimeError("frozen recipe or split mismatch")',
            '        historical = json.loads(C.cell_path("EEGNet", task, fold).read_text(encoding="utf-8"))\n'
            '        frozen_task_split = historical.get("split_sha256")\n'
            '        if (frozen_recipe != record["recipe"] or\n'
            '                data["split_sha256"] != frozen_task_split or\n'
            '                source["hashes"]["split_sha256"] != frozen_task_split):\n'
            '            raise RuntimeError("frozen recipe or Phase-1 task split mismatch")',
            1,
        ),
        (
            '            "correction_provenance": "V3_TERMINAL_PATH_FIX_V1_V2_FAILURES_PRESERVED",',
            '            "correction_provenance": PATCH_ID,\n'
            '            "split_hash_semantics": "SEARCH record split is checked against SEARCH data; task split is checked against COMPLETE Phase-1 cell and prior coupling hashes",\n'
            '            "superseded_v3_fail_closed_sha256": C.digest(cell / "checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.FAIL_CLOSED.json") if (cell / "checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.FAIL_CLOSED.json").exists() else None,',
            1,
        ),
        (
            '"implementation_sha256": C.digest(Path(__file__)),',
            '"implementation_sha256": C.digest(Path(__file__)),\n'
            '            "base_implementation_sha256": BASE_SOURCE_SHA256,\n'
            '            "implementation_patch_id": PATCH_ID,',
            2,
        ),
        ('NATIVE_PT_V3_COMPLETE', 'NATIVE_PT_V4_COMPLETE', 1),
    ]
    for old, new, expected in substitutions:
        count = source.count(old)
        if count != expected:
            raise RuntimeError(f"V4 patch anchor count mismatch: expected {expected}, found {count}")
        source = source.replace(old, new)
    compile(source, str(HERE), "exec")
    return source


def main() -> None:
    namespace = {
        "__name__": "__main__",
        "__file__": str(HERE),
        "BASE_SOURCE_SHA256": BASE_SOURCE_SHA256,
        "PATCH_ID": PATCH_ID,
    }
    exec(compile(transformed_source(), str(HERE), "exec"), namespace)


if __name__ == "__main__":
    main()
