from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "persist_eeg_p3closure_p4"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


integrity = json.loads((BASE / "INTEGRITY.json").read_text(encoding="utf-8"))
hash_errors = []
for relative, expected in integrity["files_sha256"].items():
    path = BASE / relative
    if not path.exists() or digest(path) != expected:
        hash_errors.append(relative)
p3_freeze = json.loads((BASE / "p3_closure" / "P3_FROZEN.json").read_text(encoding="utf-8"))
p3_hash_errors = []
for name, expected in p3_freeze["files_sha256"].items():
    path = BASE / "p3_closure" / name
    if not path.exists() or digest(path) != expected:
        p3_hash_errors.append(name)
development = []
for version in ("V0", "V1", "V2", "V3"):
    result = json.loads(
        (BASE / "p4" / "development" / version / "fold-0" / "seed-0" / "DEVELOPMENT_RESULT.json").read_text(encoding="utf-8")
    )
    development.append(
        {
            "version": version,
            "status": result["status"],
            "held_out_test_used": result["held_out_test_used"],
            "passed_checks": sum(bool(value) for value in result["checks"].values()),
            "total_checks": len(result["checks"]),
        }
    )
test_markers = [str(path.relative_to(BASE)) for path in BASE.glob("**/TEST_ACCESS_STARTED.json")]
test_markers += [str(path.relative_to(BASE)) for path in BASE.glob("**/TEST_COMPLETE.json")]
lock_exists = (BASE / "p4" / "P4_LOCKED_METHOD.json").exists()
report = json.loads((BASE / "p4" / "P4_FINAL_REPORT.json").read_text(encoding="utf-8"))
complete = json.loads((BASE / "COMPLETE.json").read_text(encoding="utf-8"))
verification = {
    "status": "PASS" if not hash_errors and not p3_hash_errors and not test_markers and not lock_exists else "FAIL",
    "integrity_hash_errors": hash_errors,
    "p3_freeze_hash_errors": p3_hash_errors,
    "test_access_markers": test_markers,
    "locked_method_exists": lock_exists,
    "development": development,
    "final_p4_status": report["status"],
    "complete": complete,
}
print(json.dumps(verification, indent=2))
if verification["status"] != "PASS":
    raise SystemExit(2)
