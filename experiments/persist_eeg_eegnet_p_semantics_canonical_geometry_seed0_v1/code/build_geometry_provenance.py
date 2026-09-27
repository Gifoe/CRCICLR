"""Fail-closed consolidation of the five fresh TRAIN-only geometry records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = ROOT / "protocol"
    lock = json.loads((protocol / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    records = []
    for fold in lock["folds"]:
        path = protocol / f"fold{fold}_geometry_source.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["fold"] != fold:
            raise RuntimeError(f"fold mismatch: {path}")
        if record["checkpoint_sha256"] != lock["checkpoints_sha256"][str(fold)]:
            raise RuntimeError(f"checkpoint mismatch: {path}")
        if record["checkpoint_normalizer_sha256"] != lock["canonical_normalizer_sha256"][str(fold)]:
            raise RuntimeError(f"normalizer mismatch: {path}")
        if record["checkpoint_split_sha256"] != lock["checkpoint_split_sha256"]:
            raise RuntimeError(f"split mismatch: {path}")
        if record["population"] != "TRAIN_GEOMETRY only" or len(record["fit_subjects"]) != 26:
            raise RuntimeError(f"fitting population mismatch: {path}")
        for field in ("checkpoint_validation_eeg_reads", "outer_development_eeg_reads", "final_heldout_eeg_reads"):
            if record[field] != 0:
                raise RuntimeError(f"forbidden geometry read: {field} {path}")
        for field in ("checkpoint_validation_used_for_fit", "outer_development_used_for_fit", "final_heldout_accessed"):
            if record[field] is not False:
                raise RuntimeError(f"forbidden geometry fit: {field} {path}")
        if set(record["stage_geometry_sha256"]) != set(lock["analysis"]["stages"]):
            raise RuntimeError(f"stage geometry incomplete: {path}")
        if len(record["model_state_sha256_before_after"]) != 64:
            raise RuntimeError(f"missing model-state checksum: {path}")
        records.append({"fold": fold, "source_file": path.name, "source_sha256": sha256(path), **record})
    output = {
        "schema": "P_SEMANTICS_FRESH_INNER_TRAIN_GEOMETRY_PROVENANCE_V1",
        "protocol_lock_sha256": sha256(protocol / "PROTOCOL_LOCK.json"),
        "distinction": "Frozen checkpoint provenance and this fresh TRAIN-only P/PathFit geometry are separate objects.",
        "folds": records,
        "final_heldout_eeg_reads": 0,
    }
    destination = protocol / "GEOMETRY_PROVENANCE.json"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(sha256(destination))


if __name__ == "__main__":
    main()
