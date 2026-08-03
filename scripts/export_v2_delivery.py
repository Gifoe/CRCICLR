#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
SOURCE = ROOT / "outputs/v2_joint_certified"
DESTINATION = ROOT / "repo/delivery/v2_joint_certified"
INCLUDE_SUFFIXES = {".md", ".csv", ".json"}
EXCLUDE_PARTS = {"logs", "state", "hashes", "decisions", "models", "u_states", "counterfactuals", "context_features", "figures"}
MAX_BYTES = 5_000_000


def main() -> None:
    copied = []
    for source in SOURCE.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in INCLUDE_SUFFIXES:
            continue
        relative = source.relative_to(SOURCE)
        if set(relative.parts) & EXCLUDE_PARTS or source.stat().st_size > MAX_BYTES:
            continue
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({"path": str(relative), "bytes": source.stat().st_size,
                       "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    payload = {"source_root": str(SOURCE), "files": copied,
               "excluded": "EEG data, token embeddings, checkpoints, joblib models, parquet subject records, decisions/states/logs, and files >5 MB"}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "DELIVERY_MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (DESTINATION / "README.md").write_text(
        "# HSC-TTA v2 delivery\n\nThis directory mirrors the small reports, CSV summaries, and provenance JSON produced on the AutoDL server. It deliberately excludes EEG data, subject-level parquet artifacts, token caches, checkpoints, serialized predictors, action states, and logs. Recreate those artifacts with `scripts/run_v2_full_development.sh`.\n",
        encoding="utf-8")
    print(f"copied {len(copied)} small artifacts to {DESTINATION}")


if __name__ == "__main__":
    main()
