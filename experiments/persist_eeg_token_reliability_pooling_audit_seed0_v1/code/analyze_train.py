"""TRAIN-only reliability, null, and split-half audits from frozen token caches.

This stage deliberately has no OUTER path. It does not train a decoder or
select an aggregation method; it creates source-only mechanism evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from token_scores import (cosine_reliability, decomposition, directions,
                          label_null, magnitude_reliability, pairing_null,
                          reproducibility, spread)


RUNTIME = Path(os.environ["TOKEN_AUDIT_RUNTIME"]).resolve()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(fold: int):
    source = RUNTIME / f"fold{fold}" / "train"
    manifest_path = source / "EXTRACTION_PROVENANCE.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("TRAIN extraction is not complete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["role"] != "train" or manifest["fold"] != fold or manifest["formal_final_heldout_eeg_reads"] != 0:
        raise RuntimeError("TRAIN extraction role/provenance mismatch")
    for name, expected in manifest["files"].items():
        if sha(source / name) != expected:
            raise RuntimeError(f"TRAIN extraction artifact hash mismatch: {name}")
    z = np.load(source / "tokens.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(source / "labels.npy", allow_pickle=False)
    subjects = np.load(source / "subjects.npy", allow_pickle=False)
    sessions = np.load(source / "sessions.npy", allow_pickle=False)
    if tuple(z.shape[1:]) != (248, 200):
        raise RuntimeError("unexpected CBraMod token grid")
    ordered = sorted(set(subjects.tolist()))
    d, counts = directions(z, y, subjects, sessions, ordered)
    r = cosine_reliability(d)
    magnitude = magnitude_reliability(d)
    r_z = (r - r.mean()) / max(r.std(), 1e-12)
    m_z = (magnitude - magnitude.mean()) / max(magnitude.std(), 1e-12)
    combined_descriptive = r_z + m_z
    output = RUNTIME / f"fold{fold}" / "reliability_v1"
    output.mkdir(parents=False, exist_ok=False)
    index = pd.DataFrame([{"token_index": k, "channel_index": k // 4, "temporal_patch_index": k % 4}
                          for k in range(248)])
    table = index.assign(fold=fold, R_cos=r, R_magnitude=magnitude,
                         R_combined_descriptive=combined_descriptive,
                         train_subject_count=len(ordered))
    table.to_csv(output / "TOKEN_RELIABILITY.csv", index=False)
    null = pairing_null(d, 500, fold=fold)
    null += label_null(z, y, subjects, sessions, ordered, 500, fold=fold)
    pd.DataFrame(null).to_csv(output / "TOKEN_RELIABILITY_NULL.csv", index=False)
    pd.DataFrame(reproducibility(d, 200, fold=fold)).to_csv(
        output / "TOKEN_RELIABILITY_REPRODUCIBILITY.csv", index=False)
    pd.DataFrame([{"fold": fold, **decomposition(r, 62, 4)}]).to_csv(
        output / "TOKEN_STRUCTURE_DECOMPOSITION.csv", index=False)
    summary = {"fold": fold, "status": "COMPLETE_TRAIN_RELIABILITY_ONLY",
               "source_extraction_sha256": sha(manifest_path), "train_subject_count": len(ordered),
               "class_counts_by_subject_session": counts.tolist(),
               "real": spread(r), "pairing_null_draws": 500, "label_null_draws": 500,
               "split_half_draws": 200, "formal_final_heldout_eeg_reads": 0,
               "files": {name: sha(output / name) for name in (
                   "TOKEN_RELIABILITY.csv", "TOKEN_RELIABILITY_NULL.csv",
                   "TOKEN_RELIABILITY_REPRODUCIBILITY.csv", "TOKEN_STRUCTURE_DECOMPOSITION.csv")}}
    (output / "TRAIN_RELIABILITY_SEAL.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fold": fold, "status": summary["status"], "spread": summary["real"]["std"],
                      "gap": summary["real"]["top_bottom_quartile_gap"],
                      "seal_sha256": sha(output / "TRAIN_RELIABILITY_SEAL.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    run(parser.parse_args().fold)
