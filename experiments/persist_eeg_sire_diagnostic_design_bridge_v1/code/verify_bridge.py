#!/usr/bin/env python3
"""Independent row-count, provenance, freeze and policy-estimator checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from source_audit import EXP, sha, write
from prepare_and_freeze_selection import CANDIDATES, select
from evaluate_frozen_selection import check_freeze, policy_results


def main():
    a = EXP / "outputs/subspace_decomposition"
    b = EXP / "outputs/prospective_selection"
    source = json.loads((EXP / "protocol/SOURCE_MANIFEST.json").read_text())
    freeze = check_freeze()
    cells = pd.read_csv(a / "CELL_RESULTS.csv")
    if len(cells) != 40 or cells[["task", "model", "fold", "seed"]].drop_duplicates().shape[0] != 40:
        raise RuntimeError("Part-A cell count/identity failed")
    if not (pd.read_csv(a / "REPLAY_AUDIT.csv").status == "PASS").all():
        raise RuntimeError("Table-12 replay failed")
    checkpoint = pd.read_csv(a / "CHECKPOINT_AUDIT.csv")
    if len(checkpoint) != 40 or not (checkpoint.canonical_session_replay == "PASS").all():
        raise RuntimeError("Part-A basis/checkpoint audit failed")
    manifest_a = {(e["task"], e["variant"], e["fold"], e["seed"]): e for e in source["entries"] if e["part"] == "A"}
    for r in checkpoint.itertuples(index=False):
        src = manifest_a[(r.task, r.model, int(r.fold), int(r.seed))]
        if r.checkpoint_sha256 != src["checkpoint_sha256"] or r.normalizer_sha256 != src["normalizer_sha256"]:
            raise RuntimeError("Part-A source hash mismatch")
    dev = pd.read_csv(b / "DEVELOPMENT_DIAGNOSTICS.csv", dtype={"subject_id": str})
    if len(dev) != 360 or dev[["fold", "seed", "variant", "subject_id"]].drop_duplicates().shape[0] != 360:
        raise RuntimeError("development profile incomplete")
    expected = {(e["fold"], e["seed"], e["variant"]): e for e in source["entries"] if e["part"] == "B"}
    for (fold, seed, variant), group in dev.groupby(["fold", "seed", "variant"]):
        src = expected[(int(fold), int(seed), variant)]
        if set(group.subject_id) != set(src["subject_split"]["inner_val_subjects"]):
            raise RuntimeError("development diagnostic subject leakage")
        if group.checkpoint_sha256.nunique() != 1 or group.checkpoint_sha256.iloc[0] != src["checkpoint_sha256"]:
            raise RuntimeError("development checkpoint drift")
    profile = pd.read_csv(b / "PER_FOLD_CANDIDATE_PROFILE.csv")
    selected = select(profile)
    for new, old in zip(selected, freeze["choices"]):
        for field in ("fold", "diagnostic_choice", "validation_BA_choice", "reason", "admissible_candidates"):
            if new[field] != old[field]:
                raise RuntimeError(f"frozen choice does not replay: {field}")
    outer = pd.read_csv(b / "OUTER_SUBJECT_RESULTS.csv", dtype={"subject_id": str})
    if len(outer) != 960 or outer.subject_id.nunique() != 40:
        raise RuntimeError("outer evaluation count failed")
    if outer[["fold", "seed", "variant", "subject_id", "session"]].drop_duplicates().shape[0] != 960:
        raise RuntimeError("duplicate outer cell")
    final_heldout = set("4,12,13,17,18,24,25,29,36,37,39,42,51,54".split(","))
    if set(outer.subject_id) & final_heldout:
        raise RuntimeError("final-heldout leaked into Part B")
    for (fold, seed, variant), group in outer.groupby(["fold", "seed", "variant"]):
        src = expected[(int(fold), int(seed), variant)]
        if set(group.subject_id) != set(src["subject_split"]["outer_dev_subjects"]):
            raise RuntimeError("outer subject mismatch")
        if group.checkpoint_sha256.nunique() != 1 or group.checkpoint_sha256.iloc[0] != src["checkpoint_sha256"]:
            raise RuntimeError("outer checkpoint drift")
        if group.groupby("subject_id").session.nunique().ne(2).any():
            raise RuntimeError("outer session pair missing")
    _, recomputed, _ = policy_results(outer, freeze)
    stored = pd.read_csv(b / "POLICY_SUBJECT_RESULTS.csv", dtype={"subject_id": str})
    keys = ["fold", "subject_id", "policy"]
    joined = recomputed.merge(stored, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
    if len(joined) != 200:
        raise RuntimeError("policy subject rows incomplete")
    for metric in ("future_BA", "future_macro_F1", "WSBA"):
        if not np.allclose(joined[f"{metric}_new"], joined[f"{metric}_old"], atol=1e-12):
            raise RuntimeError(f"policy estimator replay failed: {metric}")
    if (b / "SELECTION_FREEZE.json").stat().st_mtime_ns >= (b / "OUTER_SUBJECT_RESULTS.csv").stat().st_mtime_ns:
        raise RuntimeError("freeze timestamp not before outer-result timestamp")
    files = [EXP / "protocol/SOURCE_MANIFEST.json", EXP / "protocol/SELECTOR_SPEC.json",
             b / "SELECTION_FREEZE.json", b / "OUTER_SUBJECT_RESULTS.csv", a / "COMPLETION.json",
             b / "COMPLETION.json"]
    write(EXP / "outputs/INTEGRITY_AUDIT.json", {
        "status": "PASS", "part_a_cells": 40, "part_b_development_rows": 360,
        "part_b_outer_rows": 960, "outer_biological_subjects": 40,
        "full_B0_protocol_matched": False, "neural_training_runs": 0,
        "table12_replay": "PASS", "frozen_choices_replay": "PASS",
        "outer_policy_replay": "PASS", "final_heldout_leakage": False,
        "freeze_mtime_ns": (b / "SELECTION_FREEZE.json").stat().st_mtime_ns,
        "outer_result_mtime_ns": (b / "OUTER_SUBJECT_RESULTS.csv").stat().st_mtime_ns,
        "critical_sha256": {str(path): sha(path) for path in files},
    })
    print("BRIDGE_INTEGRITY_PASS", flush=True)


if __name__ == "__main__":
    main()
