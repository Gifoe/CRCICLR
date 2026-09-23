"""Complete the fixed-stage G1 signature using separately audited TRAIN controls.

V1 and each random draw remain immutable. V2 is a derived, explicitly
provenanced correction that includes random excess in S1/S2 mechanism drift.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("MECHANISM_RUNTIME", str(ROOT.parent / "pc_mechanism_closure_runtime")))
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from audit_subject_session_eegnet_v2 import (  # noqa: E402
    SIGNATURE_COMPONENTS, bootstrap_correlation,
)
from run_mediation_cell_v1 import (  # noqa: E402
    verified_context, AMENDMENT_SHA, ANALYSIS_SHA, ORIGINAL_IMPL_SHA,
)

MODEL, TASK, FOLD = "EEGNet", "OpenBMI_MI", 0
STAGE = "depth_point_elu_pool2"
BASE_SOURCE_SHA = "463861faddff7b0c72e473eacbc7f3d8fb47c19b720e1d0f734e725207bf167d"
DRAW_SOURCE_SHA = "d7d055dd5a03c17f07d5ae58ff436ff44af74a837e1e9e2f68162b2c481ef986"
COMPONENTS = (*SIGNATURE_COMPONENTS, "interaction_random_excess")


def main() -> None:
    global TASK, FOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_SSVEP"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--base-sha", required=True)
    args = parser.parse_args()
    TASK, FOLD = args.task, args.fold
    base_sha = args.base_sha.lower()
    cell = RUNTIME / "cells" / MODEL.lower() / TASK.lower() / f"fold{FOLD}_seed0"
    result = cell / "SUBJECT_SESSION_AUDIT_V2.json"
    failure = cell / "SUBJECT_SESSION_AUDIT_V2.FAIL_CLOSED.json"
    if result.exists() or failure.exists():
        raise RuntimeError("derived subject/session V2 has terminal evidence")
    try:
        gate_path, _, _, protocol_sha = verified_context(MODEL, TASK, FOLD)
        base_path = cell / "SUBJECT_SESSION_AUDIT_V1.json"
        if C.digest(base_path) != base_sha:
            raise RuntimeError("immutable V1 subject/session evidence SHA mismatch")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        if (base["status"] != "SUBJECT_SESSION_AUDIT_COMPLETE_POST_OUTCOME_DESCRIPTIVE" or
                base["implementation_sha256"] != BASE_SOURCE_SHA or
                base["upstream_protocol_sha256"] != protocol_sha or
                base["upstream_implementation_sha256"] != ORIGINAL_IMPL_SHA or
                base["analysis_lock_sha256"] != ANALYSIS_SHA or
                base["final_heldout_accessed"] is not False or
                base["signature_stage"] != STAGE):
            raise RuntimeError("immutable V1 identity/scope mismatch")
        stage_path = cell / "directional_v4" / f"{STAGE}.json"
        stage_sha = C.digest(stage_path)
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage["status"] != "STAGE_COMPLETE" or len(stage["random_controls"]) != 20:
            raise RuntimeError("frozen directional controls missing")
        train_random: dict[tuple[str, int], list[float]] = defaultdict(list)
        draw_hashes = {}
        for draw in range(20):
            path = cell / "train_random_subject_v1" / f"draw_{draw:02d}.json"
            fail = cell / "train_random_subject_v1" / f"draw_{draw:02d}.FAIL_CLOSED.json"
            if fail.exists() or not path.exists():
                raise RuntimeError(f"random TRAIN draw {draw} incomplete/failed")
            evidence = json.loads(path.read_text(encoding="utf-8"))
            frozen = stage["random_controls"][draw]
            if (evidence["status"] != "TRAIN_RANDOM_SUBJECT_DRAW_COMPLETE" or
                    int(evidence["draw"]) != draw or int(frozen["draw"]) != draw or
                    evidence["implementation_sha256"] != DRAW_SOURCE_SHA or
                    evidence["directional_stage_sha256"] != stage_sha or
                    evidence["final_subset_sha256"] != frozen["final_subset_sha256"] or
                    evidence["mapping_sha256"] != frozen["mapping_sha256"] or
                    evidence["frozen_train_top10"] != frozen["train_top10"] or
                    evidence["analysis_lock_sha256"] != ANALYSIS_SHA or
                    evidence["projector_amendment_sha256"] != AMENDMENT_SHA or
                    evidence["upstream_protocol_sha256"] != protocol_sha or
                    evidence["upstream_gate_sha256"] != C.digest(gate_path) or
                    evidence["outer_development_accessed"] is not False or
                    evidence["final_heldout_accessed"] is not False or
                    len(evidence["train_subject_session_rows"]) != 52):
                raise RuntimeError(f"random TRAIN draw {draw} identity/scope mismatch")
            seen = set()
            for row in evidence["train_subject_session_rows"]:
                key = (str(row["subject"]), int(row["session"]))
                if key in seen or row["split"] != "TRAIN" or int(row["direction_count"]) != 10:
                    raise RuntimeError(f"random TRAIN draw {draw} duplicate/invalid group")
                seen.add(key)
                train_random[key].append(float(row["random_top10_interaction_norm"]))
            draw_hashes[str(draw)] = C.digest(path)
        if len(train_random) != 52 or any(len(values) != 20 for values in train_random.values()):
            raise RuntimeError("TRAIN random-control group/draw coverage incomplete")
        signatures = base["subject_session_signature_rows"]
        if len(signatures) != 68:
            raise RuntimeError("V1 biological subject/session coverage incomplete")
        by_key = {}
        for row in signatures:
            key = (str(row["split"]), str(row["subject"]), int(row["session"]))
            if key in by_key:
                raise RuntimeError("duplicate V1 biological subject/session")
            by_key[key] = row
            if key[0] == "TRAIN":
                values = train_random[(key[1], key[2])]
                if len(values) != 20:
                    raise RuntimeError(f"TRAIN random draw count invalid for {key}")
                row["rank_matched_random_interaction_norm"] = float(np.mean(values))
                row["interaction_random_excess"] = (
                    float(row["protected_top10_interaction_norm"]) -
                    row["rank_matched_random_interaction_norm"])
                row["rank_matched_random_draws"] = 20
                row.pop("random_excess_missing_reason", None)
            elif key[0] == "OUTER_DEVELOPMENT":
                if row["interaction_random_excess"] is None:
                    raise RuntimeError("V1 outer random excess missing")
                row["rank_matched_random_draws"] = 20
            else:
                raise RuntimeError("illegal V1 split")
        if len([key for key in by_key if key[0] == "TRAIN"]) != 52:
            raise RuntimeError("V2 TRAIN subject/session coverage incomplete")
        vectors = {key: np.asarray([row[name] for name in COMPONENTS], np.float64)
                   for key, row in by_key.items()}
        if any(not np.isfinite(value).all() for value in vectors.values()):
            raise RuntimeError("nonfinite corrected mechanism signature")
        train = np.stack([value for key, value in vectors.items() if key[0] == "TRAIN"])
        mu, sd = train.mean(0), np.maximum(train.std(0), 1e-6)
        drift = base["subject_session_drift_rows"]
        if len(drift) != 34:
            raise RuntimeError("V1 paired drift coverage incomplete")
        for row in drift:
            if row["status"] != "PAIRED":
                raise RuntimeError("V1 unpaired subject session")
            split, subject = str(row["split"]), str(row["subject"])
            a, b = ((vectors[(split, subject, session)] - mu) / sd for session in (1, 2))
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            row["mechanism_cosine"] = float(np.dot(a, b) / denominator) if denominator > 1e-12 else None
            row["mechanism_standardized_euclidean"] = float(np.sqrt(np.mean((b - a) ** 2)))
            row["mechanism_component_drift"] = dict(zip(COMPONENTS, map(float, b - a)))
            row["random_excess_included_in_drift"] = True
            row.pop("random_excess_excluded_from_drift", None)
        future = {str(row["subject"]): row for row in base["future_subject_performance"]}
        outer = {str(row["subject"]): row for row in drift if row["split"] == "OUTER_DEVELOPMENT"}
        if len(future) != 8 or set(future) != set(outer):
            raise RuntimeError("future biological subject alignment invalid")
        subjects = sorted(future, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
        relation = {}
        for measure in ("mechanism_standardized_euclidean", "P_standardized_euclidean",
                        "C_standardized_euclidean", "H_standardized_euclidean"):
            x = np.asarray([outer[subject][measure] for subject in subjects], np.float64)
            for target in ("future_BA", "future_CE", "future_error_rate"):
                y = np.asarray([future[subject][target] for subject in subjects], np.float64)
                relation[f"{measure}__{target}"] = bootstrap_correlation(x, y, f"{measure}/{target}")
        C.write_json(result, {
            "status": "SUBJECT_SESSION_AUDIT_COMPLETE_POST_OUTCOME_DESCRIPTIVE",
            "model": MODEL, "task": TASK, "fold": FOLD, "seed": 0,
            "signature_stage": STAGE, "signature_component_count": len(COMPONENTS),
            "drift_components": COMPONENTS, "subject_session_signature_rows": signatures,
            "subject_session_drift_rows": drift,
            "future_subject_performance": list(future.values()),
            "drift_performance_descriptive_correlations": relation,
            "standardization_unit": "TRAIN_BIOLOGICAL_SUBJECT_SESSION",
            "bootstrap_unit": "BIOLOGICAL_SUBJECT", "bootstrap_resamples": 2000,
            "outer_scope": "EVALUATION_ONLY_POST_OUTCOME_DIAGNOSTIC_NOT_PROSPECTIVE",
            "provenance": "V1_PLUS_EXACT_FROZEN_TOP10_20_TRAIN_RANDOM_SUBJECT_DRAWS",
            "base_subject_session_audit_v1_sha256": base_sha,
            "directional_v4_stage_sha256": stage_sha,
            "train_random_draw_sha256": draw_hashes,
            "upstream_gate_sha256": C.digest(gate_path),
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "projector_amendment_sha256": AMENDMENT_SHA,
            "analysis_lock_sha256": ANALYSIS_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "outer_development_accessed": True,
            "final_heldout_accessed": False,
        })
        print("SUBJECT_SESSION_AUDIT_V2_COMPLETE", len(signatures), len(drift), flush=True)
    except Exception as exc:
        C.write_json(failure, {"status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}",
                               "implementation_sha256": C.digest(Path(__file__)),
                               "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
