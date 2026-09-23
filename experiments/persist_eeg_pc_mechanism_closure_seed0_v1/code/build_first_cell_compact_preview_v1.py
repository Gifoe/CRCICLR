"""Audit and render a first-cell preview from immutable phase-2 evidence.

This is not the final 20-cell output or a cross-model/task inference. All
source JSON is checked before compact CSVs are written. No dataset is read.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("MECHANISM_RUNTIME", str(ROOT.parent / "pc_mechanism_closure_runtime")))
sys.path.insert(0, str(ROOT / "experiments" / "persist_eeg_protected_complement_coupling_seed0_v1" / "code"))
import run_coupling as C  # noqa: E402
from run_mediation_cell_v1 import (  # noqa: E402
    AMENDMENT_SHA, ANALYSIS_SHA, ORIGINAL_IMPL_SHA,
)

MODEL, TASK, FOLD = "EEGNet", "OpenBMI_MI", 0
EPOCHS = (6, 15, 30, 45, 49, 60)
SUBJECT_V2_SHA = "77ac5f25183151a407c273b7627ddb3389068923da1134e038dfb7f2086cc45c"


def checked(path: Path, status: str, *, gate_sha: str, protocol_sha: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required evidence absent: {path.name}")
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row.get("status") != status or row.get("model") != MODEL or
            row.get("task") != TASK or int(row.get("fold", -1)) != FOLD or
            row.get("final_heldout_accessed") is not False or
            row.get("upstream_gate_sha256") != gate_sha or
            row.get("upstream_protocol_sha256") != protocol_sha or
            row.get("upstream_implementation_sha256") != ORIGINAL_IMPL_SHA):
        raise RuntimeError(f"evidence status/identity/scope invalid: {path.name}")
    return row


def avg(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def regimes(signatures: list[dict], future: list[dict], components: list[str]):
    train = [row for row in signatures if row["split"] == "TRAIN"]
    outer = [row for row in signatures if row["split"] == "OUTER_DEVELOPMENT"]
    if len(train) != 52 or len(outer) != 16:
        raise RuntimeError("subject/session signature coverage mismatch")
    a = np.asarray([[row[key] for key in components] for row in train], np.float64)
    b = np.asarray([[row[key] for key in components] for row in outer], np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("nonfinite mechanism-regime input")
    mu, sd = a.mean(0), np.maximum(a.std(0), 1e-6)
    a, b = (a - mu) / sd, (b - mu) / sd
    full = PCA(n_components=min(8, a.shape[1], len(a)), svd_solver="full").fit(a)
    rank = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.9) + 1)
    rank = min(max(rank, 1), 8)
    a, b = full.transform(a)[:, :rank], full.transform(b)[:, :rank]
    choices = []
    for k in (2, 3, 4):
        model = KMeans(n_clusters=k, n_init=20, random_state=0).fit(a)
        choices.append((float(silhouette_score(a, model.labels_)), k, model))
    score, k, model = max(choices, key=lambda item: (item[0], -item[1]))
    result = []
    for split, rows, projected in (("TRAIN", train, a), ("OUTER_DEVELOPMENT", outer, b)):
        for row, assignment in zip(rows, model.predict(projected)):
            result.append({"split": split, "subject": row["subject"],
                           "session": row["session"], "cluster": int(assignment),
                           "cluster_definition": "TRAIN_ONLY_PCA_KMEANS_SECONDARY_EXPLORATORY"})
    future_by_subject = {str(row["subject"]): row for row in future}
    for row in result:
        row["future_BA"] = (future_by_subject[str(row["subject"])]["future_BA"]
                            if row["split"] == "OUTER_DEVELOPMENT" and row["session"] == 2
                            else None)
    metadata = {"status": "SECONDARY_TRAIN_DEFINED_REGIMES", "selected_k": k,
                "train_silhouette": score, "pca_rank": rank,
                "train_explained_variance": float(np.sum(full.explained_variance_ratio_[:rank])),
                "k_candidates": {str(value): quality for quality, value, _ in choices},
                "outer_used_for_selection": False}
    return result, metadata


def main() -> None:
    cell = RUNTIME / "cells" / MODEL.lower() / TASK.lower() / f"fold{FOLD}_seed0"
    output = RUNTIME / "compact_preview" / "eegnet_openbmi_mi_fold0_seed0_v1"
    if output.exists():
        raise RuntimeError("first-cell preview path already has evidence")
    output.mkdir(parents=True)
    failure = output / "FAIL_CLOSED.json"
    try:
        gate_path = EXP / "gate" / "GATE_AUDIT.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "PASSED":
            raise RuntimeError("upstream gate not PASSED")
        gate_sha = C.digest(gate_path)
        protocol_sha = "3ddb5006bf7159b00a0c80fed31856849ccdaa69ecf5060b0161c8673327e15c"
        if (C.digest(EXP / "protocol" / "FINAL_PROJECTOR_AMENDMENT.md") != AMENDMENT_SHA or
                C.digest(EXP / "protocol" / "MECHANISM_CLOSURE_ANALYSIS_LOCK.md") != ANALYSIS_SHA):
            raise RuntimeError("frozen protocol/amendment hash mismatch")
        subject_path = cell / "SUBJECT_SESSION_AUDIT_V2.json"
        if C.digest(subject_path) != SUBJECT_V2_SHA:
            raise RuntimeError("subject/session V2 SHA mismatch")
        subject = checked(subject_path, "SUBJECT_SESSION_AUDIT_COMPLETE_POST_OUTCOME_DESCRIPTIVE",
                          gate_sha=gate_sha, protocol_sha=protocol_sha)
        med = checked(cell / "MEDIATION_V2.json",
                      "MEDIATION_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES",
                      gate_sha=gate_sha, protocol_sha=protocol_sha)
        dire = checked(cell / "DIRECTIONAL_V4.json",
                       "DIRECTIONAL_PHASE_COMPLETE_PENDING_OTHER_REQUIRED_ANALYSES",
                       gate_sha=gate_sha, protocol_sha=protocol_sha)
        endpoint = checked(cell / "REPLICA_ENDPOINT_AUDIT_V1.json",
                           "REPLICA_ENDPOINTS_AUDITED_PENDING_NATIVE_P_AND_MECHANISM",
                           gate_sha=gate_sha, protocol_sha=protocol_sha)
        backtrace = checked(cell / "FINAL_P_BACKTRACE_V1.json",
                            "FINAL_P_BACKTRACE_COMPLETE_REPLICA_ONLY",
                            gate_sha=gate_sha, protocol_sha=protocol_sha)
        replay_path = cell / "replay_v2" / "REPLAY_AUDIT.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        if replay.get("final_heldout_accessed") is not False:
            raise RuntimeError("replay heldout scope invalid")
        if not (len(med["stage_sha256"]) == len(dire["stage_sha256"]) == 5):
            raise RuntimeError("first-cell stage count mismatch")
        adjacent, long_range, directional, random = [], [], [], []
        stage_rows = {}
        for stage, expected_sha in med["stage_sha256"].items():
            p = cell / "mediation_v2" / f"{stage}.json"
            if C.digest(p) != expected_sha:
                raise RuntimeError(f"mediation stage hash mismatch: {stage}")
            rows = checked(p, "STAGE_COMPLETE", gate_sha=gate_sha, protocol_sha=protocol_sha)
            for row in rows["subject_session_rows"]:
                if row["transition_kind"] == "ADJACENT":
                    adjacent.append(row)
                elif row["transition_kind"] == "LONG_RANGE":
                    long_range.append(row)
                else:
                    raise RuntimeError("unknown mediation transition")
            stage_rows[stage] = rows
        for stage, expected_sha in dire["stage_sha256"].items():
            p = cell / "directional_v4" / f"{stage}.json"
            if C.digest(p) != expected_sha:
                raise RuntimeError(f"directional stage hash mismatch: {stage}")
            rows = checked(p, "STAGE_COMPLETE", gate_sha=gate_sha, protocol_sha=protocol_sha)
            if len(rows["random_controls"]) != 20 or len(rows["protected_top10"]) != 10:
                raise RuntimeError("directional top-10/random coverage incomplete")
            for row in rows["protected_train_subject_session_rows"] + rows["protected_outer_subject_session_rows"]:
                directional.append({"stage": stage, **row})
            for control in rows["random_controls"]:
                random.append({"stage": stage, **{key: value for key, value in control.items()
                                                    if key != "outer_subject_session_rows"},
                               "train_top10": json.dumps(control["train_top10"]),
                               "final_subset": json.dumps(control["final_subset"])})
        if len(adjacent) != 4 * 68 or len(long_range) != 3 * 68 or len(random) != 100:
            raise RuntimeError("mediation/directional compact coverage invalid")
        epochs = {int(row["epoch"]): row for row in endpoint["schedule"]}
        backs = {int(row["epoch"]): row for row in backtrace["checkpoints"]}
        if set(epochs) != set(backs) or set(epochs) != set(EPOCHS):
            raise RuntimeError("frozen checkpoint schedule mismatch")
        trajectory, mechanism, backtrace_rows = [], [], []
        for epoch in EPOCHS:
            p = cell / "checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.json"
            native = checked(p, "NATIVE_CHECKPOINT_PT_COMPLETE_PENDING_BACKTRACE_AND_MECHANISM",
                             gate_sha=gate_sha, protocol_sha=protocol_sha)
            p = cell / "checkpoint_mechanism_v1" / f"epoch_{epoch:03d}.json"
            ck = checked(p, "CHECKPOINT_MECHANISM_COMPLETE_REPLICA_ONLY",
                         gate_sha=gate_sha, protocol_sha=protocol_sha)
            if (native["trajectory_provenance"] != "RETRAINED_REPLICA_TRAJECTORY" or
                    ck["trajectory_provenance"] != "RETRAINED_REPLICA_TRAJECTORY" or
                    ck["checkpoint_native_pt_audit_sha256"] != C.digest(cell / "checkpoint_native_pt_v3" / f"epoch_{epoch:03d}.json") or
                    ck["final_p_backtrace_audit_sha256"] != C.digest(cell / "FINAL_P_BACKTRACE_V1.json") or
                    ck["pair_count"] != 832 or ck["subject_session_count"] != 52):
                raise RuntimeError(f"replica checkpoint mechanism identity invalid: {epoch}")
            outer = epochs[epoch]["outer_development"]
            trajectory.append({
                "model": MODEL, "task": TASK, "fold": FOLD, "epoch": epoch,
                "schedule_tags": ";".join(epochs[epoch]["schedule_tags"]),
                "trajectory_provenance": "RETRAINED_REPLICA_TRAJECTORY",
                "protected_rank": native["protected_rank"],
                "persistence_supported_blocks": json.dumps(native["persistence_supported_blocks"]),
                "native_finite_dependence": native["native_finite_subject_equal_centered_logit_rms"],
                "selected_excess_CE_harm_mean": native["selected_excess_CE_harm_mean"],
                "outer_BA": outer["subject_equal_BA"],
                "outer_macro_F1": outer["subject_equal_macro_F1"],
                "outer_CE": outer["subject_equal_CE"],
                "outer_used_for_selection": False,
            })
            for row in ck["subject_session_mechanism"]:
                mechanism.append({"model": MODEL, "task": TASK, "fold": FOLD,
                                  "epoch": epoch, "trajectory_provenance": "RETRAINED_REPLICA_TRAJECTORY",
                                  **row})
            for row in backs[epoch]["subject_session_recovery"]:
                backtrace_rows.append({"model": MODEL, "task": TASK, "fold": FOLD,
                                       "epoch": epoch, "trajectory_provenance": "RETRAINED_REPLICA_TRAJECTORY",
                                       **row})
        if len(mechanism) != 6 * 52:
            raise RuntimeError("checkpoint mechanism compact coverage incomplete")
        signatures = subject["subject_session_signature_rows"]
        drift = []
        for row in subject["subject_session_drift_rows"]:
            flat = {key: value for key, value in row.items() if key != "mechanism_component_drift"}
            flat.update({f"drift_{key}": value for key, value in row["mechanism_component_drift"].items()})
            drift.append(flat)
        regime_rows, regime_meta = regimes(signatures, subject["future_subject_performance"],
                                           subject["drift_components"])
        outputs = {
            "PC_MEDIATION_TRANSITIONS.csv": adjacent,
            "PC_LONG_RANGE_MEDIATION.csv": long_range,
            "PC_DIRECTIONAL_INTERACTION.csv": directional,
            "PC_DIRECTIONAL_RANDOM_CONTROL.csv": random,
            "TRAINING_TRAJECTORY_AUDIT.csv": trajectory,
            "TRAINING_PC_MECHANISM.csv": mechanism,
            "FINAL_P_BACKTRACE.csv": backtrace_rows,
            "SUBJECT_SESSION_MECHANISM_SIGNATURE.csv": signatures,
            "SUBJECT_SESSION_MECHANISM_DRIFT.csv": drift,
            "MECHANISM_REGIMES.csv": regime_rows,
        }
        for name, rows in outputs.items():
            if not rows:
                raise RuntimeError(f"empty compact output: {name}")
            C.write_csv(output / name, rows)
        report = (
            "# First-cell mechanism preview — NOT the 20-cell conclusion\n\n"
            "EEGNet / OpenBMI MI / seed0 / fold0 only. All representations and directions "
            "were TRAIN-defined; outer-development is evaluation-only. Final-heldout was not accessed.\n\n"
            f"- Adjacent transition rows: {len(adjacent)}; long-range rows: {len(long_range)}.\n"
            f"- Directional subject/session rows: {len(directional)}; fixed random controls: {len(random)}.\n"
            f"- Replica checkpoints: {len(trajectory)} (not the historical training path).\n"
            f"- Biological subject/session signatures: {len(signatures)}; paired drift rows: {len(drift)}.\n"
            f"- Secondary TRAIN-defined regime clustering: K={regime_meta['selected_k']}, "
            f"TRAIN silhouette={regime_meta['train_silhouette']:.4f}; no outer tuning.\n\n"
            "The final layer uses the historical canonical oblique projector; intermediate layers "
            "use TRAIN-only orthogonal projectors, as disclosed in the frozen amendment. "
            "Local context sensitivity in the subject audit is a finite-swap cosine, not a Jacobian. "
            "Future performance associations are post-outcome descriptions, not prospective prediction "
            "or biological causation. Cross-fold, cross-architecture, cross-task, and final case "
            "classification are pending the other 19 cells.\n"
        )
        report_path = output / "MECHANISM_CLOSURE_REPORT.md"
        report_path.write_text(report, encoding="utf-8")
        C.write_json(output / "PROVENANCE.json", {
            "status": "FIRST_CELL_COMPACT_PREVIEW_COMPLETE_NOT_20_CELL_RESULT",
            "model": MODEL, "task": TASK, "fold": FOLD, "seed": 0,
            "trajectory_provenance": "RETRAINED_REPLICA_TRAJECTORY",
            "source_cell": str(cell), "source_subject_v2_sha256": SUBJECT_V2_SHA,
            "source_mediation_manifest_sha256": C.digest(cell / "MEDIATION_V2.json"),
            "source_directional_manifest_sha256": C.digest(cell / "DIRECTIONAL_V4.json"),
            "source_replay_audit_sha256": C.digest(replay_path),
            "source_endpoint_audit_sha256": C.digest(cell / "REPLICA_ENDPOINT_AUDIT_V1.json"),
            "source_backtrace_sha256": C.digest(cell / "FINAL_P_BACKTRACE_V1.json"),
            "output_rows": {name: len(rows) for name, rows in outputs.items()},
            "output_sha256": {name: C.digest(output / name) for name in outputs},
            "report_sha256": C.digest(report_path),
            "regime_metadata": regime_meta,
            "upstream_gate_sha256": gate_sha,
            "upstream_protocol_sha256": protocol_sha,
            "upstream_implementation_sha256": ORIGINAL_IMPL_SHA,
            "projector_amendment_sha256": AMENDMENT_SHA,
            "analysis_lock_sha256": ANALYSIS_SHA,
            "implementation_sha256": C.digest(Path(__file__)),
            "outer_development_accessed": True, "final_heldout_accessed": False,
        })
        print("FIRST_CELL_COMPACT_PREVIEW_COMPLETE", len(outputs), flush=True)
    except Exception as exc:
        C.write_json(failure, {"status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}",
                               "implementation_sha256": C.digest(Path(__file__)),
                               "final_heldout_accessed": False})
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
