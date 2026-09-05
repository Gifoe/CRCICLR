"""Outcome-only evaluation for the locked GeoSR RAPID_TRIAGE run.

This script refuses to load outcome data until both dataset workers and the
merged pre-outcome lock validate.  It then writes an additional access lock,
evaluates the two retained checkpoints per dataset once, and emits a compact
directional STOP/RESTORE_FULL_PROTOCOL decision.  The result is explicitly
not a formal seed-0 scientific claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import audit_primitives as ap
import run_geosr as g


DATASETS = ("OpenBMI", "WBCIC")
METHODS = ("SUBJECT_BALANCED_ERM", "GEOSR")
FOLD = 0
SEED = 0
CLEAR_POSITIVE_MIN_PP = 0.5


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_pre_outcome(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = root / "RAPID_TRIAGE_PRE_OUTCOME_LOCK.json"
    manifest_path = root / "runtime" / "seed-0" / "PREFLIGHT_MANIFEST.json"
    if not lock_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("merged RAPID_TRIAGE pre-outcome lock/manifest missing")
    lock = read_json(lock_path)
    if lock.get("outcome_labels_read") is not False or lock.get("outcome_labels_read_before_lock") is not False:
        raise RuntimeError("pre-outcome lock indicates outcome access")
    if lock.get("datasets") != list(DATASETS) or lock.get("folds") != [FOLD] or lock.get("methods") != list(METHODS):
        raise RuntimeError("RAPID_TRIAGE merged scope mismatch")
    if lock.get("exact_refit") is not False or lock.get("wbcic_final_refit_teacher") is not False:
        raise RuntimeError("RAPID_TRIAGE training semantics mismatch")
    manifest = read_json(manifest_path)
    if file_sha(manifest_path) != lock.get("manifest_sha256"):
        raise RuntimeError("RAPID_TRIAGE manifest hash mismatch")
    for dataset in DATASETS:
        wr = root / "workers" / f"{dataset}_fold0"
        worker_lock_path = wr / "PRE_OUTCOME_RAPID_TRIAGE_WORKER_LOCK.json"
        marker_path = wr / "RAPID_TRIAGE_WORKER_COMPLETE.json"
        if not worker_lock_path.is_file() or not marker_path.is_file():
            raise RuntimeError(f"worker lock/marker missing: {dataset}")
        if file_sha(worker_lock_path) != lock["worker_lock_sha256"][dataset]:
            raise RuntimeError(f"worker lock changed: {dataset}")
        worker_lock = read_json(worker_lock_path); marker = read_json(marker_path)
        if worker_lock.get("outcome_labels_read") is not False or worker_lock.get("exact_refit") is not False:
            raise RuntimeError(f"worker pre-outcome semantics invalid: {dataset}")
        if marker.get("methods") != list(METHODS) or marker.get("fold") != FOLD:
            raise RuntimeError(f"worker marker scope mismatch: {dataset}")
        entry = manifest[f"{dataset}/fold-0/seed-0"]
        for method in METHODS:
            ck = Path(entry["checkpoints"][method]["path"])
            try:
                ck.resolve().relative_to(wr.resolve())
            except ValueError as exc:
                raise RuntimeError(f"checkpoint escapes worker root: {ck}") from exc
            meta_path = g.checkpoint_meta_path(ck)
            if not ck.is_file() or not meta_path.is_file():
                raise RuntimeError(f"checkpoint/meta missing: {dataset} {method}")
            if file_sha(ck) != entry["checkpoints"][method]["sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {dataset} {method}")
            meta = read_json(meta_path)
            if meta.get("method") != method or meta.get("stage") != "rapid_triage_initial_selection":
                raise RuntimeError(f"checkpoint stage mismatch: {dataset} {method}")
    return lock, manifest


def write_access_lock(root: Path, pre_lock: dict[str, Any]) -> Path:
    rule = {
        "schema": "PERSIST_EEG_GEOSR_RAPID_TRIAGE_OUTCOME_RULE_V1",
        "clear_positive": {
            "mean_BA_delta_pp_min": CLEAR_POSITIVE_MIN_PP,
            "mean_macro_F1_delta_pp_min": 0.0,
            "nonnegative_subject_BA_fraction_min": 0.5,
        },
        "both_datasets_required": True,
        "mixed_direction": "STOP_AS_INCONCLUSIVE",
        "positive_action": "RESTORE_FULL_PROTOCOL",
        "negative_action": "STOP",
        "final_claim_authorized": False,
    }
    rule_path = root / "RAPID_TRIAGE_OUTCOME_RULE.json"
    write_json(rule_path, rule)
    access = {
        "schema": "PERSIST_EEG_GEOSR_RAPID_TRIAGE_OUTCOME_ACCESS_LOCK_V1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_outcome_lock_sha256": file_sha(root / "RAPID_TRIAGE_PRE_OUTCOME_LOCK.json"),
        "outcome_rule_sha256": file_sha(rule_path), "amendment_sha256": pre_lock["amendment_sha256"],
        "datasets": list(DATASETS), "folds": [FOLD], "methods": list(METHODS),
        "both_workers_complete": True, "outcome_labels_read": False,
        "outcome_labels_read_before_lock": False, "WBCIC_outer_10_opened": False,
        "OpenBMI_sealed_holdout_opened": False, "screen_only": True,
        "final_claim_authorized": False,
    }
    path = root / "RAPID_TRIAGE_OUTCOME_ACCESS_LOCK.json"
    write_json(path, access)
    check = read_json(path)
    if check.get("outcome_labels_read") is not False or check.get("both_workers_complete") is not True:
        raise RuntimeError("outcome access lock verification failed")
    return path


def evaluate(root: Path, manifest: dict[str, Any], device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        roles, _, _ = ap.load_roles(dataset)
        role = roles[FOLD]
        # First outcome-subject label materialization: strictly post access-lock.
        data = ap.load_ab_data(dataset, set(role["outcome"]))
        entry = manifest[f"{dataset}/fold-0/seed-0"]
        for method in METHODS:
            ck = Path(entry["checkpoints"][method]["path"])
            rows.extend([{**r, "method": method} for r in g.eval_checkpoint(data, ck, role["outcome"], dataset, FOLD, SEED, device)])
        del data
    frame = pd.DataFrame(rows)
    results = root / "results"; results.mkdir(parents=True, exist_ok=True)
    write_csv(results / "RAPID_TRIAGE_OUTCOME_PER_SUBJECT.csv", frame)

    summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    dataset_decisions: dict[str, Any] = {}
    for dataset in DATASETS:
        f = frame[frame.dataset == dataset]
        sb = f[f.method == "SUBJECT_BALANCED_ERM"].set_index("subject_id")
        geo = f[f.method == "GEOSR"].set_index("subject_id").reindex(sb.index)
        d_ba = geo.BA - sb.BA; d_f1 = geo.macro_F1 - sb.macro_F1
        for subject in sb.index:
            delta_rows.append({"dataset": dataset, "fold": FOLD, "seed": SEED, "subject_id": subject,
                               "SB_ERM_BA": float(sb.loc[subject, "BA"]), "GEOSR_BA": float(geo.loc[subject, "BA"]),
                               "delta_BA_pp": float(d_ba.loc[subject] * 100.0),
                               "SB_ERM_macro_F1": float(sb.loc[subject, "macro_F1"]), "GEOSR_macro_F1": float(geo.loc[subject, "macro_F1"]),
                               "delta_macro_F1_pp": float(d_f1.loc[subject] * 100.0)})
        ba_delta_pp = float(d_ba.mean() * 100.0); f1_delta_pp = float(d_f1.mean() * 100.0)
        nonnegative = float(np.mean(d_ba >= 0)); positive = float(np.mean(d_ba > 0))
        clear = bool(ba_delta_pp >= CLEAR_POSITIVE_MIN_PP and f1_delta_pp >= 0.0 and nonnegative >= 0.5)
        dataset_decisions[dataset] = {"clear_positive": clear, "mean_BA_delta_pp": ba_delta_pp,
                                      "mean_macro_F1_delta_pp": f1_delta_pp,
                                      "positive_subject_fraction": positive,
                                      "nonnegative_subject_fraction": nonnegative,
                                      "worst_subject_delta_pp": float(d_ba.min() * 100.0)}
        for method in METHODS:
            z = f[f.method == method]
            summary_rows.append({"dataset": dataset, "fold": FOLD, "seed": SEED, "method": method,
                                 "mean_subject_BA": float(z.BA.mean()), "mean_macro_F1": float(z.macro_F1.mean()),
                                 "mean_accuracy": float(z.accuracy.mean()), "mean_NLL": float(z.NLL.mean()),
                                 "n_subjects": int(z.subject_id.nunique())})
    write_csv(results / "RAPID_TRIAGE_SUBJECT_DELTAS.csv", delta_rows)
    write_csv(results / "RAPID_TRIAGE_PERFORMANCE_SUMMARY.csv", summary_rows)

    both_clear = bool(all(dataset_decisions[d]["clear_positive"] for d in DATASETS))
    both_nonpositive = bool(all(dataset_decisions[d]["mean_BA_delta_pp"] <= 0.0 for d in DATASETS))
    if both_clear:
        terminal = "RAPID_TRIAGE_RESTORE_FULL_PROTOCOL"
    elif both_nonpositive:
        terminal = "RAPID_TRIAGE_STOP_NO_POSITIVE_DIRECTION"
    else:
        terminal = "RAPID_TRIAGE_STOP_INCONCLUSIVE_OR_MIXED"
    result = {
        "schema": "PERSIST_EEG_GEOSR_RAPID_TRIAGE_RESULT_V1", "terminal": terminal,
        "restore_full_protocol": both_clear, "dataset_decisions": dataset_decisions,
        "cross_dataset_direction_consistent": bool(all(dataset_decisions[d]["mean_BA_delta_pp"] > 0 for d in DATASETS) or both_nonpositive),
        "screen_only": True, "final_claim_authorized": False, "outcome_after_lock": True,
        "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False,
        "scientific_definition_changed": True,
    }
    write_json(results / "RAPID_TRIAGE_RESULT.json", result)
    lines = ["# GeoSR RAPID_TRIAGE", "", f"Terminal: `{terminal}`", "",
             "This is a one-fold, two-method directional screen under the locked amendment; it is not the formal seed-0 result.", "",
             "|Dataset|SB-ERM BA|GeoSR BA|Delta BA (pp)|Delta Macro-F1 (pp)|Positive subjects|Nonnegative subjects|Clear positive|",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    by_method = {(r["dataset"], r["method"]): r for r in summary_rows}
    for dataset in DATASETS:
        m = dataset_decisions[dataset]
        lines.append(f"|{dataset}|{by_method[(dataset, 'SUBJECT_BALANCED_ERM')]['mean_subject_BA']:.4f}|{by_method[(dataset, 'GEOSR')]['mean_subject_BA']:.4f}|{m['mean_BA_delta_pp']:.3f}|{m['mean_macro_F1_delta_pp']:.3f}|{m['positive_subject_fraction']:.3f}|{m['nonnegative_subject_fraction']:.3f}|{m['clear_positive']}|")
    lines += ["", "Only `RAPID_TRIAGE_RESTORE_FULL_PROTOCOL` authorizes resuming the original full protocol.", ""]
    (results / "RAPID_TRIAGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(results / "VALIDATION.json", {"pass": True, "terminal": terminal, "outcome_after_lock": True,
                                               "both_workers_complete": True, "screen_only": True,
                                               "final_claim_authorized": False, "WBCIC_outer_10_opened": False,
                                               "OpenBMI_sealed_holdout_opened": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); root = args.root.resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    pre_lock, manifest = validate_pre_outcome(root)
    access_lock = write_access_lock(root, pre_lock)
    # Revalidate access-lock bytes immediately before the first outcome load.
    if read_json(access_lock).get("outcome_labels_read") is not False:
        raise RuntimeError("outcome access lock changed")
    result = evaluate(root, manifest, torch.device(args.device))
    legality = read_json(root / "DATA_LEGALITY_AUDIT.json")
    legality.update({"canonical_outcome_labels_read": True, "outcome_labels_read_after_lock": True,
                     "outcome_access_lock_sha256": file_sha(access_lock)})
    write_json(root / "DATA_LEGALITY_AUDIT.json", legality)
    print(result["terminal"], flush=True)


if __name__ == "__main__":
    main()
