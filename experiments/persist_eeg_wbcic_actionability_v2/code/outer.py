"""One-time sealed WBCIC outer evaluation after every method choice is frozen."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cache import process_session
from core import (
    EPS,
    MODEL,
    OUT,
    PROTOCOL,
    RESULTS,
    balanced_accuracy_score,
    bootstrap_mean,
    ce_rows,
    clean,
    infer,
    load_model,
    random_bases,
    sha256_file,
    stable_seed,
    write_csv,
    write_json,
)
from pipeline import block_union, residual_harmful, reproduce, write_report


OUTER_CACHE = OUT / "cache" / "wbcic_outer_S3"


def build_outer_cache(raw_root: Path, subjects: list[str], workers: int, batch_size: int) -> dict[str, Any]:
    OUTER_CACHE.mkdir(parents=True, exist_ok=True)
    materialized = {path.name for path in OUTER_CACHE.iterdir() if path.is_dir()}
    if not materialized.issubset(set(subjects)):
        raise RuntimeError("DATA_SCOPE_VIOLATION: outer cache contains a non-outer subject")
    jobs = [(subject, 2, str(raw_root.resolve()), batch_size, str(OUTER_CACHE)) for subject in subjects]
    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for job in jobs:
            rows.append(process_session(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_session, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                print(f"[outer-cache {index}/{len(jobs)}] {row['subject']} n={row['n_trials']}", flush=True)
    materialized = {path.name for path in OUTER_CACHE.iterdir() if path.is_dir()}
    if materialized != set(subjects):
        raise RuntimeError("DATA_SCOPE_VIOLATION: outer cache is incomplete or contaminated")
    audit = {
        "status": "OUTER_S3_CACHE_COMPLETE_AFTER_FINAL_LOCK",
        "subject_count": len(subjects),
        "session_count": len(rows),
        "trial_count": int(sum(int(row["n_trials"]) for row in rows)),
        "sessions": [2],
        "cache_root": str(OUTER_CACHE.resolve()),
    }
    write_json(PROTOCOL / "OUTER_CACHE_SCOPE_AUDIT.json", audit)
    return audit


def evaluate(raw_root: Path, device: torch.device, workers: int, batch_size: int) -> dict[str, Any]:
    final_lock_path = PROTOCOL / "FINAL_OUTER_EVALUATION_LOCK.json"
    if not final_lock_path.is_file():
        raise RuntimeError("Outer evaluation is forbidden before FINAL_OUTER_EVALUATION_LOCK.json")
    final_lock = json.loads(final_lock_path.read_text(encoding="utf-8"))
    if (
        final_lock.get("status") != "AGDI_PRIMARY_PASS_OUTER_LOCKED"
        or final_lock.get("outer_evaluation_authorized_once") is not True
        or final_lock.get("outer_subject_ids_present") is not False
    ):
        raise RuntimeError("Outer evaluation lock is not authorized")
    result_path = RESULTS / "WBCIC_OUTER_RESULT.json"
    if result_path.exists():
        raise RuntimeError("One-time WBCIC outer result already exists; retraining/re-evaluation is forbidden")

    # This is the first and only point at which the sealed subject IDs are opened.
    sealed_path = PROTOCOL / "OUTER_SPLIT_LOCK.json"
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    subjects = list(map(str, sealed.get("outer_subjects", [])))
    if len(subjects) != 10 or sealed.get("outer_evaluation_authorized") is not False:
        raise RuntimeError("Malformed sealed outer split")
    build_outer_cache(raw_root, subjects, workers, batch_size)

    checkpoint = Path(final_lock["model_checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path(__file__).resolve().parents[3] / checkpoint
    basis_path = Path(final_lock["basis"])
    if not basis_path.is_absolute():
        basis_path = Path(__file__).resolve().parents[3] / basis_path
    if sha256_file(checkpoint) != final_lock["model_checkpoint_sha256"] or sha256_file(basis_path) != final_lock["basis_sha256"]:
        raise RuntimeError("Frozen final model/basis hash mismatch")
    model, payload = load_model(checkpoint, device)
    if payload["model_state_sha256"] != final_lock["model_state_sha256"]:
        raise RuntimeError("Frozen model state mismatch")
    with np.load(basis_path, allow_pickle=False) as item:
        basis = item["basis"].astype(np.float64)
    protected = block_union(basis, final_lock["protected_blocks"])
    harmful_raw = block_union(basis, final_lock["harmful_blocks"])
    harmful, overlap = residual_harmful(protected, harmful_raw)
    if harmful.shape[1] == 0:
        raise RuntimeError("AGDI_NO_OP_PROTECTED_OVERLAP")
    excluded = np.concatenate([protected, harmful], axis=1) if protected.size else harmful
    if 32 - int(np.linalg.matrix_rank(excluded)) < harmful.shape[1]:
        raise RuntimeError("AGDI_FAIL_NO_SPECIFICITY")
    controls = random_bases(harmful.shape[1], 999, "OUTER_AGDI", excluded=excluded)
    arrays = infer(model, subjects, [2], device, workers, cache_root=OUTER_CACHE)
    h = arrays["embeddings"].astype(np.float64)
    y = arrays["labels"].astype(int)
    sid = arrays["subject_index"].astype(int)
    weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
    bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
    alpha = float(final_lock["selected_alpha"])
    after_weight = weight @ (np.eye(32) - alpha * harmful @ harmful.T)
    base_logits = h @ weight.T + bias
    agdi_logits = h @ after_weight.T + bias
    random_logits = [h @ (weight @ (np.eye(32) - alpha * control @ control.T)).T + bias for control in controls]
    base_prediction = base_logits.argmax(1)
    agdi_prediction = agdi_logits.argmax(1)
    base_ce = ce_rows(base_logits, y)
    agdi_ce = ce_rows(agdi_logits, y)
    subject_rows: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects):
        mask = sid == index
        base_ba = balanced_accuracy_score(y[mask], base_prediction[mask])
        agdi_ba = balanced_accuracy_score(y[mask], agdi_prediction[mask])
        random_ba = np.asarray([balanced_accuracy_score(y[mask], value.argmax(1)[mask]) for value in random_logits])
        subject_rows.append(
            {
                "subject": subject,
                "n_S3_trials": int(mask.sum()),
                "baseline_BA": base_ba,
                "AGDI_BA": agdi_ba,
                "random_BA_mean": float(random_ba.mean()),
                "delta_BA": agdi_ba - base_ba,
                "delta_BA_random": float(random_ba.mean() - base_ba),
                "delta_BA_specific": float(agdi_ba - random_ba.mean()),
                "baseline_NLL": float(base_ce[mask].mean()),
                "AGDI_NLL": float(agdi_ce[mask].mean()),
            }
        )
    delta = np.asarray([row["delta_BA"] for row in subject_rows], dtype=np.float64)
    specific = np.asarray([row["delta_BA_specific"] for row in subject_rows], dtype=np.float64)
    delta_summary = bootstrap_mean(delta, stable_seed("outer", "delta"))
    specific_summary = bootstrap_mean(specific, stable_seed("outer", "specific"))
    centered_before = weight - weight.mean(axis=0, keepdims=True)
    centered_after = after_weight - after_weight.mean(axis=0, keepdims=True)
    harmful_ratio = float(
        np.sum((centered_after @ harmful) ** 2) / max(np.sum((centered_before @ harmful) ** 2), EPS)
    )
    protected_error = 0.0
    if protected.size:
        protected_error = float(
            np.linalg.norm(after_weight @ protected - weight @ protected)
            / max(np.linalg.norm(weight @ protected), EPS)
        )
    random_ratios = []
    for control in controls:
        before = float(np.sum((centered_before @ control) ** 2))
        after = float(np.sum((centered_after @ control) ** 2))
        random_ratios.append(after / max(before, EPS))
    random_ratio = float(np.mean(random_ratios))
    outer_pass = bool(
        delta_summary[0] >= 0.005
        and delta_summary[1] > 0
        and specific_summary[1] > 0
        and harmful_ratio < 1
        and protected_error <= float(final_lock["protected_tolerance"])
        and final_lock["random_equivalence_ratio"][0]
        <= random_ratio
        <= final_lock["random_equivalence_ratio"][1]
    )
    worst_count = max(1, int(np.ceil(0.2 * len(delta))))
    result = {
        "terminal_state": "WBCIC_OUTER_PASS" if outer_pass else "WBCIC_OUTER_FAIL",
        "outer_test_used": True,
        "outer_test_runs": 1,
        "subject_count": len(subjects),
        "alpha": alpha,
        "delta_BA_mean": delta_summary[0],
        "delta_BA_CI95": [delta_summary[1], delta_summary[2]],
        "specific_gain_mean": specific_summary[0],
        "specific_gain_CI95": [specific_summary[1], specific_summary[2]],
        "subject_nonharm_fraction": float(np.mean(delta >= 0)),
        "median_delta_BA": float(np.median(delta)),
        "worst_subject_delta_BA": float(np.min(delta)),
        "CVaR20_delta_BA": float(np.sort(delta)[:worst_count].mean()),
        "harmful_dependence_ratio": harmful_ratio,
        "protected_relative_error": protected_error,
        "random_dependence_ratio": random_ratio,
        "overlap": overlap,
        "model_retrained_after_outer": False,
    }
    write_csv(RESULTS / "WBCIC_OUTER_SUBJECT_RESULTS.csv", subject_rows)
    write_json(result_path, result)
    previous = json.loads((OUT / "FINAL_DECISION.json").read_text(encoding="utf-8"))
    final = {
        **previous,
        **result,
        "scientific_conclusion": (
            "Frozen AGDI passed the one-time zero-shot unseen-subject future-session WBCIC outer evaluation."
            if outer_pass
            else "Frozen AGDI failed the one-time WBCIC outer efficacy and/or specificity gate."
        ),
        "next_action": "RUN_OPENBMI_NO_OP_CLOSURE" if outer_pass else "STOP_WBCIC_OUTER_FAIL",
    }
    write_json(OUT / "FINAL_DECISION.json", final)
    write_report(final)
    reproduce()
    print(json.dumps(clean(result), indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluate", nargs="?")
    parser.add_argument("--raw-root", type=Path, default=Path(r"D:\nips-temp\TotalP\P2\nm000348_v1.0.4_bids"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    evaluate(args.raw_root, device, max(1, args.workers), max(1, args.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
