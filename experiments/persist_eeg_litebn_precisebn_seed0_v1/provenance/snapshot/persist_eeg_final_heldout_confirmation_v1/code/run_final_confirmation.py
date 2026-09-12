"""Fail-closed preflight and one-shot final held-out evaluation for LOGIT50."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from load_final_carriers import (
    CHANNELS, DATASETS, EXP, FUTURE_SESSION, MODEL_NAMES, OUTPUTS, PROTOCOL,
    V8_SPLIT, audit_checkpoints, carrier_folds, checkpoint_records, clean,
    holdout_memberships, load_eval_subject, load_frozen_model, normalize_to_device,
    normalizer_provenance, sha256, signal_schema, write_json,
)


def code_hashes() -> dict[str, str]:
    files = ("load_final_carriers.py", "run_final_confirmation.py", "aggregate_final_confirmation.py")
    return {name: sha256(Path(__file__).parent / name) for name in files}


def model_spec() -> dict[str, Any]:
    return {
        "selected_final_predictor": "FROZEN_LOGIT50",
        "anchor": "canonical_EEGNet",
        "expressive_carrier": "CompactLite_BN",
        "fusion_type": "fixed_logit_average",
        "eegnet_weight": 0.5,
        "litebn_weight": 0.5,
        "adaptive_fusion": False,
        "test_time_adaptation": False,
        "target_labels_required": False,
        "carrier_training_in_this_experiment": False,
        "fusion_training_in_this_experiment": False,
    }


def _by_pair(records: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, dict[str, Any]]]:
    pairs: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in records:
        key = (row["dataset"], int(row["fold"]), int(row["seed"]))
        pairs.setdefault(key, {})[row["model"]] = row
    if len(pairs) != 30 or any(set(value) != set(MODEL_NAMES) for value in pairs.values()):
        raise RuntimeError("dataset x fold x seed model mapping is not exact")
    return pairs


def _no_training_audit() -> dict[str, Any]:
    return {
        "optimizer_instantiated": False,
        "backward_called": False,
        "parameters_modified": False,
        "checkpoint_selected_using_holdout": False,
        "normalizer_fit_on_holdout": False,
        "calibration_on_holdout": False,
        "tta_used": False,
        "fusion_weight_modified": False,
        "inference_context": "torch.no_grad",
        "models_eval_only": True,
        "heldout_labels_read_during_preflight": False,
        "heldout_subject_source_sessions_used": False,
    }


def preflight() -> None:
    """Check all immutable inputs without opening held-out label contents."""
    PROTOCOL.mkdir(parents=True, exist_ok=True); OUTPUTS.mkdir(parents=True, exist_ok=True)
    memberships, v8 = holdout_memberships()
    heldout_schema = {dataset: {subject: signal_schema(dataset, subject, FUTURE_SESSION[dataset]) for subject in memberships[dataset]} for dataset in DATASETS}
    normalizers, _ = normalizer_provenance()
    records = checkpoint_records(); audited = audit_checkpoints(records); pairs = _by_pair(audited)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shapes: list[dict[str, Any]] = []
    for (dataset, fold, seed), pair in sorted(pairs.items()):
        eeg = load_frozen_model("EEGNet", CHANNELS[dataset], Path(pair["EEGNet"]["checkpoint_path"]), device)
        lite = load_frozen_model("LiteBN", CHANNELS[dataset], Path(pair["LiteBN"]["checkpoint_path"]), device)
        with torch.no_grad():
            x = torch.zeros((2, CHANNELS[dataset], 1000), device=device)
            ze, _ = eeg(x); zl, _ = lite(x); z50 = (ze.float() + zl.float()) / 2.0
        exact = torch.equal(z50, 0.5 * ze.float() + 0.5 * zl.float())
        frozen = (not eeg.training and not lite.training and not any(p.requires_grad for p in eeg.parameters()) and not any(p.requires_grad for p in lite.parameters()))
        if ze.shape != (2, 2) or zl.shape != (2, 2) or not exact or not frozen:
            raise RuntimeError(f"synthetic/freeze/fusion test failed: {dataset} f{fold} s{seed}")
        shapes.append({"dataset": dataset, "fold": fold, "seed": seed, "EEGNet_logits_shape": list(ze.shape), "LiteBN_logits_shape": list(zl.shape), "LOGIT50_exact": True, "eval_only_and_frozen": True})
        del eeg, lite, x, ze, zl, z50
    if device.type == "cuda": torch.cuda.empty_cache()
    manifest = {
        "source": str(V8_SPLIT), "source_sha256": sha256(V8_SPLIT), "membership_source_is_metadata_only": True,
        "heldout_labels_read_in_preflight": False,
        "OpenBMI": {"subject_ids": memberships["OpenBMI"], "n_subjects": 14, "evaluation_session": "S2", "session_index": 2, "forbidden_heldout_sessions_for_training_normalization": [1]},
        "WBCIC": {"subject_ids": memberships["WBCIC"], "n_subjects": 10, "evaluation_session": "S3", "session_index": 2, "forbidden_heldout_sessions_for_training_normalization": [0, 1]},
    }
    preflight_tests = {
        "pass": True,
        "preflight_only": True,
        "heldout_label_arrays_opened": False,
        "all_required_checkpoints_exist": True,
        "all_checkpoint_hashes_match_frozen_provenance": True,
        "exact_fixed_logit_average": True,
        "alpha_exactly_0_5": True,
        "no_learned_fusion_head": True,
        "all_models_eval_only_and_frozen": True,
        "no_optimizer_instantiated": True,
        "no_backward_path": True,
        "heldout_subjects_absent_from_normalizer_fit": True,
        "source_only_normalizers_reproduced_and_hash_verified": True,
        "output_dimensions_correct": True,
        "exact_dataset_fold_seed_pair_mapping": True,
        "missing_replicates": 0,
        "expected_pairs": 30,
        "expected_checkpoints": 60,
        "synthetic_tests": shapes,
        "heldout_signal_schema_only": heldout_schema,
        "code_sha256": code_hashes(),
    }
    write_json(PROTOCOL / "FINAL_MODEL_SPEC.json", model_spec())
    write_json(PROTOCOL / "FINAL_HOLDOUT_MANIFEST.json", manifest)
    write_json(PROTOCOL / "SOURCE_CHECKPOINTS.json", records)
    write_json(PROTOCOL / "CHECKPOINT_HASH_AUDIT.json", {"pass": True, "records": audited})
    write_json(PROTOCOL / "PREPROCESSING_PROVENANCE.json", {"pass": True, "source_sessions": {d: list((1,) if d == "OpenBMI" else (0, 1)) for d in DATASETS}, "normalizers": normalizers})
    write_json(PROTOCOL / "NO_TRAINING_AUDIT.json", _no_training_audit())
    write_json(PROTOCOL / "PREFLIGHT_TESTS.json", preflight_tests)
    print("FINAL_CONFIRMATION_PREFLIGHT_PASS", flush=True)


def require_preflight() -> dict[str, Any]:
    path = PROTOCOL / "PREFLIGHT_TESTS.json"
    if not path.is_file(): raise RuntimeError("FINAL_CONFIRMATION_PROTOCOL_INVALID: missing preflight")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if not recorded.get("pass") or recorded.get("heldout_label_arrays_opened"):
        raise RuntimeError("FINAL_CONFIRMATION_PROTOCOL_INVALID: preflight invalid")
    if recorded.get("code_sha256") != code_hashes():
        raise RuntimeError("FINAL_CONFIRMATION_PROTOCOL_INVALID: code changed after preflight")
    audit = json.loads((PROTOCOL / "CHECKPOINT_HASH_AUDIT.json").read_text(encoding="utf-8"))
    if not audit.get("pass") or len(audit.get("records", [])) != 60:
        raise RuntimeError("FINAL_CONFIRMATION_PROTOCOL_INVALID: checkpoint audit invalid")
    return recorded


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {"BA": float(balanced_accuracy_score(y, prediction)), "macro_F1": float(f1_score(y, prediction, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, prediction))}


def _logits(model: torch.nn.Module, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            z, _ = model(normalize_to_device(x[start:start + 128], mean, std, device))
            chunks.append(z.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_once() -> None:
    """Open held-out labels once and evaluate every fixed model pair before aggregation."""
    require_preflight()
    memberships, _ = holdout_memberships()
    _, normalizer_values = normalizer_provenance()
    records = audit_checkpoints(checkpoint_records()); pairs = _by_pair(records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heldout: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {dataset: {} for dataset in DATASETS}
    for dataset in DATASETS:
        for subject in memberships[dataset]:
            heldout[dataset][subject] = load_eval_subject(dataset, subject)
    rows: list[dict[str, Any]] = []
    comp: dict[str, dict[str, int]] = {d: {k: 0 for k in ("eegnet_only_correct", "litebn_only_correct", "both_correct", "both_wrong", "trials")} for d in DATASETS}
    for dataset in DATASETS:
        for fold in range(5):
            mean, std = normalizer_values[f"{dataset}:{fold}"]
            for seed in range(3):
                pair = pairs[(dataset, fold, seed)]
                eeg = load_frozen_model("EEGNet", CHANNELS[dataset], Path(pair["EEGNet"]["checkpoint_path"]), device)
                lite = load_frozen_model("LiteBN", CHANNELS[dataset], Path(pair["LiteBN"]["checkpoint_path"]), device)
                for subject in memberships[dataset]:
                    x, y = heldout[dataset][subject]
                    ze = _logits(eeg, x, mean, std, device); zl = _logits(lite, x, mean, std, device); z50 = (ze + zl) / 2.0
                    predictions = {"EEGNet": ze.argmax(axis=1), "LiteBN": zl.argmax(axis=1), "LOGIT50": z50.argmax(axis=1)}
                    for method, prediction in predictions.items():
                        rows.append({"dataset": dataset, "subject_id": subject, "fold": fold, "seed": seed, "method": method, "BA": _metric(y, prediction)["BA"], "macro_F1": _metric(y, prediction)["macro_F1"], "accuracy": _metric(y, prediction)["accuracy"], "trials": int(len(y))})
                    ep, lp = predictions["EEGNet"], predictions["LiteBN"]
                    comp[dataset]["eegnet_only_correct"] += int(((ep == y) & (lp != y)).sum())
                    comp[dataset]["litebn_only_correct"] += int(((lp == y) & (ep != y)).sum())
                    comp[dataset]["both_correct"] += int(((ep == y) & (lp == y)).sum())
                    comp[dataset]["both_wrong"] += int(((ep != y) & (lp != y)).sum())
                    comp[dataset]["trials"] += int(len(y))
                del eeg, lite
                if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    expected = (14 + 10) * 15 * 3
    if len(frame) != expected or frame.duplicated(["dataset", "subject_id", "fold", "seed", "method"]).any():
        raise RuntimeError("FINAL_CONFIRMATION_PROTOCOL_INVALID: missing or duplicate replicate result")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUTS / "REPLICATE_SUBJECT_RESULTS.csv", index=False)
    write_json(OUTPUTS / "COMPLEMENTARITY.json", {d: {**value, **{f"{key}_fraction": value[key] / value["trials"] for key in ("eegnet_only_correct", "litebn_only_correct", "both_correct", "both_wrong")}} for d, value in comp.items()})
    from aggregate_final_confirmation import aggregate
    aggregate()
    print("FINAL_HELDOUT_EVALUATION_AND_AGGREGATION_COMPLETE", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.preflight: preflight()
    else: evaluate_once()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FINAL_CONFIRMATION_PROTOCOL_INVALID: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
