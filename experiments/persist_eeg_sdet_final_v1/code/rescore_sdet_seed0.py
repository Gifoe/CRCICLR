"""Replay the frozen SDET seed-0 objects on the corrected future-only endpoint.

This is intentionally a scoring-only entry point.  It never fits EEGNet, never
constructs episodes, and never recomputes the population proposal.  It loads
the checkpoint paths, SEARCH-derived normalizers, and U/V/p/g_pop/a_pop saved
by the completed seed-0 run, then opens TEST labels only for the physical
future session (OpenBMI S2, WBCIC S3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

import run_sdet_seed0 as base


TIE_TOL = 1e-8
FUTURE_SESSION = 2
EXPECTED_TRIALS = {"OpenBMI": 100, "WBCIC": 200}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def load_frozen_objects(device: torch.device) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training_log = read_json(base.OUT / "SEED0_TRAINING_LOG.json")
    intervention = read_json(base.OUT / "SEED0_INTERVENTION.json")
    config = read_json(base.OUT / "protocol" / "SDET_SEED0_CONFIG.json")
    if int(training_log.get("seed", -1)) != 0 or int(config.get("seed", -1)) != 0:
        raise RuntimeError("frozen seed provenance is not seed 0")
    if not bool(intervention.get("frozen_before_test_labels", False)):
        raise RuntimeError("intervention provenance does not prove freeze-before-test")
    return training_log, intervention, config


def restore_model_and_module(
    dataset: str,
    bundle: Any,
    training_entry: dict[str, Any],
    intervention_entry: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    checkpoint = Path(training_entry["baseline"]["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen checkpoint missing: {checkpoint}")
    model = base.VanillaEEGNet(bundle.channels).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    module = base.SDETModule().to(device)
    arrays = {}
    for key in ("U", "V", "p", "g_pop", "a_pop"):
        if key not in intervention_entry:
            raise RuntimeError(f"missing frozen intervention object: {dataset}/{key}")
        arrays[key] = as_float_array(intervention_entry[key])
    with torch.no_grad():
        module.U.copy_(torch.from_numpy(arrays["U"]).to(device))
        module.V.copy_(torch.from_numpy(arrays["V"]).to(device))
        module.p.copy_(torch.from_numpy(arrays["p"]).to(device))
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    # Ensure the replayed module is exactly the serialized intervention, not a
    # newly optimized or re-normalized approximation.
    checks = {
        "U_match": bool(np.array_equal(module.U.detach().cpu().numpy(), arrays["U"])),
        "V_match": bool(np.array_equal(module.V.detach().cpu().numpy(), arrays["V"])),
        "p_match": bool(np.array_equal(module.p.detach().cpu().numpy(), arrays["p"])),
        "a_pop_finite": bool(np.isfinite(arrays["a_pop"]).all()),
        "g_pop_finite": bool(np.isfinite(arrays["g_pop"]).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen intervention mismatch: {dataset} {checks}")
    info = {
        "dataset": dataset,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "intervention_objects": {key: arrays[key].tolist() for key in arrays},
        "intervention_checks": checks,
        "normalizer_source": "SEED0_TRAINING_LOG.json SEARCH population statistics",
    }
    return model, module, info


def score_dataset(
    dataset: str,
    bundle: Any,
    model: torch.nn.Module,
    module: torch.nn.Module,
    intervention_entry: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    a_pop = torch.from_numpy(as_float_array(intervention_entry["a_pop"])).to(device)
    rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    probability_shifts: list[float] = []
    for subject in bundle.test_subjects:
        indices = bundle.test_indices([subject], sessions=(FUTURE_SESSION,))
        expected = EXPECTED_TRIALS[dataset]
        if len(indices) != expected:
            raise RuntimeError(f"future endpoint count mismatch: {dataset} {subject} n={len(indices)} expected={expected}")
        x = base.prepare(bundle.test_accessor, indices, mean, std, device)
        labels = bundle.load_test_labels(indices)
        with torch.no_grad():
            z = model.forward_features(x)
            base_probability = torch.softmax(model.head(z), dim=1).cpu().numpy()
            z_sdet = module.transformed(z, a_pop)
            sdet_probability = torch.softmax(model.head(z_sdet), dim=1).cpu().numpy()
        base_pred = np.argmax(base_probability, axis=1)
        sdet_pred = np.argmax(sdet_probability, axis=1)
        base_correct = base_pred == labels
        sdet_correct = sdet_pred == labels
        changed = base_pred != sdet_pred
        beneficial = (~base_correct) & sdet_correct & changed
        harmful = base_correct & (~sdet_correct) & changed
        neutral = changed & ~(beneficial | harmful)
        probability_shifts.extend(np.abs(base_probability[:, 1] - sdet_probability[:, 1]).tolist())
        flip_rows.append({
            "dataset": dataset,
            "subject_id": subject,
            "future_session": FUTURE_SESSION,
            "future_trials": int(len(labels)),
            "prediction_changed_trials": int(changed.sum()),
            "beneficial_flips": int(beneficial.sum()),
            "harmful_flips": int(harmful.sum()),
            "neutral_flips": int(neutral.sum()),
        })
        delta = float(balanced_accuracy_score(labels, sdet_pred) - balanced_accuracy_score(labels, base_pred))
        delta_f1 = float(f1_score(labels, sdet_pred, average="macro") - f1_score(labels, base_pred, average="macro"))
        if delta > TIE_TOL:
            status = "improved"
        elif delta < -TIE_TOL:
            status = "harmed"
        else:
            status = "tie"
        rows.append({
            "dataset": dataset,
            "subject_id": subject,
            "future_session": FUTURE_SESSION,
            "trials": int(len(labels)),
            "baseline_BA": float(balanced_accuracy_score(labels, base_pred)),
            "sdet_BA": float(balanced_accuracy_score(labels, sdet_pred)),
            "delta_BA": delta,
            "delta_BA_pp": delta * 100.0,
            "baseline_F1": float(f1_score(labels, base_pred, average="macro")),
            "sdet_F1": float(f1_score(labels, sdet_pred, average="macro")),
            "delta_F1": delta_f1,
            "status": status,
        })
    frame = pd.DataFrame(rows)
    deltas = frame["delta_BA"].to_numpy(dtype=float)
    improved = deltas > TIE_TOL
    harmed = deltas < -TIE_TOL
    tied = np.abs(deltas) <= TIE_TOL
    flips = pd.DataFrame(flip_rows)
    shift = np.asarray(probability_shifts, dtype=float)
    aggregate = {
        "dataset": dataset,
        "n_subjects": int(len(frame)),
        "n_future_trials": int(frame["trials"].sum()),
        "baseline_mean_subject_BA": float(frame["baseline_BA"].mean()),
        "sdet_mean_subject_BA": float(frame["sdet_BA"].mean()),
        "delta_BA_pp": float(frame["delta_BA"].mean() * 100.0),
        "baseline_macro_F1": float(frame["baseline_F1"].mean()),
        "sdet_macro_F1": float(frame["sdet_F1"].mean()),
        "delta_macro_F1": float(frame["delta_F1"].mean()),
        "mean_subject_delta_BA_pp": float(frame["delta_BA"].mean() * 100.0),
        "median_subject_delta_BA_pp": float(np.median(deltas) * 100.0),
        "improved_subjects": int(improved.sum()),
        "tied_subjects": int(tied.sum()),
        "harmed_subjects": int(harmed.sum()),
        "improved_subject_ratio": float(improved.mean()),
        "tied_subject_ratio": float(tied.mean()),
        "harmed_subject_ratio": float(harmed.mean()),
        "NTR0": float(np.mean(deltas >= -TIE_TOL)),
        "NTR0_5": float(np.mean(deltas >= 0.005 - TIE_TOL)),
        "worst_quartile_subject_delta_BA_pp": float(np.quantile(deltas, 0.25) * 100.0),
        "best_quartile_subject_delta_BA_pp": float(np.quantile(deltas, 0.75) * 100.0),
        "minimum_subject_delta_BA_pp": float(np.min(deltas) * 100.0),
        "maximum_subject_delta_BA_pp": float(np.max(deltas) * 100.0),
    }
    flips_aggregate = {
        "dataset": dataset,
        "total_future_trials": int(len(shift)),
        "prediction_changed_trials": int(flips["prediction_changed_trials"].sum()),
        "prediction_change_ratio": float(flips["prediction_changed_trials"].sum() / len(shift)),
        "beneficial_flips": int(flips["beneficial_flips"].sum()),
        "harmful_flips": int(flips["harmful_flips"].sum()),
        "neutral_flips": int(flips["neutral_flips"].sum()),
        "net_beneficial_flips": int(flips["beneficial_flips"].sum() - flips["harmful_flips"].sum()),
        "subjects_with_zero_prediction_change": int((flips["prediction_changed_trials"] == 0).sum()),
    }
    probability = {
        "dataset": dataset,
        "total_future_trials": int(len(shift)),
        "mean_absolute_probability_change_class1": float(np.mean(shift)),
        "median_absolute_probability_change_class1": float(np.median(shift)),
        "p95_absolute_probability_change_class1": float(np.quantile(shift, 0.95)),
        "max_absolute_probability_change_class1": float(np.max(shift)),
        "argmax_unchanged_but_probability_shift_gt_1e-3": bool(flips_aggregate["prediction_changed_trials"] == 0 and np.quantile(shift, 0.95) > 1e-3),
    }
    return rows, {**aggregate, **flips_aggregate}, probability


def choose_terminal(aggregates: list[dict[str, Any]], probabilities: list[dict[str, Any]]) -> str:
    deltas = [float(item["delta_BA_pp"]) for item in aggregates]
    collapse = any(float(item["sdet_mean_subject_BA"]) < 0.5 or float(item["harmed_subject_ratio"]) > 0.5 or float(item["worst_quartile_subject_delta_BA_pp"]) < -10.0 for item in aggregates)
    near_zero = all(abs(delta) < 0.1 for delta in deltas)
    almost_no_prediction_change = all(float(item["prediction_change_ratio"]) <= 0.01 for item in aggregates)
    tiny_probability_shift = all(float(item["p95_absolute_probability_change_class1"]) <= 1e-3 for item in probabilities)
    if collapse:
        return "SDET_SEED0_FUTURE_MIXED_STOP"
    if near_zero and almost_no_prediction_change and tiny_probability_shift:
        return "SDET_SEED0_FUTURE_NOOP_STOP"
    if all(delta > 0 for delta in deltas):
        return "SDET_SEED0_FUTURE_POSITIVE_STOP"
    if all(delta < 0 for delta in deltas):
        return "SDET_SEED0_FUTURE_NEGATIVE_STOP"
    return "SDET_SEED0_FUTURE_MIXED_STOP"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_log, intervention, config = load_frozen_objects(device)
    split, split_sha = base.load_split()
    training_by_dataset = {entry["dataset"]: entry for entry in training_log["datasets"]}
    intervention_by_dataset = intervention["datasets"]
    prepared: dict[str, tuple[Any, Any, Any, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {
        "protocol": "SDET_SEED0_CORRECTED_FUTURE_ONLY_RESCORING",
        "frozen_retraining": False,
        "hyperparameters_changed": False,
        "seed_changed": False,
        "test_subjects_changed": False,
        "a_pop_reconstructed": False,
        "normalization_reestimated_on_test": False,
        "split_manifest_sha256": split_sha,
        "source_training_log_sha256": base.sha256_file(base.OUT / "SEED0_TRAINING_LOG.json"),
        "source_intervention_sha256": base.sha256_file(base.OUT / "SEED0_INTERVENTION.json"),
        "datasets": {},
    }
    for dataset in ("OpenBMI", "WBCIC"):
        bundle = base.load_bundle(dataset, split)
        entry = training_by_dataset[dataset]
        intervention_entry = intervention_by_dataset[dataset]
        model, module, artifact_info = restore_model_and_module(dataset, bundle, entry, intervention_entry, device)
        mean = np.asarray(entry["normalizer_mean"], dtype=np.float32)
        std = np.asarray(entry["normalizer_std"], dtype=np.float32)
        if mean.shape != (bundle.channels,) or std.shape != (bundle.channels,):
            raise RuntimeError(f"normalizer shape mismatch: {dataset}")
        prepared[dataset] = (bundle, model, module, mean, std, intervention_entry)
        provenance["datasets"][dataset] = artifact_info

    all_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    for dataset in ("OpenBMI", "WBCIC"):
        bundle, model, module, mean, std, intervention_entry = prepared[dataset]
        rows, aggregate, probability = score_dataset(dataset, bundle, model, module, intervention_entry, mean, std, device)
        all_rows.extend(rows)
        aggregates.append(aggregate)
        probability_rows.append(probability)
    expected_totals = {"OpenBMI": 14 * 100, "WBCIC": 10 * 200}
    for aggregate in aggregates:
        if aggregate["n_future_trials"] != expected_totals[aggregate["dataset"]]:
            raise RuntimeError(f"future total mismatch: {aggregate}")
    terminal = choose_terminal(aggregates, probability_rows)
    base.write_csv(base.OUT / "SEED0_FUTURE_TEST_PER_SUBJECT.csv", all_rows)
    base.write_csv(base.OUT / "SEED0_FUTURE_TEST_RESULTS.csv", aggregates)
    flip_rows = [{
        "dataset": item["dataset"],
        "total_future_trials": item["total_future_trials"],
        "prediction_changed_trials": item["prediction_changed_trials"],
        "prediction_change_ratio": item["prediction_change_ratio"],
        "beneficial_flips": item["beneficial_flips"],
        "harmful_flips": item["harmful_flips"],
        "neutral_flips": item["neutral_flips"],
        "net_beneficial_flips": item["net_beneficial_flips"],
        "subjects_with_zero_prediction_change": item["subjects_with_zero_prediction_change"],
    } for item in aggregates]
    base.write_csv(base.OUT / "SEED0_FUTURE_PREDICTION_FLIPS.csv", flip_rows)
    base.write_csv(base.OUT / "SEED0_FUTURE_PROBABILITY_SHIFT.csv", probability_rows)
    provenance.update({
        "endpoint": {"OpenBMI": "physical session 2 (S2 future)", "WBCIC": "physical session 2 (S3 future)"},
        "future_trials": expected_totals,
        "protocol_valid": True,
        "terminal": terminal,
        "aggregates": aggregates,
        "probability_diagnostics": probability_rows,
        "checkpoint_and_intervention_replay_verified": True,
    })
    base.write_json(base.OUT / "SEED0_FUTURE_RESCORING_PROVENANCE.json", provenance)
    by_dataset = {item["dataset"]: item for item in aggregates}
    prob_by_dataset = {item["dataset"]: item for item in probability_rows}
    decision = [
        "# SDET Seed-0 Corrected Future-Session Evaluation",
        "",
        "Protocol correction: previous TEST scoring pooled history + future sessions; this corrected replay uses future session only.",
        "",
        "Frozen model/intervention retrained: NO",
        "Hyperparameters changed: NO",
        "Seed changed: NO",
        "TEST subjects changed: NO",
        "a_pop changed: NO",
        "Normalization reestimated on TEST: NO",
        "",
        "OpenBMI:",
        f"EEGNet future BA: {by_dataset['OpenBMI']['baseline_mean_subject_BA']:.6f}",
        f"SDET future BA: {by_dataset['OpenBMI']['sdet_mean_subject_BA']:.6f}",
        f"Delta BA: {by_dataset['OpenBMI']['delta_BA_pp']:+.3f} pp",
        f"Macro-F1 delta: {by_dataset['OpenBMI']['delta_macro_F1']:+.6f}",
        f"Improved / Tie / Harmed: {by_dataset['OpenBMI']['improved_subjects']} / {by_dataset['OpenBMI']['tied_subjects']} / {by_dataset['OpenBMI']['harmed_subjects']}",
        f"Prediction flips: {by_dataset['OpenBMI']['prediction_changed_trials']} / {by_dataset['OpenBMI']['n_future_trials']}",
        f"Beneficial flips: {by_dataset['OpenBMI']['beneficial_flips']}",
        f"Harmful flips: {by_dataset['OpenBMI']['harmful_flips']}",
        "",
        "WBCIC:",
        f"EEGNet future BA: {by_dataset['WBCIC']['baseline_mean_subject_BA']:.6f}",
        f"SDET future BA: {by_dataset['WBCIC']['sdet_mean_subject_BA']:.6f}",
        f"Delta BA: {by_dataset['WBCIC']['delta_BA_pp']:+.3f} pp",
        f"Macro-F1 delta: {by_dataset['WBCIC']['delta_macro_F1']:+.6f}",
        f"Improved / Tie / Harmed: {by_dataset['WBCIC']['improved_subjects']} / {by_dataset['WBCIC']['tied_subjects']} / {by_dataset['WBCIC']['harmed_subjects']}",
        f"Prediction flips: {by_dataset['WBCIC']['prediction_changed_trials']} / {by_dataset['WBCIC']['n_future_trials']}",
        f"Beneficial flips: {by_dataset['WBCIC']['beneficial_flips']}",
        f"Harmful flips: {by_dataset['WBCIC']['harmful_flips']}",
        "",
        f"Both datasets positive: {all(float(item['delta_BA_pp']) > 0 for item in aggregates)}",
        f"At least one dataset negative: {any(float(item['delta_BA_pp']) < 0 for item in aggregates)}",
        f"Subjects with zero prediction change: OpenBMI {by_dataset['OpenBMI']['subjects_with_zero_prediction_change']}/{by_dataset['OpenBMI']['n_subjects']}; WBCIC {by_dataset['WBCIC']['subjects_with_zero_prediction_change']}/{by_dataset['WBCIC']['n_subjects']}",
        f"Probability shift with unchanged argmax: OpenBMI {prob_by_dataset['OpenBMI']['argmax_unchanged_but_probability_shift_gt_1e-3']}; WBCIC {prob_by_dataset['WBCIC']['argmax_unchanged_but_probability_shift_gt_1e-3']}",
        "Improvement concentration: inspect the paired subject table; no target-specific adapter was used.",
        "New protocol invalidity: false.",
        "",
        "Previous SEED0_DECISION.md is superseded for scientific interpretation because it pooled history and future sessions in TEST scoring.",
        "",
        f"Terminal state: **{terminal}**",
    ]
    (base.OUT / "SEED0_FUTURE_TEST_DECISION.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    print(f"TERMINAL={terminal}")
    for item in aggregates:
        print(f"{item['dataset']}: baseline={item['baseline_mean_subject_BA']:.6f} sdet={item['sdet_mean_subject_BA']:.6f} delta_pp={item['delta_BA_pp']:+.6f} trials={item['n_future_trials']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
