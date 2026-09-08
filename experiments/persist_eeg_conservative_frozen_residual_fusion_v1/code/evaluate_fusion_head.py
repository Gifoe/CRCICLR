"""Evaluate CFRF only after the full preregistered training grid is present."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

EXP = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("R2EEG_REPO", EXP.parents[1])).resolve()
RUNTIME = Path(os.environ.get("CFRF_RUNTIME", "/root/rivermind-data/cfrf_v1_runtime")).resolve()
SOURCE_RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
CARRIER_EXP = REPO / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1"
OUT = EXP / "outputs"
sys.path[:0] = [str(EXP / "code"), str(REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code"), str(REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code")]
import run_carrier_screen as carrier  # noqa: E402
import run_stage1 as v1  # noqa: E402
from eegnet_locked import EEGNet  # noqa: E402
from cfrf_model import CFRF, GlobalAlpha  # noqa: E402
from feature_cache import SignalCache  # noqa: E402
from frozen_carriers import extract, load_carrier, sha256_file, state_sha256  # noqa: E402

METHODS = ("GLOBAL-ALPHA", "CFRF-v1")


def write_json(path: Path, value: Any) -> None:
    def clean(x: Any) -> Any:
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating, float)): return float(x)
        if isinstance(x, (np.bool_, bool)): return bool(x)
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        return x
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint_path(dataset: str, fold: int, seed: int, model: str) -> Path:
    return SOURCE_RUNTIME / f"{dataset.lower()}_fold{fold}_seed{seed}_{model.lower()}" / "selected_best.pt"


def head_path(dataset: str, fold: int, seed: int, method: str, raw: bool = False) -> Path:
    directory = RUNTIME / "cells" / f"{dataset.lower()}_fold{fold}_seed{seed}_{method.lower().replace('-', '_')}"
    if method == "GLOBAL-ALPHA": return directory / "epoch30_scalar.pt"
    return directory / ("epoch30_raw.pt" if raw else "epoch30_ema.pt")


def metrics(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(1)
    return {"BA": float(balanced_accuracy_score(y, prediction)), "macro_F1": float(f1_score(y, prediction, average="macro", zero_division=0)), "accuracy": float(accuracy_score(y, prediction))}


@torch.no_grad()
def evaluate_subjects(eeg: torch.nn.Module, lite: torch.nn.Module, global_head: GlobalAlpha, cfrf: CFRF, raw_cfrf: CFRF, signal: SignalCache, bundle: Any, subjects: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []; alpha_rows: list[dict[str, Any]] = []; raw_rows: list[dict[str, Any]] = []
    eeg.eval(); lite.eval(); global_head.eval(); cfrf.eval(); raw_cfrf.eval()
    for subject in subjects:
        indices = np.asarray(bundle.indices([str(subject)], (2,)), dtype=np.int64); x, _ = signal.by_global(indices); y = bundle.labels(indices)
        logits: dict[str, list[np.ndarray]] = {key: [] for key in ("EEGNet", "LiteBN", "Frozen_LOGIT50", "GLOBAL-ALPHA", "CFRF-v1", "CFRF-v1-RAW")}
        alphas: list[np.ndarray] = []
        for start in range(0, len(indices), 256):
            z_a, z_e, h_a, h_e = extract(eeg, lite, x[start:start + 256])
            z_global, _, _ = global_head(z_a, z_e); z_cfrf, alpha, _ = cfrf(h_a, h_e, z_a, z_e); z_raw, _, _ = raw_cfrf(h_a, h_e, z_a, z_e)
            for key, value in (("EEGNet", z_a), ("LiteBN", z_e), ("Frozen_LOGIT50", .5 * z_a + .5 * z_e), ("GLOBAL-ALPHA", z_global), ("CFRF-v1", z_cfrf), ("CFRF-v1-RAW", z_raw)):
                logits[key].append(value.float().cpu().numpy())
            alphas.append(alpha.float().cpu().numpy())
        values = {key: np.concatenate(parts) for key, parts in logits.items()}; primary = {key: metrics(y, values[key]) for key in values if key != "CFRF-v1-RAW"}
        row: dict[str, Any] = {"dataset": bundle.name, "fold": None, "seed": None, "subject_id": str(subject), "n_trials": int(len(y))}
        for key, result in primary.items():
            for metric_name, metric_value in result.items(): row[f"{key}_{metric_name}"] = metric_value
        rows.append(row)
        raw_rows.append({"dataset": bundle.name, "subject_id": str(subject), "CFRF_raw_BA": metrics(y, values["CFRF-v1-RAW"])["BA"], "CFRF_ema_BA": primary["CFRF-v1"]["BA"]})
        alpha_values = np.concatenate(alphas)
        alpha_rows.append({"dataset": bundle.name, "subject_id": str(subject), "n_trials": int(len(alpha_values)), "mean_alpha": float(alpha_values.mean()), "std_alpha": float(alpha_values.std(ddof=0)), "median_alpha": float(np.median(alpha_values)), "p05_alpha": float(np.quantile(alpha_values, .05)), "p95_alpha": float(np.quantile(alpha_values, .95)), "fraction_alpha_lt_040": float((alpha_values < .40).mean()), "fraction_alpha_gt_060": float((alpha_values > .60).mean()), "fraction_alpha_near_050": float((np.abs(alpha_values - .5) < .05).mean())})
    return rows, alpha_rows, raw_rows


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--seeds", default="0"); args = parser.parse_args(); seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((CARRIER_EXP / "protocol/FIVEFOLD_SPLIT.json").read_text())
    missing = [f"{dataset}/fold{fold}/seed{seed}/{method}" for dataset in ("OpenBMI", "WBCIC") for fold in range(5) for seed in seeds for method in METHODS if not head_path(dataset, fold, seed, method).is_file()]
    if missing: raise RuntimeError("outer evaluation forbidden before every requested training cell completes: " + repr(missing))
    rows: list[dict[str, Any]] = []; alpha_rows: list[dict[str, Any]] = []; raw_rows: list[dict[str, Any]] = []; invariants: list[dict[str, Any]] = []
    for dataset in ("OpenBMI", "WBCIC"):
        bundle = v1.load_bundle(dataset, list(map(str, split["search_subjects"][dataset])))
        for fold_spec in split["folds"][dataset]:
            fold = int(fold_spec["fold_id"]); mean, std, _ = v1.normalizer(bundle, list(map(str, fold_spec["inner_train_subjects"])))
            outer = list(map(str, fold_spec["outer_dev_subjects"])); outer_indices = np.asarray(bundle.indices(outer, (2,)), dtype=np.int64)
            signal = SignalCache(bundle, outer_indices, mean, std, device, v1.prepare)
            for seed in seeds:
                eeg_path, lite_path = checkpoint_path(dataset, fold, seed, "EEGNet"), checkpoint_path(dataset, fold, seed, "LiteBN")
                eeg = load_carrier("EEGNet", bundle.channels, eeg_path, device, EEGNet, carrier.CompactLite); lite = load_carrier("LiteBN", bundle.channels, lite_path, device, EEGNet, carrier.CompactLite)
                before = {"EEGNet": state_sha256(eeg), "LiteBN": state_sha256(lite)}
                global_head = GlobalAlpha().to(device); global_head.load_state_dict(torch.load(head_path(dataset, fold, seed, "GLOBAL-ALPHA"), map_location=device, weights_only=False)["state"], strict=True)
                cfrf = CFRF(131).to(device); cfrf.load_state_dict(torch.load(head_path(dataset, fold, seed, "CFRF-v1"), map_location=device, weights_only=False)["state"], strict=True)
                raw_cfrf = CFRF(131).to(device); raw_cfrf.load_state_dict(torch.load(head_path(dataset, fold, seed, "CFRF-v1", raw=True), map_location=device, weights_only=False)["state"], strict=True)
                part_rows, part_alpha, part_raw = evaluate_subjects(eeg, lite, global_head, cfrf, raw_cfrf, signal, bundle, outer)
                for row in part_rows: row.update({"fold": fold, "seed": seed})
                for row in part_alpha: row.update({"fold": fold, "seed": seed})
                for row in part_raw: row.update({"fold": fold, "seed": seed})
                rows.extend(part_rows); alpha_rows.extend(part_alpha); raw_rows.extend(part_raw)
                after = {"EEGNet": state_sha256(eeg), "LiteBN": state_sha256(lite)}
                invariants.append({"dataset": dataset, "fold": fold, "seed": seed, "carrier_state_before": before, "carrier_state_after": after, "state_identical": before == after, "source_checkpoint_sha256": {"EEGNet": sha256_file(eeg_path), "LiteBN": sha256_file(lite_path)}})
    if not all(row["state_identical"] for row in invariants): raise RuntimeError("CFRF_CARRIER_STATE_INVARIANT_VIOLATION")
    OUT.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(OUT / "PHASE_A_SUBJECT_RESULTS.csv", index=False); write_json(OUT / "FUSION_ALPHA_DIAGNOSTICS.json", alpha_rows); write_json(OUT / "RAW_EPOCH30_SECONDARY.json", raw_rows); write_json(OUT / "CARRIER_STATE_INVARIANTS.json", invariants)
    print("CFRF_OUTER_DEVELOPMENT_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
