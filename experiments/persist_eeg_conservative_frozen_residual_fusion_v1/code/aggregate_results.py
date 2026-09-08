"""Subject-level CFRF Phase-A aggregation and frozen decision gates."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "outputs"
RUNTIME = Path(__import__("os").environ.get("CFRF_RUNTIME", "/root/rivermind-data/cfrf_v1_runtime")).resolve()


def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bootstrap(values: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(0); draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return {"mean_pp": float(values.mean() * 100), "median_pp": float(np.median(values) * 100), "ci_low_pp": float(np.quantile(draws, .025) * 100), "ci_high_pp": float(np.quantile(draws, .975) * 100)}


def harm(delta: np.ndarray) -> dict[str, float]:
    return {"gain_ge_plus_1pp": float((delta >= .01).mean()), "tie_abs_lt_0_5pp": float((np.abs(delta) < .005).mean()), "harm_le_minus_1pp": float((delta <= -.01).mean()), "harm_le_minus_3pp": float((delta <= -.03).mean()), "harm_le_minus_5pp": float((delta <= -.05).mean())}


def main() -> int:
    subject_path = OUT / "PHASE_A_SUBJECT_RESULTS.csv"
    if not subject_path.is_file(): raise FileNotFoundError(subject_path)
    frame = pd.read_csv(subject_path)
    expected = {"EEGNet_BA", "LiteBN_BA", "Frozen_LOGIT50_BA", "GLOBAL-ALPHA_BA", "CFRF-v1_BA"}
    if not expected <= set(frame.columns): raise RuntimeError("incomplete subject results")
    dataset_rows: list[dict[str, Any]] = []; fold_rows: list[dict[str, Any]] = []; harm_profile: dict[str, Any] = {}
    for dataset in ("OpenBMI", "WBCIC"):
        ds = frame[frame.dataset == dataset].copy()
        frozen_delta = ds["Frozen_LOGIT50_BA"].to_numpy() - ds["EEGNet_BA"].to_numpy()
        for method in ("GLOBAL-ALPHA", "CFRF-v1"):
            delta_eeg = ds[f"{method}_BA"].to_numpy() - ds["EEGNet_BA"].to_numpy(); delta_frozen = ds[f"{method}_BA"].to_numpy() - ds["Frozen_LOGIT50_BA"].to_numpy(); stats_eeg, stats_frozen = bootstrap(delta_eeg), bootstrap(delta_frozen)
            folds = []
            for fold in range(5):
                group = ds[ds.fold == fold]; feeg = group[f"{method}_BA"].to_numpy() - group["EEGNet_BA"].to_numpy(); ffrozen = group[f"{method}_BA"].to_numpy() - group["Frozen_LOGIT50_BA"].to_numpy(); folds.append((feeg.mean() * 100, ffrozen.mean() * 100))
                fold_rows.append({"dataset": dataset, "fold": fold, "seed": 0, "method": method, "mean_subject_BA": float(group[f"{method}_BA"].mean()), "mean_subject_macro_F1": float(group[f"{method}_macro_F1"].mean()), "mean_subject_accuracy": float(group[f"{method}_accuracy"].mean()), "delta_vs_EEGNet_pp": float(feeg.mean() * 100), "delta_vs_Frozen_LOGIT50_pp": float(ffrozen.mean() * 100), "median_subject_delta_vs_EEGNet_pp": float(np.median(feeg) * 100), "median_subject_delta_vs_Frozen_LOGIT50_pp": float(np.median(ffrozen) * 100), "harm_le_minus_1pp_vs_EEGNet": harm(feeg)["harm_le_minus_1pp"], "harm_le_minus_1pp_vs_Frozen_LOGIT50": float((ffrozen <= -.01).mean()), "n_subjects": int(len(group))})
            dataset_rows.append({"dataset": dataset, "method": method, "EEGNet_BA": float(ds.EEGNet_BA.mean()), "LiteBN_BA": float(ds.LiteBN_BA.mean()), "Frozen_LOGIT50_BA": float(ds.Frozen_LOGIT50_BA.mean()), "method_BA": float(ds[f"{method}_BA"].mean()), "mean_subject_BA": float(ds[f"{method}_BA"].mean()), "median_subject_BA": float(np.median(ds[f"{method}_BA"])), "gain_vs_EEGNet_pp": stats_eeg["mean_pp"], "median_subject_delta_vs_EEGNet_pp": stats_eeg["median_pp"], "bootstrap_ci_vs_EEGNet_low_pp": stats_eeg["ci_low_pp"], "bootstrap_ci_vs_EEGNet_high_pp": stats_eeg["ci_high_pp"], "gain_vs_LOGIT50_pp": stats_frozen["mean_pp"], "median_subject_delta_vs_LOGIT50_pp": stats_frozen["median_pp"], "bootstrap_ci_vs_LOGIT50_low_pp": stats_frozen["ci_low_pp"], "bootstrap_ci_vs_LOGIT50_high_pp": stats_frozen["ci_high_pp"], "improved_subjects_vs_LOGIT50": int((delta_frozen > 1e-8).sum()), "harmed_subjects_vs_LOGIT50": int((delta_frozen < -1e-8).sum()), "tied_subjects_vs_LOGIT50": int((np.abs(delta_frozen) <= 1e-8).sum()), "positive_folds_vs_EEGNet": int(sum(value[0] > 0 for value in folds)), "worst_fold_vs_EEGNet_pp": float(min(value[0] for value in folds)), "worst_fold_vs_LOGIT50_pp": float(min(value[1] for value in folds)), "harm_le_minus_1pp_vs_EEGNet": harm(delta_eeg)["harm_le_minus_1pp"]})
            harm_profile[f"{dataset}/{method}"] = {"vs_EEGNet": harm(delta_eeg), "vs_Frozen_LOGIT50": {"harm_le_minus_1pp": float((delta_frozen <= -.01).mean()), "gain_ge_plus_1pp": float((delta_frozen >= .01).mean())}}
        harm_profile[f"{dataset}/Frozen_LOGIT50_vs_EEGNet"] = harm(frozen_delta)
    folds = pd.DataFrame(fold_rows); results = pd.DataFrame(dataset_rows); folds.to_csv(OUT / "PHASE_A_FOLD_RESULTS.csv", index=False); results.to_csv(OUT / "PHASE_A_DATASET_RESULTS.csv", index=False); write_json(OUT / "HARM_PROFILE.json", harm_profile)
    def row(dataset: str, method: str) -> pd.Series: return results[(results.dataset == dataset) & (results.method == method)].iloc[0]
    o, w, og, wg = row("OpenBMI", "CFRF-v1"), row("WBCIC", "CFRF-v1"), row("OpenBMI", "GLOBAL-ALPHA"), row("WBCIC", "GLOBAL-ALPHA")
    frozen_w_harm = harm_profile["WBCIC/Frozen_LOGIT50_vs_EEGNet"]["harm_le_minus_1pp"]
    cfrf_w_harm = harm_profile["WBCIC/CFRF-v1"]["vs_EEGNet"]["harm_le_minus_1pp"]
    global_w_harm = harm_profile["WBCIC/GLOBAL-ALPHA"]["vs_EEGNet"]["harm_le_minus_1pp"]
    open_preservation = bool(o.gain_vs_EEGNet_pp >= 3.5 and o.positive_folds_vs_EEGNet == 5 and o.gain_vs_LOGIT50_pp >= -.20 and o.worst_fold_vs_LOGIT50_pp >= -1.0)
    wbcic_preservation = bool(w.gain_vs_EEGNet_pp >= 1.0 and w.positive_folds_vs_EEGNet >= 4 and w.worst_fold_vs_EEGNet_pp > -2.0 and w.median_subject_delta_vs_EEGNet_pp > 0)
    performance_added = bool(w.gain_vs_LOGIT50_pp >= .30)
    robustness_added = bool(cfrf_w_harm <= frozen_w_harm - .05 and w.gain_vs_LOGIT50_pp >= -.10)
    cross_dataset_added = bool(o.gain_vs_LOGIT50_pp > 0 and w.gain_vs_LOGIT50_pp > 0 and (o.gain_vs_LOGIT50_pp + w.gain_vs_LOGIT50_pp) / 2 >= .25 and cfrf_w_harm <= frozen_w_harm)
    added = performance_added or robustness_added or cross_dataset_added
    samplewise = bool(w.method_BA >= wg.method_BA + .002 or (cfrf_w_harm <= global_w_harm - .05 and w.method_BA >= wg.method_BA - .001))
    materially_hurts = bool(o.gain_vs_LOGIT50_pp < -.20 or w.gain_vs_LOGIT50_pp < -.10)
    all_pass = open_preservation and wbcic_preservation and added and samplewise
    if all_pass: terminal, selected = "CFRF_STAGE1_SEED0_PASS_EXPAND", None
    elif materially_hurts: terminal, selected = "CFRF_HURTS_VALIDATED_FUSION_USE_FROZEN_LOGIT50", "FROZEN_LOGIT50"
    elif not samplewise and (open_preservation and wbcic_preservation and added): terminal, selected = "CFRF_SAMPLEWISE_HEAD_NOT_JUSTIFIED_USE_FROZEN_LOGIT50", "FROZEN_LOGIT50"
    else: terminal, selected = "CFRF_NO_ADDED_VALUE_USE_FROZEN_LOGIT50", "FROZEN_LOGIT50"
    gate = {"openbmi_preservation": open_preservation, "wbcic_preservation": wbcic_preservation, "added_value_performance": performance_added, "added_value_robustness": robustness_added, "added_value_cross_dataset": cross_dataset_added, "added_value": added, "samplewise_head_justified": samplewise, "materially_hurts_validated_fusion": materially_hurts, "phase_a_pass": all_pass, "terminal": terminal, "selected_final_predictor_if_stop": selected, "wbcic_frozen_harm_le_minus_1pp": frozen_w_harm, "wbcic_cfrf_harm_le_minus_1pp": cfrf_w_harm, "wbcic_global_alpha_harm_le_minus_1pp": global_w_harm}
    write_json(OUT / "PHASE_A_GATE.json", gate)
    global_scalars = []
    for dataset in ("OpenBMI", "WBCIC"):
        for fold in range(5):
            path = RUNTIME / "cells" / f"{dataset.lower()}_fold{fold}_seed0_global_alpha" / "epoch30_scalar.pt"; state = __import__("torch").load(path, map_location="cpu", weights_only=False)["state"]; q = float(state["q"]); alpha = .5 + .25 * math.tanh(q); global_scalars.append({"dataset": dataset, "fold": fold, "seed": 0, "q": q, "alpha": alpha})
    write_json(OUT / "GLOBAL_ALPHA_RESULTS.json", global_scalars)
    lines = ["# CFRF-v1 Phase-A Decision", "", "| Metric | OpenBMI | WBCIC |", "|---|---:|---:|"]
    for label, key, fmt in [("EEGNet BA", "EEGNet_BA", ".4f"), ("LiteBN BA", "LiteBN_BA", ".4f"), ("Frozen LOGIT50 BA", "Frozen_LOGIT50_BA", ".4f"), ("GLOBAL-ALPHA BA", "method_BA", ".4f"), ("CFRF BA", "method_BA", ".4f"), ("CFRF gain vs EEGNet (pp)", "gain_vs_EEGNet_pp", ".3f"), ("CFRF gain vs LOGIT50 (pp)", "gain_vs_LOGIT50_pp", ".3f"), ("Median subject gain vs EEGNet (pp)", "median_subject_delta_vs_EEGNet_pp", ".3f"), ("Positive folds vs EEGNet", "positive_folds_vs_EEGNet", ".0f"), ("Harm <= -1 pp vs EEGNet", "harm_le_minus_1pp_vs_EEGNet", ".3f")]:
        left, right = (og[key], wg[key]) if label == "GLOBAL-ALPHA BA" else (o[key], w[key]); lines.append(f"| {label} | {left:{fmt}} | {right:{fmt}} |")
    lines += ["", f"1. Did CFRF preserve the existing fusion advantage? {'YES' if open_preservation and wbcic_preservation else 'PARTIAL' if not materially_hurts else 'NO'}", f"2. Did CFRF improve WBCIC mean performance? {'YES' if w.gain_vs_LOGIT50_pp > 0 else 'NO'}", f"3. Did CFRF reduce the WBCIC harmful-subject tail? {'YES' if cfrf_w_harm < frozen_w_harm else 'NO'}", f"4. Did CFRF outperform GLOBAL-ALPHA? {'YES' if samplewise else 'NO'}", f"5. Is the sample-wise head justified? {'YES' if samplewise else 'NO'}", f"6. Does CFRF proceed to multiseed? {'YES' if all_pass else 'NO'}", f"7. If NO, is Frozen LOGIT50 selected as fallback final predictor? {'YES' if selected == 'FROZEN_LOGIT50' else 'N/A'}", f"8. Terminal: **{terminal}**", "", "Next action: Freeze the selected final predictor and run the separate final held-out-subject confirmation on OpenBMI holdout subjects and WBCIC true outer subjects."]
    (OUT / "PHASE_A_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if selected:
        write_json(OUT / "FINAL_MODEL_SELECTION.json", {"selected_final_predictor": selected, "basis": terminal, "next_action": "Freeze the selected final predictor and run the separate final held-out-subject confirmation on OpenBMI holdout subjects and WBCIC true outer subjects."})
    print(terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
