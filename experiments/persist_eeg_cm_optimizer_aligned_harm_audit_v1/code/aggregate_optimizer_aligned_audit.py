#!/usr/bin/env python3
"""Aggregate optimizer-aligned source-only harm observations with subject bootstrap."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(os.environ.get("OPTALIGN_REPO", "/root/rivermind-data/CRCICLR_OPTALIGN_WORK")).resolve()
OUT = REPO / "experiments/persist_eeg_cm_optimizer_aligned_harm_audit_v1/outputs"
SEED, DRAWS = 0, 10_000
DATASETS, SPACES, KS = ("OpenBMI", "WBCIC"), ("C", "M", "CM"), (1, 2, 4)


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def subject_key(value: str) -> tuple[int, str]:
    text = str(value)
    try:
        return (int(text.replace("sub-", "")), text)
    except ValueError:
        return (10**9, text)


def point_metrics(frame: pd.DataFrame, score: str) -> tuple[float, float]:
    y = frame.harm_label.to_numpy(dtype=int)
    values = frame[score].to_numpy(dtype=float)
    if len(np.unique(y)) != 2:
        raise RuntimeError(f"undefined AUROC for {score}: only one harm class")
    return float(roc_auc_score(y, values)), float(spearmanr(values, frame.future_harm.to_numpy(dtype=float)).statistic)


def _weighted_midranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Midranks of a bootstrap-expanded vector, without materializing duplicates."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    group_id_sorted = np.cumsum(np.r_[True, sorted_values[1:] != sorted_values[:-1]]) - 1
    ws = weights[:, order]
    totals = np.add.reduceat(ws, starts, axis=1)
    before = np.c_[np.zeros(len(weights), dtype=float), np.cumsum(totals, axis=1)[:, :-1]]
    mid = before + (totals + 1.0) / 2.0
    rank_sorted = mid[:, group_id_sorted]
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    return rank_sorted[:, inverse]


def _weighted_auc_from_ranks(ranks: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    positives = weights @ labels.astype(float)
    totals = weights.sum(axis=1)
    negatives = totals - positives
    numerator = (weights * labels[None, :] * ranks).sum(axis=1) - positives * (positives + 1.0) / 2.0
    return numerator / (positives * negatives)


def _weighted_spearman(score_ranks: np.ndarray, harm_ranks: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = weights.sum(axis=1)
    mean_x = (weights * score_ranks).sum(axis=1) / total
    mean_y = (weights * harm_ranks).sum(axis=1) / total
    dx, dy = score_ranks - mean_x[:, None], harm_ranks - mean_y[:, None]
    covariance = (weights * dx * dy).sum(axis=1)
    denominator = np.sqrt((weights * dx * dx).sum(axis=1) * (weights * dy * dy).sum(axis=1))
    return covariance / denominator


def cluster_bootstrap(frame: pd.DataFrame, dataset: str, space: str, k: int) -> pd.DataFrame:
    """Exact cluster resampling at B subject, retaining each subject's row cluster."""
    subjects = sorted(frame.subject_B.astype(str).unique().tolist(), key=subject_key)
    lookup = {subject: number for number, subject in enumerate(subjects)}
    group_index = frame.subject_B.astype(str).map(lookup).to_numpy(dtype=int)
    rng = np.random.default_rng(stable_seed("optimizer-aligned-subject-bootstrap", dataset, space, k, SEED))
    sampled = rng.integers(0, len(subjects), size=(DRAWS, len(subjects)))
    counts = np.zeros((DRAWS, len(subjects)), dtype=float)
    for row in range(DRAWS):
        counts[row] = np.bincount(sampled[row], minlength=len(subjects))
    weights = counts[:, group_index]
    labels = frame.harm_label.to_numpy(dtype=int)
    harm = frame.future_harm.to_numpy(dtype=float)
    harm_ranks = _weighted_midranks(harm, weights)
    output: dict[str, Any] = {"dataset": dataset, "space": space, "K": k, "draw": np.arange(DRAWS, dtype=int)}
    for label, column in (("same", "c_same"), ("different", "c_different"), ("global", "c_global")):
        score_ranks = _weighted_midranks(frame[column].to_numpy(dtype=float), weights)
        output[f"{label}_auroc"] = _weighted_auc_from_ranks(score_ranks, labels, weights)
        output[f"{label}_spearman"] = _weighted_spearman(score_ranks, harm_ranks, weights)
    result = pd.DataFrame(output)
    result["auroc_advantage"] = result.same_auroc - result.different_auroc
    result["spearman_advantage"] = result.same_spearman - result.different_spearman
    if not np.isfinite(result.drop(columns=["dataset", "space"]).to_numpy(dtype=float)).all():
        raise RuntimeError(f"non-finite subject bootstrap: {dataset}/{space}/K{k}")
    return result


def ci(draws: pd.DataFrame, column: str) -> tuple[float, float]:
    return float(draws[column].quantile(.025)), float(draws[column].quantile(.975))


def fold_consistency(frame: pd.DataFrame, dataset: str, space: str) -> pd.DataFrame:
    rows = []
    for fold, part in frame.groupby("fold", sort=True):
        same, _ = point_metrics(part, "c_same")
        different, _ = point_metrics(part, "c_different")
        rows.append({"dataset": dataset, "space": space, "K": 4, "fold": int(fold),
                     "same_auroc": same, "different_auroc": different,
                     "auroc_advantage": same - different, "nonnegative_advantage": same - different >= 0.0})
    return pd.DataFrame(rows)


def summaries(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_rows, global_rows, bootstraps, fold_rows = [], [], [], []
    for dataset in DATASETS:
        for space in SPACES:
            for k in KS:
                frame = observations[(observations.dataset == dataset) & (observations.space == space) & (observations.K == k)].copy()
                same_auc, same_rho = point_metrics(frame, "c_same")
                diff_auc, diff_rho = point_metrics(frame, "c_different")
                global_auc, global_rho = point_metrics(frame, "c_global")
                draws = cluster_bootstrap(frame, dataset, space, k)
                bootstraps.append(draws)
                same_auc_ci, same_rho_ci = ci(draws, "same_auroc"), ci(draws, "same_spearman")
                adv_auc_ci, adv_rho_ci = ci(draws, "auroc_advantage"), ci(draws, "spearman_advantage")
                row = {
                    "dataset": dataset, "space": space, "K": k, "observations": len(frame),
                    "target_subjects": int(frame.subject_B.nunique()), "harm_rate": float(frame.harm_label.mean()),
                    "same_auroc": same_auc, "same_auroc_ci95_lower": same_auc_ci[0], "same_auroc_ci95_upper": same_auc_ci[1],
                    "same_spearman": same_rho, "same_spearman_ci95_lower": same_rho_ci[0], "same_spearman_ci95_upper": same_rho_ci[1],
                    "different_auroc": diff_auc, "different_spearman": diff_rho,
                    "auroc_advantage": same_auc - diff_auc, "auroc_advantage_ci95_lower": adv_auc_ci[0], "auroc_advantage_ci95_upper": adv_auc_ci[1],
                    "spearman_advantage": same_rho - diff_rho, "spearman_advantage_ci95_lower": adv_rho_ci[0], "spearman_advantage_ci95_upper": adv_rho_ci[1],
                    "primary_K4": k == 4,
                }
                primary_rows.append(row)
                global_rows.append({"dataset": dataset, "space": space, "K": k, "observations": len(frame),
                                    "global_auroc": global_auc, "global_auroc_ci95_lower": ci(draws, "global_auroc")[0],
                                    "global_auroc_ci95_upper": ci(draws, "global_auroc")[1], "global_spearman": global_rho,
                                    "global_spearman_ci95_lower": ci(draws, "global_spearman")[0],
                                    "global_spearman_ci95_upper": ci(draws, "global_spearman")[1]})
                if k == 4:
                    fold_rows.append(fold_consistency(frame, dataset, space))
                print(f"BOOTSTRAP_COMPLETE {dataset} space={space} K={k}", flush=True)
    primary = pd.DataFrame(primary_rows)
    bootstrap = pd.concat(bootstraps, ignore_index=True)
    folds = pd.concat(fold_rows, ignore_index=True)
    global_summary = pd.DataFrame(global_rows)
    krows = []
    for (dataset, space), part in primary.groupby(["dataset", "space"], sort=False):
        ordered = part.set_index("K").loc[list(KS)]
        krows.append({"dataset": dataset, "space": space,
                      "K1_AUROC": ordered.loc[1, "same_auroc"], "K2_AUROC": ordered.loc[2, "same_auroc"], "K4_AUROC": ordered.loc[4, "same_auroc"],
                      "K1_Spearman": ordered.loc[1, "same_spearman"], "K2_Spearman": ordered.loc[2, "same_spearman"], "K4_Spearman": ordered.loc[4, "same_spearman"],
                      "Delta_AUROC_K2_minus_K1": ordered.loc[2, "same_auroc"] - ordered.loc[1, "same_auroc"],
                      "Delta_AUROC_K4_minus_K2": ordered.loc[4, "same_auroc"] - ordered.loc[2, "same_auroc"],
                      "Delta_Spearman_K2_minus_K1": ordered.loc[2, "same_spearman"] - ordered.loc[1, "same_spearman"],
                      "Delta_Spearman_K4_minus_K2": ordered.loc[4, "same_spearman"] - ordered.loc[2, "same_spearman"]})
    return primary, global_summary, bootstrap, folds, pd.DataFrame(krows)


def decision_and_report(summary: pd.DataFrame, global_summary: pd.DataFrame, folds: pd.DataFrame,
                        ksummary: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    decisions, comparisons = {}, []
    previous = {"OpenBMI": (0.6432259213, 0.2599282625), "WBCIC": (0.6034500357, 0.2213600083)}
    for space in SPACES:
        decisions[space] = {"datasets": {}, "dual_mi_status": "NOT_DUAL_MI_SUPPORTED"}
        for dataset in DATASETS:
            row = summary[(summary.dataset == dataset) & (summary.space == space) & (summary.K == 4)].iloc[0].to_dict()
            f = folds[(folds.dataset == dataset) & (folds.space == space)]
            nonnegative = int(f.nonnegative_advantage.sum())
            passed = bool(row["same_auroc"] >= .60 and row["same_auroc_ci95_lower"] > .50
                          and row["same_spearman"] > 0 and row["same_spearman_ci95_lower"] > 0
                          and row["auroc_advantage"] > 0 and row["auroc_advantage_ci95_lower"] > 0
                          and row["spearman_advantage"] > 0 and row["spearman_advantage_ci95_lower"] > 0
                          and nonnegative >= 4)
            row.update({"fold_nonnegative_auroc_advantage": nonnegative, "folds_total": int(len(f)), "pass": passed})
            decisions[space]["datasets"][dataset] = row
        decisions[space]["dual_mi_status"] = "DUAL_MI_SUPPORTED" if all(
            decisions[space]["datasets"][dataset]["pass"] for dataset in DATASETS) else "NOT_DUAL_MI_SUPPORTED"
    supported = [space for space in SPACES if decisions[space]["dual_mi_status"] == "DUAL_MI_SUPPORTED"]
    terminal = ("MULTIPLE_SPACES_SUPPORTED" if len(supported) > 1 else
                f"OPTIMIZER_ALIGNED_{supported[0]}_SUPPORTED" if len(supported) == 1 else
                "NO_OPTIMIZER_ALIGNED_DUAL_MI_SIGNAL")
    for dataset in DATASETS:
        for space in SPACES:
            new = summary[(summary.dataset == dataset) & (summary.space == space)]
            k1, k4 = new[new.K == 1].iloc[0], new[new.K == 4].iloc[0]
            old_auc, old_rho = previous[dataset]
            comparisons.append({"dataset": dataset, "space": space,
                                "previous_cosine_AUROC": old_auc if space == "CM" else np.nan,
                                "new_K1_optimizer_aligned_AUROC": k1.same_auroc,
                                "new_K4_optimizer_aligned_AUROC": k4.same_auroc,
                                "previous_cosine_Spearman": old_rho if space == "CM" else np.nan,
                                "new_K1_optimizer_aligned_Spearman": k1.same_spearman,
                                "new_K4_optimizer_aligned_Spearman": k4.same_spearman,
                                "K4_same_minus_different_AUROC_advantage": k4.auroc_advantage,
                                "K4_same_minus_different_Spearman_advantage": k4.spearman_advantage,
                                "K4_pass": decisions[space]["datasets"][dataset]["pass"],
                                "previous_cosine_note": "previous metric existed for CM only" if space != "CM" else "exact previous CM-GA raw-cosine value"})
    comparison = pd.DataFrame(comparisons)
    worst = {}
    for space in SPACES:
        values = [decisions[space]["datasets"][dataset] for dataset in DATASETS]
        worst[space] = {"worst_dataset_auroc": min(value["same_auroc"] for value in values),
                        "worst_dataset_auroc_advantage": min(value["auroc_advantage"] for value in values),
                        "dual_mi_status": decisions[space]["dual_mi_status"]}
    strongest = max(SPACES, key=lambda space: worst[space]["worst_dataset_auroc"])
    simplest = next((space for space in ("C", "M", "CM") if space in supported), None)
    result = {
        "terminal": terminal, "supported_spaces": supported, "simplest_supported_space": simplest,
        "strongest_worst_dataset_space": strongest, "space_summary": worst, "space_decisions": decisions,
        "DEVELOPMENT_OUTER_ACCESSED": "NO", "EXPOSED_BENCHMARK_ACCESSED": "NO",
        "NEW_SEALED_TEST_ACCESSED": "NO", "EEG_MODEL_TRAINED": "NO", "seed1_run": False, "seed2_run": False,
        "bootstrap": {"unit": "target biological subject B", "draws": DRAWS, "method": "exact cluster resampling with retained within-B rows"},
    }
    lines = ["# Optimizer-aligned LiteBN C/M harm-audit report", "", f"Terminal: `{terminal}`.", "",
             "Only canonical inner-training subjects and source sessions were accessed. No model training, development outer, exposed benchmark, heldout, or sealed test access occurred.", "",
             "## Primary table (K=1/2/4; pass is evaluated only at K=4)", "",
             "| Dataset | Space | K | same AUROC | AUROC CI | same Spearman | Spearman CI | different AUROC | AUROC advantage | advantage CI | Spearman advantage | advantage CI | Pass |",
             "|---|---|---:|---:|---|---:|---|---:|---:|---|---:|---|---|"]
    for _, row in summary.sort_values(["dataset", "space", "K"]).iterrows():
        flag = decisions[row.space]["datasets"][row.dataset]["pass"] if int(row.K) == 4 else "diagnostic"
        lines.append(f"| {row.dataset} | {row.space} | {int(row.K)} | {row.same_auroc:.4f} | [{row.same_auroc_ci95_lower:.4f}, {row.same_auroc_ci95_upper:.4f}] | {row.same_spearman:.4f} | [{row.same_spearman_ci95_lower:.4f}, {row.same_spearman_ci95_upper:.4f}] | {row.different_auroc:.4f} | {row.auroc_advantage:+.4f} | [{row.auroc_advantage_ci95_lower:+.4f}, {row.auroc_advantage_ci95_upper:+.4f}] | {row.spearman_advantage:+.4f} | [{row.spearman_advantage_ci95_lower:+.4f}, {row.spearman_advantage_ci95_upper:+.4f}] | {flag} |")
    lines += ["", "## K effect", "", "| Dataset | Space | K1 AUROC | K2 AUROC | K4 AUROC | K1 rho | K2 rho | K4 rho |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in ksummary.sort_values(["dataset", "space"]).iterrows():
        lines.append(f"| {row.dataset} | {row.space} | {row.K1_AUROC:.4f} | {row.K2_AUROC:.4f} | {row.K4_AUROC:.4f} | {row.K1_Spearman:.4f} | {row.K2_Spearman:.4f} | {row.K4_Spearman:.4f} |")
    lines += ["", "## Space decision", "", "| Space | OpenBMI K4 pass | WBCIC K4 pass | Worst-dataset AUROC | Worst-dataset AUROC advantage | Dual-MI status |",
              "|---|---|---|---:|---:|---|"]
    for space in SPACES:
        lines.append(f"| {space} | {decisions[space]['datasets']['OpenBMI']['pass']} | {decisions[space]['datasets']['WBCIC']['pass']} | {worst[space]['worst_dataset_auroc']:.4f} | {worst[space]['worst_dataset_auroc_advantage']:+.4f} | {worst[space]['dual_mi_status']} |")
    cm_new = {dataset: decisions["CM"]["datasets"][dataset] for dataset in DATASETS}
    cm_k = ksummary[ksummary.space == "CM"].set_index("dataset")
    cm_improve = all(cm_k.loc[dataset, "K4_AUROC"] > previous[dataset][0] for dataset in DATASETS)
    k4_beats_k1 = {space: bool((ksummary[ksummary.space == space].K4_AUROC > ksummary[ksummary.space == space].K1_AUROC).all()) for space in SPACES}
    same_better = {
        space: bool(all(
            decisions[space]["datasets"][dataset]["auroc_advantage_ci95_lower"] > 0
            and decisions[space]["datasets"][dataset]["spearman_advantage_ci95_lower"] > 0
            for dataset in DATASETS
        ))
        for space in SPACES
    }
    global_k4 = global_summary[global_summary.K == 4].set_index(["dataset", "space"])
    consensus_useful = {
        space: bool(all(
            global_k4.loc[(dataset, space), "global_auroc"] >= .60
            and global_k4.loc[(dataset, space), "global_auroc_ci95_lower"] > .50
            and global_k4.loc[(dataset, space), "global_spearman"] > 0
            and global_k4.loc[(dataset, space), "global_spearman_ci95_lower"] > 0
            for dataset in DATASETS
        ))
        for space in SPACES
    }
    lines += ["", "## Required answers", "",
              f"1. The audit tests the proposed mismatch directly; CM K=4 is {'higher than the old cosine AUROC on both datasets, which is suggestive but not causal proof of a partial mismatch effect' if cm_improve else 'not higher than the old cosine AUROC on both datasets, so a mismatch explanation is not supported'}.",
              f"2. For CM, optimizer-aligned K=4 AUROC {'exceeds' if cm_improve else 'does not exceed'} the previous raw-cosine AUROC on both MI datasets; gate status remains `{decisions['CM']['dual_mi_status']}`.",
              "3. K=4 versus K=1 is descriptive: " + "; ".join(f"{space}={'higher on both datasets' if k4_beats_k1[space] else 'mixed or lower'}" for space in SPACES) + ".",
              "4. Same-subject information versus the different-subject control: " + "; ".join(f"{space}={'both advantage CIs are positive on both datasets' if same_better[space] else 'does not establish both positive advantage CIs across datasets'}" for space in SPACES) + ".",
              "5. Generic multi-subject consensus: " + "; ".join(f"{space}={'has positive K=4 AUROC and Spearman bootstrap support on both datasets' if consensus_useful[space] else 'does not have that support on both datasets'}" for space in SPACES) + ". It remains secondary and does not override target-subject gates.",
              f"6. The strongest worst-dataset K=4 same-AUROC is `{strongest}` ({worst[strongest]['worst_dataset_auroc']:.4f}).",
              f"7. M alone passes both MI datasets: `{decisions['M']['dual_mi_status'] == 'DUAL_MI_SUPPORTED'}`.",
              f"8. CM passes both MI datasets: `{decisions['CM']['dual_mi_status'] == 'DUAL_MI_SUPPORTED'}`.",
              f"9. A later guarded residual-training experiment is {'justified only as a separate protocol' if supported else 'not justified by this audit'}; this branch does not train one.",
              f"10. The gradient-guard route should {'not be stopped solely on this audit' if supported else 'be stopped under the locked dual-MI rule'}.",
              "", "All confidence intervals are 10,000-draw target-biological-subject cluster bootstrap intervals."]
    (OUT / "FINAL_OPTIMIZER_ALIGNED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result, comparison


def main() -> None:
    path = OUT / "HARM_OBSERVATIONS.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    observations = pd.read_csv(path)
    expected = set(DATASETS) | set(SPACES)
    if set(observations.dataset.unique()) != set(DATASETS) or set(observations.space.unique()) != set(SPACES):
        raise RuntimeError("unexpected task/space scope")
    if not observations.B_blocks_disjoint.all() or not observations.controls_exclude_A_B.all():
        raise RuntimeError("block/control audit failed")
    summary, global_summary, bootstrap, folds, ksummary = summaries(observations)
    result, comparison = decision_and_report(summary, global_summary, folds, ksummary)
    write_csv(OUT / "SAME_VS_DIFFERENT_SUMMARY.csv", summary)
    write_csv(OUT / "GLOBAL_CONSENSUS_SUMMARY.csv", global_summary)
    write_csv(OUT / "SUBJECT_BOOTSTRAP.csv", bootstrap)
    write_csv(OUT / "FOLD_SIGN_CONSISTENCY.csv", folds)
    write_csv(OUT / "K1_K2_K4_SUMMARY.csv", ksummary)
    write_csv(OUT / "PREVIOUS_VS_OPTIMIZER_ALIGNED.csv", comparison)
    write_json(OUT / "FINAL_OPTIMIZER_ALIGNED_DECISION.json", result)
    print(f"OPTIMIZER_ALIGNED_AGGREGATION_COMPLETE terminal={result['terminal']}", flush=True)


if __name__ == "__main__":
    main()
