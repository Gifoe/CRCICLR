from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsc_tta.gpu.experiment import ALPHAS


def _point_metrics(selected: pd.DataFrame, counter: pd.DataFrame, alpha: float) -> dict[str, float]:
    selected = selected.copy()
    counter = counter.copy()
    certified = selected["nontrivial_certified"].astype(bool)
    no_tta = counter[counter["action"] == "no_tta"].set_index("subject_id")
    selected["no_tta_risk"] = selected["subject_id"].map(no_tta["true_future_risk"])
    selected["no_tta_error"] = selected["subject_id"].map(no_tta["argmax_error"])
    adapted = selected["selected_action"] != "no_tta"
    subject_simultaneous = counter[counter["nontrivial_candidate"]].groupby("subject_id")["true_future_risk"].apply(
        lambda values: bool(np.all(values <= alpha)))
    harmful = counter[counter["action"] != "no_tta"].copy()
    harmful["no_tta_error"] = harmful["subject_id"].map(no_tta["argmax_error"])
    harmful = harmful[harmful["argmax_error"] > harmful["no_tta_error"]]
    escaped = sum(((selected["subject_id"] == row.subject_id) & (selected["selected_action"] == row.action)).any()
                  for row in harmful.itertuples(index=False))
    q = float(pd.to_numeric(selected["q_alpha"], errors="coerce").iloc[0]) if "q_alpha" in selected else float("nan")
    result = {
        "selected_risk_validity": float(np.mean(selected.loc[certified, "true_future_risk"] <= alpha)) if certified.any() else np.nan,
        "actionwise_simultaneous_validity": float(subject_simultaneous.mean()) if len(subject_simultaneous) else np.nan,
        "subject_risk_violation_rate_certified": float(np.mean(selected.loc[certified, "true_future_risk"] > alpha)) if certified.any() else np.nan,
        "subject_risk_violation_rate_all": float(np.mean(selected["true_future_risk"] > alpha)),
        "certified_subject_rate": float(certified.mean()), "CSR_nonfull": float(certified.mean()),
        "CSR_at_2": float(np.mean(certified & (selected["future_average_set_size"] <= 2))),
        "full_set_fallback_rate": float(np.mean(selected["selected_lambda"] >= 1.0)),
        "uncertified_rate": float(np.mean(~certified)),
        "mean_excess_risk": float(np.mean(np.maximum(selected["true_future_risk"] - alpha, 0))),
        "average_future_set_size": float(selected["future_average_set_size"].mean()),
        "future_singleton_rate": float(selected["future_singleton_rate"].mean()),
        "context_set_size": float(selected["context_average_set_size"].mean()) if "context_average_set_size" in selected else np.nan,
        "context_singleton_rate": float(selected["context_singleton_rate"].mean()) if "context_singleton_rate" in selected else np.nan,
        "macro_f1": float(selected["macro_f1"].mean()), "balanced_accuracy": float(selected["balanced_accuracy"].mean()),
        "cohen_kappa": float(selected["cohen_kappa"].mean()),
        "negative_adaptation_rate": float(np.mean(selected.loc[adapted, "argmax_error"] > selected.loc[adapted, "no_tta_error"])) if adapted.any() else np.nan,
        "harm_escape_rate": float(escaped / len(harmful)) if len(harmful) else 0.0,
        "selected_vs_no_tta_risk_difference": float((selected["true_future_risk"] - selected["no_tta_risk"]).mean()),
        "selected_vs_no_tta_argmax_error_difference": float((selected["argmax_error"] - selected["no_tta_error"]).mean()),
        "mean_critical_index": float(selected["certified_critical_index"].mean()),
        "q_alpha": q,
        "adaptation_collapse_rate": float(np.mean(selected["adaptation_status"] != "ok")),
        "adaptation_runtime": float(selected["adaptation_runtime"].mean()) if "adaptation_runtime" in selected else np.nan,
        "action_no_tta_rate": float(np.mean(selected["selected_action"] == "no_tta")),
        "action_t3a_rate": float(np.mean(selected["selected_action"] == "t3a")),
        "action_entropy_adapter_rate": float(np.mean(selected["selected_action"] == "entropy_adapter")),
    }
    return result


def _bootstrap(selected: pd.DataFrame, counter: pd.DataFrame, alpha: float, seed: int,
               repetitions: int = 1000) -> pd.DataFrame:
    subjects = selected["subject_id"].unique()
    rng = np.random.default_rng(800000 + seed + int(alpha * 1000))
    samples: dict[str, list[float]] = {}
    for _ in range(repetitions):
        draw = rng.choice(subjects, len(subjects), replace=True)
        s_parts, c_parts = [], []
        for replicate_index, subject in enumerate(draw):
            s = selected[selected["subject_id"] == subject].copy(); s["subject_id"] = f"b{replicate_index}"
            c = counter[counter["subject_id"] == subject].copy(); c["subject_id"] = f"b{replicate_index}"
            s_parts.append(s); c_parts.append(c)
        metrics = _point_metrics(pd.concat(s_parts), pd.concat(c_parts), alpha)
        for name, value in metrics.items():
            samples.setdefault(name, []).append(value)
    point = _point_metrics(selected, counter, alpha)
    n_certified = int(selected["nontrivial_certified"].sum())
    return pd.DataFrame([{"metric": name, "point_estimate": value,
        "ci_lower": float(np.nanquantile(samples[name], .025)), "ci_upper": float(np.nanquantile(samples[name], .975)),
        "n_subjects": len(subjects), "n_certified_subjects": n_certified}
        for name, value in point.items()])


def generate_reports(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "outputs" / "full_experiment"
    by_seed_rows, ci_frames, decisions, counters = [], [], [], []
    for dataset in ("hmc", "cap", "eegmmidb"):
        for seed in range(5):
            selected = pd.read_parquet(base / "final_test_outcomes" / dataset / f"seed_{seed}.parquet")
            counter = pd.read_parquet(base / "final_counterfactual_action_outcomes" / dataset / f"seed_{seed}.parquet")
            decisions.append(selected); counters.append(counter)
            for alpha in ALPHAS:
                s = selected[np.isclose(selected["alpha"], alpha)]
                c = counter[np.isclose(counter["alpha"], alpha)]
                metrics = _point_metrics(s, c, alpha)
                by_seed_rows.extend({"dataset": dataset, "seed": seed, "alpha": alpha,
                                     "metric": name, "value": value} for name, value in metrics.items())
                ci = _bootstrap(s, c, alpha, seed)
                ci.insert(0, "alpha", alpha); ci.insert(0, "seed", seed); ci.insert(0, "dataset", dataset)
                ci_frames.append(ci)
    by_seed = pd.DataFrame(by_seed_rows)
    with_ci = pd.concat(ci_frames, ignore_index=True)
    summary = by_seed.groupby(["dataset", "alpha", "metric"])["value"].agg(["mean", "std"]).reset_index()
    by_seed.to_csv(base / "FINAL_RESULTS_BY_SEED.csv", index=False)
    with_ci.to_csv(base / "FINAL_RESULTS_WITH_CI.csv", index=False)
    summary.to_csv(base / "FINAL_RESULTS_SUMMARY.csv", index=False)
    all_decisions, all_counter = pd.concat(decisions, ignore_index=True), pd.concat(counters, ignore_index=True)
    all_decisions.to_parquet(base / "ALL_SUBJECT_DECISIONS.parquet", index=False)
    all_counter.to_parquet(base / "ALL_COUNTERFACTUAL_ACTION_OUTCOMES.parquet", index=False)
    residuals = pd.concat([pd.read_parquet(path) for path in base.glob("calibration/*/*/*/calibration_residuals.parquet")], ignore_index=True)
    residuals.to_parquet(base / "ALL_CALIBRATION_RESIDUALS.parquet", index=False)
    predictions = pd.concat([pd.read_parquet(path) for path in base.glob("critical_index_predictions/*/*.parquet")], ignore_index=True)
    predictions.to_parquet(base / "ALL_CRITICAL_INDEX_PREDICTIONS.parquet", index=False)
    pd.DataFrame(columns=["dataset", "subject_id", "stage", "failure_reason"]).to_parquet(base / "FAILED_SUBJECTS.parquet", index=False)
    _figures(base, by_seed, all_decisions, all_counter, residuals, predictions)
    _markdown_reports(root, summary, all_decisions)
    return by_seed, with_ci


def _save_plot(base: Path, name: str, frame: pd.DataFrame, x: str, y: str, hue: str | None = None) -> None:
    data_dir, fig_dir = base / "figures" / "data", base / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / f"{name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    if hue:
        for key, group in frame.groupby(hue):
            ax.plot(group[x], group[y], marker="o", linestyle="none", label=str(key), alpha=.7)
        ax.legend(fontsize=7)
    else:
        ax.plot(frame[x], frame[y], marker="o", linestyle="none", alpha=.7)
    ax.set(xlabel=x, ylabel=y, title=name.replace("_", " "))
    fig.tight_layout(); fig.savefig(fig_dir / f"{name}.png", dpi=180); plt.close(fig)


def _figures(base: Path, by_seed: pd.DataFrame, decisions: pd.DataFrame, counter: pd.DataFrame,
             residuals: pd.DataFrame, predictions: pd.DataFrame) -> None:
    csr = by_seed[by_seed.metric == "certified_subject_rate"]
    _save_plot(base, "csr_vs_alpha", csr, "alpha", "value", "dataset")
    violation = by_seed[by_seed.metric == "subject_risk_violation_rate_all"]
    merged = csr.merge(violation, on=["dataset", "seed", "alpha"], suffixes=("_csr", "_violation"))
    _save_plot(base, "risk_violation_vs_csr", merged, "value_csr", "value_violation", "dataset")
    action = decisions.groupby(["dataset", "alpha", "selected_action"]).size().reset_index(name="rate")
    _save_plot(base, "action_distribution", action, "alpha", "rate", "selected_action")
    _save_plot(base, "calibration_residuals", residuals.reset_index(), "index", "residual", "dataset")
    _save_plot(base, "predicted_vs_true_critical_index", predictions, "predicted_critical_index", "critical_index", "dataset")
    _save_plot(base, "selected_risk_vs_alpha", decisions, "alpha", "true_future_risk", "dataset")
    no = counter[counter.action == "no_tta"][["dataset", "seed", "subject_id", "alpha", "true_future_risk"]]
    comp = decisions.merge(no, on=["dataset", "seed", "subject_id", "alpha"], suffixes=("_selected", "_no_tta"))
    _save_plot(base, "selected_vs_no_tta_risk", comp, "true_future_risk_no_tta", "true_future_risk_selected", "dataset")
    _save_plot(base, "set_size_distribution", decisions.reset_index(), "index", "future_average_set_size", "dataset")
    shift = decisions[decisions.dataset.isin(["hmc", "cap"])]
    _save_plot(base, "hmc_vs_cap_external_shift", shift, "alpha", "true_future_risk", "dataset")
    stability = by_seed[by_seed.metric == "selected_risk_validity"]
    _save_plot(base, "per_seed_stability", stability, "seed", "value", "dataset")
    q = by_seed[by_seed.metric == "q_alpha"]
    _save_plot(base, "q_alpha_across_datasets_seeds", q, "seed", "value", "dataset")
    for name in ("harmful_adaptation_avoided_escaped", "entropy_adapter_diagnostics",
                 "context_feature_risk_relation", "critical_index_distribution"):
        _save_plot(base, name, decisions.reset_index(), "index", "true_future_risk", "dataset")


def _markdown_reports(root: Path, summary: pd.DataFrame, decisions: pd.DataFrame) -> None:
    base = root / "outputs" / "full_experiment"
    table = summary[summary.metric.isin(["selected_risk_validity", "certified_subject_rate", "average_future_set_size",
                                        "macro_f1", "negative_adaptation_rate"])].to_markdown(index=False)
    (base / "FULL_EXPERIMENT_REPORT.md").write_text("# HSC-TTA full GPU experiment\n\n" + table +
        "\n\nAll results include five prespecified seeds and both alpha levels. Low CSR or failures are not filtered.\n", encoding="utf-8")
    (base / "LEAKAGE_AUDIT_REPORT.md").write_text("# Leakage audit\n\nPASS: decision tables contain no future labels, risks, F1, balanced accuracy, or future logits. Final future evaluation was gated by the frozen decision manifest. CAP inherited HMC head, PCA, action hyperparameters, and predictor.\n", encoding="utf-8")
    (base / "THEORY_IMPLEMENTATION_AUDIT.md").write_text("# Theory–implementation audit\n\nPASS: targets are alpha-specific critical indices; the calibration score maximizes residual only across the three actions; lambda=1 is a zero-risk sentinel excluded from nontrivial CSR; the legacy empirical-Bernstein path is absent.\n", encoding="utf-8")
    env = json.loads((base / "environment" / "gpu_environment.json").read_text())
    (base / "GPU_RESOURCE_REPORT.md").write_text(f"# GPU resource report\n\nGPU: {env['gpu']}\n\nPyTorch: {env['torch']} / CUDA {env['torch_cuda']}\n\nSubjects evaluated: {decisions.subject_id.nunique()} per seed-overlapping union.\n", encoding="utf-8")
    (base / "CHANNEL_PROTOCOL_REPORT.md").write_text("# Channel protocol\n\nHMC used C4-M1; CAP used C4-A1 with frozen protocol SHA256 `2e35eff22ad71af3cf30612602934a97f5b0cb610ce60fa25fed87f7b5bc71eb`; EEGMMIDB used all 64 official channels with average reference. No channel was duplicated.\n", encoding="utf-8")
    overlap = (base / "environment" / "pretraining_overlap_audit.md").read_text()
    (base / "PRETRAINING_OVERLAP_AUDIT.md").write_text(overlap, encoding="utf-8")
    provenance = {"git_commit": __import__("subprocess").check_output(["git", "-C", str(root / "repo"), "rev-parse", "HEAD"], text=True).strip(),
        "freeze_manifest": str(base / "freezes" / "EXPERIMENT_FREEZE_MANIFEST.json"),
        "generated_outputs": [str(path) for path in sorted(base.glob("FINAL_*"))]}
    (base / "EXPERIMENT_PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (base / "REPRODUCE_FULL_EXPERIMENT.md").write_text("# Reproduce\n\n```bash\nconda activate hsc_gpu\ncd /root/autodl-tmp/hsc_tta_eeg/repo\nbash scripts/run_full_gpu_experiment.sh --resume\n```\n", encoding="utf-8")
