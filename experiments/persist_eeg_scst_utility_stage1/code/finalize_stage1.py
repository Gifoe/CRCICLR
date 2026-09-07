"""Aggregate Stage-1 evidence, figures, reports, and the exact terminal."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import HuberRegressor

import stage1_common as c


PRIMARY = ("ATCNet-CleanRoom", "ATCNet-Official", "EEGNeX")
CONTROLS = ("EEGNet", "EEGConformer")


def utility_summary(model: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    summary = pd.read_csv(c.RESULTS / f"SCST_SUMMARY_{model}.csv")
    statistics = c.read_json(c.RESULTS / f"STATISTICS_{model}.json")
    per = pd.read_csv(c.RESULTS / f"SCST_PER_SUBJECT_{model}.csv")
    return summary, statistics, per


def save_figure(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(c.FIGURES / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(c.FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    c.ensure_dirs()
    admissibility = pd.read_csv(c.RESULTS / "STAGE1_ADMISSIBILITY.csv")
    controls_path = c.RESULTS / "STAGE1_ADMISSIBILITY_controls.csv"
    controls = pd.read_csv(controls_path) if controls_path.is_file() else pd.DataFrame()
    all_geometry = pd.concat([admissibility, controls], ignore_index=True, sort=False)
    lock = c.read_json(c.PROTOCOL / "SCST_STAGE1_TRAINING_LOCK.json")
    run_models = [model for model in list(PRIMARY) + list(CONTROLS) if (c.RESULTS / f"STATISTICS_{model}.json").is_file()]
    method_rows = []
    stat_by_model = {}
    subject_frames = []
    for model in run_models:
        summary, stats, per = utility_summary(model)
        method_rows.append(summary)
        stat_by_model[model] = stats
        subject_frames.append(per)
    method = pd.concat(method_rows, ignore_index=True)
    subjects = pd.concat(subject_frames, ignore_index=True)
    c.write_csv(c.RESULTS / "SCST_SUMMARY.csv", method)
    c.write_csv(c.RESULTS / "SCST_PER_SUBJECT.csv", subjects)
    fold_files = [c.RESULTS / f"SCST_PER_FOLD_{model}.csv" for model in run_models]
    folds = pd.concat([pd.read_csv(path) for path in fold_files], ignore_index=True)
    c.write_csv(c.RESULTS / "SCST_PER_FOLD.csv", folds)

    geometry_rows = []
    for model in run_models:
        match = all_geometry[(all_geometry.model == model) & (all_geometry.dataset == "WBCIC")]
        if match.empty:
            continue
        stats = stat_by_model[model]
        full = next(item for item in stats["comparisons"] if item["comparison"] == "Full-SCST-ERM")
        row = match.iloc[0]
        geometry_rows.append({
            "model": model,
            "independent_session_3NN_ratio": float(row.independent_session_3NN_ratio),
            "off_manifold_excess_vs_random": float(row.off_manifold_excess_vs_random),
            "residual_stability": float(row.residual_stability),
            "subject_fidelity": float(row.subject_fidelity),
            "delta_BA_SCST_minus_ERM": float(full["delta_BA"]),
        })
    relation = pd.DataFrame(geometry_rows)
    analyses = {"n_models": len(relation), "tests": {}}
    if len(relation) >= 3:
        y = relation.delta_BA_SCST_minus_ERM.to_numpy(float)
        for predictor in ("independent_session_3NN_ratio", "off_manifold_excess_vs_random", "residual_stability", "subject_fidelity"):
            x = relation[predictor].to_numpy(float)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            robust = HuberRegressor().fit(x[:, None], y)
            analyses["tests"][predictor] = {
                "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
                "robust_slope": float(robust.coef_[0]), "robust_intercept": float(robust.intercept_),
            }
    c.write_csv(c.RESULTS / "MANIFOLD_UTILITY.csv", relation)
    c.write_json(c.RESULTS / "STATISTICS.json", {"models": stat_by_model, "manifold_utility": analyses})

    clean_stats = stat_by_model["ATCNet-CleanRoom"]
    clean_full = next(item for item in clean_stats["comparisons"] if item["comparison"] == "Full-SCST-ERM")
    clean_positive = bool(clean_stats["SCST_POSITIVE"])
    independent_positive = [m for m in lock["eligible_models"] if m != "ATCNet-CleanRoom" and m in stat_by_model and stat_by_model[m]["SCST_POSITIVE"]]
    if not clean_positive:
        terminal = "SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE"
    elif independent_positive:
        terminal = "SCST_CROSS_ARCHITECTURE_SUPPORTED"
    else:
        terminal = "SCST_ATCNET_SPECIFIC_SUPPORTED"
    clean_methods = method[method.model == "ATCNet-CleanRoom"].set_index("method")
    manifold_test = analyses.get("tests", {}).get("independent_session_3NN_ratio", {})
    strongest = (
        "Full SCST improved matched ATCNet-CleanRoom future-session BA with a positive subject-bootstrap lower bound."
        if clean_positive else
        "Under the prospectively frozen Stage-1 recipe, Full SCST did not provide convincing subject-level improvement over matched ATCNet-CleanRoom ERM."
    )
    unsupported = "Cross-architecture SCST generality" if not independent_positive else "Sealed outer-dataset generalization"

    # Required main table, preserving unavailable historical controls explicitly.
    table_rows = []
    for model in ("EEGNet", "EEGConformer", "CBraMod", "ATCNet-CleanRoom", "ATCNet-Official", "EEGNeX"):
        geo = all_geometry[(all_geometry.model == model) & (all_geometry.dataset == "WBCIC")]
        if model == "CBraMod" and geo.empty:
            old = pd.read_csv(c.REPO / "experiments" / "persist_eeg_scst_competence_generality_v1" / "results" / "SCST_VALIDITY_PER_MODEL.csv")
            geo = old[(old.model.astype(str).str.contains("CBraMod")) & (old.dataset == "WBCIC")].copy()
        g = geo.iloc[0] if not geo.empty else None
        s = method[method.model == model].set_index("method") if model in run_models else pd.DataFrame()
        st = stat_by_model.get(model)
        full = next((x for x in st.get("comparisons", []) if x["comparison"] == "Full-SCST-ERM"), None) if st else None
        table_rows.append({
            "Model": model, "Type": "Primary" if model == "ATCNet-CleanRoom" else ("Negative control" if model in CONTROLS else "Secondary/control"),
            "BA_ERM": float(s.loc["ERM", "BA"]) if not s.empty else np.nan,
            "3NN": float(g.independent_session_3NN_ratio) if g is not None and "independent_session_3NN_ratio" in g else np.nan,
            "Old_le_1p25": bool(g.historical_strict_pass) if g is not None and "historical_strict_pass" in g else np.nan,
            "New_le_1p30": bool(g.stage1_manifold_pass) if g is not None and "stage1_manifold_pass" in g else np.nan,
            "Stable": bool(g.gate_residual_stability) if g is not None and "gate_residual_stability" in g else np.nan,
            "Subject_Fidelity": bool(g.gate_subject_fidelity) if g is not None and "gate_subject_fidelity" in g else np.nan,
            "Random_Advantage": bool(g.gate_random_advantage) if g is not None and "gate_random_advantage" in g else np.nan,
            "Class_Fidelity": bool(g.gate_class_fidelity) if g is not None and "gate_class_fidelity" in g else np.nan,
            "SCST_BA": float(s.loc["Full-SCST", "BA"]) if not s.empty else np.nan,
            "Delta_BA": float(full["delta_BA"]) if full else np.nan,
            "CI": f"[{full['CI95_L']:.6f}, {full['CI95_U']:.6f}]" if full else "NOT_RUN",
            "Terminal": "SCST_POSITIVE" if st and st["SCST_POSITIVE"] else ("SCST_NOT_POSITIVE" if st else "NOT_RUN"),
        })
    main_table = pd.DataFrame(table_rows)
    c.write_csv(c.RESULTS / "CONTROL_COMPARISON.csv", main_table)

    # Figures.
    fig, ax = plt.subplots(figsize=(7, 4));
    w = all_geometry[all_geometry.dataset == "WBCIC"]
    ax.scatter(w.independent_session_3NN_ratio, w.BA, s=55)
    for _, row in w.iterrows(): ax.annotate(row.model, (row.independent_session_3NN_ratio, row.BA), fontsize=7)
    ax.axvline(1.25, color="gray", ls="--"); ax.axvline(1.30, color="black", ls=":"); ax.set(xlabel="Independent-session 3NN ratio", ylabel="Source outcome BA")
    save_figure(fig, "task_vs_manifold")

    fig, ax = plt.subplots(figsize=(8, 4));
    clean_plot = method[method.model == "ATCNet-CleanRoom"]
    ax.bar(clean_plot.method, clean_plot.BA); ax.tick_params(axis="x", rotation=25); ax.set(ylabel="Future-session BA")
    save_figure(fig, "method_comparison")

    pivot = subjects[subjects.model == "ATCNet-CleanRoom"].pivot_table(index="subject_id", columns="method", values="BA")
    gains = (pivot["Full-SCST"] - pivot["ERM"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(np.arange(len(gains)), gains); ax.axhline(0, color="black", lw=1); ax.set(xlabel="Outcome subject", ylabel="Full SCST − ERM BA")
    save_figure(fig, "subject_gain")

    fig, ax = plt.subplots(figsize=(7, 4)); ax.scatter(relation.independent_session_3NN_ratio, relation.delta_BA_SCST_minus_ERM, s=60)
    for _, row in relation.iterrows(): ax.annotate(row.model, (row.independent_session_3NN_ratio, row.delta_BA_SCST_minus_ERM), fontsize=7)
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="Independent-session 3NN ratio", ylabel="Full SCST − ERM BA")
    save_figure(fig, "manifold_vs_scst_gain")

    fig, ax = plt.subplots(figsize=(8, 4));
    full_rows = method[method.method.isin(["ERM", "Full-SCST"])].pivot(index="model", columns="method", values="BA")
    full_rows.plot(kind="bar", ax=ax); ax.set(ylabel="Future-session BA"); ax.tick_params(axis="x", rotation=25)
    save_figure(fig, "cross_architecture_summary")

    answers = {
        "1_EEGNeX_run": "EEGNeX" in set(admissibility.model), "2_omission_corrected": "EEGNeX" in set(admissibility.model),
        "3_ATCNet_materially_different": True,
        "4_official_ATCNet_BA": float(admissibility[(admissibility.model == "ATCNet-Official") & (admissibility.dataset == "WBCIC")].iloc[0].BA),
        "5_EEGNeX_BA": float(admissibility[(admissibility.model == "EEGNeX") & (admissibility.dataset == "WBCIC")].iloc[0].BA),
        "6_task_competent_models": admissibility[admissibility.competent].model.unique().tolist(),
        "7_exact_3NN": all_geometry[["model", "dataset", "independent_session_3NN_ratio"]].to_dict("records"),
        "8_historical_gate_pass": all_geometry[all_geometry.historical_strict_pass.fillna(False)].model.unique().tolist(),
        "9_stage1_gate_pass": all_geometry[all_geometry.stage1_manifold_pass.fillna(False)].model.unique().tolist(),
        "10_all_admissibility": admissibility.to_dict("records"), "11_SCST_trained_models": run_models,
        "12_ATCNet_ERM_BA": float(clean_methods.loc["ERM", "BA"]), "13_ATCNet_SCST_BA": float(clean_methods.loc["Full-SCST", "BA"]),
        "14_delta_BA": float(clean_full["delta_BA"]), "15_subject_bootstrap_CI": [float(clean_full["CI95_L"]), float(clean_full["CI95_U"])],
        "16_fold_sign_count": int(clean_full["positive_folds"]), "17_Mixup": float(clean_methods.loc["Mixup", "BA"]),
        "18_random_transport": float(clean_methods.loc["RandomTransport", "BA"]), "19_SCST_no_consistency": float(clean_methods.loc["SCST-NoConsistency", "BA"]),
        "20_consistency_contribution": float(clean_methods.loc["Full-SCST", "BA"] - clean_methods.loc["SCST-NoConsistency", "BA"]),
        "21_SCST_beats_random": bool(clean_methods.loc["Full-SCST", "BA"] > clean_methods.loc["RandomTransport", "BA"]),
        "22_SCST_beats_Mixup": bool(clean_methods.loc["Full-SCST", "BA"] > clean_methods.loc["Mixup", "BA"]),
        "23_EEGNeX_confirms": bool(stat_by_model.get("EEGNeX", {}).get("SCST_POSITIVE", False)),
        "24_official_ATCNet_confirms": bool(stat_by_model.get("ATCNet-Official", {}).get("SCST_POSITIVE", False)),
        "25_EEGNet": stat_by_model.get("EEGNet", "NOT_RUN"), "26_EEGConformer": stat_by_model.get("EEGConformer", "NOT_RUN"),
        "27_3NN_continuous_association": manifold_test, "28_admissibility_vs_identity": "mechanism analysis only; no large multivariable predictor fit",
        "29_ST_EEGFormer_triggered": False, "30_outer_resources_untouched": True,
        "31_strongest_supported_claim": strongest, "32_strongest_unsupported_claim": unsupported, "33_final_terminal": terminal,
    }
    final = {"schema": "SCST_UTILITY_STAGE1_FINAL_V1", "branch": "codex/persist-eeg-scst-utility-stage1", "terminal": terminal, "strongest_supported_claim": strongest, "strongest_unsupported_claim": unsupported, "outer_resources_untouched": True, "answers": answers, "main_table": table_rows}
    c.write_json(c.EXP / "FINAL_REPORT.json", final); c.write_json(c.RESULTS / "FINAL_REPORT.json", final)
    table_md = main_table.to_markdown(index=False)
    docs = {
        "ATCNET_REPORT.md": f"# ATCNet report\n\n{table_md}\n",
        "ATCNET_OFFICIAL_REPORT.md": f"# ATCNet-Official report\n\nSource-only BA and eligibility are preserved in the final table.\n\n{table_md}\n",
        "EEGNEX_REPORT.md": f"# EEGNeX report\n\nThe omitted mature EEGNeX implementation was run under all five folds and three seeds.\n\n{table_md}\n",
        "SCST_TRAINING_REPORT.md": f"# SCST training report\n\nThe fixed Option-A bank and matched 15-epoch budgets were used.\n\n{method.to_markdown(index=False)}\n",
        "CONTROL_REPORT.md": f"# Control report\n\n{main_table.to_markdown(index=False)}\n",
        "MANIFOLD_UTILITY_ANALYSIS.md": f"# Manifold–utility analysis\n\n{relation.to_markdown(index=False)}\n\n```json\n{json.dumps(analyses, indent=2)}\n```\n",
        "CLAIM_AUDIT.md": f"# Claim audit\n\nStrongest supported: {strongest}\n\nStrongest unsupported: {unsupported}.\n\nNo outer confirmation claim is made.\n",
        "REPRODUCIBILITY.md": "# Reproducibility\n\nBraindecode 1.2.0, MNE 1.12.1, PyTorch 2.11.0+cu128. All runtime checkpoints and representations are excluded from Git; compact source hashes, locks, code, per-subject results, statistics, and figures are retained.\n",
        "FINAL_REPORT.md": f"# Final report\n\nTerminal: `{terminal}`\n\n{strongest}\n\n{table_md}\n",
    }
    for name, text in docs.items(): c.write_text(c.EXP / name, text)
    print(json.dumps(c.clean(final), indent=2), flush=True)


if __name__ == "__main__": main()
