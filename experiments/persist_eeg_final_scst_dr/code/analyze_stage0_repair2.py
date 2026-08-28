from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_stage0 as v0_analysis
import common as c
import run_stage0_repair2 as repair


def load_units(name: str) -> pd.DataFrame:
    paths = sorted((c.RUNTIME / "stage0_repair2_units").glob(f"*/fold-*/{name}"))
    expected = len(c.SETTINGS) * 5
    if len(paths) != expected:
        raise RuntimeError(f"{name}: expected {expected} Repair-2 files, found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def summarize(
    stability: pd.DataFrame,
    subject: pd.DataFrame,
    class_fidelity: pd.DataFrame,
    manifold: pd.DataFrame,
) -> pd.DataFrame:
    gate = c.protocol()["per_setting_layer_gates"]
    rows: list[dict[str, Any]] = []
    for setting in c.SETTINGS:
        st = stability[(stability.setting_id == setting) & (stability.layer == repair.LAYER)]
        matched = st[st.control == "matched_subject_same_class"].groupby(
            ["fold", "source_subject", "class_label"], as_index=False
        ).cosine.mean().rename(columns={"cosine": "matched"})
        mismatch = st[st.control == "mismatched_subject_same_class"].groupby(
            ["fold", "source_subject", "class_label"], as_index=False
        ).cosine.mean().rename(columns={"cosine": "mismatch"})
        effect = matched.merge(
            mismatch, on=["fold", "source_subject", "class_label"], validate="one_to_one"
        )
        effect["effect"] = effect.matched - effect.mismatch
        stability_mean, stability_lo, stability_hi = v0_analysis.ci_subject(
            effect, "effect", (setting, repair.LAYER, "stability")
        )

        sf = subject[subject.setting_id == setting]
        scst_sf = sf[sf.method == "scst"]
        sf_mean, sf_lo, sf_hi = v0_analysis.ci_subject(
            scst_sf, "relative_target_affinity_improvement", (setting, repair.LAYER, "subject")
        )
        sf_random = v0_analysis.method_effect(
            sf, "scst", "norm_matched_random", "relative_target_affinity_improvement", "effect"
        )
        random_mean, random_lo, random_hi = v0_analysis.ci_subject(
            sf_random, "effect", (setting, repair.LAYER, "subject_vs_random")
        )

        cf = class_fidelity[class_fidelity.setting_id == setting]
        scst_cf = cf[cf.method == "scst"]
        probe_ba = float(scst_cf.independent_probe_BA.mean())
        accuracy_change, accuracy_lo, accuracy_hi = v0_analysis.ci_subject(
            scst_cf, "accuracy_change", (setting, repair.LAYER, "class_accuracy")
        )
        logp_change, logp_lo, logp_hi = v0_analysis.ci_subject(
            scst_cf, "true_log_probability_change", (setting, repair.LAYER, "class_logp")
        )

        mf = manifold[manifold.setting_id == setting]
        scst_mf = mf[mf.method == "scst"]
        clean_mf = mf[mf.method == "no_transport"]
        random_mf = mf[mf.method == "norm_matched_random"]
        manifold_ratio = float(scst_mf.knn_distance.mean() / max(clean_mf.knn_distance.mean(), 1e-12))
        scst_off = float(scst_mf.off_manifold.astype(float).mean())
        random_off = float(random_mf.off_manifold.astype(float).mean())
        off_excess = scst_off - random_off

        gates = {
            "competence": probe_ba >= float(gate["independent_probe_BA_min"]),
            "stability": stability_mean > 0 and stability_lo > 0,
            "subject_fidelity": sf_mean > 0 and sf_lo > 0 and random_lo > 0,
            "class_fidelity": -accuracy_change <= float(gate["class_accuracy_loss_max"])
            and logp_change >= float(gate["class_true_log_probability_change_min"]),
            "manifold": manifold_ratio <= float(gate["manifold_knn_ratio_to_clean_max"])
            and off_excess <= float(gate["manifold_off_rate_excess_over_random_max"]),
        }
        rows.append(
            {
                "setting_id": setting,
                "dataset": c.SETTINGS[setting]["dataset"],
                "backbone": c.SETTINGS[setting]["backbone"],
                "layer": repair.LAYER,
                "independent_probe_BA": probe_ba,
                "stability_effect_mean": stability_mean,
                "stability_CI_low": stability_lo,
                "stability_CI_high": stability_hi,
                "subject_affinity_improvement_mean": sf_mean,
                "subject_affinity_CI_low": sf_lo,
                "subject_affinity_CI_high": sf_hi,
                "subject_advantage_over_random_mean": random_mean,
                "subject_advantage_over_random_CI_low": random_lo,
                "subject_advantage_over_random_CI_high": random_hi,
                "class_accuracy_change": accuracy_change,
                "class_accuracy_CI_low": accuracy_lo,
                "class_accuracy_CI_high": accuracy_hi,
                "class_logp_change": logp_change,
                "class_logp_CI_low": logp_lo,
                "class_logp_CI_high": logp_hi,
                "manifold_knn_ratio_to_clean": manifold_ratio,
                "scst_off_manifold_rate": scst_off,
                "random_off_manifold_rate": random_off,
                "off_manifold_excess_over_random": off_excess,
                **{f"gate_{key}": value for key, value in gates.items()},
                "all_gates_pass": all(gates.values()),
            }
        )
    return pd.DataFrame(rows)


def compact_metric(frame: pd.DataFrame, values: list[str]) -> pd.DataFrame:
    keys = ["setting_id", "layer", "source_subject", "class_label", "method"]
    grouped = frame.groupby(keys, as_index=False).agg(
        **{value: (value, "mean") for value in values},
        pair_count=("target_subject", "size"),
        fold_count=("fold", "nunique"),
    )
    return grouped


def distribution_rows() -> pd.DataFrame:
    buckets: dict[tuple[str, str, str, str], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    paths = sorted((c.RUNTIME / "stage0_repair2_units").glob("*/fold-*/ALPHA_VALUES.npz"))
    if len(paths) != len(c.SETTINGS) * 5:
        raise RuntimeError(f"expected 20 alpha NPZ files, found {len(paths)}")
    for path in paths:
        setting = path.parent.parent.name
        payload = np.load(path, allow_pickle=False)
        subjects = payload["subject_values"].astype(str)
        for unit in ("centroid", "trial"):
            alpha = payload[f"{unit}_alpha"].astype(np.float64)
            norm = payload[f"{unit}_norm_ratio"].astype(np.float64)
            classes = payload[f"{unit}_class"].astype(int)
            source_code = payload[f"{unit}_source_code"].astype(int)
            buckets[(setting, unit, "setting", "ALL")].append((alpha, norm))
            for label in sorted(np.unique(classes)):
                mask = classes == label
                buckets[(setting, unit, "class", str(label))].append((alpha[mask], norm[mask]))
            for code in sorted(np.unique(source_code)):
                mask = source_code == code
                buckets[(setting, unit, "source_subject", str(subjects[code]))].append((alpha[mask], norm[mask]))

    rows: list[dict[str, Any]] = []
    for (setting, unit, scope, value), parts in sorted(buckets.items()):
        alpha = np.concatenate([part[0] for part in parts])
        norm = np.concatenate([part[1] for part in parts])
        rows.append(
            {
                "setting_id": setting,
                "query_unit": unit,
                "scope": scope,
                "scope_value": value,
                "candidate_count": len(alpha),
                "fraction_alpha_zero": float(np.mean(alpha == 0.0)),
                "alpha_mean": float(alpha.mean()),
                "alpha_median": float(np.median(alpha)),
                "alpha_q25": float(np.quantile(alpha, 0.25)),
                "alpha_q75": float(np.quantile(alpha, 0.75)),
                "fraction_alpha_max": float(np.mean(alpha == repair.ALPHA_MAX)),
                "realized_norm_ratio_mean": float(norm.mean()),
                "realized_norm_ratio_median": float(np.median(norm)),
            }
        )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, alpha: pd.DataFrame) -> None:
    c.FIGURES.mkdir(parents=True, exist_ok=True)
    setting_rows = alpha[(alpha.scope == "setting") & (alpha.query_unit == "centroid")].set_index("setting_id").loc[list(c.SETTINGS)]
    labels = [setting.replace("_MI_", "\n") for setting in c.SETTINGS]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    axes[0].bar(np.arange(len(labels)), setting_rows.alpha_mean, color="#2166ac")
    axes[0].scatter(np.arange(len(labels)), setting_rows.alpha_median, color="black", s=30, label="median")
    axes[0].axhline(0.25, color="#b2182b", ls="--", lw=0.9)
    axes[0].set_xticks(np.arange(len(labels)), labels, fontsize=8)
    axes[0].set_ylabel("Realized alpha_star")
    axes[0].set_title("A  Source-support-constrained step")
    axes[0].legend(frameon=False, fontsize=8)
    ordered = summary.set_index("setting_id").loc[list(c.SETTINGS)]
    colors = np.where(ordered.gate_manifold, "#2166ac", "#b2182b")
    axes[1].bar(np.arange(len(labels)), ordered.manifold_knn_ratio_to_clean, color=colors)
    axes[1].axhline(1.25, color="black", ls="--", lw=0.9, label="frozen gate")
    axes[1].set_xticks(np.arange(len(labels)), labels, fontsize=8)
    axes[1].set_ylabel("Independent-session 3NN / clean")
    axes[1].set_title("B  Non-circular manifold validation")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(c.FIGURES / "FIGURE_STAGE0_REPAIR2_SUPPORT_AND_MANIFOLD.png", dpi=300)
    fig.savefig(c.FIGURES / "FIGURE_STAGE0_REPAIR2_SUPPORT_AND_MANIFOLD.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for row in ordered.itertuples():
        ax.scatter(row.subject_affinity_improvement_mean, row.class_accuracy_change, s=82, label=row.setting_id)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(-0.02, color="#b2182b", ls="--", lw=0.9)
    ax.set_xlabel("Target-subject affinity improvement")
    ax.set_ylabel("Independent-probe accuracy change")
    ax.set_title("Repair-2 subject fidelity versus class fidelity")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(c.FIGURES / "FIGURE_STAGE0_REPAIR2_FIDELITY.png", dpi=300)
    fig.savefig(c.FIGURES / "FIGURE_STAGE0_REPAIR2_FIDELITY.pdf")
    plt.close(fig)


def main() -> None:
    repair.verify_repair2_freeze()
    stability = pd.read_csv(c.RESULTS / "TRANSPORT_STABILITY.csv")
    subject = load_units("SUBJECT_FIDELITY.csv")
    class_fidelity = load_units("CLASS_FIDELITY.csv")
    manifold = load_units("MANIFOLD_VALIDITY.csv")
    alpha = distribution_rows()
    summary = summarize(stability, subject, class_fidelity, manifold)
    supported = bool(summary.all_gates_pass.all()) and len(summary) == len(c.SETTINGS)
    terminal = "TRANSPORT_VALIDITY_SUPPORTED" if supported else "TRANSPORT_VALIDITY_NOT_SUPPORTED"

    compact_subject = compact_metric(
        subject,
        ["target_distance", "clean_target_distance", "relative_target_affinity_improvement", "perturbation_norm"],
    )
    compact_class = compact_metric(
        class_fidelity,
        ["independent_probe_BA", "clean_accuracy", "transported_accuracy", "accuracy_change", "true_log_probability_change", "mean_perturbation_norm"],
    )
    compact_manifold = compact_metric(
        manifold, ["knn_distance", "off_manifold", "perturbation_norm"]
    )
    c.write_csv(c.RESULTS / "STAGE0_REPAIR2_LAYER_SUMMARY.csv", summary)
    c.write_csv(c.RESULTS / "STAGE0_REPAIR2_ALPHA_DISTRIBUTION.csv", alpha)
    c.write_csv(c.RESULTS / "STAGE0_REPAIR2_SUBJECT_FIDELITY.csv", compact_subject)
    c.write_csv(c.RESULTS / "STAGE0_REPAIR2_CLASS_FIDELITY.csv", compact_class)
    c.write_csv(c.RESULTS / "STAGE0_REPAIR2_MANIFOLD_VALIDITY.csv", compact_manifold)
    setting_alpha = alpha[(alpha.scope == "setting") & (alpha.query_unit == "centroid")]
    result = {
        "schema": "SCST_DR_STAGE0_REPAIR2_FINAL_V1",
        "terminal": terminal,
        "transport_validity_supported": supported,
        "stage1_authorized": supported,
        "selected_layer": repair.LAYER if supported else None,
        "operator": "source_support_constrained",
        "alpha_max": repair.ALPHA_MAX,
        "all_four_competent_settings_retained": True,
        "outer_or_future_performance_accessed": False,
        "layer_summary": summary.to_dict(orient="records"),
        "setting_centroid_alpha_distribution": setting_alpha.to_dict(orient="records"),
    }
    c.write_json(c.RESULTS / "STAGE0_REPAIR2_FINAL_RESULT.json", result)
    c.write_json(
        c.RESULTS / "STAGE0_REPAIR2_STATISTICS.json",
        {
            "schema": "SCST_DR_STAGE0_REPAIR2_STATISTICS_V1",
            "terminal": terminal,
            "bootstrap_unit": "source_subject",
            "bootstrap_resamples": 10000,
            "all_original_gates_unchanged": True,
            "summary": summary.to_dict(orient="records"),
        },
    )
    make_figures(summary, alpha)

    alpha_table = alpha[(alpha.scope == "setting")].to_markdown(index=False, floatfmt=".5f")
    report = f"""# Stage-0 Repair 2 report

## Terminal

`{terminal}`

The source-support operator, code, grid, controls, gates, and sealed state were
hash-locked before these outcomes.  Source Session 1 alone defined residuals,
support geometry, class radii, and every `alpha_star`; Session 2 was used only
for independent validation.

## Four-setting validity

{summary.to_markdown(index=False, floatfmt='.5f')}

## Alpha-star diagnostics

{alpha_table}

All original gates, including the absolute 1.25 independent-session manifold
ratio and subject-level bootstrap unit, were retained.
"""
    c.write_text(c.EXP / "STAGE0_REPAIR2_REPORT.md", report)

    ledger_path = c.EXP / "ITERATION_LEDGER.md"
    ledger = ledger_path.read_text(encoding="utf-8")
    pending = "- Actual result: pending.\n- Decision: pending."
    decision = (
        "- Decision: retain Repair 2 and authorize the prospectively frozen SCST development stage."
        if supported
        else "- Decision: reject transport validity permanently; do not train SCST, do not create Repair 3, and do not open outer resources."
    )
    replacement = f"- Actual result: validated `{terminal}`; all-setting pass count `{int(summary.all_gates_pass.sum())}/4`.\n{decision}"
    if pending not in ledger:
        raise RuntimeError("Repair-2 pending ledger block is absent or already resolved")
    c.write_text(ledger_path, ledger.replace(pending, replacement, 1))
    print(terminal, flush=True)


if __name__ == "__main__":
    main()
