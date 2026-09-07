from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as c


def load_units(name: str) -> pd.DataFrame:
    paths = sorted((c.RUNTIME / "stage0_units").glob(f"*/fold-*/*/{name}"))
    expected = len(c.SETTINGS) * 5 * 2
    if len(paths) != expected:
        raise RuntimeError(f"{name}: expected {expected} unit files, found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def ci_subject(frame: pd.DataFrame, value: str, seed_parts: tuple[Any, ...], resamples: int = 10000) -> tuple[float, float, float]:
    per_subject = frame.groupby("source_subject", as_index=False)[value].mean()
    values = per_subject[value].to_numpy(np.float64)
    if len(values) < 4 or not np.isfinite(values).all():
        return float(np.nanmean(values)), float("nan"), float("nan")
    rng = np.random.default_rng(c.stable_seed("SCST_DR_STAGE0_BOOTSTRAP_V1", *seed_parts))
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    distribution = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(distribution, 0.025)), float(np.quantile(distribution, 0.975))


def method_effect(frame: pd.DataFrame, method_a: str, method_b: str, value: str, output: str) -> pd.DataFrame:
    key = ["setting_id", "fold", "layer", "source_subject", "target_subject", "class_label"]
    a = frame[frame.method == method_a][key + [value]].rename(columns={value: "a"})
    b = frame[frame.method == method_b][key + [value]].rename(columns={value: "b"})
    merged = a.merge(b, on=key, validate="one_to_one")
    merged[output] = merged.a - merged.b
    return merged


def summarize(stability: pd.DataFrame, subject: pd.DataFrame, class_fidelity: pd.DataFrame, manifold: pd.DataFrame) -> pd.DataFrame:
    protocol = c.protocol()
    gate = protocol["per_setting_layer_gates"]
    rows: list[dict[str, Any]] = []
    for setting in c.SETTINGS:
        for layer in protocol["candidate_layers"]:
            st = stability[(stability.setting_id == setting) & (stability.layer == layer)]
            matched = st[st.control == "matched_subject_same_class"].groupby(
                ["fold", "source_subject", "class_label"], as_index=False
            ).cosine.mean().rename(columns={"cosine": "matched"})
            mismatch = st[st.control == "mismatched_subject_same_class"].groupby(
                ["fold", "source_subject", "class_label"], as_index=False
            ).cosine.mean().rename(columns={"cosine": "mismatch"})
            stability_effect = matched.merge(mismatch, on=["fold", "source_subject", "class_label"], validate="one_to_one")
            stability_effect["effect"] = stability_effect.matched - stability_effect.mismatch
            stability_mean, stability_lo, stability_hi = ci_subject(stability_effect, "effect", (setting, layer, "stability"))

            sf = subject[(subject.setting_id == setting) & (subject.layer == layer)]
            scst_sf = sf[sf.method == "scst"]
            sf_mean, sf_lo, sf_hi = ci_subject(scst_sf, "relative_target_affinity_improvement", (setting, layer, "subject"))
            sf_vs_random = method_effect(sf, "scst", "norm_matched_random", "relative_target_affinity_improvement", "effect")
            sfr_mean, sfr_lo, sfr_hi = ci_subject(sf_vs_random, "effect", (setting, layer, "subject_vs_random"))

            cf = class_fidelity[(class_fidelity.setting_id == setting) & (class_fidelity.layer == layer)]
            scst_cf = cf[cf.method == "scst"]
            probe_ba = float(scst_cf.independent_probe_BA.mean())
            accuracy_change, accuracy_lo, accuracy_hi = ci_subject(scst_cf, "accuracy_change", (setting, layer, "class_accuracy"))
            logp_change, logp_lo, logp_hi = ci_subject(scst_cf, "true_log_probability_change", (setting, layer, "class_logp"))

            mf = manifold[(manifold.setting_id == setting) & (manifold.layer == layer)]
            scst_mf = mf[mf.method == "scst"]
            clean_mf = mf[mf.method == "no_transport"]
            random_mf = mf[mf.method == "norm_matched_random"]
            manifold_ratio = float(scst_mf.knn_distance.mean() / max(clean_mf.knn_distance.mean(), 1e-12))
            scst_off = float(scst_mf.off_manifold.astype(float).mean())
            random_off = float(random_mf.off_manifold.astype(float).mean())
            off_excess = scst_off - random_off

            gates = {
                "competence": probe_ba >= float(gate["independent_probe_BA_min"]),
                "stability": stability_mean > float(gate["stability_matched_minus_mismatched_mean_min"]) and stability_lo > 0,
                "subject_fidelity": sf_mean > float(gate["subject_fidelity_relative_improvement_mean_min"]) and sf_lo > 0 and sfr_lo > 0,
                "class_fidelity": -accuracy_change <= float(gate["class_accuracy_loss_max"]) and logp_change >= float(gate["class_true_log_probability_change_min"]),
                "manifold": manifold_ratio <= float(gate["manifold_knn_ratio_to_clean_max"]) and off_excess <= float(gate["manifold_off_rate_excess_over_random_max"]),
            }
            rows.append({
                "setting_id": setting, "dataset": c.SETTINGS[setting]["dataset"],
                "backbone": c.SETTINGS[setting]["backbone"], "layer": layer,
                "independent_probe_BA": probe_ba,
                "stability_effect_mean": stability_mean, "stability_CI_low": stability_lo, "stability_CI_high": stability_hi,
                "subject_affinity_improvement_mean": sf_mean, "subject_affinity_CI_low": sf_lo, "subject_affinity_CI_high": sf_hi,
                "subject_advantage_over_random_mean": sfr_mean, "subject_advantage_over_random_CI_low": sfr_lo, "subject_advantage_over_random_CI_high": sfr_hi,
                "class_accuracy_change": accuracy_change, "class_accuracy_CI_low": accuracy_lo, "class_accuracy_CI_high": accuracy_hi,
                "class_logp_change": logp_change, "class_logp_CI_low": logp_lo, "class_logp_CI_high": logp_hi,
                "manifold_knn_ratio_to_clean": manifold_ratio, "scst_off_manifold_rate": scst_off,
                "random_off_manifold_rate": random_off, "off_manifold_excess_over_random": off_excess,
                **{f"gate_{name}": value for name, value in gates.items()},
                "all_gates_pass": all(gates.values()),
            })
    return pd.DataFrame(rows)


def choose_terminal(summary: pd.DataFrame) -> tuple[str, dict[str, str]]:
    selections: dict[str, str] = {}
    for setting in c.SETTINGS:
        group = summary[summary.setting_id == setting].set_index("layer")
        if bool(group.loc["final_embedding", "all_gates_pass"]):
            selections[setting] = "final_embedding"
        elif bool(group.loc["pre_embedding", "all_gates_pass"]):
            selections[setting] = "pre_embedding"
    if len(selections) == len(c.SETTINGS):
        return "TRANSPORT_VALIDITY_SUPPORTED", selections
    failed_settings = [setting for setting in c.SETTINGS if setting not in selections]
    if any(not summary[summary.setting_id == setting].gate_subject_fidelity.any() for setting in failed_settings):
        return "TRANSPORT_NOT_SUBJECT_FAITHFUL", selections
    if any(not summary[summary.setting_id == setting].gate_class_fidelity.any() for setting in failed_settings):
        return "TRANSPORT_NOT_CLASS_PRESERVING", selections
    if any(not summary[summary.setting_id == setting].gate_manifold.any() for setting in failed_settings):
        return "TRANSPORT_OFF_MANIFOLD", selections
    return "TRANSPORT_NOT_IDENTIFIABLE", selections


def figures(summary: pd.DataFrame) -> None:
    c.FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [setting.replace("_MI_", "\n") for setting in c.SETTINGS]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for offset, layer in enumerate(("pre_embedding", "final_embedding")):
        frame = summary[summary.layer == layer].set_index("setting_id").loc[list(c.SETTINGS)]
        x = np.arange(len(frame)) + (offset - 0.5) * 0.32
        y = frame.stability_effect_mean.to_numpy()
        yerr = np.vstack([y - frame.stability_CI_low.to_numpy(), frame.stability_CI_high.to_numpy() - y])
        axes[0].errorbar(x, y, yerr=yerr, marker="o", capsize=3, linestyle="none", label=layer)
        axes[1].scatter(frame.subject_affinity_improvement_mean, frame.class_accuracy_change, s=65, label=layer)
        for setting, row in frame.iterrows():
            axes[1].annotate(setting.split("_MI_")[-1][:4] + "/" + setting.split("_")[0][:2],
                             (row.subject_affinity_improvement_mean, row.class_accuracy_change), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(np.arange(len(labels)), labels, fontsize=8)
    axes[0].set_ylabel("Matched − mismatched residual cosine")
    axes[0].set_title("A  Cross-session residual stability")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(-0.02, color="#b2182b", lw=0.9, ls="--", label="class-loss gate")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Target-subject affinity improvement")
    axes[1].set_ylabel("Independent-probe accuracy change")
    axes[1].set_title("B  Subject fidelity vs class fidelity")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(c.FIGURES / "FIGURE_1_TRANSPORT_CONCEPT_VALIDATION.png", dpi=300)
    fig.savefig(c.FIGURES / "FIGURE_1_TRANSPORT_CONCEPT_VALIDATION.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    selected = summary.copy()
    colors = np.where(selected.all_gates_pass, "#2166ac", "#b2182b")
    ax.scatter(selected.subject_affinity_improvement_mean, selected.class_logp_change,
               c=colors, s=80, alpha=0.9)
    for row in selected.itertuples():
        ax.annotate(f"{row.setting_id.split('_')[0][:2]}-{row.backbone[:4]}-{row.layer[:3]}",
                    (row.subject_affinity_improvement_mean, row.class_logp_change), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(-0.05, color="#b2182b", lw=0.9, ls="--")
    ax.set_xlabel("Relative target-subject affinity improvement")
    ax.set_ylabel("Independent true-class log-probability change")
    ax.set_title("Source-only transport validity across predeclared layers")
    fig.savefig(c.FIGURES / "FIGURE_2_SUBJECT_VS_CLASS_FIDELITY.png", dpi=300)
    fig.savefig(c.FIGURES / "FIGURE_2_SUBJECT_VS_CLASS_FIDELITY.pdf")
    plt.close(fig)


def main() -> None:
    freeze = c.read_json(c.EXP / "protocol" / "PRE_STAGE0_FREEZE.json")
    if freeze.get("frozen_before_first_transport_metric") is not True:
        raise RuntimeError("invalid pre-Stage0 freeze")
    stability = load_units("TRANSPORT_STABILITY.csv")
    subject = load_units("SUBJECT_FIDELITY.csv")
    class_fidelity = load_units("CLASS_FIDELITY.csv")
    manifold = load_units("MANIFOLD_VALIDITY.csv")
    c.write_csv(c.RESULTS / "TRANSPORT_STABILITY.csv", stability)
    c.write_csv(c.RESULTS / "SUBJECT_FIDELITY.csv", subject)
    c.write_csv(c.RESULTS / "CLASS_FIDELITY.csv", class_fidelity)
    c.write_csv(c.RESULTS / "MANIFOLD_VALIDITY.csv", manifold)
    summary = summarize(stability, subject, class_fidelity, manifold)
    terminal, selections = choose_terminal(summary)
    c.write_csv(c.RESULTS / "STAGE0_LAYER_SUMMARY.csv", summary)
    result = {
        "schema": "SCST_DR_STAGE0_FINAL_V1", "terminal": terminal,
        "transport_validity_supported": terminal == "TRANSPORT_VALIDITY_SUPPORTED",
        "selected_layers": selections, "setting_count": len(c.SETTINGS),
        "settings_with_passing_layer": len(selections), "all_four_competent_settings_retained": True,
        "outer_or_future_performance_accessed": False, "stage1_authorized": terminal == "TRANSPORT_VALIDITY_SUPPORTED",
        "layer_summary": summary.to_dict(orient="records"),
    }
    c.write_json(c.RESULTS / "STAGE0_FINAL_RESULT.json", result)
    figures(summary)

    table = summary.to_markdown(index=False, floatfmt=".5f")
    report = f"""# SCST-DR transport validity report

## Terminal

`{terminal}`

This is a source-only Stage-0 result.  No outcome-subject future-session metric,
OpenBMI internal holdout, WBCIC Session 3 development outcome, or WBCIC outer
subject was used.

## Layer audit

{table}

Selected passing layers: `{json.dumps(selections, sort_keys=True)}`.

The gate is conjunctive.  A setting/layer passes only when the independent task
probe is competent, matched subject-class residuals are more stable than
mismatched residuals with a subject-bootstrap CI above zero, target-subject
affinity improves and beats norm-matched random transport, class accuracy/log
probability remain within the frozen tolerances, and the centroid support test
does not show excess off-manifold behavior.
"""
    c.write_text(c.EXP / "TRANSPORT_VALIDITY_REPORT.md", report)
    c.write_text(c.EXP / "TRANSPORT_LAYER_AUDIT.md", "# Transport layer audit\n\n" + table + f"\n\nDeterministic selections: `{json.dumps(selections, sort_keys=True)}`. Future performance was not computed.")
    ledger = c.EXP / "ITERATION_LEDGER.md"
    original = ledger.read_text(encoding="utf-8")
    original = original.replace("- Actual result: pending Stage-0 execution.\n- Decision: pending.", f"- Actual result: `{terminal}`; see `results/STAGE0_LAYER_SUMMARY.csv`.\n- Decision: {'proceed to Stage 1 under the frozen gate' if terminal == 'TRANSPORT_VALIDITY_SUPPORTED' else 'stop constructive model development unless a single predeclared, mechanism-level repair is justified' }.")
    c.write_text(ledger, original)
    print(terminal, flush=True)


if __name__ == "__main__":
    main()
