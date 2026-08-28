from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_stage0 as v0_analysis
import common as c
import run_stage0_repair1 as repair


def load_repair_units(name: str) -> pd.DataFrame:
    paths = sorted((c.RUNTIME / "stage0_repair1_units").glob(f"*/fold-*/*/{name}"))
    expected = len(c.SETTINGS) * 5 * 2
    if len(paths) != expected:
        raise RuntimeError(f"{name}: expected {expected} Repair-1 unit files, found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def selections(frame: pd.DataFrame) -> dict[str, str]:
    selected: dict[str, str] = {}
    for setting in c.SETTINGS:
        group = frame[frame.setting_id == setting].set_index("layer")
        if bool(group.loc["final_embedding", "all_gates_pass"]):
            selected[setting] = "final_embedding"
        elif bool(group.loc["pre_embedding", "all_gates_pass"]):
            selected[setting] = "pre_embedding"
    return selected


def failure_terminal(summary: pd.DataFrame) -> str:
    # This rule is global and conjunctive: a scientific failure is assigned only
    # when at least one retained setting cannot satisfy that gate at any frozen
    # alpha/layer.  Otherwise the candidates fail to identify one coherent
    # all-gate transport and the conservative terminal is NOT_IDENTIFIABLE.
    for gate, terminal in (
        ("gate_subject_fidelity", "TRANSPORT_NOT_SUBJECT_FAITHFUL"),
        ("gate_class_fidelity", "TRANSPORT_NOT_CLASS_PRESERVING"),
        ("gate_manifold", "TRANSPORT_OFF_MANIFOLD"),
    ):
        for setting in c.SETTINGS:
            if not bool(summary[summary.setting_id == setting][gate].any()):
                return terminal
    return "TRANSPORT_NOT_IDENTIFIABLE"


def make_figures(summary: pd.DataFrame) -> None:
    c.FIGURES.mkdir(parents=True, exist_ok=True)
    settings = list(c.SETTINGS)
    labels = [setting.replace("_MI_", "\n") for setting in settings]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), constrained_layout=True)
    markers = {"pre_embedding": "s", "final_embedding": "o"}
    colors = {0.25: "#2166ac", 0.50: "#b2182b"}
    for alpha in repair.ALPHAS:
        for layer in repair.LAYERS:
            frame = (
                summary[(summary.alpha == alpha) & (summary.layer == layer)]
                .set_index("setting_id")
                .loc[settings]
            )
            x = np.arange(len(settings)) + (-0.12 if alpha == 0.25 else 0.12)
            y = frame.subject_affinity_improvement_mean.to_numpy()
            yerr = np.vstack(
                [y - frame.subject_affinity_CI_low.to_numpy(), frame.subject_affinity_CI_high.to_numpy() - y]
            )
            axes[0].errorbar(
                x,
                y,
                yerr=yerr,
                marker=markers[layer],
                color=colors[alpha],
                capsize=2,
                linestyle="none",
                alpha=0.82,
                label=f"alpha={alpha:.2f}, {layer}",
            )
            axes[1].scatter(
                frame.subject_affinity_improvement_mean,
                frame.class_accuracy_change,
                marker=markers[layer],
                color=colors[alpha],
                s=62,
                alpha=0.82,
                label=f"alpha={alpha:.2f}, {layer}",
            )
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(np.arange(len(labels)), labels, fontsize=8)
    axes[0].set_ylabel("Target-subject affinity improvement")
    axes[0].set_title("A  Magnitude repair restores subject fidelity")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].axhline(-0.02, color="#b2182b", lw=0.9, ls="--")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Target-subject affinity improvement")
    axes[1].set_ylabel("Independent-probe accuracy change")
    axes[1].set_title("B  Subject versus class fidelity")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(c.FIGURES / "FIGURE_STAGE0_REPAIR1_ALPHA_FIDELITY.png", dpi=300)
    fig.savefig(c.FIGURES / "FIGURE_STAGE0_REPAIR1_ALPHA_FIDELITY.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    width = 0.18
    positions = np.arange(len(settings))
    offset_index = 0
    for alpha in repair.ALPHAS:
        for layer in repair.LAYERS:
            frame = (
                summary[(summary.alpha == alpha) & (summary.layer == layer)]
                .set_index("setting_id")
                .loc[settings]
            )
            offset = (offset_index - 1.5) * width
            ax.bar(
                positions + offset,
                frame.manifold_knn_ratio_to_clean,
                width,
                color=colors[alpha],
                alpha=0.55 if layer == "pre_embedding" else 0.9,
                hatch="//" if layer == "pre_embedding" else None,
                label=f"alpha={alpha:.2f}, {layer}",
            )
            offset_index += 1
    ax.axhline(1.25, color="#b2182b", lw=1.0, ls="--", label="frozen gate")
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("3NN manifold distance / clean")
    ax.set_title("Source-only on-manifold validity after magnitude repair")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(c.FIGURES / "FIGURE_STAGE0_REPAIR1_MANIFOLD.png", dpi=300)
    fig.savefig(c.FIGURES / "FIGURE_STAGE0_REPAIR1_MANIFOLD.pdf")
    plt.close(fig)


def main() -> None:
    repair.verify_repair_freeze()
    v0_validation = c.read_json(c.RESULTS / "STAGE0_VALIDATION.json")
    if v0_validation.get("pass") is not True or v0_validation.get("stage0_terminal") != "TRANSPORT_NOT_SUBJECT_FAITHFUL":
        raise RuntimeError("Repair-1 analysis requires the validated V0 failure")

    stability = pd.read_csv(c.RESULTS / "TRANSPORT_STABILITY.csv")
    v0_subject = pd.read_csv(c.RESULTS / "SUBJECT_FIDELITY.csv")
    v0_class = pd.read_csv(c.RESULTS / "CLASS_FIDELITY.csv")
    v0_manifold = pd.read_csv(c.RESULTS / "MANIFOLD_VALIDITY.csv")
    repair_subject = load_repair_units("SUBJECT_FIDELITY.csv")
    repair_class = load_repair_units("CLASS_FIDELITY.csv")
    repair_manifold = load_repair_units("MANIFOLD_VALIDITY.csv")

    summaries: list[pd.DataFrame] = []
    terminals_by_alpha: dict[str, str] = {}
    selections_by_alpha: dict[str, dict[str, str]] = {}
    for alpha in repair.ALPHAS:
        method = repair.alpha_method(alpha)
        subject = pd.concat(
            [v0_subject[v0_subject.method != "scst"], repair_subject[repair_subject.method == method].assign(method="scst")],
            ignore_index=True,
        )
        class_fidelity = pd.concat(
            [v0_class[v0_class.method != "scst"], repair_class[repair_class.method == method].assign(method="scst")],
            ignore_index=True,
        )
        manifold = pd.concat(
            [v0_manifold[v0_manifold.method != "scst"], repair_manifold[repair_manifold.method == method].assign(method="scst")],
            ignore_index=True,
        )
        summary = v0_analysis.summarize(stability, subject, class_fidelity, manifold)
        summary.insert(4, "alpha", alpha)
        selected = selections(summary)
        terminal = "TRANSPORT_VALIDITY_SUPPORTED" if len(selected) == len(c.SETTINGS) else v0_analysis.choose_terminal(summary)[0]
        terminals_by_alpha[f"{alpha:.2f}"] = terminal
        selections_by_alpha[f"{alpha:.2f}"] = selected
        summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True)
    eligible_alphas = [
        alpha
        for alpha in repair.ALPHAS
        if len(selections_by_alpha[f"{alpha:.2f}"]) == len(c.SETTINGS)
    ]
    if eligible_alphas:
        selected_alpha = 0.50 if 0.50 in eligible_alphas else 0.25
        terminal = "TRANSPORT_VALIDITY_SUPPORTED"
        selected_layers = selections_by_alpha[f"{selected_alpha:.2f}"]
    else:
        selected_alpha = None
        terminal = failure_terminal(all_summary)
        selected_layers = {}

    c.write_csv(c.RESULTS / "STAGE0_REPAIR1_LAYER_SUMMARY.csv", all_summary)
    result = {
        "schema": "SCST_DR_STAGE0_REPAIR1_FINAL_V1",
        "terminal": terminal,
        "transport_validity_supported": terminal == "TRANSPORT_VALIDITY_SUPPORTED",
        "stage1_authorized": terminal == "TRANSPORT_VALIDITY_SUPPORTED",
        "selected_alpha": selected_alpha,
        "eligible_alphas": eligible_alphas,
        "selected_layers": selected_layers,
        "terminals_by_alpha": terminals_by_alpha,
        "selections_by_alpha": selections_by_alpha,
        "only_change_from_v0": "transport_magnitude",
        "v0_terminal": v0_validation["stage0_terminal"],
        "all_four_competent_settings_retained": True,
        "outer_or_future_performance_accessed": False,
        "layer_summary": all_summary.to_dict(orient="records"),
    }
    c.write_json(c.RESULTS / "STAGE0_REPAIR1_FINAL_RESULT.json", result)
    make_figures(all_summary)

    table = all_summary.to_markdown(index=False, floatfmt=".5f")
    report = f"""# SCST-DR Stage-0 magnitude-only repair

## Terminal

`{terminal}`

Validated V0 terminal: `{v0_validation['stage0_terminal']}`.

Eligible global alphas: `{json.dumps(eligible_alphas)}`.  Selected alpha:
`{selected_alpha}`.  Deterministic selected layers:
`{json.dumps(selected_layers, sort_keys=True)}`.

This repair changed only the global residual multiplier from alpha=1 to the
prelocked source-only candidates 0.25 and 0.50.  Every setting, fold, layer,
centroid, scaling hash, independent class probe, control, manifold estimator,
gate, and bootstrap definition was retained.  No outcome/future-session row or
sealed subject was loaded.

## Frozen-gate results

{table}
"""
    c.write_text(c.EXP / "STAGE0_REPAIR1_REPORT.md", report)
    existing_report_path = c.EXP / "TRANSPORT_VALIDITY_REPORT.md"
    if existing_report_path.is_file():
        existing = existing_report_path.read_text(encoding="utf-8")
        marker = "## Magnitude-only Repair 1"
        if marker not in existing:
            existing += f"\n\n{marker}\n\nFinal repair terminal: `{terminal}`.  See `STAGE0_REPAIR1_REPORT.md`.\n"
            c.write_text(existing_report_path, existing)

    ledger_path = c.EXP / "ITERATION_LEDGER.md"
    ledger = ledger_path.read_text(encoding="utf-8")
    pending = "- Actual result: pending.\n- Decision: pending."
    replacement = (
        f"- Actual result: validated `{terminal}`; eligible global alphas "
        f"`{json.dumps(eligible_alphas)}` and selected alpha `{selected_alpha}`.\n"
        + (
            "- Decision: retain the magnitude repair and authorize matched Stage-1 development."
            if terminal == "TRANSPORT_VALIDITY_SUPPORTED"
            else "- Decision: reject the residual transport hypothesis, stop model training, and do not open outer resources."
        )
    )
    if pending not in ledger:
        raise RuntimeError("Repair-1 ledger pending block is absent or already resolved")
    c.write_text(ledger_path, ledger.replace(pending, replacement, 1))
    print(terminal, flush=True)


if __name__ == "__main__":
    main()
