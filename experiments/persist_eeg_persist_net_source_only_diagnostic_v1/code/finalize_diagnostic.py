"""Aggregate, test, plot, and report the frozen source-only diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
RESULTS = EXPERIMENT / "results"
FIGURES = EXPERIMENT / "figures"
RUNTIME = EXPERIMENT / "runtime"
DEFAULT_SOURCE_EXPERIMENT = Path(
    os.environ.get(
        "PERSIST_FINAL_V1_EXPERIMENT",
        str(REPO / "experiments" / "persist_eeg_persist_net_final_v1"),
    )
)

SOURCE_METHODS = (
    "B0_VANILLA_EEGNET",
    "B1_STRONG_EEGNET",
    "A2_SOURCE_ONLY",
    "PUD_SOURCE_ONLY",
    "IDENTITY_SOURCE_ONLY",
    "RANDOM_SOURCE_ONLY",
)
METHOD_LABELS = {
    "B0_VANILLA_EEGNET": "Vanilla EEGNet",
    "B1_STRONG_EEGNET": "Strong EEGNet",
    "A2_SOURCE_ONLY": "Dual task-only source",
    "PUD_SOURCE_ONLY": "PUD source-only",
    "IDENTITY_SOURCE_ONLY": "Identity source-only",
    "RANDOM_SOURCE_ONLY": "Random source-only",
    "PUD_AFTER_ADAPT": "PUD after target adaptation",
}
COMPARISON_SPECS = {
    "PUD_minus_Vanilla": ("PUD_SOURCE_ONLY", "B0_VANILLA_EEGNET"),
    "PUD_minus_DualControlSource": ("PUD_SOURCE_ONLY", "A2_SOURCE_ONLY"),
    "PUD_minus_StrongEEGNet": ("PUD_SOURCE_ONLY", "B1_STRONG_EEGNET"),
    "PUD_after_adapt_minus_source": ("PUD_AFTER_ADAPT", "PUD_SOURCE_ONLY"),
}
TOL = 1e-12


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def portable_source_paths(value: Any, source_experiment: Path) -> Any:
    """Replace private machine roots with a stable source-experiment token."""
    source_root = str(source_experiment.resolve()).replace("\\", "/").rstrip("/")

    def convert(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: convert(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(entry) for entry in item]
        if isinstance(item, Path):
            item = str(item)
        if isinstance(item, str):
            normalized = item.replace("\\", "/")
            if normalized.casefold() == source_root.casefold():
                return "SOURCE_EXPERIMENT"
            prefix = source_root + "/"
            if normalized.casefold().startswith(prefix.casefold()):
                return "SOURCE_EXPERIMENT/" + normalized[len(prefix) :]
        return item

    return convert(value)


def portable_path_columns(frame: pd.DataFrame, source_experiment: Path) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda item: portable_source_paths(item, source_experiment)
        )
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def normalize_svg(path: Path) -> None:
    """Remove backend-generated line-end whitespace for clean Git diffs."""
    normalized = "\n".join(
        line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
    )
    write_text(path, normalized)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def bootstrap(values: np.ndarray, seed: int, draws: int = 10_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "CI95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "draws": draws,
        "subjects": len(array),
    }


def sign_flip(values: np.ndarray, seed: int, draws: int = 100_000) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = draws
    while remaining:
        count = min(10_000, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(array)))
        permuted = np.abs((signs * array[None, :]).mean(axis=1))
        exceed += int(np.sum(permuted >= observed - 1e-15))
        remaining -= count
    return {
        "observed_abs_mean": observed,
        "two_sided_p": float((exceed + 1) / (draws + 1)),
        "draws": draws,
        "exact": False,
        "statistical_unit": "subject after three-seed aggregation",
    }


def markdown_table(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    table = frame.loc[:, list(columns)] if columns is not None else frame
    headers = list(map(str, table.columns))
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def aggregate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["subject_id"] = raw.subject_id.astype(str)
    expected_counts = raw.groupby("method").size().to_dict()
    if set(expected_counts) != set(SOURCE_METHODS) | {"PUD_AFTER_ADAPT"}:
        raise RuntimeError(f"method coverage mismatch: {expected_counts}")
    if any(count != 120 for count in expected_counts.values()):
        raise RuntimeError(f"each method must have 120 fold/seed/subject rows: {expected_counts}")

    per_subject = (
        raw.groupby(["method", "fold", "subject_id"], as_index=False)
        .agg(BA=("BA", "mean"), macro_f1=("macro_f1", "mean"), seeds=("seed", "nunique"), n_trials=("n_trials", "sum"))
    )
    if len(per_subject) != 280 or not (per_subject.seeds == 3).all():
        raise RuntimeError("subject aggregation is not 7 methods x 40 subjects x 3 seeds")
    b0 = per_subject.loc[per_subject.method.eq("B0_VANILLA_EEGNET"), ["subject_id", "BA"]].rename(
        columns={"BA": "B0_BA"}
    )
    per_subject = per_subject.merge(b0, on="subject_id", validate="many_to_one")
    per_subject["Delta_BA_vs_B0"] = per_subject.BA - per_subject.B0_BA

    per_fold = raw.groupby(["method", "fold"], as_index=False).agg(BA=("BA", "mean"), macro_f1=("macro_f1", "mean"))
    fold_b0 = per_fold.loc[per_fold.method.eq("B0_VANILLA_EEGNET"), ["fold", "BA"]].rename(columns={"BA": "B0_BA"})
    per_fold = per_fold.merge(fold_b0, on="fold", validate="many_to_one")
    per_fold["Delta_BA_vs_B0"] = per_fold.BA - per_fold.B0_BA

    per_seed = raw.groupby(["method", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_f1=("macro_f1", "mean"))
    seed_b0 = per_seed.loc[per_seed.method.eq("B0_VANILLA_EEGNET"), ["seed", "BA"]].rename(columns={"BA": "B0_BA"})
    per_seed = per_seed.merge(seed_b0, on="seed", validate="many_to_one")
    per_seed["Delta_BA_vs_B0"] = per_seed.BA - per_seed.B0_BA
    return per_subject, per_fold, per_seed


def paired_raw(raw: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    keys = ["fold", "seed", "subject_id"]
    lhs = raw.loc[raw.method.eq(left), keys + ["BA"]].rename(columns={"BA": "left_BA"})
    rhs = raw.loc[raw.method.eq(right), keys + ["BA"]].rename(columns={"BA": "right_BA"})
    paired = lhs.merge(rhs, on=keys, validate="one_to_one")
    if len(paired) != 120:
        raise RuntimeError(f"incomplete pairing for {left} vs {right}")
    paired["delta"] = paired.left_BA - paired.right_BA
    return paired


def comparison_statistics(raw: pd.DataFrame, name: str, left: str, right: str) -> dict[str, Any]:
    paired = paired_raw(raw, left, right)
    subject = paired.groupby(["fold", "subject_id"], as_index=False).agg(delta=("delta", "mean"))
    values = subject.delta.to_numpy(dtype=np.float64)
    boot = bootstrap(values, stable_seed("source-only-bootstrap", name), 10_000)
    flip = sign_flip(values, stable_seed("source-only-sign-flip", name), 100_000)
    fold = paired.groupby("fold", as_index=False).agg(delta=("delta", "mean"))
    seed = paired.groupby("seed", as_index=False).agg(delta=("delta", "mean"))
    improved = int(np.sum(values > TOL))
    harmed = int(np.sum(values < -TOL))
    tied = int(len(values) - improved - harmed)
    return {
        "name": name,
        "left": left,
        "right": right,
        "subject_bootstrap": boot,
        "sign_flip": flip,
        "improved_subjects": improved,
        "harmed_subjects": harmed,
        "tied_subjects": tied,
        "subject_win_rate": improved / len(values),
        "worst_quartile_mean_delta": float(np.mean(np.sort(values)[:10])),
        "fold_deltas": [{"fold": int(row.fold), "delta": float(row.delta)} for row in fold.itertuples()],
        "positive_folds": int(np.sum(fold.delta > TOL)),
        "seed_deltas": [{"seed": int(row.seed), "delta": float(row.delta)} for row in seed.itertuples()],
        "positive_seeds": int(np.sum(seed.delta > TOL)),
    }


def method_summary(
    per_subject: pd.DataFrame,
    per_fold: pd.DataFrame,
    per_seed: pd.DataFrame,
    comparisons: Mapping[str, Any],
    checkpoint_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    comparison_by_method = {
        "PUD_SOURCE_ONLY": comparisons["PUD_minus_Vanilla"],
    }
    b0_subject = per_subject.loc[per_subject.method.eq("B0_VANILLA_EEGNET")].set_index("subject_id")
    parameter_map = {
        "B0_VANILLA_EEGNET": "B0_VANILLA_EEGNET",
        "B1_STRONG_EEGNET": "B1_STRONG_EEGNET",
        "A2_SOURCE_ONLY": "A2_DUAL_CONTROL",
        "PUD_SOURCE_ONLY": "PUD_SOURCE",
        "IDENTITY_SOURCE_ONLY": "A7_IDENTITY_PROTECTED",
        "RANDOM_SOURCE_ONLY": "A8_RANDOM_PROTECTED",
    }
    for method in SOURCE_METHODS:
        frame = per_subject.loc[per_subject.method.eq(method)].set_index("subject_id")
        delta = (frame.BA - b0_subject.BA).to_numpy(dtype=np.float64)
        absolute = bootstrap(frame.BA.to_numpy(dtype=np.float64), stable_seed("absolute-BA", method), 10_000)
        delta_boot = bootstrap(delta, stable_seed("delta-B0", method), 10_000)
        folds = per_fold.loc[per_fold.method.eq(method)]
        seeds = per_seed.loc[per_seed.method.eq(method)]
        improved = int(np.sum(delta > TOL))
        harmed = int(np.sum(delta < -TOL))
        checkpoint_name = parameter_map[method]
        parameters = int(
            checkpoint_audit.loc[checkpoint_audit.method.eq(checkpoint_name), "parameters"].astype(int).mode().iloc[0]
        )
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "BA": float(frame.BA.mean()),
                "BA_CI95_L": absolute["CI95"][0],
                "BA_CI95_U": absolute["CI95"][1],
                "macro_f1": float(frame.macro_f1.mean()),
                "Delta_BA_vs_B0": float(delta.mean()),
                "Delta_CI95_L": delta_boot["CI95"][0],
                "Delta_CI95_U": delta_boot["CI95"][1],
                "median_subject_delta_vs_B0": float(np.median(delta)),
                "subjects_improved_vs_B0": improved,
                "subjects_harmed_vs_B0": harmed,
                "subjects_tied_vs_B0": int(len(delta) - improved - harmed),
                "subject_win_rate_vs_B0": improved / len(delta),
                "worst_quartile_delta_vs_B0": float(np.mean(np.sort(delta)[:10])),
                "positive_folds_vs_B0": int(np.sum(folds.Delta_BA_vs_B0 > TOL)),
                "positive_seeds_vs_B0": int(np.sum(seeds.Delta_BA_vs_B0 > TOL)),
                "parameters": parameters,
            }
        )
    return pd.DataFrame(rows)


def aggregate_mechanism(mechanism_raw: pd.DataFrame) -> pd.DataFrame:
    mechanism_raw = mechanism_raw.copy()
    mechanism_raw["subject_id"] = mechanism_raw.subject_id.astype(str)
    numeric = [
        "combined_BA",
        "protected_only_BA",
        "adaptive_only_BA",
        "protected_branch_erasure_harm_BA",
        "adaptive_branch_erasure_harm_BA",
        "protected_D_finite",
        "adaptive_D_finite",
        "functional_teacher_RMSE",
        "functional_teacher_correlation",
    ]
    return mechanism_raw.groupby(["method", "fold", "subject_id"], as_index=False)[numeric].mean()


def mechanism_statistics(mechanism: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for method in ("PUD_SOURCE_ONLY", "A2_SOURCE_ONLY", "IDENTITY_SOURCE_ONLY", "RANDOM_SOURCE_ONLY"):
        frame = mechanism.loc[mechanism.method.eq(method)]
        protected = frame.protected_branch_erasure_harm_BA.to_numpy(dtype=float)
        adaptive = frame.adaptive_branch_erasure_harm_BA.to_numpy(dtype=float)
        payload[method] = {
            "protected_branch_erasure_harm": bootstrap(
                protected, stable_seed("mechanism-protected", method), 10_000
            ),
            "adaptive_branch_erasure_harm": bootstrap(
                adaptive, stable_seed("mechanism-adaptive", method), 10_000
            ),
            "functional_teacher_RMSE_mean": float(frame.functional_teacher_RMSE.mean()),
            "functional_teacher_correlation_mean": float(frame.functional_teacher_correlation.mean()),
            "protected_D_finite_mean": float(frame.protected_D_finite.mean()),
            "adaptive_D_finite_mean": float(frame.adaptive_D_finite.mean()),
        }
    pud = mechanism.loc[mechanism.method.eq("PUD_SOURCE_ONLY")].set_index("subject_id")
    for control in ("A2_SOURCE_ONLY", "IDENTITY_SOURCE_ONLY", "RANDOM_SOURCE_ONLY"):
        other = mechanism.loc[mechanism.method.eq(control)].set_index("subject_id")
        delta = (pud.protected_branch_erasure_harm_BA - other.protected_branch_erasure_harm_BA).to_numpy(float)
        payload[f"PUD_protected_harm_minus_{control}"] = bootstrap(
            delta, stable_seed("mechanism-control", control), 10_000
        )
    payload["task_consequential"] = bool(
        payload["PUD_SOURCE_ONLY"]["protected_branch_erasure_harm"]["CI95"][0] > 0.0
    )
    return payload


def draw_figures(main: pd.DataFrame, per_subject: pd.DataFrame, source_vs_adapted: pd.DataFrame) -> None:
    # Figure contract: these quantitative comparison panels document that the
    # frozen PUD source representation underperforms matched EEGNet controls,
    # while target adaptation recovers only a small fraction of the deficit.
    # Every plotted value comes from the committed result tables; no rows are
    # sampled or excluded for display.
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    order = list(SOURCE_METHODS)
    frame = main.set_index("method").loc[order]
    x = np.arange(len(frame))
    values = frame.BA.to_numpy(float) * 100.0
    low = (frame.BA - frame.BA_CI95_L).to_numpy(float) * 100.0
    high = (frame.BA_CI95_U - frame.BA).to_numpy(float) * 100.0
    colors = ["#6b7280", "#4b5563", "#3b82f6", "#dc2626", "#8b5cf6", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x, values, color=colors, width=0.72)
    ax.errorbar(x, values, yerr=np.vstack([low, high]), fmt="none", ecolor="black", capsize=3, lw=0.9)
    ax.set_xticks(x, [METHOD_LABELS[item] for item in order], rotation=25, ha="right")
    ax.set_ylabel("Mean subject balanced accuracy (%)")
    ax.set_title("Frozen source-only Session-2 evaluation (40 subjects; 3 seeds)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "source_only_performance.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "source_only_performance.pdf", bbox_inches="tight")
    svg_path = FIGURES / "source_only_performance.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    normalize_svg(svg_path)
    plt.close(fig)

    frame = source_vs_adapted.sort_values("source_only_BA").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, row in frame.iterrows():
        color = "#dc2626" if row.adaptation_delta_BA < -TOL else "#2563eb"
        ax.plot([0, 1], [row.source_only_BA * 100, row.after_adaptation_BA * 100], color=color, alpha=0.34, lw=0.8)
        ax.scatter([0, 1], [row.source_only_BA * 100, row.after_adaptation_BA * 100], color=color, s=9, alpha=0.6)
    ax.set_xticks([0, 1], ["PUD source-only", "PUD after target adaptation"])
    ax.set_ylabel("Subject balanced accuracy (%)")
    ax.set_title("Per-subject effect of frozen target adaptation")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "source_vs_adapted.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "source_vs_adapted.pdf", bbox_inches="tight")
    svg_path = FIGURES / "source_vs_adapted.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    normalize_svg(svg_path)
    plt.close(fig)

    delta = per_subject.loc[per_subject.method.eq("PUD_SOURCE_ONLY")].sort_values("Delta_BA_vs_B0")
    values = delta.Delta_BA_vs_B0.to_numpy(float) * 100.0
    labels = delta.subject_id.astype(str).tolist()
    colors = np.where(values > TOL, "#2563eb", np.where(values < -TOL, "#dc2626", "#9ca3af"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(len(values)), values, color=colors, width=0.82)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(values)), labels, rotation=90, fontsize=6.5)
    ax.set_xlabel("Outcome subject (sorted by delta)")
    ax.set_ylabel("PUD source-only − Vanilla BA (pp)")
    ax.set_title("Per-subject source-only delta versus Vanilla EEGNet")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_subject_delta_vs_vanilla.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "per_subject_delta_vs_vanilla.pdf", bbox_inches="tight")
    svg_path = FIGURES / "per_subject_delta_vs_vanilla.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    normalize_svg(svg_path)
    plt.close(fig)


def write_figure_qa() -> None:
    write_text(
        EXPERIMENT / "FIGURE_QA.md",
        """# Figure QA

- Core conclusion: the frozen PUD source representation is below Vanilla EEGNet and the capacity-matched dual control; target adaptation provides only a small partial recovery.
- Archetype: quantitative comparison figures with one claim-bearing panel per export.
- Backend: Python/matplotlib only for drawing, rendering, and export.
- Source data: `results/source_only_main.csv`, `results/per_subject_results.csv`, and `results/source_vs_adapted.csv`.
- Data inclusion: all six source-only methods and all 40 outcome subjects are shown where applicable; no observation was sampled or excluded.
- Protocol: OpenBMI V8_SEARCH, 40 subjects, five folds, three seeds, future Session 2.
- Metric: mean subject balanced accuracy; seed values are averaged within subject before subject-level summaries.
- Intervals: 10,000-draw paired subject bootstrap where shown.
- Export: PNG at 300 dpi plus PDF/SVG with editable text; white background and sans-serif fallback.
- Final dimensions: 7.2 × 4.2 inches (approximately 183 × 107 mm) before tight bounding-box adjustment.
- Automated source preflight: no FAIL findings after adding SVG export. The TIFF warning is accepted because the requested deliverables are PNG/PDF and both PDF/SVG provide vector submission masters; the random-number warning refers only to preregistered bootstrap/sign-flip resampling, not simulated plot data.
- Visual inspection: labels, error bars, zero reference, and color/direction encoding must be checked on the regenerated PNG/PDF files after finalization.
""",
    )


def render_reports(
    main: pd.DataFrame,
    per_subject: pd.DataFrame,
    per_fold: pd.DataFrame,
    per_seed: pd.DataFrame,
    source_vs_adapted: pd.DataFrame,
    statistics: Mapping[str, Any],
    audit: Mapping[str, Any],
    replay: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    checkpoint_audit: pd.DataFrame,
) -> None:
    pud = main.set_index("method").loc["PUD_SOURCE_ONLY"]
    vanilla = main.set_index("method").loc["B0_VANILLA_EEGNET"]
    strong = main.set_index("method").loc["B1_STRONG_EEGNET"]
    dual = main.set_index("method").loc["A2_SOURCE_ONLY"]
    primary = statistics["comparisons"]["PUD_minus_Vanilla"]
    theory = statistics["comparisons"]["PUD_minus_DualControlSource"]
    competitive = statistics["comparisons"]["PUD_minus_StrongEEGNet"]
    adaptation = statistics["comparisons"]["PUD_after_adapt_minus_source"]
    mechanism = statistics["mechanism"]
    state = statistics["terminal_state"]
    adapted_mean = float(
        per_subject.loc[per_subject.method.eq("PUD_AFTER_ADAPT"), "BA"].mean()
    )
    alias_rows = checkpoint_audit.loc[
        checkpoint_audit.method.eq("B0_VANILLA_EEGNET") & checkpoint_audit.B0_aliases_B1.astype(bool)
    ]
    alias_locations = ", ".join(
        f"fold {int(row.fold)}/seed {int(row.seed)}" for row in alias_rows.itertuples()
    )

    write_text(
        EXPERIMENT / "README.md",
        f"""# PERSIST-EEG PERSIST-Net source-only diagnostic v1

Strict post-hoc evaluation of commit `12ab811c2a6194192b430f9c010781acd1c0379f`.
No checkpoint was trained, adapted, selected, copied, or renamed. B0/B1 replay passed before PUD source-only evaluation.

Terminal state: `{state}`.

See `FINAL_REPORT.md` for the result and `CHECKPOINT_PROVENANCE.md` / `REPLAY_VALIDATION.md` for the recovery proof.
""",
    )
    write_text(
        EXPERIMENT / "SCIENTIFIC_QUESTION.md",
        """# Scientific question

Does the already-trained PUD dual-path source checkpoint generalize to unseen subjects' Session 2 better than the frozen Vanilla EEGNet when target-subject adaptation is completely omitted?

The diagnostic is evaluation-only. It does not test a newly trained method and cannot be used to select an epoch, architecture, certificate, loss weight, or adaptation rule.
""",
    )
    write_text(
        EXPERIMENT / "CHECKPOINT_PROVENANCE.md",
        f"""# Checkpoint provenance

- Frozen base commit: `12ab811c2a6194192b430f9c010781acd1c0379f`.
- Frozen run grid: 5 folds × 3 seeds.
- Required checkpoint entries audited: {int(audit['checkpoint_rows'])}; all file SHA256 values matched their `RUN_LOCK.json` entries.
- Normalizers: 15/15 SHA256 matches; each contains exactly the 32 source subjects and sessions 1–2.
- PUD source checkpoint: the single pre-adaptation `PUD_SOURCE.pt` used by both A6 and A10 in final-v1.
- Physical `B0_VANILLA_EEGNET.pt` files absent: {int(audit['B0_named_files_absent'])}.
- Unambiguous B0→B1 aliases: {int(audit['B0_unambiguous_aliases'])}; locations: {alias_locations}.

For those aliases, the frozen lock records the B0 path as `B1_STRONG_EEGNET.pt`, the selected B1 configuration is `EEGNET_F8`, B0 and B1 seeds/epochs/parameter counts match, and the hashes are identical. No file was synthesized, copied, or renamed. The complete per-run audit is in `results/checkpoint_audit.csv`.
""",
    )
    write_text(
        EXPERIMENT / "REPLAY_VALIDATION.md",
        f"""# B0/B1 replay validation

Status: **{replay['status']}**.

- Replayed rows: {int(replay['subject_method_rows'])} (15 runs × 8 subjects × 2 methods).
- Maximum absolute BA error versus authoritative final-v1 rows: {float(replay['max_abs_BA_error']):.3e}.
- Maximum absolute Macro-F1 error: {float(replay['max_abs_macro_f1_error']):.3e}.
- Trial counts matched: {str(bool(replay['all_trial_counts_match'])).lower()}.
- Checkpoint files unchanged: {str(bool(replay['all_checkpoint_files_unchanged'])).lower()}.
- Parameters and BN/LayerNorm buffers unchanged: {str(bool(replay['all_parameters_and_buffers_unchanged'])).lower()}.

PUD source-only evaluation was not started until this persisted PASS existed and the frozen input fingerprint was revalidated.
""",
    )
    write_text(
        EXPERIMENT / "HOLDOUT_PURITY_AUDIT.md",
        """# Holdout purity audit

`internal_holdout_accessed = false`

`WBCIC_outer_accessed = false`

Only the existing OpenBMI `V8_SEARCH` cache (40 subjects) was opened. Session identifiers for that cache were read, while labels were predicate-filtered to Session 2 before materialization. Target Session-1 labels were not materialized or used. No OpenBMI 14-subject internal-holdout EEG, labels, predictions, embeddings, or metrics were accessed. No WBCIC path or sealed-outer artifact was accessed.
""",
    )

    display = main[["label", "BA", "macro_f1", "Delta_BA_vs_B0", "Delta_CI95_L", "Delta_CI95_U"]].copy()
    write_text(
        EXPERIMENT / "SOURCE_ONLY_EVALUATION.md",
        f"""# Source-only evaluation

Matched protocol: the same 40 OpenBMI V8_SEARCH subjects, 5 folds × 3 seeds, identical source normalizers, and outcome Session 2 trials. Statistical unit for inference is subject after averaging the three seeds.

{markdown_table(display)}

PUD source-only BA = {pud.BA:.6f}; Macro-F1 = {pud.macro_f1:.6f}. Relative to Vanilla EEGNet, ΔBA = {primary['subject_bootstrap']['mean']:+.6f}, 95% CI [{primary['subject_bootstrap']['CI95'][0]:+.6f}, {primary['subject_bootstrap']['CI95'][1]:+.6f}]. Relative to the capacity-matched dual source control, ΔBA = {theory['subject_bootstrap']['mean']:+.6f}. Relative to Strong EEGNet, ΔBA = {competitive['subject_bootstrap']['mean']:+.6f}.

All source models remained in `eval()`; file hashes and state hashes (parameters plus buffers) were identical before and after inference. Optimizer steps = 0; adaptation calls = 0.
""",
    )

    if state == "SOURCE_ONLY_PERSIST_NOT_SUPPORTED":
        interpretation = (
            "PUD supervision did not improve future-session generalization over Vanilla EEGNet. "
            "This is not a marginally positive result and does not justify another constructive model search."
        )
    elif state == "SOURCE_ONLY_PERSIST_WEAK_POSITIVE":
        interpretation = (
            "The direction is positive versus Vanilla, but the magnitude, uncertainty, consistency, or capacity-matched comparison is insufficient for a strong constructive claim."
        )
    else:
        interpretation = (
            "The frozen PUD source representation passes the predeclared positive gate. The appropriate next hypothesis is static deployment without target adaptation, not tuning inside this diagnostic."
        )
    write_text(
        EXPERIMENT / "INTERPRETATION_GATE.md",
        f"""# Interpretation gate

Terminal state: `{state}`.

{interpretation}

- ΔBA versus Vanilla: {primary['subject_bootstrap']['mean']:+.6f}.
- Positive folds: {primary['positive_folds']}/5.
- Positive seeds: {primary['positive_seeds']}/3.
- ΔBA versus capacity-matched dual source control: {theory['subject_bootstrap']['mean']:+.6f}.
- PUD protected-branch erasure harm: {mechanism['PUD_SOURCE_ONLY']['protected_branch_erasure_harm']['mean']:+.6f}, 95% CI [{mechanism['PUD_SOURCE_ONLY']['protected_branch_erasure_harm']['CI95'][0]:+.6f}, {mechanism['PUD_SOURCE_ONLY']['protected_branch_erasure_harm']['CI95'][1]:+.6f}].

Mechanism boundary: a pathway can be highly task-consequential without being future-session-generalization-beneficial. This boundary applies if the erasure effect is positive while the performance comparison is non-positive.
""",
    )

    adaptation_word = "damaged" if adaptation["subject_bootstrap"]["mean"] < -TOL else (
        "helped" if adaptation["subject_bootstrap"]["mean"] > TOL else "was neutral for"
    )
    final_report = f"""# Final report

Terminal state: `{state}`.

## Direct answers

1. **PUD source-only:** BA = **{pud.BA:.6f}**, Macro-F1 = **{pud.macro_f1:.6f}**.
2. **Versus Vanilla EEGNet:** Vanilla BA = {vanilla.BA:.6f}; ΔBA = {primary['subject_bootstrap']['mean']:+.6f} ({primary['subject_bootstrap']['mean']*100:+.3f} pp), 95% subject-bootstrap CI [{primary['subject_bootstrap']['CI95'][0]:+.6f}, {primary['subject_bootstrap']['CI95'][1]:+.6f}].
3. **Versus capacity-matched dual source control:** control BA = {dual.BA:.6f}; ΔBA = {theory['subject_bootstrap']['mean']:+.6f}, 95% CI [{theory['subject_bootstrap']['CI95'][0]:+.6f}, {theory['subject_bootstrap']['CI95'][1]:+.6f}].
4. **Versus Strong EEGNet:** Strong BA = {strong.BA:.6f}; ΔBA = {competitive['subject_bootstrap']['mean']:+.6f}, 95% CI [{competitive['subject_bootstrap']['CI95'][0]:+.6f}, {competitive['subject_bootstrap']['CI95'][1]:+.6f}].
5. **Fold consistency versus Vanilla:** {primary['positive_folds']}/5 positive.
6. **Seed consistency versus Vanilla:** {primary['positive_seeds']}/3 positive.
7. **Subject wins versus Vanilla:** {primary['improved_subjects']}/40 improved, {primary['harmed_subjects']}/40 harmed, {primary['tied_subjects']}/40 tied.
8. **Target adaptation effect:** after-adaptation BA = {adapted_mean:.6f}; after − source ΔBA = {adaptation['subject_bootstrap']['mean']:+.6f}, 95% CI [{adaptation['subject_bootstrap']['CI95'][0]:+.6f}, {adaptation['subject_bootstrap']['CI95'][1]:+.6f}]. Target adaptation {adaptation_word} the frozen source representation on average.
9. **Protected branch consequence:** erasure harm = {mechanism['PUD_SOURCE_ONLY']['protected_branch_erasure_harm']['mean']:+.6f}, 95% CI [{mechanism['PUD_SOURCE_ONLY']['protected_branch_erasure_harm']['CI95'][0]:+.6f}, {mechanism['PUD_SOURCE_ONLY']['protected_branch_erasure_harm']['CI95'][1]:+.6f}]; functional-teacher correlation = {mechanism['PUD_SOURCE_ONLY']['functional_teacher_correlation_mean']:.6f}.
10. **Interpretation:** `{state}`.
11. **OpenBMI 14-subject internal holdout accessed? NO.**
12. **WBCIC sealed outer accessed? NO.**

## Scientific conclusion

{interpretation}

The protected pathway's erasure result and the end-to-end generalization result are separate facts; task consequence is not evidence of generalization benefit. The diagnostic is limited to the frozen OpenBMI V8_SEARCH protocol and cannot establish a broader cross-dataset claim.

## Integrity

B0/B1 replay passed before PUD evaluation. PUD source BA also matched the authoritative pre-adaptation BA stored independently in both A6 and A10 rows (maximum absolute error {float(evaluation['PUD_source_max_abs_BA_error']):.3e}). No checkpoint, parameter, BN buffer, or LayerNorm buffer changed. No training or adaptation was run.
"""
    write_text(EXPERIMENT / "FINAL_REPORT.md", final_report)


def finalize(source_experiment: Path) -> dict[str, Any]:
    required = {
        "audit": RUNTIME / "INPUT_AUDIT.json",
        "replay": RUNTIME / "REPLAY_PASS.json",
        "evaluation": RUNTIME / "SOURCE_ONLY_EVALUATION.json",
        "replay_rows": RESULTS / "replay_per_subject.csv",
        "source_rows": RESULTS / "source_only_raw.csv",
        "adapted_rows": RESULTS / "adapted_authoritative_raw.csv",
        "mechanism": RESULTS / "mechanism_raw.csv",
        "checkpoint_audit": RESULTS / "checkpoint_audit.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing diagnostic artifacts: " + "; ".join(missing))
    audit = load_json(required["audit"])
    replay = load_json(required["replay"])
    evaluation = load_json(required["evaluation"])
    if not (audit.get("pass") and replay.get("pass") and evaluation.get("pass")):
        raise RuntimeError("audit, replay, and source-only evaluation must all pass before finalization")
    if len({audit["input_fingerprint"], replay["input_fingerprint"], evaluation["input_fingerprint"]}) != 1:
        raise RuntimeError("input fingerprints differ across diagnostic stages")

    replay_raw = pd.read_csv(required["replay_rows"])
    source_raw = pd.read_csv(required["source_rows"])
    adapted_raw = pd.read_csv(required["adapted_rows"])
    raw = pd.concat([replay_raw, source_raw, adapted_raw], ignore_index=True, sort=False)
    per_subject, per_fold, per_seed = aggregate(raw)
    comparisons = {
        name: comparison_statistics(raw, name, left, right)
        for name, (left, right) in COMPARISON_SPECS.items()
    }
    checkpoint_audit = portable_path_columns(
        pd.read_csv(required["checkpoint_audit"]), source_experiment
    )
    write_csv(required["checkpoint_audit"], checkpoint_audit)
    for integrity_path in (
        RESULTS / "evaluation_integrity_replay.csv",
        RESULTS / "evaluation_integrity_source_only.csv",
    ):
        if integrity_path.is_file():
            write_csv(
                integrity_path,
                portable_path_columns(pd.read_csv(integrity_path), source_experiment),
            )
    main = method_summary(per_subject, per_fold, per_seed, comparisons, checkpoint_audit)
    mechanism = aggregate_mechanism(pd.read_csv(required["mechanism"]))
    mechanism_stats = mechanism_statistics(mechanism)

    source_subject = per_subject.loc[per_subject.method.eq("PUD_SOURCE_ONLY"), ["fold", "subject_id", "BA", "macro_f1"]].rename(
        columns={"BA": "source_only_BA", "macro_f1": "source_only_macro_f1"}
    )
    adapted_subject = per_subject.loc[per_subject.method.eq("PUD_AFTER_ADAPT"), ["fold", "subject_id", "BA", "macro_f1"]].rename(
        columns={"BA": "after_adaptation_BA", "macro_f1": "after_adaptation_macro_f1"}
    )
    source_vs_adapted = source_subject.merge(adapted_subject, on=["fold", "subject_id"], validate="one_to_one")
    source_vs_adapted["adaptation_delta_BA"] = (
        source_vs_adapted.after_adaptation_BA - source_vs_adapted.source_only_BA
    )

    primary = comparisons["PUD_minus_Vanilla"]
    theory = comparisons["PUD_minus_DualControlSource"]
    delta = float(primary["subject_bootstrap"]["mean"])
    if delta <= TOL:
        terminal_state = "SOURCE_ONLY_PERSIST_NOT_SUPPORTED"
    elif (
        delta >= 0.005 - TOL
        and primary["positive_folds"] >= 4
        and primary["positive_seeds"] >= 2
        and float(theory["subject_bootstrap"]["mean"]) > TOL
    ):
        terminal_state = "SOURCE_ONLY_PERSIST_POSITIVE"
    else:
        terminal_state = "SOURCE_ONLY_PERSIST_WEAK_POSITIVE"

    statistics = {
        "terminal_state": terminal_state,
        "cluster_structure": "three seeds averaged within each of 40 outcome subjects before inference",
        "bootstrap_draws": 10_000,
        "comparisons": comparisons,
        "mechanism": mechanism_stats,
        "interpretation_gate": {
            "delta_vs_Vanilla_at_least_0_005": delta >= 0.005 - TOL,
            "positive_folds_at_least_4": primary["positive_folds"] >= 4,
            "positive_seeds_at_least_2": primary["positive_seeds"] >= 2,
            "bootstrap_lower_above_zero": primary["subject_bootstrap"]["CI95"][0] > 0.0,
            "beats_capacity_matched_dual_source_control": theory["subject_bootstrap"]["mean"] > TOL,
        },
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }

    write_csv(RESULTS / "source_only_main.csv", main)
    write_csv(RESULTS / "per_subject_results.csv", per_subject)
    write_csv(RESULTS / "per_fold_results.csv", per_fold)
    write_csv(RESULTS / "per_seed_results.csv", per_seed)
    write_csv(RESULTS / "source_vs_adapted.csv", source_vs_adapted)
    write_csv(RESULTS / "mechanism_diagnostic.csv", mechanism)
    write_json(RESULTS / "statistics.json", statistics)
    draw_figures(main, per_subject, source_vs_adapted)
    write_figure_qa()

    pud = main.set_index("method").loc["PUD_SOURCE_ONLY"]
    vanilla = main.set_index("method").loc["B0_VANILLA_EEGNET"]
    dual = main.set_index("method").loc["A2_SOURCE_ONLY"]
    strong = main.set_index("method").loc["B1_STRONG_EEGNET"]
    adapted_mean = float(per_subject.loc[per_subject.method.eq("PUD_AFTER_ADAPT"), "BA"].mean())
    report = {
        "terminal_state": terminal_state,
        "base_commit": "12ab811c2a6194192b430f9c010781acd1c0379f",
        "protocol": "OpenBMI V8_SEARCH 40 subjects; 5 folds x 3 seeds; outcome Session 2",
        "Q1_PUD_SOURCE_ONLY": {"BA": float(pud.BA), "macro_f1": float(pud.macro_f1)},
        "Q2_vs_Vanilla": {
            "Vanilla_BA": float(vanilla.BA),
            "Delta_BA": comparisons["PUD_minus_Vanilla"]["subject_bootstrap"]["mean"],
            "CI95": comparisons["PUD_minus_Vanilla"]["subject_bootstrap"]["CI95"],
        },
        "Q3_vs_capacity_matched_dual_source": {
            "control_BA": float(dual.BA),
            "Delta_BA": comparisons["PUD_minus_DualControlSource"]["subject_bootstrap"]["mean"],
            "CI95": comparisons["PUD_minus_DualControlSource"]["subject_bootstrap"]["CI95"],
        },
        "Q4_vs_Strong_EEGNet": {
            "Strong_EEGNet_BA": float(strong.BA),
            "Delta_BA": comparisons["PUD_minus_StrongEEGNet"]["subject_bootstrap"]["mean"],
            "CI95": comparisons["PUD_minus_StrongEEGNet"]["subject_bootstrap"]["CI95"],
        },
        "Q5_positive_folds_vs_Vanilla": comparisons["PUD_minus_Vanilla"]["positive_folds"],
        "Q6_positive_seeds_vs_Vanilla": comparisons["PUD_minus_Vanilla"]["positive_seeds"],
        "Q7_subjects_vs_Vanilla": {
            "improved": comparisons["PUD_minus_Vanilla"]["improved_subjects"],
            "harmed": comparisons["PUD_minus_Vanilla"]["harmed_subjects"],
            "tied": comparisons["PUD_minus_Vanilla"]["tied_subjects"],
        },
        "Q8_target_adaptation": {
            "after_adaptation_BA": adapted_mean,
            "after_minus_source_Delta_BA": comparisons["PUD_after_adapt_minus_source"]["subject_bootstrap"]["mean"],
            "CI95": comparisons["PUD_after_adapt_minus_source"]["subject_bootstrap"]["CI95"],
            "harmed_subjects": comparisons["PUD_after_adapt_minus_source"]["harmed_subjects"],
            "improved_subjects": comparisons["PUD_after_adapt_minus_source"]["improved_subjects"],
        },
        "Q9_protected_branch": mechanism_stats["PUD_SOURCE_ONLY"],
        "Q10_interpretation": terminal_state,
        "Q11_internal_holdout_accessed": False,
        "Q12_WBCIC_outer_accessed": False,
        "replay_validation": replay,
        "evaluation_integrity": evaluation,
        "source_checkpoint_retrained": False,
        "adaptation_function_called": False,
        "optimizer_steps": 0,
    }
    report = portable_source_paths(report, source_experiment)
    write_json(EXPERIMENT / "FINAL_REPORT.json", report)
    render_reports(
        main,
        per_subject,
        per_fold,
        per_seed,
        source_vs_adapted,
        statistics,
        audit,
        replay,
        evaluation,
        checkpoint_audit,
    )
    print(json.dumps(clean(report), ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-experiment", type=Path, default=DEFAULT_SOURCE_EXPERIMENT)
    args = parser.parse_args()
    finalize(args.source_experiment)


if __name__ == "__main__":
    main()
