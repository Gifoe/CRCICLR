from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
P4B = REPO / "experiments" / "persist_eeg_p4b_identity_reliability_discovery_v1"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 441027

sys.path.insert(0, str(HERE))
from p4c_safety_common import dataframe_markdown, now_utc, percentile_ci, read_json, sha256, write_json  # noqa: E402


@dataclass
class RunData:
    setting: str
    fold: int
    seed: int
    subjects: list[str]
    regime: np.ndarray
    highest_index: int
    u_ba: np.ndarray
    u_f1: np.ndarray
    u_ce: np.ndarray
    control_ba: np.ndarray
    specific_ba: np.ndarray


def build_runs(subject: pd.DataFrame) -> dict[tuple[str, int, int], RunData]:
    runs: dict[tuple[str, int, int], RunData] = {}
    for key, cell in subject.groupby(["setting_id", "fold", "seed"], sort=True):
        setting, fold, seed = str(key[0]), int(key[1]), int(key[2])
        subjects = sorted(cell.outcome_subject.astype(str).unique())
        first = cell.drop_duplicates("direction_rank").sort_values("direction_rank")
        if len(first) != 8 or first.highest_identity.astype(bool).sum() != 1:
            raise RuntimeError(f"run direction/highest-I cardinality {setting}/{fold}/{seed}")
        matrices: dict[str, np.ndarray] = {}
        for column in ["U_BA", "U_F1", "U_CE", "control_U_BA_mean", "SpecificU_BA"]:
            matrix = np.empty((len(subjects), 8), dtype=np.float64)
            for index, name in enumerate(subjects):
                rows = cell[cell.outcome_subject.astype(str) == name].sort_values("direction_rank")
                if len(rows) != 8:
                    raise RuntimeError(f"subject direction cardinality {setting}/{fold}/{seed}/{name}")
                matrix[index] = rows[column].to_numpy(float)
            matrices[column] = matrix
        regime = np.array([0 if value == "REGIME_LOW" else 1 if value == "REGIME_HIGH" else -1 for value in first.regime_label], dtype=np.int64)
        highest_index = int(np.flatnonzero(first.highest_identity.astype(bool).to_numpy())[0])
        runs[(setting, fold, seed)] = RunData(setting, fold, seed, subjects, regime, highest_index, matrices["U_BA"], matrices["U_F1"], matrices["U_CE"], matrices["control_U_BA_mean"], matrices["SpecificU_BA"])
    return runs


def sampled_metrics(runs: dict[tuple[str, int, int], RunData], settings: list[str], rng: np.random.Generator) -> dict[str, float]:
    low: list[float] = []
    high: list[float] = []
    low_control: list[float] = []
    high_control: list[float] = []
    low_specific: list[float] = []
    high_specific: list[float] = []
    highest: list[float] = []
    sampled_settings = rng.choice(settings, size=len(settings), replace=True) if len(settings) > 1 else np.asarray(settings)
    for setting in sampled_settings:
        folds = sorted({key[1] for key in runs if key[0] == str(setting)})
        for fold in rng.choice(folds, size=len(folds), replace=True):
            seeds = sorted({key[2] for key in runs if key[0] == str(setting) and key[1] == int(fold)})
            for seed in rng.choice(seeds, size=len(seeds), replace=True):
                run = runs[(str(setting), int(fold), int(seed))]
                n_subject = len(run.subjects)
                direction_draw = rng.integers(0, 8, size=8)
                subject_draw = rng.integers(0, n_subject, size=(8, n_subject))
                columns = np.arange(8)[:, None]
                def draw_mean(matrix: np.ndarray) -> np.ndarray:
                    selected = matrix[:, direction_draw]
                    return selected[subject_draw, columns].mean(axis=1)
                values = draw_mean(run.u_ba)
                controls = draw_mean(run.control_ba)
                specifics = draw_mean(run.specific_ba)
                labels = run.regime[direction_draw]
                low.extend(values[labels == 0])
                high.extend(values[labels == 1])
                low_control.extend(controls[labels == 0])
                high_control.extend(controls[labels == 1])
                low_specific.extend(specifics[labels == 0])
                high_specific.extend(specifics[labels == 1])
                highest_draw = rng.integers(0, n_subject, size=n_subject)
                highest.append(float(run.u_ba[highest_draw, run.highest_index].mean()))
    if not low or not high:
        raise ValueError("bootstrap draw lacks one regime")
    u_low = float(np.mean(low))
    u_high = float(np.mean(high))
    return {
        "U_low": u_low,
        "U_high": u_high,
        "DeltaRegime": u_low - u_high,
        "control_U_low": float(np.mean(low_control)),
        "control_U_high": float(np.mean(high_control)),
        "SpecificU_low": float(np.mean(low_specific)),
        "SpecificU_high": float(np.mean(high_specific)),
        "U_HighestI": float(np.mean(highest)),
    }


def bootstrap(runs: dict[tuple[str, int, int], RunData], settings: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for draw in range(BOOTSTRAP_DRAWS):
        for _attempt in range(100):
            try:
                row = sampled_metrics(runs, settings, rng)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"bootstrap lacks estimable draw for {settings}")
        rows.append({"analysis_scope": "POOLED" if len(settings) > 1 else settings[0], "draw": draw, **row})
        if (draw + 1) % 1000 == 0:
            print(f"[bootstrap {'+'.join(settings)}] {draw + 1}/{BOOTSTRAP_DRAWS}", flush=True)
    return pd.DataFrame(rows)


def point_summary(direction: pd.DataFrame, settings: list[str]) -> dict[str, float | int | str]:
    cell = direction[direction.setting_id.isin(settings)]
    low = cell[cell.regime_label == "REGIME_LOW"]
    high = cell[cell.regime_label == "REGIME_HIGH"]
    highest = cell[cell.highest_identity.astype(bool)]
    u_low = float(low.U_BA.mean())
    u_high = float(high.U_BA.mean())
    return {
        "setting_id": "POOLED" if len(settings) > 1 else settings[0],
        "low_count": len(low),
        "high_count": len(high),
        "low_folds": low.fold.nunique(),
        "high_folds": high.fold.nunique(),
        "low_seeds": low.seed.nunique(),
        "high_seeds": high.seed.nunique(),
        "U_low": u_low,
        "U_high": u_high,
        "DeltaRegime": u_low - u_high,
        "U_F1_low": float(low.U_F1.mean()),
        "U_F1_high": float(high.U_F1.mean()),
        "U_CE_low": float(low.U_CE.mean()),
        "U_CE_high": float(high.U_CE.mean()),
        "control_U_low": float(low.control_U_BA_mean.mean()),
        "control_U_high": float(high.control_U_BA_mean.mean()),
        "SpecificU_low": float(low.SpecificU_BA.mean()),
        "SpecificU_high": float(high.SpecificU_BA.mean()),
        "U_HighestI": float(highest.U_BA.mean()),
    }


def add_cis(summary: pd.DataFrame, bootstrap_frame: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    for index, row in output.iterrows():
        draws = bootstrap_frame[bootstrap_frame.analysis_scope == row.setting_id]
        for metric in ["U_low", "U_high", "DeltaRegime", "SpecificU_low", "SpecificU_high", "U_HighestI"]:
            lower, upper = percentile_ci(draws[metric].to_numpy(float))
            output.loc[index, f"{metric}_CI_lower"] = lower
            output.loc[index, f"{metric}_CI_upper"] = upper
    return output


def make_figures(assignments: pd.DataFrame, summary: pd.DataFrame, discovery: pd.DataFrame, matched: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    order = ["S4", "S6", "POOLED"]
    cell = summary.set_index("setting_id").loc[order]
    x = np.arange(3)
    width = .34
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for offset, metric, label, color in [(-width/2, "U_low", "High-I + Low-E", "#4C78A8"), (width/2, "U_high", "High-I + High-E", "#E45756")]:
        values = cell[metric].to_numpy(float)
        errors = np.vstack([values - cell[f"{metric}_CI_lower"].to_numpy(float), cell[f"{metric}_CI_upper"].to_numpy(float) - values])
        ax.bar(x + offset, values, width, yerr=errors, capsize=3, label=label, color=color)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, order); ax.set(ylabel="Future U_BA", title="Prospective suppression safety by frozen regime"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "figure2_future_uba_regimes.png", dpi=220); plt.close(fig)

    for figure_number, metric, title, color in [(3, "U_high", "High-E suppression veto versus no-op", "#E45756"), (4, "U_low", "Low-E suppression actionability versus no-op", "#4C78A8")]:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        values = cell[metric].to_numpy(float)
        errors = np.vstack([values - cell[f"{metric}_CI_lower"].to_numpy(float), cell[f"{metric}_CI_upper"].to_numpy(float) - values])
        ax.bar(order, values, yerr=errors, capsize=4, color=color)
        ax.axhline(0, color="black", lw=1.2, label="No-op U=0"); ax.set(ylabel="Future U_BA", title=title); ax.legend(); ax.grid(axis="y", alpha=.2)
        fig.tight_layout(); fig.savefig(FIGURES / f"figure{figure_number}_{'high_e_veto' if figure_number == 3 else 'low_e_actionability'}.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    labels = []
    targets = []
    controls = []
    for setting in order:
        for regime in ["REGIME_LOW", "REGIME_HIGH"]:
            row = matched[(matched.setting_id == setting) & (matched.regime_label == regime)].iloc[0]
            labels.append(f"{setting}\n{'Low-E' if regime == 'REGIME_LOW' else 'High-E'}")
            targets.append(row.target_U_BA)
            controls.append(row.control_U_BA)
    xx = np.arange(len(labels))
    ax.bar(xx - .18, targets, .36, label="Target", color="#4C78A8")
    ax.bar(xx + .18, controls, .36, label="Matched random", color="#BAB0AC")
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(xx, labels); ax.set(ylabel="Future U_BA", title="Target versus displacement-matched random erasure"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "figure5_matched_random_specificity.png", dpi=220); plt.close(fig)

    prospective = summary[summary.setting_id.isin(["S4", "S6"])][["setting_id", "DeltaRegime", "DeltaRegime_CI_lower", "DeltaRegime_CI_upper"]].copy()
    discovery_plot = discovery.set_index("setting_id").loc[["S1", "S2", "S3", "S5"], ["DeltaRegime"]].reset_index()
    labels = discovery_plot.setting_id.tolist() + prospective.setting_id.tolist()
    values = discovery_plot.DeltaRegime.tolist() + prospective.DeltaRegime.tolist()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.scatter(discovery_plot.DeltaRegime, y[:len(discovery_plot)], color="#72B7B2", s=50, label="DISCOVERY point")
    p_y = y[len(discovery_plot):]
    p_value = prospective.DeltaRegime.to_numpy(float)
    p_err = np.vstack([p_value - prospective.DeltaRegime_CI_lower.to_numpy(float), prospective.DeltaRegime_CI_upper.to_numpy(float) - p_value])
    ax.errorbar(p_value, p_y, xerr=p_err, fmt="o", color="#E45756", capsize=4, label="PROSPECTIVE HELD-OUT 95% CI")
    ax.axvline(0, color="black", lw=.8); ax.set_yticks(y, [f"{name} — {'DISCOVERY' if name in {'S1','S2','S3','S5'} else 'PROSPECTIVE HELD-OUT'}" for name in labels]); ax.set(xlabel="DeltaRegime = U_low - U_high", title="Discovery and prospective safety-boundary effects"); ax.legend(fontsize=8); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "figure6_discovery_prospective_forest.png", dpi=220); plt.close(fig)


def main() -> None:
    protocol = read_json(EXP / "P4C_SAFETY_PROTOCOL_FROZEN.json")
    pre = read_json(EXP / "P4C_SAFETY_PREOUTCOME_FREEZE.json")
    outcome = read_json(RESULTS / "P4C_SAFETY_OUTCOME_EVALUATION_COMPLETE.json")
    if pre.get("pass") is not True or outcome.get("pass") is not True:
        raise RuntimeError("freeze/outcome prerequisite failure")
    assignments = pd.read_csv(RESULTS / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv")
    subject = pd.read_csv(RESULTS / "p4c_safety_future_utility_subject.csv")
    direction = pd.read_csv(RESULTS / "p4c_safety_future_utility_direction.csv")
    if len(direction) != 240 or direction[["U_BA", "U_F1", "U_CE", "control_U_BA_mean", "SpecificU_BA"]].isna().any().any():
        raise RuntimeError("outcome cardinality/completeness failure")
    runs = build_runs(subject)
    boot = pd.concat([
        bootstrap(runs, ["S4"], BOOTSTRAP_SEED + 4),
        bootstrap(runs, ["S6"], BOOTSTRAP_SEED + 6),
        bootstrap(runs, ["S4", "S6"], BOOTSTRAP_SEED),
    ], ignore_index=True)
    boot.to_csv(RESULTS / "p4c_safety_bootstrap.csv", index=False)
    summary = pd.DataFrame([point_summary(direction, ["S4"]), point_summary(direction, ["S6"]), point_summary(direction, ["S4", "S6"])])
    summary = add_cis(summary, boot)
    summary.to_csv(RESULTS / "p4c_safety_regime_summary.csv", index=False)
    matched_rows = []
    for _, row in summary.iterrows():
        for regime, suffix in [("REGIME_LOW", "low"), ("REGIME_HIGH", "high")]:
            matched_rows.append({
                "setting_id": row.setting_id,
                "regime_label": regime,
                "target_U_BA": row[f"U_{suffix}"],
                "control_U_BA": row[f"control_U_{suffix}"],
                "SpecificU_BA": row[f"SpecificU_{suffix}"],
                "SpecificU_BA_CI_lower": row[f"SpecificU_{suffix}_CI_lower"],
                "SpecificU_BA_CI_upper": row[f"SpecificU_{suffix}_CI_upper"],
                "control_count_per_direction": 100,
            })
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(RESULTS / "p4c_safety_matched_random_summary.csv", index=False)
    highest = summary[["setting_id", "U_HighestI", "U_HighestI_CI_lower", "U_HighestI_CI_upper"]].copy()
    highest["No_op"] = 0.0
    highest.to_csv(RESULTS / "p4c_safety_highest_identity_summary.csv", index=False)
    discovery = pd.read_csv(P4B / "results" / "regime_summary.csv")
    make_figures(assignments, summary, discovery, matched)

    point = summary.set_index("setting_id")
    pooled = point.loc["POOLED"]
    g1 = bool(pooled.DeltaRegime > 0 and pooled.DeltaRegime_CI_lower > 0)
    g2 = bool(pooled.U_high < 0 and pooled.U_high_CI_upper < 0)
    g3 = bool(point.loc["S4", "DeltaRegime"] > 0 and point.loc["S6", "DeltaRegime"] > 0)
    g4 = bool(point.loc["S4", "U_high"] < 0 and point.loc["S6", "U_high"] < 0)
    purity = bool(
        outcome.get("preoutcome_freeze_sha256") == sha256(EXP / "P4C_SAFETY_PREOUTCOME_FREEZE.json")
        and outcome.get("post_outcome_scientific_modification") is False
        and protocol.get("reserved_settings") == ["S4", "S6"]
        and read_json(P4B / "results" / "P4B_FINAL_VALIDATION.json").get("p4c_reserved_future_utility_accessed") is False
    )
    g5 = purity
    indicators = [point.loc[setting, metric] > 0 if metric == "DeltaRegime" else point.loc[setting, metric] < 0 for setting in ["S4", "S6"] for metric in ["DeltaRegime", "U_high"]]
    neither_double_reversal = all(not (point.loc[setting, "DeltaRegime"] <= 0 and point.loc[setting, "U_high"] >= 0) for setting in ["S4", "S6"])
    partial = bool(pooled.DeltaRegime > 0 and pooled.U_high < 0 and sum(indicators) >= 3 and neither_double_reversal and purity)
    if all([g1, g2, g3, g4, g5]):
        safety_status = "P4C_SAFETY_BOUNDARY_STRONG_SUPPORTED"
        bridge = "AUTHORIZED"
    elif partial:
        safety_status = "P4C_SAFETY_BOUNDARY_PARTIAL_SUPPORTED"
        bridge = "CONDITIONAL"
    else:
        safety_status = "P4C_SAFETY_BOUNDARY_NOT_SUPPORTED"
        bridge = "NOT_AUTHORIZED"
    if pooled.U_low > 0 and pooled.U_low_CI_lower > 0 and point.loc["S4", "U_low"] >= 0 and point.loc["S6", "U_low"] >= 0:
        low_status = "LOW_E_SUPPRESSION_BENEFICIAL"
    elif pooled.U_low <= 0 or pooled.U_low_CI_upper < 0 or (point.loc["S4", "U_low"] < 0 and point.loc["S6", "U_low"] < 0):
        low_status = "LOW_E_SUPPRESSION_NOT_BENEFICIAL"
    else:
        low_status = "LOW_E_SUPPRESSION_INCONCLUSIVE"
    result = {
        "schema": "P4C_SAFETY_ANALYSIS_COMPLETE_V1",
        "timestamp_utc": now_utc(),
        "pass": True,
        "SAFETY_STATUS_candidate": safety_status,
        "LOW_E_ACTIONABILITY_STATUS_candidate": low_status,
        "METHOD_LEVEL_BRIDGE_AUTHORIZATION_candidate": bridge,
        "FINAL_NEW_MODEL_AUTHORIZATION": "NOT_AUTHORIZED_AT_THIS_STAGE",
        "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5},
        "purity_pass": purity,
        "point_results": summary.to_dict("records"),
        "bootstrap_draws_per_scope": BOOTSTRAP_DRAWS,
        "bootstrap_rows": len(boot),
        "post_outcome_protocol_modification": False,
    }
    write_json(RESULTS / "P4C_SAFETY_ANALYSIS_COMPLETE.json", result)
    summary_table = dataframe_markdown(summary)
    (EXP / "REGIME_SEPARATION_PROSPECTIVE_AUDIT.md").write_text("# Prospective Regime Separation Audit\n\n" + summary_table + f"\n\nPooled DeltaRegime CI: [{pooled.DeltaRegime_CI_lower:.9f}, {pooled.DeltaRegime_CI_upper:.9f}]. Gate G1={g1}; cross-setting point replication G3={g3}.\n", encoding="utf-8")
    (EXP / "SUPPRESSION_VETO_AUDIT.md").write_text("# Suppression Veto Audit\n\nThe primary safety veto is High-I + High-E U_BA < 0.\n\n" + dataframe_markdown(summary[["setting_id", "U_high", "U_high_CI_lower", "U_high_CI_upper"]]) + f"\n\nG2={g2}; G4={g4}.\n", encoding="utf-8")
    (EXP / "MATCHED_RANDOM_SPECIFICITY_AUDIT.md").write_text("# Matched Random Specificity Audit\n\nControls use 100 deterministic full-space Gaussian directions per target and per-trial displacement-norm matching. Control draws remain nested, not independent N.\n\n" + dataframe_markdown(matched) + "\n", encoding="utf-8")
    (EXP / "LOW_ENTANGLEMENT_ACTIONABILITY_AUDIT.md").write_text("# Low-Entanglement Actionability Audit\n\n" + dataframe_markdown(summary[["setting_id", "U_low", "U_low_CI_lower", "U_low_CI_upper"]]) + f"\n\nCandidate status: `{low_status}`. Safety and beneficial actionability are evaluated separately.\n", encoding="utf-8")
    (EXP / "CROSS_SETTING_SAFETY_STABILITY.md").write_text("# Cross-Setting Safety Stability\n\n" + dataframe_markdown(summary[summary.setting_id != "POOLED"][["setting_id", "U_low", "U_high", "DeltaRegime", "SpecificU_high", "U_HighestI"]]) + f"\n\nDirectional indicators correct: {sum(indicators)}/4. No setting was dropped.\n", encoding="utf-8")
    (EXP / "THEORY_ADMISSIBILITY_NOTE.md").write_text(
        "# Minimal Theory: Nuisance Admissibility\n\nLet `z=A_y y + A_s s + epsilon`, task head `f(z)=Wz`, and subject projection `P_s`. Suppression gives `z'=(I-P_s)z`, hence `Wz'-Wz=-WP_s z`. "
        "When `WP_s` is near zero, the subject subspace is approximately decision-decoupled; when it is nonzero, erasure directly removes task-decision-relevant variation. `D_finite`, `O_task`, and `C_src` are empirical views of this coupling, so task entanglement can serve as an invariance veto. "
        "This argument does not guarantee that Low-E suppression improves future BA. It supports admissibility, not beneficial actionability.\n",
        encoding="utf-8",
    )
    (EXP / "HOLDOUT_PURITY_AUDIT.md").write_text(
        "# Holdout Purity Audit\n\n"
        f"Purity candidate: {'PASS' if purity else 'FAIL'}. S4/S6 labels, thresholds, direction bank, Highest-I and matched-control seeds were hash-frozen before first outcome access. "
        "No setting, threshold, E_task primitive, model or direction was changed afterward. OpenBMI sealed internal holdout remains UNTOUCHED; WBCIC outer 10 remains UNTOUCHED_NOT_ENUMERATED. P4A remains OPTIONAL_PARTIAL_INVARIANCE_GRID and paused.\n",
        encoding="utf-8",
    )
    print(f"P4C_SAFETY_ANALYSIS_COMPLETE candidate={safety_status} lowE={low_status}", flush=True)


if __name__ == "__main__":
    main()
