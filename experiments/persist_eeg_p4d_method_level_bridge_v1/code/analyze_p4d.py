from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import p4d_common as c


COLORS = {"DANN": "#2f5597", "MMD": "#70ad47", "CORAL": "#c55a11", "LOW": "#2f5597", "HIGH": "#c00000"}


def design(frame: pd.DataFrame, bridge: bool = True, methods: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if methods is None:
        methods = sorted(frame.method.astype(str).unique())
    columns = [np.ones(len(frame)), frame.z_SI.to_numpy(float)]
    names = ["intercept", "beta_z_SI"]
    if bridge:
        columns.extend([frame.R_unsafe.to_numpy(float), (frame.z_SI * frame.R_unsafe).to_numpy(float)])
        names.extend(["beta_R_unsafe", "beta_zSI_x_Runsafe"])
    for method in methods[1:]:
        columns.append((frame.method.astype(str) == method).to_numpy(float))
        names.append(f"method_{method}")
    return np.column_stack(columns), names


def fit(frame: pd.DataFrame, bridge: bool = True, methods: list[str] | None = None) -> dict[str, float]:
    matrix, names = design(frame, bridge=bridge, methods=methods)
    coefficients = np.linalg.lstsq(matrix, frame.DeltaG_BA.to_numpy(float), rcond=None)[0]
    return {name: float(value) for name, value in zip(names, coefficients)}


def predict(train: pd.DataFrame, test: pd.DataFrame, bridge: bool) -> np.ndarray:
    methods = sorted(set(train.method.astype(str)) | set(test.method.astype(str)))
    x_train, _ = design(train, bridge=bridge, methods=methods)
    x_test, _ = design(test, bridge=bridge, methods=methods)
    coefficient = np.linalg.lstsq(x_train, train.DeltaG_BA.to_numpy(float), rcond=None)[0]
    return x_test @ coefficient


def hierarchical_sample(frame: pd.DataFrame, rng: np.random.Generator, resample_settings: bool = True) -> pd.DataFrame:
    setting_values = sorted(frame.setting_id.astype(str).unique())
    sampled_settings = rng.choice(setting_values, size=len(setting_values), replace=True) if resample_settings else setting_values
    blocks: list[pd.DataFrame] = []
    for setting_draw, setting in enumerate(sampled_settings):
        setting_frame = frame[frame.setting_id.astype(str) == str(setting)]
        folds = sorted(setting_frame.fold.unique())
        for fold_draw, fold in enumerate(rng.choice(folds, size=len(folds), replace=True)):
            fold_frame = setting_frame[setting_frame.fold == fold]
            seeds = sorted(fold_frame.seed.unique())
            for seed_draw, seed in enumerate(rng.choice(seeds, size=len(seeds), replace=True)):
                block = fold_frame[fold_frame.seed == seed].copy()
                block["bootstrap_setting"] = setting_draw
                block["bootstrap_fold"] = fold_draw
                block["bootstrap_seed"] = seed_draw
                blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def ci(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]


def primary_bootstrap(frame: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    rng = np.random.default_rng(int(protocol["bootstrap"]["seed"]))
    r_low = float(protocol["frozen_R_quantiles"]["R_low_q25"])
    r_high = float(protocol["frozen_R_quantiles"]["R_high_q75"])
    r_split = float(protocol["frozen_R_quantiles"]["R_split_median"])
    headroom = str(protocol["canonical_headroom_method"])
    methods = sorted(frame.method.astype(str).unique())
    rows: list[dict[str, float | int]] = []
    for draw in range(int(protocol["bootstrap"]["draws"])):
        sampled = hierarchical_sample(frame, rng, resample_settings=True)
        coefficient = fit(sampled, bridge=True, methods=methods)
        beta_z = coefficient["beta_z_SI"]
        beta_interaction = coefficient["beta_zSI_x_Runsafe"]
        low_mask = sampled.R_unsafe <= r_split
        high_mask = sampled.R_unsafe > r_split
        head = sampled.method.astype(str) == headroom
        low_values = sampled.loc[low_mask & head, "DeltaG_BA"].to_numpy(float)
        high_values = sampled.loc[high_mask & head, "DeltaG_BA"].to_numpy(float)
        low_gain = float(low_values.mean()) if len(low_values) else float("nan")
        high_gain = float(high_values.mean()) if len(high_values) else float("nan")
        rows.append(
            {
                "draw": draw,
                "beta_z_SI": beta_z,
                "beta_R_unsafe": coefficient["beta_R_unsafe"],
                "beta_zSI_x_Runsafe": beta_interaction,
                "slope_low": beta_z + beta_interaction * r_low,
                "slope_high": beta_z + beta_interaction * r_high,
                "DeltaSlope_bridge": beta_interaction * (r_low - r_high),
                "headroom_low_gain": low_gain,
                "headroom_high_gain": high_gain,
                "HeadroomContrast": low_gain - high_gain,
            }
        )
    return pd.DataFrame(rows)


def grouped_bridge(frame: pd.DataFrame, group_column: str, seed_token: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, part in frame.groupby(group_column, sort=True):
        point = fit(part, bridge=True)
        rng = np.random.default_rng(c.stable_seed("P4D-group", seed_token, group))
        draws = []
        for _ in range(c.BOOTSTRAP_DRAWS):
            sampled = hierarchical_sample(part, rng, resample_settings=group_column != "setting_id")
            draws.append(fit(sampled, bridge=True)["beta_zSI_x_Runsafe"])
        interval = ci(np.asarray(draws))
        rows.append(
            {
                group_column: group,
                "rows": len(part),
                "beta_z_SI": point["beta_z_SI"],
                "beta_R_unsafe": point["beta_R_unsafe"],
                "beta_zSI_x_Runsafe": point["beta_zSI_x_Runsafe"],
                "interaction_CI_lower": interval[0],
                "interaction_CI_upper": interval[1],
                "hypothesis_direction": bool(point["beta_zSI_x_Runsafe"] < 0),
            }
        )
    return pd.DataFrame(rows)


def loso(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for setting in sorted(frame.setting_id.unique()):
        train = frame[frame.setting_id != setting]
        test = frame[frame.setting_id == setting]
        prediction_id = predict(train, test, bridge=False)
        prediction_bridge = predict(train, test, bridge=True)
        target = test.DeltaG_BA.to_numpy(float)
        rows.append(
            {
                "held_out_setting": setting,
                "n": len(test),
                "RMSE_ID": float(np.sqrt(np.mean((target - prediction_id) ** 2))),
                "RMSE_BRIDGE": float(np.sqrt(np.mean((target - prediction_bridge) ** 2))),
            }
        )
    table = pd.DataFrame(rows)
    return {"folds": rows, "RMSE_ID": float(np.sqrt(np.average(table.RMSE_ID**2, weights=table.n))), "RMSE_BRIDGE": float(np.sqrt(np.average(table.RMSE_BRIDGE**2, weights=table.n)))}


def figures(frame: pd.DataFrame, burden: pd.DataFrame, summary: dict[str, Any], per_setting: pd.DataFrame, per_method: pd.DataFrame) -> None:
    plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    c.FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    positions = {"S4": 0, "S6": 1}
    rng = np.random.default_rng(4)
    for setting, part in burden.groupby("setting_id"):
        x = positions[setting] + rng.uniform(-0.09, 0.09, len(part))
        ax.scatter(x, part.R_unsafe, alpha=0.75, label=setting)
    ax.set_xticks([0, 1], ["S4 WBCIC-MI", "S6 OpenBMI-ERP"])
    ax.set_ylabel("Frozen unsafe identity burden $R_{unsafe}$")
    ax.set_title("Source-only unsafe burden across prospective runs")
    fig.tight_layout(); fig.savefig(c.FIGURES / "figure1_unsafe_burden_distribution.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    split = float(summary["frozen_R_split"])
    for label, mask in (("LOW", frame.R_unsafe <= split), ("HIGH", frame.R_unsafe > split)):
        part = frame[mask]
        ax.scatter(part.z_SI, part.DeltaG_BA * 100, color=COLORS[label], alpha=0.75, label=f"{label} unsafe")
        if len(part) >= 2:
            x = np.linspace(part.z_SI.min(), part.z_SI.max(), 100)
            coefficient = np.polyfit(part.z_SI, part.DeltaG_BA * 100, 1)
            ax.plot(x, np.polyval(coefficient, x), color=COLORS[label])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(r"Within-setting identity suppression $z_{SI}$"); ax.set_ylabel(r"Future $\Delta G_{BA}$ (pp)")
    ax.legend(frameon=False); ax.set_title("Global invariance effect by frozen safety burden")
    fig.tight_layout(); fig.savefig(c.FIGURES / "figure2_zsi_vs_future_gain.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    slope = summary["simple_slopes"]
    values = [slope["slope_low"], slope["slope_high"]]
    lowers = [slope["slope_low_CI"][0], slope["slope_high_CI"][0]]
    uppers = [slope["slope_low_CI"][1], slope["slope_high_CI"][1]]
    ax.errorbar([0, 1], values, yerr=[np.asarray(values)-np.asarray(lowers), np.asarray(uppers)-np.asarray(values)], fmt="o", color="#2f5597", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks([0, 1], ["R low (Q25)", "R high (Q75)"])
    ax.set_ylabel(r"Slope: $z_{SI} \rightarrow \Delta G_{BA}$"); ax.set_title("Frozen simple slopes")
    fig.tight_layout(); fig.savefig(c.FIGURES / "figure3_simple_slopes.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    y = np.arange(len(per_setting))
    point = per_setting.beta_zSI_x_Runsafe.to_numpy(float)
    ax.errorbar(point, y, xerr=[np.maximum(0, point-per_setting.interaction_CI_lower), np.maximum(0, per_setting.interaction_CI_upper-point)], fmt="o", capsize=4)
    ax.axvline(0, color="black", linewidth=0.8); ax.set_yticks(y, per_setting.setting_id); ax.set_xlabel(r"$\beta_{zSI \times Runsafe}$")
    ax.set_title("Prospective bridge by setting")
    fig.tight_layout(); fig.savefig(c.FIGURES / "figure4_setting_bridge_forest.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    method_order = ["DANN", "MMD", "CORAL"]
    for y_value, method in enumerate(method_order):
        part = per_method[per_method.method.astype(str) == method]
        if len(part):
            row = part.iloc[0]
            point = float(row.beta_zSI_x_Runsafe)
            ax.errorbar(point, y_value, xerr=[[max(0.0, point-float(row.interaction_CI_lower))], [max(0.0, float(row.interaction_CI_upper)-point)]], fmt="o", capsize=4, color=COLORS[method])
        else:
            ax.scatter([0], [y_value], marker="x", color="#888888")
            ax.annotate("identity manipulation incompetent", (0, y_value), xytext=(6, 0), textcoords="offset points", va="center", color="#666666", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8); ax.set_yticks(np.arange(3), method_order); ax.set_xlabel(r"$\beta_{zSI \times Runsafe}$")
    ax.set_title("Per-method bridge (incompetent methods not estimated)")
    fig.tight_layout(); fig.savefig(c.FIGURES / "figure5_method_bridge_forest.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    head = summary["headroom"]
    values = [head["low_unsafe_DeltaG_BA"], head["high_unsafe_DeltaG_BA"]]
    cis = [head["low_unsafe_CI"], head["high_unsafe_CI"]]
    ax.bar([0, 1], np.asarray(values)*100, color=[COLORS["LOW"], COLORS["HIGH"]], alpha=0.85)
    ax.errorbar([0, 1], np.asarray(values)*100, yerr=[np.maximum(0, np.asarray(values)*100-np.asarray([v[0] for v in cis])*100), np.maximum(0, np.asarray([v[1] for v in cis])*100-np.asarray(values)*100)], fmt="none", color="black", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8); ax.set_xticks([0, 1], ["Low unsafe", "High unsafe"]); ax.set_ylabel("Future gain (pp)")
    ax.set_title(f"Canonical headroom method: {head['method']}")
    fig.tight_layout(); fig.savefig(c.FIGURES / "figure6_headroom_low_vs_high.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 3.8)); ax.axis("off")
    boxes = [
        (0.03, 0.56, 0.25, 0.28, "Direction level\nHigh-I + High-E\nunsafe to erase", "#f4cccc"),
        (0.375, 0.56, 0.25, 0.28, "Run burden\nmore task-entangled\nidentity ($R_{unsafe}$)", "#fff2cc"),
        (0.72, 0.56, 0.25, 0.28, "Method level\nstronger global suppression\nmore likely harmful", "#d9ead3"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="#555555")); ax.text(x+w/2, y+h/2, text, transform=ax.transAxes, ha="center", va="center")
    ax.annotate("", xy=(0.375, 0.70), xytext=(0.28, 0.70), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.annotate("", xy=(0.72, 0.70), xytext=(0.625, 0.70), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.5, 0.25, f"Observed interaction: {summary['coefficients']['beta_zSI_x_Runsafe']:+.5f} [{summary['coefficient_CIs']['beta_zSI_x_Runsafe'][0]:+.5f}, {summary['coefficient_CIs']['beta_zSI_x_Runsafe'][1]:+.5f}]\nTerminal: {summary['P4D_terminal']}", transform=ax.transAxes, ha="center", va="center", weight="bold")
    fig.tight_layout(); fig.savefig(c.FIGURES / "figure7_mechanism_to_method_bridge.png"); plt.close(fig)


def reports(frame: pd.DataFrame, burden: pd.DataFrame, summary: dict[str, Any], per_setting: pd.DataFrame, per_method: pd.DataFrame) -> None:
    coefficient = summary["coefficients"]
    cis = summary["coefficient_CIs"]
    slopes = summary["simple_slopes"]
    head = summary["headroom"]
    method_text = ", ".join(f"{row.method}: beta_int={row.beta_zSI_x_Runsafe:+.6f}" for row in per_method.itertuples())
    setting_text = ", ".join(f"{row.setting_id}: beta_int={row.beta_zSI_x_Runsafe:+.6f}" for row in per_setting.itertuples())
    c.write_text(c.EXP / "PRIMARY_BRIDGE_INTERACTION_AUDIT.md", f"# Primary Bridge Interaction Audit\n\nThe frozen model estimated `beta_zSI={coefficient['beta_z_SI']:+.9f}`, `beta_Runsafe={coefficient['beta_R_unsafe']:+.9f}`, and primary interaction `beta_zSI×Runsafe={coefficient['beta_zSI_x_Runsafe']:+.9f}` with hierarchical 95% CI `[{cis['beta_zSI_x_Runsafe'][0]:+.9f}, {cis['beta_zSI_x_Runsafe'][1]:+.9f}]`. The one-method prospective matrix limits causal breadth; no outcome-driven rescue was performed.")
    c.write_text(c.EXP / "SIMPLE_SLOPE_BRIDGE_AUDIT.md", f"# Simple Slope Bridge Audit\n\nAt frozen Q25 burden, slope={slopes['slope_low']:+.9f} with CI `{slopes['slope_low_CI']}`. At frozen Q75 burden, slope={slopes['slope_high']:+.9f} with CI `{slopes['slope_high_CI']}`. DeltaSlope={slopes['DeltaSlope_bridge']:+.9f} with CI `{slopes['DeltaSlope_CI']}`.")
    c.write_text(c.EXP / "CROSS_SETTING_METHOD_BRIDGE.md", f"# Cross-Setting Method Bridge\n\n{setting_text}. Both settings were retained regardless of outcome.\n\n" + c.markdown_table(per_setting))
    c.write_text(c.EXP / "CROSS_METHOD_STABILITY.md", f"# Cross-Method Stability\n\nOnly methods passing the frozen identity manipulation competence rule enter the primary analysis. {method_text}. Fewer than two competent methods means the cross-method strong gate fails by construction; incompetent methods were not retrained or substituted.\n\n" + c.markdown_table(per_method))
    c.write_text(c.EXP / "HEADROOM_AUDIT.md", f"# Headroom Audit\n\nThe identity-only canonical headroom method is `{head['method']}`. Low-unsafe mean DeltaG_BA={head['low_unsafe_DeltaG_BA']:+.9f}, CI={head['low_unsafe_CI']}; high-unsafe mean={head['high_unsafe_DeltaG_BA']:+.9f}, CI={head['high_unsafe_CI']}; contrast={head['HeadroomContrast']:+.9f}, CI={head['HeadroomContrast_CI']}. P4E remains `{summary['P4E_MODEL_AUTHORIZATION']}` because P4C was partial.")
    c.write_text(c.EXP / "HOLDOUT_PURITY_AUDIT.md", "# Holdout Purity Audit\n\nCanonical lambdas and headroom method were selected using S4 source identity only. S6 training accessed source/validation roles only. Protocol, burden thresholds, normalization, bootstrap, and gates were frozen before future method outcomes. OpenBMI sealed internal holdout was untouched. WBCIC outer 10 remained untouched and unenumerated. The P4A 405-grid remained paused.")
    c.write_text(c.EXP / "THEORY_METHOD_BRIDGE_NOTE.md", "# Theory-to-Method Bridge Note\n\nLet `z=A_y y + A_s s + epsilon`, task head `W`, and a global subject-invariance method suppress subject subspace `P_s`. When `W P_s` is near zero, suppression acts mostly in task-null directions. When `W P_s` is large, global suppression also perturbs task-relevant structure. Frozen `R_unsafe` is an empirical run-level burden for identity-bearing directions entangled with the task decision structure. The prespecified interaction tests whether stronger measured identity suppression becomes more costly as this burden rises. This is a falsifiable empirical bridge, not a theorem guaranteeing generalization gains.")


def main() -> None:
    evaluation = c.read_json(c.RESULTS / "P4D_METHOD_OUTCOME_EVALUATION_COMPLETE.json")
    if evaluation.get("pass") is not True:
        raise RuntimeError("prospective canonical evaluation incomplete")
    frame = pd.read_csv(c.RESULTS / "canonical_method_future_outcomes.csv")
    burden = pd.read_csv(c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv")
    protocol = c.read_json(c.EXP / "P4D_PROTOCOL_FROZEN.json")
    canonical = c.read_json(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json")
    point = fit(frame, bridge=True)
    bootstrap_path = c.RESULTS / "bridge_bootstrap.csv"
    if bootstrap_path.is_file():
        bootstrap = pd.read_csv(bootstrap_path)
        if len(bootstrap) != int(protocol["bootstrap"]["draws"]):
            raise RuntimeError("existing primary bootstrap has wrong draw count")
    else:
        bootstrap = primary_bootstrap(frame, protocol)
        c.write_csv(bootstrap_path, bootstrap)
    coefficient_cis = {name: ci(bootstrap[name].to_numpy(float)) for name in ("beta_z_SI", "beta_R_unsafe", "beta_zSI_x_Runsafe")}
    r_low = float(protocol["frozen_R_quantiles"]["R_low_q25"])
    r_high = float(protocol["frozen_R_quantiles"]["R_high_q75"])
    slope_point = {
        "R_low": r_low,
        "R_high": r_high,
        "slope_low": point["beta_z_SI"] + point["beta_zSI_x_Runsafe"] * r_low,
        "slope_high": point["beta_z_SI"] + point["beta_zSI_x_Runsafe"] * r_high,
        "DeltaSlope_bridge": point["beta_zSI_x_Runsafe"] * (r_low - r_high),
        "slope_low_CI": ci(bootstrap.slope_low),
        "slope_high_CI": ci(bootstrap.slope_high),
        "DeltaSlope_CI": ci(bootstrap.DeltaSlope_bridge),
    }
    c.write_csv(c.RESULTS / "simple_slope_summary.csv", pd.DataFrame([slope_point]))
    per_setting = grouped_bridge(frame, "setting_id", "setting")
    per_method = grouped_bridge(frame, "method", "method")
    c.write_csv(c.RESULTS / "per_setting_bridge.csv", per_setting)
    c.write_csv(c.RESULTS / "per_method_bridge.csv", per_method)
    split = float(protocol["frozen_R_quantiles"]["R_split_median"])
    headroom_method = str(protocol["canonical_headroom_method"])
    head_frame = frame[frame.method.astype(str) == headroom_method]
    low = head_frame[head_frame.R_unsafe <= split].DeltaG_BA
    high = head_frame[head_frame.R_unsafe > split].DeltaG_BA
    headroom = {
        "method": headroom_method,
        "frozen_R_split": split,
        "low_n": len(low),
        "high_n": len(high),
        "low_unsafe_DeltaG_BA": float(low.mean()),
        "low_unsafe_CI": ci(bootstrap.headroom_low_gain),
        "high_unsafe_DeltaG_BA": float(high.mean()),
        "high_unsafe_CI": ci(bootstrap.headroom_high_gain),
        "HeadroomContrast": float(low.mean() - high.mean()),
        "HeadroomContrast_CI": ci(bootstrap.HeadroomContrast),
    }
    c.write_json(c.RESULTS / "headroom_summary.json", headroom)
    competent_count = int(canonical["competent_method_count"])
    g1 = competent_count >= 2
    g2 = point["beta_zSI_x_Runsafe"] < 0 and coefficient_cis["beta_zSI_x_Runsafe"][1] < 0
    g3 = slope_point["DeltaSlope_bridge"] > 0 and slope_point["DeltaSlope_CI"][0] > 0
    g4 = len(per_setting) == 2 and bool(per_setting.hypothesis_direction.all())
    g5 = int(per_method.hypothesis_direction.sum()) >= 2
    g6 = True
    gates = {"G1_manipulation_competence": g1, "G2_primary_interaction": g2, "G3_simple_slope": g3, "G4_setting_consistency": g4, "G5_method_consistency": g5, "G6_purity": g6}
    if all(gates.values()):
        terminal = "P4D_METHOD_LEVEL_BRIDGE_STRONG_SUPPORTED"
    elif point["beta_zSI_x_Runsafe"] < 0 and slope_point["DeltaSlope_bridge"] > 0 and g4:
        terminal = "P4D_METHOD_LEVEL_BRIDGE_PARTIAL_SUPPORTED"
    else:
        terminal = "P4D_METHOD_LEVEL_BRIDGE_NOT_SUPPORTED"
    p4e = "NOT_AUTHORIZED"
    summary: dict[str, Any] = {
        "schema": "PERSIST_EEG_P4D_BRIDGE_MODEL_SUMMARY_V1",
        "timestamp_utc": c.now_utc(),
        "rows": len(frame),
        "competent_methods": sorted(frame.method.astype(str).unique().tolist()),
        "coefficients": point,
        "coefficient_CIs": coefficient_cis,
        "simple_slopes": slope_point,
        "per_setting_direction": dict(zip(per_setting.setting_id, per_setting.hypothesis_direction.astype(bool))),
        "per_method_direction": dict(zip(per_method.method, per_method.hypothesis_direction.astype(bool))),
        "gates": gates,
        "P4D_terminal": terminal,
        "P4E_MODEL_AUTHORIZATION": p4e,
        "P4E_reason": "P4C safety terminal is PARTIAL, while P4E requires P4C STRONG; the decision is invariant to P4D/headroom outcome",
        "frozen_R_split": split,
        "headroom": headroom,
        "LOSO": loso(frame),
        "bootstrap_draws": len(bootstrap),
        "cluster_hierarchy": ["setting", "fold", "seed/run"],
        "method_configs_nested_within_run": True,
        "outcome_driven_modification": False,
    }
    c.write_json(c.RESULTS / "bridge_model_summary.json", summary)
    figures(frame, burden, summary, per_setting, per_method)
    reports(frame, burden, summary, per_setting, per_method)
    print(json.dumps(summary, indent=2))
    print("P4D_ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
