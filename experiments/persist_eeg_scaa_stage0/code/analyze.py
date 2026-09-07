from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common as c


BOOTSTRAPS = 10_000
LCB_BOOTSTRAPS = 2_000


def qci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return math.nan
    if method == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    return float(stats.spearmanr(x, y).statistic)


def bootstrap_correlation(x: np.ndarray, y: np.ndarray, method: str, seed: int) -> tuple[float, float, float]:
    point = safe_corr(x, y, method)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))
    samples = np.asarray([safe_corr(x[idx], y[idx], method) for idx in indices], dtype=float)
    low, high = qci(samples)
    return point, low, high


def bootstrap_harm_policy(frame: pd.DataFrame, seed: int) -> dict[str, tuple[float, float, float]]:
    d2 = frame.Delta_S2_BA.to_numpy(float)
    d3 = frame.Delta_S3_BA.to_numpy(float)
    anchor = frame.anchor_S3_BA.to_numpy(float)
    adapted = frame.adapted_S3_BA.to_numpy(float)

    def values(idx: np.ndarray) -> dict[str, float]:
        x2, x3 = d2[idx], d3[idx]
        cert = x2 > 0
        always_harm = float(np.mean(x3 < 0))
        certified_harm = float(np.mean(x3[cert] < 0)) if cert.any() else math.nan
        reduction_abs = always_harm - certified_harm if math.isfinite(certified_harm) else math.nan
        reduction_rel = 1 - certified_harm / always_harm if always_harm > 0 and math.isfinite(certified_harm) else math.nan
        gated = np.where(cert, adapted[idx], anchor[idx])
        return {
            "harm_always": always_harm,
            "harm_certified": certified_harm,
            "harm_reduction_absolute": reduction_abs,
            "harm_reduction_relative": reduction_rel,
            "coverage": float(cert.mean()),
            "anchor_policy_BA": float(anchor[idx].mean()),
            "always_adapt_policy_BA": float(adapted[idx].mean()),
            "S2_gated_policy_BA": float(gated.mean()),
            "gated_minus_anchor": float(gated.mean() - anchor[idx].mean()),
            "gated_minus_always": float(gated.mean() - adapted[idx].mean()),
        }

    point = values(np.arange(len(frame)))
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, len(frame), size=len(frame))
        current = values(sample)
        for key, value in current.items():
            draws[key].append(value)
    return {key: (value, *qci(np.asarray(draws[key]))) for key, value in point.items()}


def aggregate_subjects(seed_frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "anchor_S1_validation_confidence", "anchor_S1_validation_BA", "adapted_S1_validation_BA",
        "parameter_relative_change", "anchor_S2_BA", "adapted_S2_BA", "Delta_S2_BA",
        "anchor_S3_BA", "adapted_S3_BA", "Delta_S3_BA", "Delta_S2_macro_F1", "Delta_S3_macro_F1",
    ]
    per_backbone = seed_frame.groupby(["backbone", "subject_id"], as_index=False)[numeric].mean()
    per_backbone["S1_validation_BA_delta"] = per_backbone.adapted_S1_validation_BA - per_backbone.anchor_S1_validation_BA
    per_backbone.insert(0, "scope", per_backbone.backbone)
    pooled = per_backbone.groupby("subject_id", as_index=False)[numeric + ["S1_validation_BA_delta"]].mean()
    pooled.insert(0, "scope", "Pooled")
    pooled.insert(1, "backbone", "Pooled")
    return pd.concat([per_backbone, pooled], ignore_index=True)


def paired_lcb(trials: dict[str, np.ndarray], subject: int, backbones: list[str], seed: int) -> float:
    models: list[np.ndarray] = []
    reference_rows = None
    reference_labels = None
    for backbone in backbones:
        for model_seed in c.SEEDS:
            mask = (
                (trials["backbone"] == backbone)
                & (trials["subject_id"] == subject)
                & (trials["session_id"] == 1)
                & (trials["seed"] == model_seed)
            )
            order = np.argsort(trials["row_index"][mask])
            rows = trials["row_index"][mask][order]
            labels = trials["label"][mask][order]
            diff = (
                (trials["adapted_pred"][mask][order] == labels).astype(float)
                - (trials["anchor_pred"][mask][order] == labels).astype(float)
            )
            if reference_rows is None:
                reference_rows, reference_labels = rows, labels
            elif not np.array_equal(rows, reference_rows) or not np.array_equal(labels, reference_labels):
                raise RuntimeError(f"trial alignment failure for subject {subject}")
            models.append(diff)
    assert reference_labels is not None
    rng = np.random.default_rng(seed)
    samples: dict[int, np.ndarray] = {}
    for label in (0, 1):
        positions = np.flatnonzero(reference_labels == label)
        samples[label] = rng.choice(positions, size=(LCB_BOOTSTRAPS, len(positions)), replace=True)
    utilities = np.zeros(LCB_BOOTSTRAPS, dtype=float)
    for diff in models:
        utilities += 0.5 * (
            diff[samples[0]].mean(axis=1) + diff[samples[1]].mean(axis=1)
        ) / len(models)
    return float(np.quantile(utilities, 0.10))


def save_figures(subjects: pd.DataFrame, policy: pd.DataFrame, harm: pd.DataFrame) -> None:
    c.FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"EEGNet": "#2671b8", "EEGConformer": "#d95f02", "Pooled": "#4d4d4d"}

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    for ax, scope in zip(axes, ("EEGNet", "EEGConformer")):
        frame = subjects[subjects.scope == scope]
        x, y = 100 * frame.Delta_S2_BA, 100 * frame.Delta_S3_BA
        limit = max(1.0, float(np.max(np.abs(np.r_[x, y]))) * 1.12)
        ax.axhline(0, color="0.65", lw=0.8)
        ax.axvline(0, color="0.65", lw=0.8)
        ax.plot([-limit, limit], [-limit, limit], ls="--", lw=0.8, color="0.55")
        ax.scatter(x, y, s=22, alpha=0.82, color=colors[scope], edgecolor="white", linewidth=0.3)
        rho = stats.spearmanr(x, y).statistic
        ax.set(title=f"{scope} (Spearman {rho:.2f})", xlabel=r"$\Delta$ S2 (pp)", ylabel=r"$\Delta$ S3 (pp)", xlim=(-limit, limit), ylim=(-limit, limit))
    for suffix in ("png", "pdf"):
        fig.savefig(c.FIGURES / f"utility_transfer_scatter.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    labels = ["Anchor", "Always Adapt", "S2-Gated Adapt"]
    x = np.arange(3)
    width = 0.22
    for i, scope in enumerate(("EEGNet", "EEGConformer", "Pooled")):
        row = policy[policy.scope == scope].iloc[0]
        vals = 100 * np.array([row.anchor_policy_BA, row.always_adapt_policy_BA, row.S2_gated_policy_BA])
        ax.bar(x + (i - 1) * width, vals, width, label=scope, color=colors[scope])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean S3 balanced accuracy (%)")
    ax.legend(frameon=False, ncol=3)
    for suffix in ("png", "pdf"):
        fig.savefig(c.FIGURES / f"policy_comparison.{suffix}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    scopes = ["EEGNet", "EEGConformer", "Pooled"]
    xx = np.arange(3)
    axes[0].bar(xx - 0.18, 100 * harm.set_index("scope").loc[scopes].harm_always, 0.36, label="Always Adapt", color="#999999")
    axes[0].bar(xx + 0.18, 100 * harm.set_index("scope").loc[scopes].harm_certified, 0.36, label="S2-positive", color="#2ca25f")
    axes[0].set(xticks=xx, xticklabels=scopes, ylabel="S3 negative-transfer rate (%)", title="Future harm")
    axes[0].legend(frameon=False)
    axes[1].bar(xx, 100 * harm.set_index("scope").loc[scopes].coverage, 0.55, color=[colors[s] for s in scopes])
    axes[1].axhline(25, ls="--", lw=0.8, color="0.5")
    axes[1].set(xticks=xx, xticklabels=scopes, ylabel="Certificate coverage (%)", title="Coverage", ylim=(0, 100))
    for suffix in ("png", "pdf"):
        fig.savefig(c.FIGURES / f"harm_coverage.{suffix}", dpi=220)
    plt.close(fig)

    pooled = subjects[subjects.scope == "Pooled"].sort_values("subject_id")
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    x = np.arange(len(pooled))
    d2, d3 = 100 * pooled.Delta_S2_BA.to_numpy(), 100 * pooled.Delta_S3_BA.to_numpy()
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [d2[i], d3[i]], color="#bdbdbd", lw=0.7)
    ax.scatter(x, d2, s=15, label=r"$\Delta$ S2", color="#2671b8")
    ax.scatter(x, d3, s=15, label=r"$\Delta$ S3", color="#d95f02")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set(xticks=x, xticklabels=pooled.subject_id.astype(str), xlabel="Development subject", ylabel="Utility (pp)")
    ax.legend(frameon=False, ncol=2)
    for suffix in ("png", "pdf"):
        fig.savefig(c.FIGURES / f"per_subject_transfer.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    lock = c.read_json(c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json")
    execution = c.read_json(c.RUNTIME / "UTILITY_EXECUTION.json")
    if not execution.get("complete") or execution.get("S2_or_S3_used_for_training_or_selection") is not False:
        raise RuntimeError("frozen utility execution is absent or impure")
    if execution["protocol_lock_sha256"] != c.sha256(c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json"):
        raise RuntimeError("protocol lock differs from utility execution")
    seed_frame = pd.read_csv(c.RESULTS / "PER_SUBJECT_SEED_UTILITY.csv", dtype={"subject_id": str})
    if len(seed_frame) != 246:
        raise RuntimeError("expected 246 seed-level utility rows")
    subjects = aggregate_subjects(seed_frame)

    trials_npz = np.load(c.RUNTIME / "TRIAL_PREDICTIONS.npz", allow_pickle=False)
    trials = {key: trials_npz[key] for key in trials_npz.files}
    for scope in ("EEGNet", "EEGConformer", "Pooled"):
        backbones = list(c.BACKBONES) if scope == "Pooled" else [scope]
        mask = subjects.scope == scope
        for idx, row in subjects[mask].iterrows():
            subjects.loc[idx, "LCB_S2_90"] = paired_lcb(
                trials,
                int(row.subject_id),
                backbones,
                c.stable_seed("SCAA-LCB", scope, row.subject_id),
            )
    c.write_csv(c.RESULTS / "PER_SUBJECT_UTILITY.csv", subjects)

    correlation_rows = []
    sign_rows = []
    harm_rows = []
    policy_rows = []
    lcb_rows = []
    control_rows = []
    summary_rows = []
    tests: dict[str, dict] = {}
    scope_frames: dict[str, pd.DataFrame] = {}
    for scope in ("EEGNet", "EEGConformer", "Pooled"):
        frame = subjects[subjects.scope == scope].copy().sort_values("subject_id")
        scope_frames[scope] = frame
        x, y = frame.Delta_S2_BA.to_numpy(float), frame.Delta_S3_BA.to_numpy(float)
        corr_result = {}
        for method in ("pearson", "spearman"):
            point, low, high = bootstrap_correlation(x, y, method, c.stable_seed("SCAA-correlation", scope, method))
            correlation_rows.append({"scope": scope, "method": method, "estimate": point, "CI95_low": low, "CI95_high": high, "subjects": len(frame), "bootstrap_resamples": BOOTSTRAPS})
            corr_result[method] = {"estimate": point, "CI95": [low, high]}

        concordant = np.sign(x) == np.sign(y)
        count = int(concordant.sum())
        binomial = stats.binomtest(count, len(frame), 0.5, alternative="two-sided")
        exact_ci = binomial.proportion_ci(confidence_level=0.95, method="exact")
        sign_rows.append({
            "scope": scope, "subjects": len(frame), "concordant": count,
            "sign_concordance": float(concordant.mean()), "exact_CI95_low": float(exact_ci.low),
            "exact_CI95_high": float(exact_ci.high), "exact_binomial_p_two_sided": float(binomial.pvalue),
            "zero_rule": "np.sign; zero only concordant with zero",
        })

        bootstrap = bootstrap_harm_policy(frame, c.stable_seed("SCAA-harm-policy", scope))
        harm_rows.append({
            "scope": scope, "subjects": len(frame),
            **{name: bootstrap[name][0] for name in ("harm_always", "harm_certified", "harm_reduction_absolute", "harm_reduction_relative", "coverage")},
            **{f"{name}_CI95_low": bootstrap[name][1] for name in ("harm_always", "harm_certified", "harm_reduction_absolute", "harm_reduction_relative", "coverage")},
            **{f"{name}_CI95_high": bootstrap[name][2] for name in ("harm_always", "harm_certified", "harm_reduction_absolute", "harm_reduction_relative", "coverage")},
            "certificate_positive_subjects": int((x > 0).sum()),
            "S2_positive_S3_negative_reversals": int(((x > 0) & (y < 0)).sum()),
        })
        policy_rows.append({
            "scope": scope, "subjects": len(frame),
            **{name: bootstrap[name][0] for name in ("anchor_policy_BA", "always_adapt_policy_BA", "S2_gated_policy_BA", "gated_minus_anchor", "gated_minus_always")},
            **{f"{name}_CI95_low": bootstrap[name][1] for name in ("anchor_policy_BA", "always_adapt_policy_BA", "S2_gated_policy_BA", "gated_minus_anchor", "gated_minus_always")},
            **{f"{name}_CI95_high": bootstrap[name][2] for name in ("anchor_policy_BA", "always_adapt_policy_BA", "S2_gated_policy_BA", "gated_minus_anchor", "gated_minus_always")},
            "subjects_gated_policy_above_anchor": int((np.where(x > 0, frame.adapted_S3_BA, frame.anchor_S3_BA) > frame.anchor_S3_BA).sum()),
            "subjects_gated_policy_above_always": int((np.where(x > 0, frame.adapted_S3_BA, frame.anchor_S3_BA) > frame.adapted_S3_BA).sum()),
        })
        lcb_cert = frame.LCB_S2_90.to_numpy(float) > 0
        lcb_rows.append({
            "scope": scope, "confidence": "90% one-sided", "subjects": len(frame),
            "certificate_positive_subjects": int(lcb_cert.sum()), "coverage": float(lcb_cert.mean()),
            "future_harm_given_LCB_positive": float(np.mean(y[lcb_cert] < 0)) if lcb_cert.any() else math.nan,
            "primary_gate_use": False, "bootstrap_resamples_within_subject": LCB_BOOTSTRAPS,
        })
        predictors = {
            "target_history_Delta_S2": frame.Delta_S2_BA.to_numpy(float),
            "anchor_S1_validation_confidence": frame.anchor_S1_validation_confidence.to_numpy(float),
            "S1_parameter_relative_change": frame.parameter_relative_change.to_numpy(float),
            "S1_validation_BA_delta": frame.S1_validation_BA_delta.to_numpy(float),
        }
        for predictor, value in predictors.items():
            control_rows.append({"scope": scope, "predictor": predictor, "Spearman_with_Delta_S3": safe_corr(value, y, "spearman")})
        tests[scope] = {
            "correlation": corr_result,
            "sign": sign_rows[-1],
            "harm_and_coverage": harm_rows[-1],
            "policy": policy_rows[-1],
        }
        summary_rows.append({
            "scope": scope,
            "subjects": len(frame),
            "mean_Delta_S2_BA": float(x.mean()),
            "mean_Delta_S3_BA": float(y.mean()),
            "Spearman": corr_result["spearman"]["estimate"],
            "Spearman_CI95_low": corr_result["spearman"]["CI95"][0],
            "Spearman_CI95_high": corr_result["spearman"]["CI95"][1],
            "sign_concordance": float(concordant.mean()),
            "harm_always": bootstrap["harm_always"][0],
            "harm_certified": bootstrap["harm_certified"][0],
            "coverage": bootstrap["coverage"][0],
        })

    correlations = pd.DataFrame(correlation_rows)
    signs = pd.DataFrame(sign_rows)
    harms = pd.DataFrame(harm_rows)
    policies = pd.DataFrame(policy_rows)
    controls = pd.DataFrame(control_rows)
    lcb = pd.DataFrame(lcb_rows)
    summaries = pd.DataFrame(summary_rows)
    c.write_csv(c.RESULTS / "BACKBONE_SUMMARY.csv", summaries)
    c.write_csv(c.RESULTS / "UTILITY_TRANSFER_CORRELATION.csv", correlations)
    c.write_csv(c.RESULTS / "SIGN_CONCORDANCE.csv", signs)
    c.write_csv(c.RESULTS / "HARM_AND_COVERAGE.csv", harms)
    c.write_csv(c.RESULTS / "POLICY_COMPARISON.csv", policies)
    c.write_csv(c.RESULTS / "SECONDARY_LCB_ANALYSIS.csv", lcb)
    c.write_csv(c.RESULTS / "CONTROL_DIAGNOSTICS.csv", controls)

    pooled_corr = correlations[(correlations.scope == "Pooled") & (correlations.method == "spearman")].iloc[0]
    pooled_sign = signs[signs.scope == "Pooled"].iloc[0]
    pooled_harm = harms[harms.scope == "Pooled"].iloc[0]
    pooled_policy = policies[policies.scope == "Pooled"].iloc[0]
    backbone_spearman = summaries.set_index("scope").loc[["EEGNet", "EEGConformer"], "Spearman"]
    gates = {
        "A_utility_transfer": bool(pooled_corr.estimate > 0 and pooled_corr.CI95_low > 0 and (backbone_spearman > 0).all()),
        "B_sign_persistence": bool(pooled_sign.sign_concordance >= 0.65 and pooled_sign.exact_binomial_p_two_sided < 0.05 and pooled_sign.exact_CI95_low > 0.5),
        "C_reduced_future_harm": bool(pooled_harm.harm_certified < pooled_harm.harm_always and pooled_harm.harm_reduction_absolute_CI95_low > 0 and pooled_harm.harm_reduction_relative >= 0.30),
        "D_nontrivial_coverage": bool(pooled_harm.coverage >= 0.25),
        "E_policy_usefulness": bool(pooled_policy.S2_gated_policy_BA >= max(pooled_policy.anchor_policy_BA, pooled_policy.always_adapt_policy_BA) - 0.005),
    }
    supported = all(gates.values())
    partial = bool(
        not supported
        and pooled_corr.estimate > 0
        and pooled_sign.sign_concordance > 0.5
        and pooled_harm.harm_certified < pooled_harm.harm_always
        and pooled_harm.coverage >= 0.25
        and (backbone_spearman >= 0).all()
    )
    terminal = (
        "TARGET_HISTORY_UTILITY_TRANSFER_SUPPORTED" if supported
        else "TARGET_HISTORY_UTILITY_TRANSFER_PARTIAL" if partial
        else "TARGET_HISTORY_UTILITY_TRANSFER_NOT_SUPPORTED"
    )
    authorization = "SCAA_DEVELOPMENT_AUTHORIZED" if supported else "SCAA_DEVELOPMENT_NOT_AUTHORIZED"
    control_pooled = controls[controls.scope == "Pooled"].set_index("predictor").Spearman_with_Delta_S3
    history_outperforms = bool(
        abs(control_pooled["target_history_Delta_S2"])
        > max(abs(control_pooled.drop("target_history_Delta_S2")))
    )
    tests["strong_support_gates"] = gates
    tests["terminal"] = terminal
    tests["authorization"] = authorization
    tests["bootstrap_resamples"] = BOOTSTRAPS
    tests["statistical_unit"] = "subject"
    c.write_json(c.RESULTS / "STATISTICAL_TESTS.json", tests)

    save_figures(subjects, policies, harms)
    eegnet_corr = correlations[(correlations.scope == "EEGNet") & (correlations.method == "spearman")].iloc[0]
    conformer_corr = correlations[(correlations.scope == "EEGConformer") & (correlations.method == "spearman")].iloc[0]
    final = {
        "schema": "PERSIST_EEG_SCAA_STAGE0_FINAL_REPORT_V1",
        "branch": "codex/persist-eeg-scaa-stage0",
        "analysis_commit": c.git_head(),
        "protocol_lock_sha256": c.sha256(c.PROTOCOL / "SCAA_STAGE0_PROTOCOL_LOCK.json"),
        "development_subjects_only": True,
        "development_subject_count": 41,
        "outer_10_untouched_unenumerated": True,
        "target_never_seen_by_anchor": True,
        "frozen_adapter": lock["adapter"],
        "competence": c.read_json(c.PROTOCOL / "ADAPTATION_RECIPE_SELECTION.json")["selected_summary"],
        "correlations": {row.scope: {"estimate": row.estimate, "CI95": [row.CI95_low, row.CI95_high]} for row in correlations[correlations.method == "spearman"].itertuples()},
        "pooled_sign": pooled_sign.to_dict(),
        "pooled_harm_coverage": pooled_harm.to_dict(),
        "pooled_policy": pooled_policy.to_dict(),
        "secondary_LCB": lcb[lcb.scope == "Pooled"].iloc[0].to_dict(),
        "target_history_outperforms_simple_proxies_descriptively": history_outperforms,
        "strong_support_gates": gates,
        "authorization": authorization,
        "terminal": terminal,
        "strongest_supported_claim": (
            "Same-target historical adaptation utility contains prospective information about next-session utility under both frozen WBCIC backbones."
            if supported else
            "Same-target historical utility shows favorable but insufficient prospective evidence under the frozen WBCIC analysis."
            if partial else
            "The frozen WBCIC experiment does not establish reliable transfer of same-target adaptation utility to the next session."
        ),
        "stronger_claim_not_justified": "This Stage-0 does not establish that SCAA improves generalization or formally controls negative transfer.",
        "primary_limitation": "Single WBCIC development cohort, one simple head-only adapter, two correlated backbones, and no OpenBMI or sealed-outer confirmation.",
    }
    c.write_json(c.EXP / "SCAA_STAGE0_FINAL_REPORT.json", final)

    gate_lines = "\n".join(f"- {name}: `{'PASS' if value else 'FAIL'}`" for name, value in gates.items())
    report = f"""# SCAA Stage-0 final report

## Frozen experiment

1. Only the 41 WBCIC development subjects were used: **yes**.
2. The outer 10 remained untouched and unenumerated: **yes**.
3. Every target used its outcome-fold anchor, which had never seen that target: **yes**.
4. Frozen adaptation: classifier-head-only supervised AdamW, encoder and normalization frozen, LR `{lock['adapter']['learning_rate']}`, weight decay `{lock['adapter']['weight_decay']}`, maximum `{lock['adapter']['maximum_epochs']}` epochs.
5. It was selected because it passed the source/S1-only competence gate without a last-block repair.
6. Selection and protocol locking occurred before S2/S3 utility inspection: **yes**.
7. S1-only competence was nontrivial: mean BA delta `{100 * final['competence']['S1_validation_BA_delta']:+.2f}` pp, prediction-change rate `{final['competence']['prediction_change_rate']:.3f}`, catastrophic fraction `{final['competence']['catastrophic_fraction']:.3f}`.

## Prospective utility transfer

8. EEGNet Spearman: `{eegnet_corr.estimate:.4f}`.
9. EEGNet 95% subject-bootstrap CI: `[{eegnet_corr.CI95_low:.4f}, {eegnet_corr.CI95_high:.4f}]`.
10. EEGConformer Spearman: `{conformer_corr.estimate:.4f}`.
11. EEGConformer 95% CI: `[{conformer_corr.CI95_low:.4f}, {conformer_corr.CI95_high:.4f}]`.
12. Pooled within-subject Spearman: `{pooled_corr.estimate:.4f}`; CI `[{pooled_corr.CI95_low:.4f}, {pooled_corr.CI95_high:.4f}]`.
13. Pooled sign concordance: `{pooled_sign.sign_concordance:.3f}` (`{int(pooled_sign.concordant)}/41`).
14. Exact two-sided binomial p versus 0.5: `{pooled_sign.exact_binomial_p_two_sided:.6g}`; exact CI `[{pooled_sign.exact_CI95_low:.3f}, {pooled_sign.exact_CI95_high:.3f}]`.
15. Always-Adapt S3 negative-transfer rate: `{pooled_harm.harm_always:.3f}`.
16. S2-positive-certified S3 negative-transfer rate: `{pooled_harm.harm_certified:.3f}`.
17. Relative harm reduction: `{pooled_harm.harm_reduction_relative:.3f}`.
18. Certificate coverage: `{pooled_harm.coverage:.3f}` (`{int(pooled_harm.certificate_positive_subjects)}/41`).
19. Mean pooled S3 Anchor BA: `{pooled_policy.anchor_policy_BA:.4f}`.
20. Mean pooled S3 Always-Adapt BA: `{pooled_policy.always_adapt_policy_BA:.4f}`.
21. Mean pooled S3 S2-Gated BA: `{pooled_policy.S2_gated_policy_BA:.4f}`.
22. Subjects whose gated policy exceeds Anchor: `{int(pooled_policy.subjects_gated_policy_above_anchor)}`.
23. S2-positive subjects reversing negative on S3: `{int(pooled_harm.S2_positive_S3_negative_reversals)}`.
24. Backbone consistency: EEGNet/EEGConformer Spearman signs are `{'compatible' if (backbone_spearman >= 0).all() else 'incompatible'}`.
25. Target-history utility exceeds the absolute pooled Spearman of each simple S1 proxy: **{str(history_outperforms).lower()}** (descriptive, not a gate).
26. Secondary 90% LCB coverage/harm: `{float(final['secondary_LCB']['coverage']):.3f}` / `{final['secondary_LCB']['future_harm_given_LCB_positive']}`; it does not rescue the primary analysis.

## Decision

27. Strong Support gates:
{gate_lines}
28. Authorization: `{authorization}`.
29. Strongest justified claim: {final['strongest_supported_claim']}
30. Not justified: {final['stronger_claim_not_justified']}
31. Terminal: `{terminal}`.
"""
    c.write_text(c.EXP / "SCAA_STAGE0_FINAL_REPORT.md", report)
    c.write_text(c.EXP / "CLAIM_AUDIT.md", f"""# Claim audit

- Frozen primary terminal: `{terminal}`.
- Authorization: `{authorization}`.
- Strongest supported claim: {final['strongest_supported_claim']}
- Prohibited stronger claim: {final['stronger_claim_not_justified']}
- No certificate threshold, subject, seed, fold, backbone, adapter, or primary metric was changed after outcome access.
- OpenBMI and the sealed outer 10 were not accessed.
""")
    c.write_text(c.EXP / "REPRODUCIBILITY.md", f"""# Reproducibility

Run from repository root with the frozen server environment:

1. `{lock['exact_commands'][0]}`
2. `{lock['exact_commands'][1]}`
3. `{lock['exact_commands'][2]}`

The protocol lock records code, data-lock, recipe, checkpoint, and normalizer hashes. Primary inference uses 10,000 subject resamples; seeds are averaged within subject and are not treated as biological replicates. Runtime trial predictions and raw EEG are excluded from Git.
""")
    print(f"SCAA_STAGE0_ANALYSIS_COMPLETE terminal={terminal} authorization={authorization}", flush=True)


if __name__ == "__main__":
    main()
