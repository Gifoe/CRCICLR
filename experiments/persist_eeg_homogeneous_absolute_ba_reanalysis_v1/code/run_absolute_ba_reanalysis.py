"""Statistics-only, subject-equal absolute BA reanalysis of frozen ensembles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments" / "persist_eeg_matched_homogeneous_ensemble_control_v1"
OUT = ROOT / "experiments" / "persist_eeg_homogeneous_absolute_ba_reanalysis_v1"
INPUT = SOURCE / "subject_level_results.csv"

BLOCKS = ("OpenBMI_MI", "OpenBMI_ERP", "OpenBMI_SSVEP", "WBCIC_MI")
FAMILIES = (
    "EEGNet_homogeneous",
    "LiteBN_homogeneous",
    "Heterogeneous_orientation_mean",
)
EL_COMPONENTS = ("Heterogeneous_A", "Heterogeneous_B")
PAIRS = {(0, 1), (0, 2), (1, 2)}
FOLDS = set(range(5))
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260909
MATERIAL_SHIFT_PP = 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise RuntimeError("bootstrap requires a non-empty finite subject vector")
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def global_bootstrap(
    values_by_block: dict[str, np.ndarray], rng: np.random.Generator, dataset_balanced: bool
) -> dict[str, Any]:
    if set(values_by_block) != set(BLOCKS):
        raise RuntimeError("all four locked blocks are required for a global summary")
    block_means: dict[str, float] = {}
    draws_by_block: dict[str, np.ndarray] = {}
    for block in BLOCKS:
        values = np.asarray(values_by_block[block], dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise RuntimeError(f"invalid subject contrast values: {block}")
        block_means[block] = float(values.mean())
        draws_by_block[block] = values[
            rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
        ].mean(axis=1)
    if dataset_balanced:
        openbmi = sum(draws_by_block[b] for b in BLOCKS[:3]) / 3.0
        draws = 0.5 * openbmi + 0.5 * draws_by_block["WBCIC_MI"]
        mean = 0.5 * (sum(block_means[b] for b in BLOCKS[:3]) / 3.0 + block_means["WBCIC_MI"])
        method = "OpenBMI task average, then 50/50 OpenBMI/WBCIC"
    else:
        draws = sum(draws_by_block.values()) / len(BLOCKS)
        mean = float(np.mean(list(block_means.values())))
        method = "equal average of four locked blocks"
    return {
        "mean_pp": float(mean),
        "ci_low_pp": float(np.quantile(draws, 0.025)),
        "ci_high_pp": float(np.quantile(draws, 0.975)),
        "block_means_pp": block_means,
        "method": method,
    }


def validate(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "block", "subject_id", "fold", "seed_i", "seed_j", "family",
        "fusion_rule", "ensemble_BA", "orientation_count",
        "trial_identity_sha256", "label_sha256", "trials",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"source result columns missing: {sorted(missing)}")
    if set(frame.block.unique()) != set(BLOCKS):
        raise RuntimeError(f"wrong locked blocks: {sorted(frame.block.unique())}")
    if set(frame.fusion_rule.unique()) != {"LOGIT50", "PROB50"}:
        raise RuntimeError("source does not contain exactly LOGIT50 and PROB50")

    expected = {(fold, i, j) for fold in FOLDS for i, j in PAIRS}
    report: dict[str, Any] = {
        "pass": True,
        "all_four_blocks": True,
        "all_five_folds": True,
        "all_three_seed_pairs": True,
        "same_matched_subjects": True,
        "both_EL_orientations_retained_equally": True,
        "no_duplicate_primary_replicates": True,
        "alignment_hashes_match_across_families": True,
        "rules": {},
    }
    for rule in ("LOGIT50", "PROB50"):
        data = frame[frame.fusion_rule.eq(rule)].copy()
        for block in BLOCKS:
            block_data = data[data.block.eq(block)].copy()
            cohorts: dict[str, set[str]] = {}
            for family in FAMILIES:
                rows = block_data[block_data.family.eq(family)].copy()
                rows["subject_id"] = rows.subject_id.astype(str)
                counts = rows.groupby("subject_id").size()
                if rows.empty or not counts.eq(15).all():
                    raise RuntimeError(f"missing 15 fixed replicates: {rule}/{block}/{family}")
                if rows.duplicated(["subject_id", "fold", "seed_i", "seed_j"]).any():
                    raise RuntimeError(f"duplicate fixed replicate: {rule}/{block}/{family}")
                for _, subject_rows in rows.groupby("subject_id", sort=False):
                    found = set(
                        zip(
                            subject_rows.fold.astype(int),
                            subject_rows.seed_i.astype(int),
                            subject_rows.seed_j.astype(int),
                        )
                    )
                    if found != expected:
                        raise RuntimeError(f"wrong fold/pair set: {rule}/{block}/{family}")
                cohorts[family] = set(rows.subject_id)
            if len({frozenset(x) for x in cohorts.values()}) != 1:
                raise RuntimeError(f"family subject mismatch: {rule}/{block}")

            keys = ["subject_id", "fold", "seed_i", "seed_j"]
            primary = block_data[block_data.family.isin(FAMILIES)]
            alignment = primary.groupby(keys).agg(
                families=("family", "nunique"),
                identity_hashes=("trial_identity_sha256", "nunique"),
                label_hashes=("label_sha256", "nunique"),
                trial_counts=("trials", "nunique"),
            )
            if not (
                alignment.families.eq(3).all()
                and alignment.identity_hashes.eq(1).all()
                and alignment.label_hashes.eq(1).all()
                and alignment.trial_counts.eq(1).all()
            ):
                raise RuntimeError(f"trial/label ledger mismatch: {rule}/{block}")

            el = block_data[block_data.family.isin((*EL_COMPONENTS, "Heterogeneous_orientation_mean"))]
            el_wide = el.pivot(index=keys, columns="family", values="ensemble_BA")
            if set(el_wide.columns) != set((*EL_COMPONENTS, "Heterogeneous_orientation_mean")):
                raise RuntimeError(f"EL orientations incomplete: {rule}/{block}")
            if not np.allclose(
                el_wide.Heterogeneous_orientation_mean.to_numpy(),
                0.5 * (el_wide.Heterogeneous_A.to_numpy() + el_wide.Heterogeneous_B.to_numpy()),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"EL orientation average incorrect: {rule}/{block}")
            if not block_data[block_data.family.eq("Heterogeneous_orientation_mean")].orientation_count.eq(2).all():
                raise RuntimeError(f"EL orientation count is not two: {rule}/{block}")
            report["rules"][f"{rule}:{block}"] = {
                "n_subjects": int(len(next(iter(cohorts.values())))),
                "subjects": sorted(next(iter(cohorts.values()))),
                "replicates_per_subject_family": 15,
                "folds": sorted(FOLDS),
                "seed_pairs": [list(pair) for pair in sorted(PAIRS)],
            }
    return report


def aggregate_subject_family(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    data = frame[frame.fusion_rule.eq(rule) & frame.family.isin(FAMILIES)].copy()
    result = (
        data.groupby(["block", "subject_id", "family"], as_index=False)
        .agg(subject_family_BA=("ensemble_BA", "mean"), n_replicates=("ensemble_BA", "size"))
    )
    if not result.n_replicates.eq(15).all():
        raise RuntimeError("a primary subject/family lacks a fixed replicate")
    result.subject_id = result.subject_id.astype(str)
    result.subject_family_BA *= 100.0
    return result.sort_values(["block", "subject_id", "family"], kind="stable").reset_index(drop=True)


def absolute_table(
    subject: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    deltas: dict[str, dict[str, np.ndarray]] = {
        "EL_minus_EE": {}, "EL_minus_LL": {}, "EL_minus_HOMOAVG": {}
    }
    for block in BLOCKS:
        wide = (
            subject[subject.block.eq(block)]
            .pivot(index="subject_id", columns="family", values="subject_family_BA")
            .reindex(columns=FAMILIES)
        )
        if wide.isna().any().any():
            raise RuntimeError(f"incomplete absolute BA subject table: {block}")
        ee, ll, el = (wide[family].to_numpy(dtype=float) for family in FAMILIES)
        ees, lls, els = (bootstrap_mean(values, rng) for values in (ee, ll, el))
        contrast = {
            "EL_minus_EE": el - ee,
            "EL_minus_LL": el - ll,
            "EL_minus_HOMOAVG": el - 0.5 * (ee + ll),
        }
        summary = {name: bootstrap_mean(values, rng) for name, values in contrast.items()}
        for name, values in contrast.items():
            deltas[name][block] = values
        rows.append({
            "block": block,
            "EE_BA": ees["mean"], "EE_CI_LOW": ees["ci_low"], "EE_CI_HIGH": ees["ci_high"],
            "LL_BA": lls["mean"], "LL_CI_LOW": lls["ci_low"], "LL_CI_HIGH": lls["ci_high"],
            "EL_BA": els["mean"], "EL_CI_LOW": els["ci_low"], "EL_CI_HIGH": els["ci_high"],
            "EL_minus_EE_pp": summary["EL_minus_EE"]["mean"],
            "EL_minus_EE_CI_LOW": summary["EL_minus_EE"]["ci_low"],
            "EL_minus_EE_CI_HIGH": summary["EL_minus_EE"]["ci_high"],
            "EL_minus_LL_pp": summary["EL_minus_LL"]["mean"],
            "EL_minus_LL_CI_LOW": summary["EL_minus_LL"]["ci_low"],
            "EL_minus_LL_CI_HIGH": summary["EL_minus_LL"]["ci_high"],
            "EL_minus_HOMOAVG_pp": summary["EL_minus_HOMOAVG"]["mean"],
            "EL_minus_HOMOAVG_CI_LOW": summary["EL_minus_HOMOAVG"]["ci_low"],
            "EL_minus_HOMOAVG_CI_HIGH": summary["EL_minus_HOMOAVG"]["ci_high"],
            "n_subjects": int(len(wide)),
        })
    return pd.DataFrame(rows), deltas


def single_model_reference(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = frame[frame.fusion_rule.eq("LOGIT50")]
    pieces: list[pd.DataFrame] = []
    for source_family, output_family in (
        ("EEGNet_homogeneous", "EEGNet_single"),
        ("LiteBN_homogeneous", "LiteBN_single"),
    ):
        rows = primary[primary.family.eq(source_family)]
        for value in ("BA_A", "BA_B"):
            piece = rows[["block", "subject_id", value]].copy()
            piece.columns = ["block", "subject_id", "subject_single_BA"]
            piece["family"] = output_family
            pieces.append(piece)
    subject = (
        pd.concat(pieces, ignore_index=True)
        .groupby(["block", "subject_id", "family"], as_index=False)
        .agg(subject_single_BA=("subject_single_BA", "mean"), n_replicates=("subject_single_BA", "size"))
    )
    if not subject.n_replicates.eq(30).all():
        raise RuntimeError("matched single-model ledger is incomplete")
    subject.subject_id = subject.subject_id.astype(str)
    subject.subject_single_BA *= 100.0
    rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        wide = subject[subject.block.eq(block)].pivot(
            index="subject_id", columns="family", values="subject_single_BA"
        )
        es, ls = (
            bootstrap_mean(wide[family].to_numpy(dtype=float), rng)
            for family in ("EEGNet_single", "LiteBN_single")
        )
        rows.append({
            "block": block,
            "EEGNet_BA": es["mean"], "EEGNet_CI_LOW": es["ci_low"], "EEGNet_CI_HIGH": es["ci_high"],
            "LiteBN_BA": ls["mean"], "LiteBN_CI_LOW": ls["ci_low"], "LiteBN_CI_HIGH": ls["ci_high"],
            "n_subjects": int(len(wide)),
            "n_replicates_per_subject_family": 30,
        })
    return subject, pd.DataFrame(rows)


def global_summary(
    deltas: dict[str, dict[str, np.ndarray]], rng: np.random.Generator
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bootstrap": {
            "uncertainty_unit": "subject",
            "paired_contrasts": True,
            "replicates": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
        },
        "equal_block": {},
        "dataset_balanced_optional": {},
    }
    for name, values in deltas.items():
        result["equal_block"][name] = global_bootstrap(values, rng, dataset_balanced=False)
        result["dataset_balanced_optional"][name] = global_bootstrap(values, rng, dataset_balanced=True)
    return result


def complementarity(
    subject: pd.DataFrame, frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    source = frame[frame.fusion_rule.eq("LOGIT50") & frame.family.isin(FAMILIES)]
    comp = (
        source.groupby(["block", "subject_id", "family"], as_index=False)
        .agg(subject_exclusive_correct_pct=("exclusive_correct_fraction", "mean"))
    )
    comp.subject_exclusive_correct_pct *= 100.0
    comp_wide = comp.pivot(
        index=["block", "subject_id"], columns="family", values="subject_exclusive_correct_pct"
    )
    ba_wide = subject.pivot(
        index=["block", "subject_id"], columns="family", values="subject_family_BA"
    )
    rows: list[dict[str, Any]] = []
    global_values = {family: [] for family in FAMILIES}
    for block in BLOCKS:
        c, b = comp_wide.xs(block), ba_wide.xs(block)
        cmean = {family: float(c[family].mean()) for family in FAMILIES}
        bmean = {family: float(b[family].mean()) for family in FAMILIES}
        for family in FAMILIES:
            global_values[family].append(cmean[family])
        max_c, max_b = max(cmean.values()), max(bmean.values())
        c_winners = [f for f in FAMILIES if np.isclose(cmean[f], max_c, atol=1e-12)]
        b_winners = [f for f in FAMILIES if np.isclose(bmean[f], max_b, atol=1e-12)]
        rows.append({
            "block": block,
            "EE_exclusive_correct": cmean["EEGNet_homogeneous"],
            "LL_exclusive_correct": cmean["LiteBN_homogeneous"],
            "EL_exclusive_correct": cmean["Heterogeneous_orientation_mean"],
            "EE_BA": bmean["EEGNet_homogeneous"],
            "LL_BA": bmean["LiteBN_homogeneous"],
            "EL_BA": bmean["Heterogeneous_orientation_mean"],
            "highest_complementarity_family": ";".join(c_winners),
            "highest_BA_family": ";".join(b_winners),
        })
    return pd.DataFrame(rows), {
        family: float(np.mean(values)) for family, values in global_values.items()
    }


def absolute_markdown(table: pd.DataFrame, single: pd.DataFrame) -> str:
    lines = [
        "# Absolute subject-equal balanced accuracy",
        "",
        "Primary fusion is LOGIT50. All values are percentage points; brackets are 95% subject-bootstrap CIs with 10,000 resamples and fixed seed 20260909.",
        "",
        "| Block | EEGNet+EEGNet BA | LiteBN+LiteBN BA | EEGNet+LiteBN BA | EL minus EE pp | EL minus LL pp | EL minus HOMOAVG pp | n subjects |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def show(mean: float, low: float, high: float) -> str:
        return f"{mean:.3f} [{low:.3f}, {high:.3f}]"
    for _, row in table.iterrows():
        lines.append(
            "| {block} | {ee} | {ll} | {el} | {dee} | {dll} | {dh} | {n} |".format(
                block=str(row.block).replace("_", " "),
                ee=show(row.EE_BA, row.EE_CI_LOW, row.EE_CI_HIGH),
                ll=show(row.LL_BA, row.LL_CI_LOW, row.LL_CI_HIGH),
                el=show(row.EL_BA, row.EL_CI_LOW, row.EL_CI_HIGH),
                dee=show(row.EL_minus_EE_pp, row.EL_minus_EE_CI_LOW, row.EL_minus_EE_CI_HIGH),
                dll=show(row.EL_minus_LL_pp, row.EL_minus_LL_CI_LOW, row.EL_minus_LL_CI_HIGH),
                dh=show(row.EL_minus_HOMOAVG_pp, row.EL_minus_HOMOAVG_CI_LOW, row.EL_minus_HOMOAVG_CI_HIGH),
                n=int(row.n_subjects),
            )
        )
    lines += [
        "",
        "## Optional exactly matched single-model reference",
        "",
        "The values below are reconstructed only from the same frozen homogeneous-pair ledger. Each of the three seeds appears twice per fold, producing 30 matched seed/fold values per subject; no number is copied from a different experiment.",
        "",
        "| Block | EEGNet BA | LiteBN BA | n subjects |",
        "|---|---:|---:|---:|",
    ]
    for _, row in single.iterrows():
        eeg = show(row.EEGNet_BA, row.EEGNet_CI_LOW, row.EEGNet_CI_HIGH)
        lite = show(row.LiteBN_BA, row.LiteBN_CI_LOW, row.LiteBN_CI_HIGH)
        lines.append(f"| {str(row.block).replace('_', ' ')} | {eeg} | {lite} | {int(row.n_subjects)} |")
    return "\n".join(lines) + "\n"


def method_markdown(source_sha: str) -> str:
    return f"""# Method: absolute BA reanalysis

This is a statistics-only reanalysis of frozen source file
experiments/persist_eeg_matched_homogeneous_ensemble_control_v1/subject_level_results.csv
with SHA-256 {source_sha}. No EEGNet or LiteBN model was trained, loaded for
new inference, calibrated, or selected.

## Locked inputs

The four retained blocks are OpenBMI MI, OpenBMI ERP, OpenBMI SSVEP, and WBCIC
MI. The source ledger contains five historical folds, seeds {{0,1,2}}, and all
unordered pairs (0,1), (0,2), (1,2). The primary rule is LOGIT50.

EE is the fixed 50/50 fusion of two EEGNet seeds. LL is the fixed 50/50 fusion
of two LiteBN seeds. EL retains both fixed orientations, EL_A and EL_B, and
uses their arithmetic outcome mean. It is not a four-model ensemble.

## Aggregation and uncertainty

For every block, subject, and primary family, the 15 fixed fold by seed-pair BA
values are first averaged. EL has both orientations equally represented inside
each fixed replicate. The family BA is then the ordinary mean of those
subject-level values across matched held-out subjects. The subject is therefore
the final statistical unit; trials are never pooled and folds/pairs are never
treated as independent observations.

BA and contrasts are percentage points. Family CIs use 10,000 subject
bootstraps. EL minus EE, EL minus LL, and EL minus 0.5 times (EE plus LL) use
paired subject bootstrap. Global equal-block CIs resample subjects independently
within every block and then average the four block means. A separately labeled
dataset-balanced result averages the three OpenBMI task blocks before averaging
OpenBMI and WBCIC 50/50.

No best subject-specific constituent, seed pair, fold, orientation, fusion
weight, calibration, routing, or gating is used. The former post-hoc
gain-over-best-constituent quantity is excluded from the primary table.
"""


def interpretation(
    table: pd.DataFrame,
    global_result: dict[str, Any],
    comp: pd.DataFrame,
    global_comp: dict[str, float],
    prob: pd.DataFrame,
) -> str:
    equal = global_result["equal_block"]
    balanced = global_result["dataset_balanced_optional"]
    ee_blocks = int((table.EL_minus_EE_pp > 0).sum())
    ll_blocks = int((table.EL_minus_LL_pp > 0).sum())
    def show(stat: dict[str, Any]) -> str:
        return f"{stat['mean_pp']:+.3f} pp (95% CI [{stat['ci_low_pp']:+.3f}, {stat['ci_high_pp']:+.3f}])"
    contrast_to_column = {
        "EL_minus_EE": "EL_minus_EE_pp",
        "EL_minus_LL": "EL_minus_LL_pp",
        "EL_minus_HOMOAVG": "EL_minus_HOMOAVG_pp",
    }
    flags: list[str] = []
    for contrast, column in contrast_to_column.items():
        logit = float(equal[contrast]["mean_pp"])
        probability = float(prob[column].mean())
        if np.sign(logit) != np.sign(probability) or abs(logit - probability) > MATERIAL_SHIFT_PP:
            flags.append(contrast)
    el_comp_all = all(
        row.highest_complementarity_family == "Heterogeneous_orientation_mean"
        for _, row in comp.iterrows()
    )
    el_ba_best = int((comp.highest_BA_family == "Heterogeneous_orientation_mean").sum())
    lines = [
        "# Final interpretation",
        "",
        "## Direct answers",
        "",
        f"1. EL is higher than EE by point estimate in {ee_blocks}/4 blocks. Equal-block EL minus EE is {show(equal['EL_minus_EE'])}.",
        f"2. EL is higher than LL by point estimate in {ll_blocks}/4 blocks. Equal-block EL minus LL is {show(equal['EL_minus_LL'])}.",
        f"3. EL exceeds EE in {ee_blocks}/4 blocks and LL in {ll_blocks}/4 blocks under identical subject-equal aggregation.",
        f"4. Global equal-block EL minus EE is {show(equal['EL_minus_EE'])}.",
        f"5. Global equal-block EL minus LL is {show(equal['EL_minus_LL'])}.",
        f"6. Global equal-block EL minus HOMOAVG is {show(equal['EL_minus_HOMOAVG'])}.",
        "",
        "## Complementarity versus utility",
        "",
        f"EL is the highest descriptive exclusive-correct family in every block: {el_comp_all}. Equal-block exclusive-correct percentages are EE={global_comp['EEGNet_homogeneous']:.3f}, LL={global_comp['LiteBN_homogeneous']:.3f}, EL={global_comp['Heterogeneous_orientation_mean']:.3f}.",
        f"EL is the highest absolute-BA family in {el_ba_best}/4 blocks. Thus descriptive complementarity does not consistently translate into the highest absolute BA against both homogeneous controls.",
        "",
        "## Global weighting and fixed-rule robustness",
        "",
    ]
    for contrast in ("EL_minus_EE", "EL_minus_LL", "EL_minus_HOMOAVG"):
        lines.append(f"- Dataset-balanced {contrast.replace('_', ' ')}: {show(balanced[contrast])}.")
    if flags:
        lines.append(
            f"- PROB50 differs materially for {', '.join(flags)} under a predeclared global check: sign change or shift over {MATERIAL_SHIFT_PP:.1f} pp. See PROB50_ROBUSTNESS.csv."
        )
    else:
        lines.append(
            f"- PROB50 does not materially change the global conclusion: no contrast sign change and no shift over {MATERIAL_SHIFT_PP:.1f} pp. See PROB50_ROBUSTNESS.csv."
        )
    lines += [
        "",
        "## Claim boundary",
        "",
        "Heterogeneous-specific superiority is not supported unless EL is convincingly superior to both EE and LL in this ordinary matched absolute-BA comparison. The EL minus LL result, rather than a post-hoc best-constituent oracle quantity, is decisive for that boundary.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    validation = validate(frame)
    source_sha = sha256(INPUT)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    subject = aggregate_subject_family(frame, "LOGIT50")
    table, deltas = absolute_table(subject, rng)
    single_subject, single_table = single_model_reference(frame, rng)
    global_result = global_summary(deltas, rng)
    comp, global_comp = complementarity(subject, frame)

    prob_subject = aggregate_subject_family(frame, "PROB50")
    prob, _ = absolute_table(prob_subject, np.random.default_rng(BOOTSTRAP_SEED + 1))

    validation.update({
        "analysis": "statistics_only_reaggregation",
        "source_results_path": str(INPUT),
        "source_results_sha256": source_sha,
        "source_rows": int(len(frame)),
        "primary_fusion_rule": "LOGIT50",
        "robustness_only_rule": "PROB50",
        "bootstrap_unit": "subject",
        "bootstrap_replicates": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "no_model_training": True,
        "no_new_inference": True,
        "no_oracle_comparison_in_primary": True,
        "subject_family_BA_unit": "percentage_points",
    })

    write_csv(OUT / "SUBJECT_LEVEL_FAMILY_BA.csv", subject)
    write_csv(OUT / "MATCHED_SINGLE_MODEL_SUBJECT_BA.csv", single_subject)
    write_csv(OUT / "MATCHED_SINGLE_MODEL_REFERENCE.csv", single_table)
    write_csv(OUT / "ABSOLUTE_BA_TABLE.csv", table)
    write_csv(OUT / "COMPLEMENTARITY_VS_UTILITY.csv", comp)
    write_csv(OUT / "PROB50_ROBUSTNESS.csv", prob)
    write_json(OUT / "GLOBAL_SUMMARY.json", global_result)
    write_json(OUT / "VALIDATION.json", validation)
    write_text(OUT / "ABSOLUTE_BA_TABLE.md", absolute_markdown(table, single_table))
    write_text(OUT / "METHOD.md", method_markdown(source_sha))
    write_text(OUT / "FINAL_INTERPRETATION.md", interpretation(table, global_result, comp, global_comp, prob))
    print("HOMOGENEOUS_ABSOLUTE_BA_REANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
