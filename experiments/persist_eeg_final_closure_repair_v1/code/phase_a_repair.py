"""Repair Phase A exclusively from frozen checkpoints and evaluation artifacts.

No model is trained, adapted, or modified by this script.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import (
    EXP,
    FIGURES,
    FINAL,
    PREVIOUS,
    RESULTS,
    SOURCE,
    append_engineering_log,
    audit_frozen_tables,
    certificate_coordinate_map,
    ensure_dirs,
    expected_subjects,
    historical_run_dir,
    protocol,
    read_json,
    sha256_file,
    subject_bootstrap,
    subject_bootstrap_corr,
    write_csv,
    write_json,
    write_md,
)


def frozen_model_hashes() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        for seed in range(3):
            run = historical_run_dir(fold, seed)
            lock = read_json(run / "RUN_LOCK.json")
            teacher = Path(str(lock["checkpoint_hashes"]["B1_STRONG_EEGNET"]["path"]))
            certificate = run / "certificate" / "PUD_CERTIFICATE.npz"
            cert_csv = run / "certificate" / "PUD_CERTIFICATION.csv"
            for kind, path in (("B1_STRONG_EEGNET", teacher), ("PUD_CERTIFICATE", certificate), ("PUD_CERTIFICATION_TABLE", cert_csv)):
                if not path.is_file():
                    raise RuntimeError(f"frozen Phase-A input is missing: {path}")
                rows.append({"fold": fold, "seed": seed, "kind": kind, "path": str(path), "sha256_before": sha256_file(path)})
    return pd.DataFrame(rows)


def verify_hashes_unchanged(before: pd.DataFrame) -> pd.DataFrame:
    after = before.copy()
    after["sha256_after"] = [sha256_file(Path(path)) for path in after.path]
    after["unchanged"] = after.sha256_before == after.sha256_after
    if not bool(after.unchanged.all()):
        raise RuntimeError("Phase A modified a frozen checkpoint/certificate")
    write_csv(RESULTS / "phase_a_frozen_hash_audit.csv", after)
    return after


def build_consequence_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    mechanism = pd.read_csv(SOURCE / "results" / "mechanism_raw.csv")
    raw = pd.read_csv(SOURCE / "results" / "source_only_raw.csv")
    replay = pd.read_csv(SOURCE / "results" / "replay_per_subject.csv")
    reliance = pd.read_csv(PREVIOUS / "results" / "reliance_metrics.csv")
    for frame in (mechanism, raw, replay, reliance):
        frame["subject_id"] = frame.subject_id.astype(str)

    keys = ["fold", "subject_id"]
    pud_mechanism = (
        mechanism[mechanism.method.eq("PUD_SOURCE_ONLY")]
        .groupby(keys, as_index=False)
        .agg(
            protected_branch_erasure_harm_BA=("protected_branch_erasure_harm_BA", "mean"),
            adaptive_branch_erasure_harm_BA=("adaptive_branch_erasure_harm_BA", "mean"),
            protected_D_finite=("protected_D_finite", "mean"),
            adaptive_D_finite=("adaptive_D_finite", "mean"),
            functional_teacher_correlation=("functional_teacher_correlation", "mean"),
            functional_teacher_RMSE=("functional_teacher_RMSE", "mean"),
            mechanism_seed_rows=("seed", "nunique"),
        )
    )
    pud_reliance = (
        reliance[reliance.method.eq("PUD_SOURCE_ONLY")]
        .groupby(keys, as_index=False)
        .agg(R_P=("R_P", "mean"), reliance_seed_rows=("seed", "nunique"))
    )

    def performance(frame: pd.DataFrame, method: str, name: str) -> pd.DataFrame:
        selected = frame[frame.method.eq(method)]
        return selected.groupby(keys, as_index=False).agg(**{name: ("BA", "mean"), f"{name}_seed_rows": ("seed", "nunique")})

    vanilla = performance(replay, "B0_VANILLA_EEGNET", "vanilla_BA")
    dual = performance(raw, "A2_SOURCE_ONLY", "dual_BA")
    pud = performance(raw, "PUD_SOURCE_ONLY", "pud_source_BA")
    adapted = performance(raw, "PUD_AFTER_ADAPT", "pud_adapted_BA")
    table = pud_mechanism.merge(pud_reliance, on=keys, validate="one_to_one")
    for frame in (vanilla, dual, pud, adapted):
        table = table.merge(frame, on=keys, validate="one_to_one")
    if len(table) != 40 or set(table.subject_id) != set(expected_subjects()):
        raise RuntimeError(f"repaired consequence table must contain exactly 40 subjects; got {len(table)}")
    seed_columns = [column for column in table if column.endswith("_seed_rows")]
    if not bool((table[seed_columns] == 3).all().all()):
        raise RuntimeError("Phase-A seed aggregation is incomplete")
    table["PUD_minus_Vanilla"] = table.pud_source_BA - table.vanilla_BA
    table["PUD_minus_Dual"] = table.pud_source_BA - table.dual_BA
    table["adaptation_gain"] = table.pud_adapted_BA - table.pud_source_BA
    table = table.sort_values(["fold", "subject_id"]).reset_index(drop=True)
    write_csv(RESULTS / "consequence_generalization_repaired.csv", table)

    comparisons = [
        ("protected_branch_erasure_harm_BA", "PUD_minus_Vanilla"),
        ("protected_D_finite", "PUD_minus_Vanilla"),
        ("functional_teacher_correlation", "PUD_minus_Vanilla"),
        ("R_P", "PUD_minus_Vanilla"),
        ("adaptive_branch_erasure_harm_BA", "PUD_minus_Vanilla"),
        ("protected_branch_erasure_harm_BA", "adaptation_gain"),
    ]
    rows = []
    for offset, (predictor, outcome) in enumerate(comparisons):
        rows.append({"predictor": predictor, "outcome": outcome, **subject_bootstrap_corr(table[predictor], table[outcome], seed=9173 + offset)})
    correlations = pd.DataFrame(rows)
    write_csv(RESULTS / "consequence_generalization_correlations.csv", correlations)

    report = [
        "The B0 join bug is fixed by reading the authoritative frozen `replay_per_subject.csv`. Three seeds are averaged inside each of the 40 subjects before correlation.",
        "",
        "| predictor → outcome | n | Pearson [95% CI] | Spearman [95% CI] | interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    for row in correlations.to_dict("records"):
        report.append(
            f"| {row['predictor']} → {row['outcome']} | {row['n']} | {row['pearson']:.3f} [{row['pearson_ci95_l']:.3f}, {row['pearson_ci95_u']:.3f}] | "
            f"{row['spearman']:.3f} [{row['spearman_ci95_l']:.3f}, {row['spearman_ci95_u']:.3f}] | {row['label']} |"
        )
    report += ["", "A bootstrap interval crossing zero is reported as uncertainty, not as proof of no relationship or statistical independence."]
    write_md(EXP / "CONSEQUENCE_GENERALIZATION_REPAIRED.md", "Consequence to generalization — repaired", "\n".join(report))
    return table, correlations


def build_reliance_gate(consequence: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    reliance = pd.read_csv(PREVIOUS / "results" / "reliance_metrics.csv")
    mechanism = pd.read_csv(SOURCE / "results" / "mechanism_raw.csv")
    reliance.subject_id = reliance.subject_id.astype(str)
    mechanism.subject_id = mechanism.subject_id.astype(str)
    keys = ["fold", "subject_id"]
    methods = ["PUD_SOURCE_ONLY", "A2_SOURCE_ONLY", "RANDOM_SOURCE_ONLY", "IDENTITY_SOURCE_ONLY"]
    compact = reliance[reliance.method.isin(methods)].groupby(["method", *keys], as_index=False).agg(R_P=("R_P", "mean"))
    harm = mechanism[mechanism.method.isin(methods)].groupby(["method", *keys], as_index=False).agg(protected_harm=("protected_branch_erasure_harm_BA", "mean"))
    compact = compact.merge(harm, on=["method", *keys], validate="one_to_one")
    wide = compact.pivot(index=keys, columns="method", values=["R_P", "protected_harm"])
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    wide = wide.reset_index().merge(consequence[[*keys, "PUD_minus_Vanilla"]], on=keys, validate="one_to_one")
    if len(wide) != 40:
        raise RuntimeError("reliance gate does not contain 40 independent subjects")
    wide["delta_R_P_PUD_minus_Dual"] = wide["R_P__PUD_SOURCE_ONLY"] - wide["R_P__A2_SOURCE_ONLY"]
    for control in ("A2_SOURCE_ONLY", "RANDOM_SOURCE_ONLY", "IDENTITY_SOURCE_ONLY"):
        wide[f"delta_harm_PUD_minus_{control}"] = wide["protected_harm__PUD_SOURCE_ONLY"] - wide[f"protected_harm__{control}"]
    write_csv(RESULTS / "reliance_repaired.csv", wide)

    b1 = subject_bootstrap(wide.delta_R_P_PUD_minus_Dual, seed=10001)
    b1_support = bool(b1["mean"] > 0 and b1["ci95_l"] > 0)
    b2_stats = {
        control: subject_bootstrap(wide[f"delta_harm_PUD_minus_{control}"], seed=11000 + offset)
        for offset, control in enumerate(("A2_SOURCE_ONLY", "RANDOM_SOURCE_ONLY", "IDENTITY_SOURCE_ONLY"))
    }
    b2_support = bool(
        b2_stats["A2_SOURCE_ONLY"]["mean"] > 0
        and b2_stats["A2_SOURCE_ONLY"]["ci95_l"] > 0
        and b2_stats["RANDOM_SOURCE_ONLY"]["mean"] > 0
        and b2_stats["IDENTITY_SOURCE_ONLY"]["mean"] > 0
    )
    b3 = subject_bootstrap_corr(wide["R_P__PUD_SOURCE_ONLY"], wide.PUD_minus_Vanilla, seed=12001)
    if b3["spearman"] < 0 and b3["spearman_ci95_u"] < 0:
        b3_status = "B3_STRONG_SUPPORT"
    elif b3["spearman"] < 0:
        b3_status = "B3_PARTIAL"
    else:
        b3_status = "B3_NOT_SUPPORT"
    if b1_support and b2_support and b3_status == "B3_STRONG_SUPPORT":
        h3 = "HARD_FACTORIZATION_BRITTLE_BOTTLENECK_STRONG"
    elif b1_support and b2_support and b3_status == "B3_PARTIAL":
        h3 = "HARD_FACTORIZATION_CONCENTRATES_RELIANCE_BUT_HARM_LINK_UNCERTAIN"
    else:
        h3 = "HARD_FACTORIZATION_BRITTLE_BOTTLENECK_NOT_SUPPORTED"
    payload = {
        "B1": {"support": b1_support, **b1},
        "B2": {"support": b2_support, "comparisons": b2_stats},
        "B3": {"status": b3_status, **b3},
        "H3": h3,
    }
    write_json(RESULTS / "reliance_gate_repaired.json", payload)
    b2_lines = []
    for control, values in b2_stats.items():
        b2_lines.append(f"- PUD minus {control} erasure harm: {values['mean']:.4f} [{values['ci95_l']:.4f}, {values['ci95_u']:.4f}]")
    body = [
        f"B1: **{'SUPPORT' if b1_support else 'NOT SUPPORT'}**. Paired R_P(PUD)−R_P(Dual) = {b1['mean']:.4f} [{b1['ci95_l']:.4f}, {b1['ci95_u']:.4f}].",
        "",
        f"B2: **{'SUPPORT' if b2_support else 'NOT SUPPORT'}**.",
        *b2_lines,
        "",
        f"B3: **{b3_status}**. R_P → PUD−Vanilla Pearson {b3['pearson']:.3f} [{b3['pearson_ci95_l']:.3f}, {b3['pearson_ci95_u']:.3f}], Spearman {b3['spearman']:.3f} [{b3['spearman_ci95_l']:.3f}, {b3['spearman_ci95_u']:.3f}].",
        "",
        f"Final H3: **{h3}**.",
        "",
        "Reliance concentration is not described as an explanation of generalization loss unless B3 supplies the preregistered signed subject-level relationship.",
    ]
    write_md(EXP / "BRITTLE_RELIANCE_REPAIRED.md", "Hard factorization and brittle reliance — repaired", "\n".join(body))
    return wide, payload


def _hierarchical_bootstrap(
    subject_rows: pd.DataFrame,
    outcome: str,
    source: str | None = None,
    draws: int = 5000,
    seed: int = 24491,
) -> dict[str, Any]:
    direction_rows = subject_rows.groupby(["fold", "seed", "certificate_direction"], as_index=False).agg(
        **({outcome: (outcome, "mean")} | ({source: (source, "first")} if source else {}))
    )
    point_mean = float(direction_rows.groupby("fold")[outcome].mean().mean())
    if source:
        point_pearson = float(np.corrcoef(direction_rows[source], direction_rows[outcome])[0, 1])
        point_spearman = float(stats.spearmanr(direction_rows[source], direction_rows[outcome]).statistic)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    pearsons = np.empty(draws, dtype=np.float64) if source else None
    spearmans = np.empty(draws, dtype=np.float64) if source else None
    folds = sorted(subject_rows.fold.unique())
    for draw in range(draws):
        fold_values: list[float] = []
        units_x: list[float] = []
        units_y: list[float] = []
        for _ in folds:
            fold = int(rng.choice(folds))
            fold_frame = subject_rows[subject_rows.fold.eq(fold)]
            seeds = sorted(fold_frame.seed.unique())
            run_values: list[float] = []
            for _ in seeds:
                run_seed = int(rng.choice(seeds))
                run_frame = fold_frame[fold_frame.seed.eq(run_seed)]
                directions = sorted(run_frame.certificate_direction.unique())
                direction_values: list[float] = []
                for _ in directions:
                    direction = int(rng.choice(directions))
                    cell = run_frame[run_frame.certificate_direction.eq(direction)]
                    sampled = cell.iloc[rng.integers(0, len(cell), size=len(cell))]
                    value_y = float(sampled[outcome].mean())
                    direction_values.append(value_y)
                    if source:
                        units_x.append(float(cell[source].iloc[0]))
                        units_y.append(value_y)
                run_values.append(float(np.mean(direction_values)))
            fold_values.append(float(np.mean(run_values)))
        samples[draw] = float(np.mean(fold_values))
        if source:
            if len(set(units_x)) < 2 or len(set(units_y)) < 2:
                pearsons[draw] = math.nan
                spearmans[draw] = math.nan
            else:
                pearsons[draw] = float(np.corrcoef(units_x, units_y)[0, 1])
                spearmans[draw] = float(stats.spearmanr(units_x, units_y).statistic)
    result: dict[str, Any] = {
        "n_folds": 5,
        "n_runs": int(direction_rows[["fold", "seed"]].drop_duplicates().shape[0]),
        "n_run_directions": int(len(direction_rows)),
        "mean": point_mean,
        "mean_ci95_l": float(np.quantile(samples, 0.025)),
        "mean_ci95_u": float(np.quantile(samples, 0.975)),
    }
    if source:
        pearsons = pearsons[np.isfinite(pearsons)]
        spearmans = spearmans[np.isfinite(spearmans)]
        result.update({
            "pearson": point_pearson,
            "spearman": point_spearman,
            "pearson_ci95_l": float(np.quantile(pearsons, 0.025)),
            "pearson_ci95_u": float(np.quantile(pearsons, 0.975)),
            "spearman_ci95_l": float(np.quantile(spearmans, 0.025)),
            "spearman_ci95_u": float(np.quantile(spearmans, 0.975)),
        })
    return result


def repair_certificate_transfer() -> tuple[pd.DataFrame, dict[str, Any]]:
    transfer = pd.read_csv(PREVIOUS / "results" / "certificate_transfer.csv")
    transfer.subject_id = transfer.subject_id.astype(str)
    repaired_frames = []
    mismatch_count = 0
    for (fold, seed), group in transfer.groupby(["fold", "seed"], sort=True):
        mapping = certificate_coordinate_map(int(fold), int(seed))
        certificate = pd.read_csv(historical_run_dir(int(fold), int(seed)) / "certificate" / "PUD_CERTIFICATION.csv").set_index("direction")
        current = group.copy()
        current["basis_column"] = current.direction.astype(int)
        current["certificate_direction"] = current.basis_column.map(mapping)
        if current.certificate_direction.isna().any():
            raise RuntimeError(f"missing PUD coordinate mapping for fold={fold} seed={seed}")
        current.certificate_direction = current.certificate_direction.astype(int)
        for output, source_column in (
            ("source_rho_repaired", "rho"),
            ("source_P_repaired", "persistence_correlation"),
            ("source_U_repaired", "utility_specific_mean"),
            ("source_D_finite_repaired", "D_finite"),
            ("source_PUD_pass_repaired", "PUD_pass"),
        ):
            current[output] = current.certificate_direction.map(certificate[source_column])
        mismatch_count += int((~np.isclose(current.source_U, current.source_U_repaired, equal_nan=True)).sum())
        repaired_frames.append(current)
    repaired = pd.concat(repaired_frames, ignore_index=True)
    if len(repaired[["fold", "seed"]].drop_duplicates()) != 15:
        raise RuntimeError("certificate repair is missing fold/seed runs")
    if not bool(repaired.source_PUD_pass_repaired.astype(bool).all()):
        raise RuntimeError("repaired transfer contains a direction outside the frozen PUD basis")
    if set(repaired.subject_id) != set(expected_subjects()):
        raise RuntimeError("certificate transfer includes restricted or missing subjects")
    repaired = repaired.rename(columns={
        "outcome_ba_harm": "future_BA_erasure_harm",
        "outcome_ce_harm": "future_CE_erasure_harm",
        "outcome_D_finite": "future_D_finite",
    })
    write_csv(RESULTS / "certificate_transfer_repaired.csv", repaired)

    consequence = {
        metric: _hierarchical_bootstrap(repaired, metric, seed=24491 + offset)
        for offset, metric in enumerate(("future_BA_erasure_harm", "future_CE_erasure_harm", "future_D_finite"))
    }
    ba = consequence["future_BA_erasure_harm"]
    ce = consequence["future_CE_erasure_harm"]
    if ba["mean"] > 0 and ce["mean"] > 0 and ba["mean_ci95_l"] > 0 and ce["mean_ci95_l"] > 0:
        consequence_status = "DIRECTION_CONSEQUENCE_TRANSFER_SUPPORTED"
    elif ba["mean"] > 0 and ce["mean"] > 0:
        consequence_status = "DIRECTION_CONSEQUENCE_TRANSFER_WEAK"
    else:
        consequence_status = "DIRECTION_CONSEQUENCE_TRANSFER_NOT_SUPPORTED"

    mappings = [
        ("source_U_repaired", "future_CE_erasure_harm"),
        ("source_D_finite_repaired", "future_D_finite"),
        ("source_P_repaired", "future_BA_erasure_harm"),
    ]
    score_rows = []
    mapping_statuses = []
    for offset, (source, outcome) in enumerate(mappings):
        values = _hierarchical_bootstrap(repaired, outcome, source=source, seed=25001 + offset)
        if values["pearson"] > 0 and values["spearman"] > 0 and values["pearson_ci95_l"] > 0 and values["spearman_ci95_l"] > 0:
            label = "POSITIVE"
        elif values["pearson"] < 0 and values["spearman"] < 0 and values["pearson_ci95_u"] < 0 and values["spearman_ci95_u"] < 0:
            label = "NEGATIVE"
        elif (values["pearson"] > 0 and values["spearman"] > 0) or (values["pearson"] < 0 and values["spearman"] < 0):
            label = "UNCERTAIN"
        else:
            label = "NOT_ESTABLISHED"
        mapping_statuses.append(label)
        score_rows.append({"source": source, "future": outcome, "mapping_status": label, **values})
    all_positive_direction = all(row["pearson"] > 0 and row["spearman"] > 0 for row in score_rows)
    all_negative_direction = all(row["pearson"] < 0 and row["spearman"] < 0 for row in score_rows)
    if all(status == "POSITIVE" for status in mapping_statuses):
        score_status = "CERTIFICATE_SCORE_TRANSFER_POSITIVE"
    elif all(status == "NEGATIVE" for status in mapping_statuses):
        score_status = "CERTIFICATE_SCORE_TRANSFER_NEGATIVE"
    elif all_positive_direction or all_negative_direction:
        score_status = "CERTIFICATE_SCORE_TRANSFER_UNCERTAIN"
    else:
        score_status = "CERTIFICATE_SCORE_TRANSFER_NOT_ESTABLISHED"

    run_rows = []
    for (fold, seed), group in repaired.groupby(["fold", "seed"]):
        direction = group.groupby("certificate_direction", as_index=False).agg(
            source_U=("source_U_repaired", "first"),
            source_D=("source_D_finite_repaired", "first"),
            source_P=("source_P_repaired", "first"),
            future_CE=("future_CE_erasure_harm", "mean"),
            future_D=("future_D_finite", "mean"),
            future_BA=("future_BA_erasure_harm", "mean"),
        )
        for source, future in (("source_U", "future_CE"), ("source_D", "future_D"), ("source_P", "future_BA")):
            p = float(direction[source].corr(direction[future], method="pearson")) if len(direction) >= 3 else math.nan
            s = float(direction[source].corr(direction[future], method="spearman")) if len(direction) >= 3 else math.nan
            run_rows.append({"fold": fold, "seed": seed, "mapping": f"{source}->{future}", "n_directions": len(direction), "pearson": p, "spearman": s})
    run_level = pd.DataFrame(run_rows)
    write_csv(RESULTS / "certificate_transfer_run_level.csv", run_level)
    score_frame = pd.DataFrame(score_rows)
    write_csv(RESULTS / "certificate_transfer_score_inference.csv", score_frame)
    fold_signs = repaired.groupby("fold", as_index=False).agg(
        future_BA_erasure_harm=("future_BA_erasure_harm", "mean"),
        future_CE_erasure_harm=("future_CE_erasure_harm", "mean"),
        future_D_finite=("future_D_finite", "mean"),
    )
    write_csv(RESULTS / "certificate_transfer_fold_signs.csv", fold_signs)
    payload = {
        "source_score_join_rows_corrected": mismatch_count,
        "consequence": consequence,
        "consequence_status": consequence_status,
        "score_mappings": score_rows,
        "score_status": score_status,
        "hierarchy": ["outer_fold", "seed_run", "direction", "outcome_subject"],
    }
    write_json(RESULTS / "certificate_transfer_repaired_inference.json", payload)
    body = [
        f"The earlier join treated PUD basis column `j` as certificate direction `j`. The repaired join reads `basis_PUD`, maps each selected basis column to its actual certified coordinate, and corrected {mismatch_count} repeated direction-subject source-score cells.",
        "",
        f"Direction consequence: **{consequence_status}**.",
        "",
        "| future consequence | mean | hierarchical 95% CI |",
        "|---|---:|---:|",
    ]
    for metric, values in consequence.items():
        body.append(f"| {metric} | {values['mean']:.4f} | [{values['mean_ci95_l']:.4f}, {values['mean_ci95_u']:.4f}] |")
    body += [
        "",
        f"Certificate-score transfer: **{score_status}**.",
        "",
        "| source → future | Pearson [95% CI] | Spearman [95% CI] | mapping status |",
        "|---|---:|---:|---|",
    ]
    for row in score_rows:
        body.append(
            f"| {row['source']} → {row['future']} | {row['pearson']:.3f} [{row['pearson_ci95_l']:.3f}, {row['pearson_ci95_u']:.3f}] | "
            f"{row['spearman']:.3f} [{row['spearman_ci95_l']:.3f}, {row['spearman_ci95_u']:.3f}] | {row['mapping_status']} |"
        )
    body += ["", "Mean future consequence and predictive transfer of source certificate scores are separate hypotheses. The former cannot be used to label the latter as supported."]
    write_md(EXP / "CERTIFICATE_TRANSFER_REPAIRED.md", "Certificate transfer — repaired hierarchical inference", "\n".join(body))
    return repaired, payload


def repair_functional_persistence() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(PREVIOUS / "results" / "functional_persistence.csv")
    raw.subject_id = raw.subject_id.astype(str)
    mapped = []
    for (fold, seed), group in raw.groupby(["fold", "seed"], sort=True):
        current = group.copy()
        current["basis_column"] = current.direction.astype(int)
        current["certificate_direction"] = current.basis_column.map(certificate_coordinate_map(int(fold), int(seed))).astype(int)
        mapped.append(current)
    raw = pd.concat(mapped, ignore_index=True)
    metrics = ["margin_mean", "margin_rms", "class0_margin_mean", "class1_margin_mean", "ba_harm", "ce_harm"]
    keys = ["fold", "seed", "subject_id", "basis", "basis_column", "certificate_direction"]
    s1 = raw[raw.session.eq(1)][keys + metrics].rename(columns={metric: f"{metric}_S1" for metric in metrics})
    s2 = raw[raw.session.eq(2)][keys + metrics].rename(columns={metric: f"{metric}_S2" for metric in metrics})
    paired = s1.merge(s2, on=keys, validate="one_to_one")
    if not len(paired) or len(paired[["fold", "seed"]].drop_duplicates()) != 15:
        raise RuntimeError("functional persistence pairing is incomplete")

    def cosine(a: pd.Series, b: pd.Series) -> float:
        frame = pd.concat([a, b], axis=1).dropna().to_numpy(dtype=float)
        if not len(frame):
            return math.nan
        denominator = np.linalg.norm(frame[:, 0]) * np.linalg.norm(frame[:, 1])
        return float(np.dot(frame[:, 0], frame[:, 1]) / denominator) if denominator > 0 else math.nan

    def summary(prefix: str) -> dict[str, Any]:
        a = paired[f"{prefix}_S1"]
        b = paired[f"{prefix}_S2"]
        valid = pd.concat([a, b], axis=1).dropna()
        return {
            "n": len(valid),
            "pearson": float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson")),
            "spearman": float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman")),
            "cosine": cosine(valid.iloc[:, 0], valid.iloc[:, 1]),
            "rms_change": float(np.sqrt(np.mean(np.square(valid.iloc[:, 1] - valid.iloc[:, 0])))),
            "sign_consistency": float((np.sign(valid.iloc[:, 0]) == np.sign(valid.iloc[:, 1])).mean()),
        }

    payload = {
        "scope": "RETROSPECTIVE / NON-DEPLOYABLE DIAGNOSTIC",
        "margin_contribution": summary("margin_mean"),
        "class0_margin_contribution": summary("class0_margin_mean"),
        "class1_margin_contribution": summary("class1_margin_mean"),
        "representation_claim": "not evaluated as physiology; only frozen task-function contribution is audited",
    }
    write_csv(RESULTS / "functional_persistence_repaired.csv", paired)
    write_json(RESULTS / "functional_persistence_repaired_summary.json", payload)
    main = payload["margin_contribution"]
    c0 = payload["class0_margin_contribution"]
    c1 = payload["class1_margin_contribution"]
    body = [
        "**RETROSPECTIVE / NON-DEPLOYABLE DIAGNOSTIC.** Session-1 outcome labels are used only in this analysis.",
        "",
        f"Overall PUD decision contribution: Pearson {main['pearson']:.3f}, Spearman {main['spearman']:.3f}, cosine {main['cosine']:.3f}, RMS change {main['rms_change']:.4f}, sign consistency {main['sign_consistency']:.3f}.",
        "",
        f"Class 0: Pearson {c0['pearson']:.3f}, cosine {c0['cosine']:.3f}, RMS change {c0['rms_change']:.4f}, sign consistency {c0['sign_consistency']:.3f}.",
        f"Class 1: Pearson {c1['pearson']:.3f}, cosine {c1['cosine']:.3f}, RMS change {c1['rms_change']:.4f}, sign consistency {c1['sign_consistency']:.3f}.",
        "",
        "These statistics address stability of task function, not physiological or biological persistence.",
    ]
    write_md(EXP / "FUNCTIONAL_PERSISTENCE_REPAIRED.md", "Functional persistence — repaired", "\n".join(body))
    return paired, payload


def retain_secondary_diagnostics() -> dict[str, Any]:
    gradient = pd.read_csv(PREVIOUS / "results" / "gradient_conflict.csv")
    gradient["diagnostic_only"] = True
    if "state_unchanged_after" not in gradient or not bool(gradient.state_unchanged_after.astype(bool).all()):
        raise RuntimeError("frozen gradient audit did not preserve model state")
    write_csv(RESULTS / "gradient_conflict_repaired.csv", gradient)
    calibration = pd.read_csv(PREVIOUS / "results" / "calibration_metrics.csv")
    write_csv(RESULTS / "calibration_repaired.csv", calibration)
    grad_summary = {
        column: {"mean": float(gradient[column].mean()), "median": float(gradient[column].median())}
        for column in ("cos_task_vs_protected", "cos_task_vs_residual", "cos_task_vs_persistence")
    }
    cal_summary = calibration.groupby("method").agg(NLL=("NLL", "mean"), Brier=("Brier", "mean"), ECE=("ECE", "mean"), margin_std=("margin_std", "mean")).reset_index()
    write_json(RESULTS / "gradient_conflict_repaired_summary.json", grad_summary)
    write_csv(RESULTS / "calibration_repaired_summary.csv", cal_summary)
    write_md(
        EXP / "GRADIENT_DIAGNOSTIC.md",
        "Gradient diagnostic",
        "The frozen no-step gradient table is retained as a diagnostic only. Every row verifies unchanged model state. It is not treated as a training-time causal explanation.\n\n"
        + "\n".join(f"- {name}: mean {values['mean']:.4f}, median {values['median']:.4f}" for name, values in grad_summary.items()),
    )
    write_md(
        EXP / "CALIBRATION_DIAGNOSTIC.md",
        "Calibration and margin diagnostic",
        "Frozen outcome calibration and margin measurements are retained without redefining performance or selecting a model. They are secondary diagnostics; the main conclusion remains balanced-accuracy generalization.\n\n"
        + "```text\n" + cal_summary.to_csv(index=False, float_format="%.4f").strip() + "\n```",
    )
    return {"gradient": grad_summary, "calibration_rows": cal_summary.to_dict("records")}


def make_phase_a_figures(consequence: pd.DataFrame, reliance: pd.DataFrame, transfer: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

    def scatter(frame: pd.DataFrame, x: str, y: str, title: str, path: str) -> None:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.scatter(frame[x], frame[y], s=25, alpha=0.8, edgecolor="none")
        if frame[x].nunique() > 1:
            slope, intercept = np.polyfit(frame[x], frame[y], 1)
            grid = np.linspace(frame[x].min(), frame[x].max(), 100)
            ax.plot(grid, slope * grid + intercept, color="#b22222", lw=1.5)
        ax.axhline(0, color="0.5", lw=0.8, ls="--")
        ax.set(xlabel=x, ylabel=y, title=title)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{path}.png", dpi=220)
        fig.savefig(FIGURES / f"{path}.pdf")
        plt.close(fig)

    scatter(consequence, "protected_branch_erasure_harm_BA", "PUD_minus_Vanilla", "Protected consequence vs future generalization", "consequence_vs_generalization_repaired")
    scatter(reliance, "R_P__PUD_SOURCE_ONLY", "PUD_minus_Vanilla", "Protected reliance vs future generalization", "reliance_vs_generalization_repaired")

    direction = transfer.groupby(["fold", "seed", "certificate_direction"], as_index=False).agg(
        source_U=("source_U_repaired", "first"),
        source_D=("source_D_finite_repaired", "first"),
        source_P=("source_P_repaired", "first"),
        future_CE=("future_CE_erasure_harm", "mean"),
        future_D=("future_D_finite", "mean"),
        future_BA=("future_BA_erasure_harm", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    for ax, (x, y, title) in zip(axes, (("source_U", "future_CE", "U → future CE harm"), ("source_D", "future_D", "D → future D"), ("source_P", "future_BA", "P → future BA harm"))):
        ax.scatter(direction[x], direction[y], s=16, alpha=0.7, edgecolor="none")
        ax.axhline(0, color="0.5", lw=0.7, ls="--")
        ax.set(xlabel=x, ylabel=y, title=title)
    fig.tight_layout()
    fig.savefig(FIGURES / "certificate_transfer_repaired.png", dpi=220)
    fig.savefig(FIGURES / "certificate_transfer_repaired.pdf")
    plt.close(fig)


def write_phase_a_reports(
    audit: dict[str, Any],
    hashes: pd.DataFrame,
    correlations: pd.DataFrame,
    reliance: dict[str, Any],
    transfer: dict[str, Any],
    functional: dict[str, Any],
) -> dict[str, Any]:
    primary = correlations[(correlations.predictor == "protected_branch_erasure_harm_BA") & (correlations.outcome == "PUD_minus_Vanilla")].iloc[0].to_dict()
    dfinite = correlations[(correlations.predictor == "protected_D_finite") & (correlations.outcome == "PUD_minus_Vanilla")].iloc[0].to_dict()
    summary = {
        "phase": "A_REPAIRED_COMPLETE",
        "B0_join_bug_fixed": True,
        "frozen_hashes_unchanged": bool(hashes.unchanged.all()),
        "protected_harm_to_generalization": primary,
        "D_finite_to_generalization": dfinite,
        "R_P_to_generalization": reliance["B3"],
        "H3": reliance["H3"],
        "direction_consequence_transfer": transfer["consequence_status"],
        "certificate_score_transfer": transfer["score_status"],
        "functional_persistence": functional,
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }
    write_json(RESULTS / "phase_a_repaired_summary.json", summary)
    provenance_lines = [
        f"Repository base SHA: `3b519a138fe5074858717c70b611926fd3708f75`.",
        "",
        "All Phase-A inputs are frozen historical artifacts. No optimizer step, adaptation, checkpoint write, or certificate rebuild was performed.",
        "",
        "| input | SHA-256 | rows |",
        "|---|---|---:|",
    ]
    for name, digest in audit["hashes"].items():
        provenance_lines.append(f"| {name} | `{digest}` | {audit['rows'].get(name, '')} |")
    provenance_lines += ["", f"Frozen model/certificate files hashed before and after: {len(hashes)}; unchanged: {int(hashes.unchanged.sum())}/{len(hashes)}."]
    write_md(EXP / "FROZEN_INPUT_PROVENANCE.md", "Frozen input provenance", "\n".join(provenance_lines))
    write_md(
        EXP / "HOLDOUT_PURITY_AUDIT.md",
        "Holdout purity audit",
        "**PASS.** Every observed subject belongs to the exact frozen 40-subject V8_SEARCH pool; all historical restricted-data flags are false. OpenBMI internal 14-subject holdout accessed: **NO**. WBCIC outer accessed: **NO**. The repair code has no loader for either restricted source.",
    )
    phase_body = [
        "1. **PUD task consequence is real in the frozen evidence.** Protected erasure harm and teacher agreement remain measurable; this does not imply future utility.",
        f"2. Protected harm → future PUD−Vanilla: Pearson {primary['pearson']:.3f} [{primary['pearson_ci95_l']:.3f}, {primary['pearson_ci95_u']:.3f}], Spearman {primary['spearman']:.3f} [{primary['spearman_ci95_l']:.3f}, {primary['spearman_ci95_u']:.3f}]; **{primary['label']}**.",
        f"3. R_P concentration → harm status: **{reliance['B3']['status']}**; final H3: **{reliance['H3']}**.",
        f"4. Frozen source-certified directions on future data: **{transfer['consequence_status']}**.",
        f"5. Predictive ranking of future consequence by source U/D/P scores: **{transfer['score_status']}**.",
        f"6. Functional stability is retrospective only: margin contribution correlation {functional['margin_contribution']['pearson']:.3f}, cosine {functional['margin_contribution']['cosine']:.3f}, RMS change {functional['margin_contribution']['rms_change']:.4f}.",
        f"7. Hard bottleneck conclusion: **{reliance['H3']}**.",
        "8. Claims weakened relative to the old closure: mean consequence is separated from score transfer; reliance concentration is not said to explain loss without B3; no independence, biological causality, or identity-free claim is made.",
    ]
    write_md(EXP / "PHASE_A_REPAIRED_FINAL.md", "Phase A repaired final", "\n\n".join(phase_body))
    return summary


def main() -> None:
    ensure_dirs()
    protocol()
    audit = audit_frozen_tables()
    if not audit["pass"]:
        raise RuntimeError(f"frozen input purity failed: {audit['issues']}")
    before = frozen_model_hashes()
    consequence, correlations = build_consequence_table()
    reliance_table, reliance = build_reliance_gate(consequence)
    transfer_table, transfer = repair_certificate_transfer()
    _, functional = repair_functional_persistence()
    retain_secondary_diagnostics()
    make_phase_a_figures(consequence, reliance_table, transfer_table)
    hashes = verify_hashes_unchanged(before)
    summary = write_phase_a_reports(audit, hashes, correlations, reliance, transfer, functional)
    append_engineering_log(
        "Phase A protocol repairs",
        "Repaired the B0 join using the frozen replay table; enforced B3 in H3; mapped PUD basis columns to their actual certified coordinate indices; separated consequence transfer from score transfer; replaced direction-subject bootstrap with fold/run/direction/subject hierarchical resampling. No training or checkpoint modification occurred.",
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
