"""Strict compact finalization of the frozen five-fold token pooling audit.

Reads only hash-sealed runtime outputs. OUTER values are summarized, never used
to update a score, subset, decoder C, q, tau, or proposed-model family.
Biological subject is the bootstrap unit. No final-heldout loader is present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aggregate import paired_subject_bootstrap, rank_subset, stable_seed


METHODS = ("MEAN_ALL", "FLATTEN_LINEAR", "ENERGY_TOPK", "RELIABLE_TOPK",
           "UTILITY_TOPK", "RELIABLE_UTILITY_TOPK", "RELIABILITY_WEIGHTED",
           "RU_WEIGHTED", "RU_TWO_STREAM", "ENERGY_TWO_STREAM")
CONTRASTS = (
    ("RELIABLE_TOPK", "MEAN_ALL"), ("UTILITY_TOPK", "MEAN_ALL"),
    ("RELIABLE_UTILITY_TOPK", "MEAN_ALL"), ("RELIABLE_UTILITY_TOPK", "UTILITY_TOPK"),
    ("RU_WEIGHTED", "MEAN_ALL"), ("RU_TWO_STREAM", "MEAN_ALL"),
    ("RU_TWO_STREAM", "RANDOM_TWO_STREAM_MEAN"),
    ("RELIABLE_UTILITY_TOPK", "FLATTEN_LINEAR"),
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_seal(path: Path, expected: str, directory: Path) -> dict:
    seal = json.loads(path.read_text(encoding="utf-8"))
    if seal.get("status") != expected or seal.get("formal_final_heldout_eeg_reads") != 0:
        raise RuntimeError(f"invalid seal status/final-heldout audit: {path}")
    for name, expected_sha in seal["files"].items():
        if Path(name).name != name or sha(directory / name) != expected_sha:
            raise RuntimeError(f"sealed artifact SHA mismatch: {directory / name}")
    return seal


def table(path: Path, *, subject: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"subject": str} if subject else None)
    if frame.empty:
        raise RuntimeError(f"empty compact source table: {path}")
    values = frame.select_dtypes(include="number").to_numpy()
    if not np.isfinite(values).all():
        raise RuntimeError(f"nonfinite compact source table: {path}")
    return frame


def token_index(repo: Path) -> pd.DataFrame:
    lock = repo / "experiments" / "persist_eeg_fm_rescue_stage0" / "protocol" / "FM_INPUT_PROTOCOL_LOCK.json"
    names = json.loads(lock.read_text(encoding="utf-8"))["OpenBMI"]["channels"]
    if len(names) != 62:
        raise RuntimeError("frozen OpenBMI channel count changed")
    return pd.DataFrame([{"token_index": 4 * channel + patch,
                          "channel_index": channel, "channel_name": name,
                          "temporal_patch_index": patch,
                          "approx_start_seconds": float(patch),
                          "approx_end_seconds": float(patch + 1),
                          "embedding_dimension": 200}
                         for channel, name in enumerate(names) for patch in range(4)])


def compact_sources(runtime: Path) -> tuple[dict, list[dict], dict[str, pd.DataFrame]]:
    evidence_dir = runtime / "train_evidence_v1"
    evidence_path = evidence_dir / "TRAIN_EVIDENCE_PROVENANCE.json"
    evidence = checked_seal(evidence_path, "COMPLETE_TRAIN_DESCRIPTIVE_NO_POOLING_SELECTION", evidence_dir)
    review_path = runtime / "train_gate_review_v1" / "TRAIN_GATE_A_B_REVIEW.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if sha(review_path) != evidence["train_gate_review_sha256"]:
        raise RuntimeError("TRAIN gate review changed after evidence aggregation")
    if review["outer_development_eeg_reads"] != 0 or review["formal_final_heldout_eeg_reads"] != 0:
        raise RuntimeError("TRAIN gate role audit mismatch")
    selection_rows, loso_rows, outer_rows, topk_rows, two_rows = [], [], [], [], []
    topk_subject_rows, two_subject_rows, fold_seals = [], [], []
    all_outer_subjects: set[str] = set()
    for fold in range(5):
        base = runtime / f"fold{fold}"
        choice_path = base / "TRAIN_SELECTION_SEAL.json"
        choice = checked_seal(choice_path, "FROZEN_TRAIN_ONLY", base / "train_selection_v1")
        eval_dir = base / "outer_evaluation_v2"
        eval_path = eval_dir / "OUTER_EVALUATION_SEAL.json"
        evaluated = checked_seal(eval_path, "COMPLETE_OUTER_EVALUATION_ONLY", eval_dir)
        if choice["fold"] != fold or evaluated["fold"] != fold or sha(choice_path) != evaluated["train_selection_seal_sha256"]:
            raise RuntimeError(f"fold {fold}: selection/evaluation identity mismatch")
        if choice["source_sha256"]["extraction"] != evaluated["train_extraction_sha256"]:
            raise RuntimeError(f"fold {fold}: TRAIN extraction identity mismatch")
        expected_sha = evidence["folds"][fold]
        if (choice["source_sha256"]["reliability"], choice["source_sha256"]["utility"]) != (
            expected_sha["reliability_seal_sha256"], expected_sha["utility_seal_sha256"]
        ):
            raise RuntimeError(f"fold {fold}: TRAIN score seal mismatch")
        choice_table = table(base / "train_selection_v1" / "TRAIN_POOLING_SELECTION.csv")
        loso = table(base / "train_selection_v1" / "TRAIN_LOSO_RESULTS.csv", subject=False)
        outer = table(eval_dir / "OUTER_POOLING_RESULTS.csv", subject=True)
        topk = table(eval_dir / "RANDOM_TOPK_NULL.csv")
        two = table(eval_dir / "RANDOM_TWO_STREAM_NULL.csv")
        topk_subject = table(eval_dir / "RANDOM_TOPK_SUBJECT.csv", subject=True)
        two_subject = table(eval_dir / "RANDOM_TWO_STREAM_SUBJECT.csv", subject=True)
        subjects = int(evaluated["outer_subject_count"])
        if (set(choice_table.method), choice_table.shape[0], set(outer.method), outer.shape[0],
            set(loso.method), len(loso), len(topk), len(two), len(topk_subject), len(two_subject)) != (
            set(METHODS), len(METHODS), set(METHODS), subjects * len(METHODS),
            set(METHODS), int(choice["train_subject_count"]) * len(METHODS),
            500, 500, 500 * subjects, 500 * subjects
        ):
            raise RuntimeError(f"fold {fold}: selected/OUTER/random row inventory mismatch")
        if set(outer.subject) != set(two_subject.subject) or set(outer.subject) != set(topk_subject.subject):
            raise RuntimeError(f"fold {fold}: matched OUTER biological-subject inventory mismatch")
        if all_outer_subjects & set(outer.subject):
            raise RuntimeError("biological subject appears in multiple OUTER folds")
        all_outer_subjects.update(outer.subject)
        if set(topk.repeat) != set(range(500)) or set(two.repeat) != set(range(500)):
            raise RuntimeError(f"fold {fold}: 500 random draw inventory incomplete")
        if evaluated["random_topk_draws"] != 500 or evaluated["random_two_stream_draws"] != 500:
            raise RuntimeError(f"fold {fold}: random control seal count mismatch")
        for name, frame, destination in (
            ("selection", choice_table, selection_rows), ("loso", loso, loso_rows),
            ("outer", outer, outer_rows), ("topk", topk, topk_rows),
            ("two", two, two_rows), ("topk_subject", topk_subject, topk_subject_rows),
            ("two_subject", two_subject, two_subject_rows),
        ):
            if set(frame.fold) != {fold}:
                raise RuntimeError(f"fold {fold}: {name} mixed fold identity")
            destination.append(frame)
        fold_seals.append({"fold": fold, "selection_seal_sha256": sha(choice_path),
                           "outer_evaluation_seal_sha256": sha(eval_path),
                           "train_subject_count": int(choice["train_subject_count"]),
                           "outer_subject_count": subjects})
    frames = {
        "TOKEN_RELIABILITY.csv": table(evidence_dir / "TOKEN_RELIABILITY.csv"),
        "TOKEN_RELIABILITY_NULL.csv": table(evidence_dir / "TOKEN_RELIABILITY_NULL.csv"),
        "TOKEN_RELIABILITY_REPRODUCIBILITY.csv": table(evidence_dir / "TOKEN_RELIABILITY_REPRODUCIBILITY.csv"),
        "TOKEN_UTILITY.csv": table(evidence_dir / "TOKEN_UTILITY.csv"),
        "TOKEN_RU_GROUPS.csv": table(evidence_dir / "TOKEN_RU_GROUPS.csv"),
        "TOKEN_STRUCTURE_DECOMPOSITION.csv": table(evidence_dir / "TOKEN_STRUCTURE_DECOMPOSITION.csv"),
        "CROSSFOLD_TOKEN_STABILITY.csv": table(evidence_dir / "CROSSFOLD_TOKEN_STABILITY.csv"),
        "TRAIN_POOLING_SELECTION.csv": pd.concat(selection_rows, ignore_index=True),
        "TRAIN_LOSO_RESULTS.csv": pd.concat(loso_rows, ignore_index=True),
        "OUTER_POOLING_RESULTS.csv": pd.concat(outer_rows, ignore_index=True),
        "RANDOM_TOPK_NULL.csv": pd.concat(topk_rows, ignore_index=True),
        "RANDOM_TWO_STREAM_NULL.csv": pd.concat(two_rows, ignore_index=True),
        "RANDOM_TOPK_SUBJECT.csv": pd.concat(topk_subject_rows, ignore_index=True),
        "RANDOM_TWO_STREAM_SUBJECT.csv": pd.concat(two_subject_rows, ignore_index=True),
    }
    return {"evidence_sha256": sha(evidence_path), "review_sha256": sha(review_path),
            "review": review, "fold_seals": fold_seals,
            "utility_erasure_convergence_warning_count": evidence["utility_erasure_convergence_warning_count"]}, fold_seals, frames


def grid_png(values: list[np.ndarray], output: Path, title: str, *, binary: bool = False) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(14, 12), constrained_layout=True)
    if binary:
        lo, hi, cmap = 0, 1, "Greys"
    else:
        combined = np.concatenate(values)
        lo, hi = float(np.quantile(combined, .02)), float(np.quantile(combined, .98))
        cmap = "coolwarm"
        if lo == hi:
            lo, hi = float(combined.min()) - 1, float(combined.max()) + 1
    for fold, (ax, vector) in enumerate(zip(axes, values)):
        image = ax.imshow(vector.reshape(62, 4), aspect="auto", origin="lower", cmap=cmap,
                          vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"Fold {fold}")
        ax.set_xticks(range(4))
        ax.set_xlabel("Temporal patch")
        if fold == 0:
            ax.set_ylabel("Channel index")
    figure.suptitle(title + " (TRAIN-derived; descriptive only)")
    figure.colorbar(image, ax=axes, fraction=.018, pad=.01)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def selected_pivot(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = frames["OUTER_POOLING_RESULTS.csv"]
    pivot = outer.pivot(index=["fold", "subject"], columns="method", values="BA")
    if pivot.isna().any().any() or set(pivot.columns) != set(METHODS):
        raise RuntimeError("OUTER subject/method metric grid incomplete")
    random_two = frames["RANDOM_TWO_STREAM_SUBJECT.csv"]
    random_mean = random_two.groupby(["fold", "subject"]).BA.mean().rename("RANDOM_TWO_STREAM_MEAN")
    pivot = pivot.join(random_mean, validate="one_to_one")
    if pivot.isna().any().any():
        raise RuntimeError("random two-stream subject control incomplete")
    biological = pivot.reset_index().groupby("subject", as_index=True)[list(pivot.columns)].mean()
    effects = pivot.reset_index().copy()
    for a, b in CONTRASTS:
        effects[f"{a}_minus_{b}"] = effects[a] - effects[b]
    fold_ba = pivot.reset_index().groupby("fold")[list(pivot.columns)].mean()
    return effects, biological, fold_ba


def bootstrap(biological: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in CONTRASTS:
        x = biological[a].to_dict()
        y = biological[b].to_dict()
        result = paired_subject_bootstrap(x, y, draws=20_000,
                                          seed=stable_seed("subject-bootstrap", a, b))
        rows.append({"method_a": a, "method_b": b, "unit": "biological_subject", **result})
    return pd.DataFrame(rows)


def decision(provenance: dict, frames: dict[str, pd.DataFrame], effects: pd.DataFrame,
             biological: pd.DataFrame, fold_ba: pd.DataFrame, boot: pd.DataFrame) -> dict:
    review = provenance["review"]
    pooled = biological.mean(axis=0).to_dict()
    ru_mean = pooled["RELIABLE_UTILITY_TOPK"] - pooled["MEAN_ALL"]
    ru_utility = pooled["RELIABLE_UTILITY_TOPK"] - pooled["UTILITY_TOPK"]
    ru_energy = pooled["RELIABLE_UTILITY_TOPK"] - pooled["ENERGY_TOPK"]
    positive_mean = int(((fold_ba["RELIABLE_UTILITY_TOPK"] - fold_ba["MEAN_ALL"]) > 0).sum())
    positive_utility = int(((fold_ba["RELIABLE_UTILITY_TOPK"] - fold_ba["UTILITY_TOPK"]) > 0).sum())
    random_topk = frames["RANDOM_TOPK_NULL.csv"]
    counts = {int(row["fold"]): int(row["outer_subject_count"]) for row in provenance["fold_seals"]}
    weighted_random = random_topk.assign(weight=random_topk.fold.map(counts))
    draws = weighted_random.groupby("repeat").apply(
        lambda x: float(np.average(x.BA, weights=x.weight)), include_groups=False)
    random_p95 = float(np.quantile(draws, .95))
    gates = {
        "A": bool(review["gate_A_pass"]),
        "B": bool(review["gate_B_numeric_pass"]),
        "C": bool(ru_mean >= .015 and positive_mean >= 4),
        "D": bool(ru_utility >= .005 and positive_utility >= 4),
        "E": bool(ru_energy > 0 and pooled["RELIABLE_UTILITY_TOPK"] > random_p95),
        "F_descriptive": bool(pooled["RU_TWO_STREAM"] > pooled["RELIABLE_UTILITY_TOPK"]),
    }
    if not gates["A"]:
        interpretation, action = "NO_MEANINGFUL_TOKEN_RELIABILITY_HETEROGENEITY", "STOP_RELIABILITY_POOLING_DIRECTION"
    elif not gates["B"]:
        interpretation, action = "TOKEN_RELIABILITY_NOT_REPRODUCIBLE", "STOP_RELIABILITY_POOLING_DIRECTION"
    elif not gates["C"]:
        interpretation, action = "TOKEN_RELIABILITY_REAL_BUT_NO_PREDICTIVE_HEADROOM", "STOP_RELIABILITY_POOLING_DIRECTION"
    elif not gates["D"]:
        interpretation, action = "TOKEN_HETEROGENEITY_REAL_BUT_UTILITY_EXPLAINS_SELECTION", "STOP_PERSISTENCE_GUIDED_POOLING"
    elif not gates["E"]:
        interpretation, action = "TOKEN_SELECTION_NOT_SPECIFIC", "STOP_RELIABILITY_POOLING_DIRECTION"
    else:
        interpretation, action = "TOKEN_RELIABILITY_POOLING_SUPPORTED", "PROCEED_RELIABILITY_AWARE_POOLING_MODEL"
    return {"schema": "TOKEN_RELIABILITY_POOLING_DECISION_V1",
            "status": "COMPLETE_WITH_ALL_FIVE_OUTER_DEVELOPMENT_FOLDS",
            "gates": gates, "primary_interpretation": interpretation, "next_action": action,
            "primary_proposed_method_locked_before_outer": "RELIABLE_UTILITY_TOPK",
            "subject_equal_BA": pooled, "RU_minus_MEAN_BA": float(ru_mean),
            "RU_minus_UTILITY_BA": float(ru_utility), "RU_minus_ENERGY_BA": float(ru_energy),
            "RU_positive_folds_vs_MEAN": positive_mean,
            "RU_positive_folds_vs_UTILITY": positive_utility,
            "random_topk_p95_pooled_subject_equal_BA": random_p95,
            "split_half_mean_spearman": review["split_half_mean_spearman_across_folds"],
            "split_half_mean_top_quartile_jaccard": review["split_half_mean_top_quartile_jaccard_across_folds"],
            "crossfold_pairwise_R_spearman_mean": float(frames["CROSSFOLD_TOKEN_STABILITY.csv"].R_spearman.mean()),
            "bootstrap_contrasts": boot.to_dict(orient="records"),
            "outer_development_warning": "historically exposed development cohort; evaluation-only in this audit",
            "formal_final_heldout_eeg_reads": 0}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def report(provenance: dict, frames: dict[str, pd.DataFrame], decision_doc: dict,
           effects: pd.DataFrame, fold_ba: pd.DataFrame, boot: pd.DataFrame) -> str:
    review = provenance["review"]
    pooled = decision_doc["subject_equal_BA"]
    r_u = [row["R_U_probe_spearman"] for row in json.loads(
        (Path(provenance["runtime"]) / "train_evidence_v1" / "TRAIN_EVIDENCE_PROVENANCE.json").read_text(encoding="utf-8"))["folds"]]
    groups = frames["TOKEN_RU_GROUPS.csv"]
    counts = groups.RU_group.value_counts().to_dict()
    structure = frames["TOKEN_STRUCTURE_DECOMPOSITION.csv"]
    null = frames["TOKEN_RELIABILITY_NULL.csv"]
    lines = ["# CBraMod pre-pooling token reliability qualification audit", "",
             "All five frozen OpenBMI_MI/seed-0 folds were analyzed. The neural backbone and task head were not retrained. "
             "TRAIN_GEOMETRY alone supplied token scores and all selections; OUTER_DEVELOPMENT was evaluation-only "
             "and historically exposed. Formal final-heldout EEG reads: **0**.", "",
             f"**Primary interpretation: `{decision_doc['primary_interpretation']}`.**",
             f"**Next action: `{decision_doc['next_action']}`.**", "",
             "The predeclared Gate A/B failures cannot be repaired by an OUTER gain. "
             "No Reliability-Aware Pooling architecture is justified by this audit.", "",
             "## Required questions", "",
             "**Q1.** Is reliability meaningfully heterogeneous? A raw quartile gap is visible, "
             "but the required null-calibrated spread test fails in all five folds.", "",
             "**Q2.** Stronger than both nulls? The subject-pairing spread null is not exceeded "
             "in any fold. Pairing empirical spread p by fold: " + ", ".join(
                 fmt(row["pairing_null_spread_empirical_p"]) for row in review["folds"]) +
             "; Session-2-label-null p: " + ", ".join(
                 fmt(row["label_null_spread_empirical_p"]) for row in review["folds"]) +
             ". A label-null signal in a subset of folds does not rescue the predeclared pairing-null Gate A.", "",
             f"**Q3.** Reproducible ranking? No: mean split-half Spearman {fmt(review['split_half_mean_spearman_across_folds'])} "
             f"and quartile Jaccard {fmt(review['split_half_mean_top_quartile_jaccard_across_folds'])}; "
             f"mean crossfold Spearman {fmt(decision_doc['crossfold_pairwise_R_spearman_mean'])}.", "",
             f"**Q4.** Reliability–utility relation? Across-fold R/U_probe Spearman mean {fmt(float(np.mean(r_u)))} "
             f"(range {fmt(min(r_u))} to {fmt(max(r_u))}). U_erase is provisional because the all-token "
             "liblinear fit emitted convergence warnings.", "",
             f"**Q5.** Both mismatched groups exist: reliable/low-utility {counts.get('RELIABLE_LOWUTILITY', 0)} "
             f"and fragile/useful {counts.get('FRAGILE_USEFUL', 0)} token-fold assignments. "
             "These are TRAIN-median-defined descriptive groups, not biological labels.", ""]
    questions = [
        ("Q6", "RELIABLE_TOPK", "MEAN_ALL", "reliability-only hard selection"),
        ("Q7", "UTILITY_TOPK", "MEAN_ALL", "utility-only hard selection"),
        ("Q8", "RELIABLE_UTILITY_TOPK", "UTILITY_TOPK", "incremental reliability beyond utility"),
        ("Q9", "RELIABLE_UTILITY_TOPK", "ENERGY_TOPK", "energy-ranked control"),
        ("Q10", "RU_WEIGHTED", "RELIABLE_UTILITY_TOPK", "soft weighting versus RU hard selection"),
        ("Q11", "RU_TWO_STREAM", "RELIABLE_UTILITY_TOPK", "two-stream versus RU pooled"),
        ("Q12", "RELIABLE_UTILITY_TOPK", "FLATTEN_LINEAR", "RU pooling versus information-preserving flatten"),
    ]
    for q, a, b, meaning in questions:
        lines.append(f"**{q}.** {meaning}: subject-equal BA {fmt(pooled[a])} vs {fmt(pooled[b])}, "
                     f"difference {fmt(pooled[a]-pooled[b])}. "
                     "This does not override the failed prerequisite gates.")
        lines.append("")
    lines.extend([
        f"For Q9, the matched RANDOM_TOPK pooled p95 is {fmt(decision_doc['random_topk_p95_pooled_subject_equal_BA'])}; "
        f"Gate E is {decision_doc['gates']['E']}.", "",
        f"For Q11, the 500-draw RANDOM_TWO_STREAM subject-mean control BA is "
        f"{fmt(pooled['RANDOM_TWO_STREAM_MEAN'])}; Gate F is descriptive only.", "",
        f"**Q13.** Descriptive TRAIN-only variance fractions (mean across folds): channel "
        f"{fmt(structure.channel_fraction.mean())}, temporal patch {fmt(structure.time_fraction.mean())}, "
        f"interaction/residual {fmt(structure.interaction_fraction.mean())}. "
        "This is not a causal or scalp-physiology claim.", "",
        "**Q14.** No credible explanatory mechanism for earlier final-embedding P/C failures "
        "can be inferred from a token reliability premise that fails its null/reproducibility gates. "
        "Token reliability and the previous Protected subspace are different objects.", "",
        f"**Q15.** Build a new pooling architecture? No. `{decision_doc['next_action']}`.", "",
        "## Biological-subject bootstrap", "",
        "20,000 paired draws; subject, not trial, is the resampling unit.", "",
        "| Contrast (BA) | Difference | 95% CI |", "| --- | ---: | ---: |",
    ])
    for row in boot.itertuples(index=False):
        lines.append(f"| {row.method_a} − {row.method_b} | {fmt(row.mean_difference)} | "
                     f"[{fmt(row.ci025)}, {fmt(row.ci975)}] |")
    lines.extend(["", "## Final fold table", "",
                  "All BA values are OUTER_DEVELOPMENT subject-equal. Positive/harmed fractions compare RU_TOPK with MEAN_ALL. "
                  "No family was selected on OUTER.", ""])
    headers = ["fold", "TRAIN n", "OUTER n", "tokens", "d_model", "R mean", "R std", "R q-gap",
               "half rho", "half J", "R/U rho", "q R", "q U", "q RU", "MEAN", "FLAT", "ENERGY",
               "R TOPK", "U TOPK", "RU TOPK", "R WEIGHT", "RU WEIGHT", "RU TWO",
               "random topk p95", "random two p95", "RU−MEAN", "RU−U", "improved", "harmed",
               "A", "B", "C", "D", "E", "interpretation"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    selection = frames["TRAIN_POOLING_SELECTION.csv"]
    rel = frames["TOKEN_RELIABILITY.csv"]
    utility = frames["TOKEN_UTILITY.csv"]
    topk_null = frames["RANDOM_TOPK_NULL.csv"]
    two_null = frames["RANDOM_TWO_STREAM_NULL.csv"]
    for fold in range(5):
        rev = review["folds"][fold]
        choices = selection[selection.fold == fold].set_index("method")
        ba = fold_ba.loc[fold]
        subset = effects[effects.fold == fold]
        gain = float(ba.RELIABLE_UTILITY_TOPK - ba.MEAN_ALL)
        gain_u = float(ba.RELIABLE_UTILITY_TOPK - ba.UTILITY_TOPK)
        a = bool(rev["gate_A_fold"])
        b = bool(rev["split_half_mean_spearman"] >= .50 and rev["split_half_mean_top_quartile_jaccard"] >= .40)
        c = bool(gain >= .015)
        d = bool(gain_u >= .005)
        e = bool(ba.RELIABLE_UTILITY_TOPK > ba.ENERGY_TOPK and
                 ba.RELIABLE_UTILITY_TOPK > topk_null[topk_null.fold == fold].BA.quantile(.95))
        values = [str(fold), str(provenance["fold_seals"][fold]["train_subject_count"]),
                  str(provenance["fold_seals"][fold]["outer_subject_count"]), "248", "200",
                  fmt(float(rel[rel.fold == fold].R_cos.mean())), fmt(rev["real_reliability_std"]),
                  fmt(rev["real_top_bottom_quartile_gap"]), fmt(rev["split_half_mean_spearman"]),
                  fmt(rev["split_half_mean_top_quartile_jaccard"]),
                  fmt(float(spearmanr(rel[rel.fold == fold].R_cos,
                                      utility[utility.fold == fold].U_probe).statistic)),
                  str(choices.loc["RELIABLE_TOPK", "parameter"]),
                  str(choices.loc["UTILITY_TOPK", "parameter"]),
                  str(choices.loc["RELIABLE_UTILITY_TOPK", "parameter"])]
        values += [fmt(float(ba[method])) for method in ("MEAN_ALL", "FLATTEN_LINEAR", "ENERGY_TOPK",
                                                     "RELIABLE_TOPK", "UTILITY_TOPK", "RELIABLE_UTILITY_TOPK",
                                                     "RELIABILITY_WEIGHTED", "RU_WEIGHTED", "RU_TWO_STREAM")]
        values += [fmt(float(topk_null[topk_null.fold == fold].BA.quantile(.95))),
                   fmt(float(two_null[two_null.fold == fold].BA.quantile(.95))),
                   fmt(gain), fmt(gain_u),
                   fmt(float((subset["RELIABLE_UTILITY_TOPK_minus_MEAN_ALL"] > 0).mean())),
                   fmt(float((subset["RELIABLE_UTILITY_TOPK_minus_MEAN_ALL"] < 0).mean())),
                   str(a), str(b), str(c), str(d), str(e),
                   "NO_MEANINGFUL_TOKEN_RELIABILITY_HETEROGENEITY" if not a else "SEE_POOLED_GATES"]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Pooled decision", "",
                  f"MEAN_ALL BA {fmt(pooled['MEAN_ALL'])}; FLATTEN_LINEAR BA {fmt(pooled['FLATTEN_LINEAR'])}; "
                  f"reliability-only BA {fmt(pooled['RELIABLE_TOPK'])}; utility-only BA {fmt(pooled['UTILITY_TOPK'])}; "
                  f"RU_TOPK BA {fmt(pooled['RELIABLE_UTILITY_TOPK'])}; two-stream BA {fmt(pooled['RU_TWO_STREAM'])}.", "",
                  f"RU−MEAN {fmt(decision_doc['RU_minus_MEAN_BA'])}; RU−UTILITY "
                  f"{fmt(decision_doc['RU_minus_UTILITY_BA'])}; paired CIs appear above.", "",
                  f"Pairing-null spread empirical p by fold: " + ", ".join(
                      fmt(row["pairing_null_spread_empirical_p"]) for row in review["folds"] ) + ".", "",
                  f"Split-half Spearman {fmt(review['split_half_mean_spearman_across_folds'])}; "
                  f"top-quartile Jaccard {fmt(review['split_half_mean_top_quartile_jaccard_across_folds'])}.", "",
                  f"Gates A/B/C/D/E: " + "/".join(str(decision_doc["gates"][k]) for k in "ABCDE") + ". "
                  f"Gate F descriptive: {decision_doc['gates']['F_descriptive']}.", "",
                  f"Primary interpretation: `{decision_doc['primary_interpretation']}`. "
                  f"Exact next action: `{decision_doc['next_action']}`.", "",
                  "Formal final-heldout EEG reads: **0**.", "",
                  "Branch: `codex/persist-eeg-token-reliability-pooling-audit-seed0-v1`. "
                  "The final commit SHA is recorded in the GitHub publication message; a commit cannot contain its own hash.", "",
                  "## Integrity notes", "",
                  "High-dimensional all-token U_erase produced convergence warnings and is descriptive/provisional. "
                  "The primary U_probe and all selection scores were TRAIN-only. OUTER is historically exposed "
                  "development data and not formal final-heldout. The score/select/evaluate task logs and bulky token "
                  "caches remain on the original server, not in Git. Prior failed attempts remain disclosed.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    repo = Path(__file__).resolve().parents[3]
    provenance, _, frames = compact_sources(runtime)
    provenance["runtime"] = str(runtime)
    output = runtime / "final_compact_v1"
    output.mkdir(exist_ok=False)
    frames["TOKEN_INDEX.csv"] = token_index(repo)
    effects, biological, fold_ba = selected_pivot(frames)
    frames["SUBJECT_LEVEL_EFFECTS.csv"] = effects
    boot = bootstrap(biological)
    frames["BOOTSTRAP_CONTRASTS.csv"] = boot
    result = decision(provenance, frames, effects, biological, fold_ba, boot)
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    r_vectors = [frames["TOKEN_RELIABILITY.csv"].query("fold == @fold").sort_values("token_index").R_cos.to_numpy()
                 for fold in range(5)]
    u_vectors = [frames["TOKEN_UTILITY.csv"].query("fold == @fold").sort_values("token_index").U_probe.to_numpy()
                 for fold in range(5)]
    selected = frames["TRAIN_POOLING_SELECTION.csv"]
    masks = []
    for fold in range(5):
        q = float(selected.query("fold == @fold and method == 'RELIABLE_UTILITY_TOPK'").parameter.iloc[0])
        r, u = r_vectors[fold], u_vectors[fold]
        score = (r-r.mean())/r.std() + (u-u.mean())/u.std()
        mask = np.zeros(248)
        mask[rank_subset(score, q)] = 1
        masks.append(mask)
    grid_png(r_vectors, output / "TOKEN_RELIABILITY_HEATMAP.png", "Cosine reliability")
    grid_png(u_vectors, output / "TOKEN_UTILITY_HEATMAP.png", "Transfer-probe utility")
    differences = [(r-r.mean())/r.std() - (u-u.mean())/u.std() for r, u in zip(r_vectors, u_vectors)]
    grid_png(differences, output / "RELIABILITY_MINUS_UTILITY_HEATMAP.png", "Standardized R minus U")
    grid_png(masks, output / "SELECTED_RU_TOKEN_HEATMAP.png", "TRAIN-selected RU tokens", binary=True)
    (output / "DECISION_SUMMARY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    exclusion = {"schema": "TOKEN_POOLING_FINAL_HELDOUT_EXCLUSION_AUDIT_V1",
                 "formal_final_heldout_eeg_reads": 0,
                 "source_roles": ["TRAIN_GEOMETRY", "OUTER_DEVELOPMENT"],
                 "checkpoint_validation": "inventory only; not loaded into selection or evaluation",
                 "outer_development": "historically exposed, evaluation only",
                 "claim_scope": "audit process and code paths, not other repository experiments"}
    (output / "FINAL_HELDOUT_EXCLUSION_AUDIT.json").write_text(
        json.dumps(exclusion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "FINAL_REPORT.md").write_text(
        report(provenance, frames, result, effects, fold_ba, boot), encoding="utf-8")
    inventory = {"schema": "TOKEN_POOLING_FINAL_COMPACT_INVENTORY_V1",
                 "status": "COMPLETE_REVIEW_REQUIRED_BEFORE_GIT_PUSH",
                 "train_evidence_provenance_sha256": provenance["evidence_sha256"],
                 "train_gate_review_sha256": provenance["review_sha256"],
                 "fold_seals": provenance["fold_seals"],
                 "formal_final_heldout_eeg_reads": 0,
                 "files": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output / "FINAL_COMPACT_INVENTORY.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": inventory["status"], "files": len(inventory["files"]),
                      "inventory_sha256": sha(output / "FINAL_COMPACT_INVENTORY.json"),
                      "primary_interpretation": result["primary_interpretation"]},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
