"""Competence-filtered multi-backbone Phase-A union diagnostic.

This program does not train a selector.  It constructs a broad exploratory
upper-bound bank from candidates already evaluated on the same V8_SEARCH
source folds.  A fixed competence rule excludes incomplete or grossly weak
experts before the subject-oracle calculation.  Because the rule is evaluated
on search outcomes, this artifact is explicitly not a prospective result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adaptation_banks.run_query_bank import _upsert
from common import DIAGNOSTICS, HEADROOM, PROTOCOL, RESEARCH_LOG, ensure_directories, write_csv, write_json
from evaluation.headroom import summarize_headroom
from protocol.datasets import assert_search_only, baseline_predictions, load_feature_fold


def _mean_subject_ba(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    values = {
        str(subject): float(balanced_accuracy_score(group.label, group.prediction))
        for subject, group in frame.groupby("subject_id", sort=False)
    }
    return float(np.mean(list(values.values()))), values


def run(
    benchmark: str,
    folds: tuple[int, ...],
    competence_tolerance_pp: float,
    minimum_positive_fraction: float,
) -> dict:
    ensure_directories()
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    benchmark_name = "OpenBMI_MI_S1_to_S2" if benchmark == "openbmi" else "WBCIC_S1S2_to_S3_authorized_development"
    fold_tag = "".join(map(str, folds))
    family_slug = f"MULTIBACKBONE_COMPETENCE_UNION_F{fold_tag}"
    family_id = f"{benchmark_name}__{family_slug}"
    expected_subjects = set().union(*(
        set(load_feature_fold(benchmark, fold, "CONFORMER_NORM").search_outcome_subjects)
        for fold in folds
    ))
    assert_search_only(tuple(expected_subjects), benchmark)
    locked, baseline_source_method = baseline_predictions(benchmark)
    baseline = locked.loc[
        locked.subject_id.astype(str).isin(expected_subjects)
        & locked.outer_fold.astype(int).isin(folds)
    ].copy()
    baseline["method_id"] = "B_STRONG_MATCHED_V7"
    baseline["family_id"] = family_id
    baseline["source_fold"] = baseline.outer_fold.astype(int)
    baseline["benchmark"] = benchmark_name
    baseline["internal_holdout_used"] = False
    expected_uids = set(baseline.trial_uid.astype(str))
    baseline_mean, baseline_subject = _mean_subject_ba(baseline)
    if set(baseline.subject_id.astype(str)) != expected_subjects:
        raise RuntimeError("Multi-backbone baseline subject mismatch")

    candidates = []
    audit_rows = []
    for path in sorted(DIAGNOSTICS.glob("*_SEARCH_PREDICTIONS.csv")):
        # Make reruns idempotent: a previously emitted union is a diagnostic
        # derivative, not a new source family.
        if "MULTIBACKBONE_COMPETENCE_UNION" in path.name:
            continue
        frame = pd.read_csv(path)
        required = {"benchmark", "family_id", "method_id", "trial_uid", "subject_id", "source_fold", "label", "probability", "prediction", "OUTER_TEST_USED"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.loc[
            frame.benchmark.astype(str).eq(benchmark_name)
            & frame.source_fold.astype(int).isin(folds)
            & ~frame.method_id.astype(str).eq("B_STRONG_MATCHED_V7")
        ].copy()
        if frame.empty:
            continue
        if frame.OUTER_TEST_USED.astype(bool).any():
            raise RuntimeError(f"Outer-test contamination in {path}")
        if "internal_holdout_used" in frame and frame.internal_holdout_used.astype(bool).any():
            raise RuntimeError(f"Internal-holdout contamination in {path}")
        for (origin_family, origin_method), group in frame.groupby(["family_id", "method_id"], sort=True):
            group = group.drop_duplicates("trial_uid")
            complete = set(group.trial_uid.astype(str)) == expected_uids
            if complete:
                mean_ba, subject_ba = _mean_subject_ba(group)
                delta = np.asarray([
                    subject_ba[subject] - baseline_subject[subject]
                    for subject in sorted(expected_subjects)
                ])
                positive_fraction = float(np.mean(delta > 0.0))
                eligible = bool(
                    mean_ba >= baseline_mean - competence_tolerance_pp / 100.0
                    and positive_fraction >= minimum_positive_fraction
                )
            else:
                mean_ba = np.nan
                positive_fraction = np.nan
                eligible = False
            method_id = f"MB{len(audit_rows):03d}__{str(origin_method)}"
            audit_rows.append({
                "benchmark": benchmark_name,
                "family_id": family_id,
                "candidate_id": method_id,
                "origin_family": str(origin_family),
                "origin_method": str(origin_method),
                "origin_file": path.name,
                "complete_expected_trials": complete,
                "mean_subject_BA": mean_ba,
                "positive_subject_fraction": positive_fraction,
                "eligible": eligible,
                "eligibility_rule": f"complete and BA >= baseline-{competence_tolerance_pp:g}pp and positive_fraction >= {minimum_positive_fraction:g}",
                "selection_scope": "V8_SEARCH outcome competence; exploratory upper-bound construction only",
                "internal_holdout_used": False,
                "OUTER_TEST_USED": False,
            })
            if eligible:
                selected = group.copy()
                selected["method_id"] = method_id
                selected["family_id"] = family_id
                selected["benchmark"] = benchmark_name
                selected["internal_holdout_used"] = False
                candidates.append(selected)
    audit = pd.DataFrame(audit_rows)
    if not candidates:
        raise RuntimeError("No complete competent multi-backbone candidates")
    primary = [str(frame.method_id.iloc[0]) for frame in candidates]
    predictions = pd.concat([baseline, *candidates], ignore_index=True)
    report = summarize_headroom(predictions, "B_STRONG_MATCHED_V7", primary)
    summary = report["summary"]
    summary.update({
        "folds": list(folds),
        "candidate_count_before_competence_filter": int(len(audit)),
        "eligible_candidate_count": int(len(primary)),
        "competence_tolerance_pp": competence_tolerance_pp,
        "minimum_positive_subject_fraction": minimum_positive_fraction,
        "baseline_source_method": baseline_source_method,
        "bank_type": "multi-backbone adapted candidate union",
        "prospective_status": "exploratory Phase-A upper bound; candidate competence screened on V8_SEARCH outcomes",
        "selector_trained": False,
    })
    tag = f"{prefix}_{family_slug}"
    write_csv(DIAGNOSTICS / f"{tag}_SEARCH_PREDICTIONS.csv", predictions)
    write_csv(DIAGNOSTICS / f"{tag}_CANDIDATE_AUDIT.csv", audit)
    write_csv(DIAGNOSTICS / f"{tag}_SUBJECT_RESULTS.csv", report["subjects"])
    write_json(HEADROOM / f"{tag}_HEADROOM.json", summary)
    write_csv(HEADROOM / f"{tag}_SUBJECT_ORACLE.csv", report["oracle"])
    write_csv(HEADROOM / f"{tag}_EXPERT_COMPETENCE.csv", report["competence"])
    write_csv(HEADROOM / f"{tag}_EXPERT_DIVERSITY.csv", report["diversity"])
    write_csv(HEADROOM / f"{tag}_ORACLE_BY_FOLD.csv", report["folds"])
    _upsert(HEADROOM / "HEADROOM_FAMILY_TABLE.csv", pd.DataFrame([summary]), ["benchmark", "family_id"])
    _upsert(HEADROOM / "EXPERT_COMPETENCE.csv", report["competence"], ["benchmark", "family_id", "method_id"])
    _upsert(HEADROOM / "EXPERT_DIVERSITY.csv", report["diversity"], ["benchmark", "family_id", "expert_left", "expert_right"])
    _upsert(HEADROOM / "SUBJECT_ORACLE.csv", report["oracle"], ["benchmark", "family_id", "subject_id"])
    _upsert(HEADROOM / "ORACLE_BY_FOLD.csv", report["folds"], ["benchmark", "family_id", "source_fold"])
    write_json(PROTOCOL / f"{tag}_LEGALITY.json", {
        "source_predictions": "V8_SEARCH-only Phase-A outputs",
        "candidate_filter": "fixed competence rule evaluated on V8_SEARCH outcomes",
        "interpretation": "exploratory upper bound, not a prospective selector result",
        "internal_holdout_used": False,
        "WBCIC_outer_split_opened": False,
        "OUTER_TEST_USED": False,
    })
    (RESEARCH_LOG / f"ITERATION_{tag}.md").write_text(
        f"# {tag}\n\nStructural diagnostic: test whether a competence-filtered union of distinct learned, geometric, normalization, metric, and raw-adaptation families contains headroom absent from each family alone.\n\n"
        f"```json\n{json.dumps(summary, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), action="append")
    parser.add_argument("--competence-tolerance-pp", type=float, default=1.0)
    parser.add_argument("--minimum-positive-fraction", type=float, default=0.25)
    args = parser.parse_args()
    folds = tuple(sorted(set(args.fold))) if args.fold else (0, 1, 2, 3, 4)
    run(args.benchmark, folds, args.competence_tolerance_pp, args.minimum_positive_fraction)


if __name__ == "__main__":
    main()
