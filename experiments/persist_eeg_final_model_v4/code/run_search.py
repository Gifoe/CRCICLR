from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from common import (
    DESIGN,
    DIAGNOSTICS,
    LEADERBOARD,
    PROTOCOL,
    RESEARCH_LOG,
    V4_SEED,
    default_openbmi_cache,
    default_wbcic_repo,
    ensure_directories,
    markdown_table,
    sha256_file,
    write_csv,
    write_json,
)
from datasets import load_openbmi
from evaluation import summarize_method
from features import build_openbmi_features
from models import linear, residual, trees
from reconstruct import audit_legality, reconstruct_openbmi
from training import OOFResult, baseline_result, run_nested_oof


def _write_protocol(data, cache_root: Path) -> None:
    payload = {
        "status": "V4_DEVELOPMENT_PROTOCOL_FROZEN",
        "frozen_before_V4_model_outcomes": True,
        "seed": V4_SEED,
        "primary_reference": {
            "method_id": "B6_ALL_RUN_LOGIT_MEAN",
            "v4_id": "M0_B_STRONG_B6",
            "definition": "arithmetic mean of all legally available frozen KEEP binary margins",
        },
        "dataset_roles": {
            "OpenBMI": "exploratory discovery/architecture sandbox; 52 historical development subjects",
            "WBCIC": "authorized 41-subject development transfer only; sealed outer forbidden",
        },
        "outer_folds": data.folds,
        "selection_rule": (
            "within each outer fold, fit on model-training subjects; choose model configuration and "
            "classification threshold on disjoint calibration subjects by mean subject BA, then "
            "worst-subject delta, harm rate, rescue precision, and switch rate; evaluate once on held-out subjects"
        ),
        "candidate_selection_reads_heldout": False,
        "trial_random_split": False,
        "primary_metric": "mean subject-balanced BA; Delta vs M0_B_STRONG_B6",
        "strong_candidate_criteria": {
            "Delta_BA_vs_B_STRONG_min": 0.0,
            "subject_bootstrap_LCB95_min": 0.0,
            "positive_fold_fraction_min": 0.6,
            "nonnegative_subject_fraction_min": 0.5,
            "preferred_Delta_BA": 0.005,
        },
        "stop_rule": (
            "stop when distinct calibrated linear, tree, residual, correctness-ranking and set/MoE "
            "families plateau without stable grouped gain, or continued tuning would mainly reuse development outcomes"
        ),
        "openbmi_cache": str(cache_root),
        "OUTER_TEST_USED": False,
    }
    write_json(PROTOCOL / "DEVELOPMENT_PROTOCOL.json", payload)


def _write_design_review() -> None:
    review = """# V4 research design review

## What V3 established

V3 did not fail because the expert pool lacked headroom. B6 reached 0.846442
mean subject BA, while the KEEP-only oracle added 7.663 pp and the full global
action oracle added 8.596 pp. The combined menu added 10.702 pp. It failed
because prospective rescue/harm selectors converted rare switches into more
harm than rescue: the best M5 policy was -0.029 pp with a CI crossing zero.
Conditional ERASE correctness was discriminable (AUROC about 0.722), but the
candidate population was still harm-dominated. That makes selection and
aggregation—not action availability—the bottleneck.

## Modelling assumptions that were too weak

1. V3 predicted rescue and harm separately, then hard-switched. Separate
   probability errors are amplified by subtraction and thresholding.
2. It concentrated on action candidates before testing the stronger generic
   control: direct cross-fitted stacking of KEEP logits.
3. Hard selection throws away agreement information and makes one erroneous
   decision worth a full label flip. A bounded residual correction has a
   smaller failure surface.
4. Expert identity and variable expert count were reduced to aggregates. A
   permutation-invariant token model can estimate joint expert correctness
   without requiring the same expert count on both benchmarks.
5. PERSIST was used mainly as a flat feature vector. It may work better as a
   KEEP prior or an ERASE constraint than as ordinary trial-level signal.

## Evidence-informed candidate order

The first controls are subject-cross-fitted logistic stacking and shallow
boosting. Stacked generalization requires out-of-fold base predictions
(Wolpert, 1992, doi:10.1016/S0893-6080(05)80023-1; Ting & Witten, 1999,
doi:10.1613/jair.594). Dynamic ensemble selection literature treats local
competence estimation as the central problem rather than assuming the most
confident classifier is best (Cruz et al., 2018,
doi:10.1016/j.inffus.2017.09.010).

If direct stacking is insufficient, V4 tests B6-anchored residual logits,
joint expert correctness, and permutation-invariant expert aggregation.
Deep Sets (Zaheer et al., 2017, arXiv:1703.06114) and Set Transformer (Lee et
al., 2019, arXiv:1810.00825) motivate shared expert-token encoders, not a large
EEG encoder retrain. Learning-to-defer work motivates a conservative default
to B6 when the estimated utility gap is small (Mozannar & Sontag, 2020,
arXiv:2006.01862). SelectiveNet (Geifman & El-Yaniv, 2019,
arXiv:1901.09192) motivates reporting risk/switch coverage rather than only
accuracy. Calibration is evaluated explicitly because confidence ordering can
be useful while absolute probabilities are wrong (Guo et al., 2017,
PMLR 70:1321-1330).

## Falsifiable hypotheses

- H1: generic dynamic KEEP aggregation beats frozen B6 on grouped subjects.
- H2: action logits add gain beyond the best dynamic KEEP control.
- H3: PERSIST adds either BA or measurable safety beyond the matched
  KEEP+ACTION model.
- H4: bounded soft residual correction is safer than hard action switching.
- H5: the same information ladder transfers to WBCIC development subjects.

OpenBMI estimates remain exploratory. WBCIC outer is not read in V4.
"""
    (DESIGN / "RESEARCH_DESIGN_REVIEW.md").write_text(review, encoding="utf-8")
    families = """# Candidate families

| Family | First test | Reason | Failure signal |
| --- | --- | --- | --- |
| Calibrated linear | KEEP, KEEP+ACTION, +PERSIST | strongest low-variance stacking control | no grouped gain |
| Shallow HGB | same ladder | limited nonlinear interactions | fold instability/overfit |
| Anchored residual | bounded correction around B6 | reduces hard-switch harm | correction collapses to zero or harms |
| Joint correctness/ranking | score all expert tokens jointly | attacks selection target directly | poor rescue precision |
| DeepSets/MoE | shared token encoder, B6 prior | variable expert count, soft aggregation | no transfer or unstable weights |
| Frozen representation gate | optional only after meta plateau | tests missing representation information | capacity-only gain |

The action grid is finite: AMPLIFY, GEOMETRY, ERASE and alpha 0.25/0.5
interpolations. ERASE is high risk and is never forced.
"""
    (DESIGN / "CANDIDATE_FAMILIES.md").write_text(families, encoding="utf-8")


def _iteration_decision(row: dict[str, Any]) -> str:
    if float(row["Delta_BA_vs_B_STRONG"]) > 0 and float(row["CI95_L"]) > 0:
        return "KEEP"
    if float(row["Delta_BA_vs_B_STRONG"]) > 0:
        return "MODIFY"
    return "ABANDON"


def _write_iteration_logs(leaderboard: pd.DataFrame, feature_map: dict[str, str]) -> None:
    rows = []
    for iteration, item in enumerate(leaderboard.itertuples(index=False), 1):
        row = item._asdict()
        decision = "REFERENCE" if row["method_id"] == "M0_B_STRONG_B6" else _iteration_decision(row)
        diagnosis = (
            "frozen constructive reference"
            if decision == "REFERENCE"
            else (
                "grouped improvement with positive subject-bootstrap lower bound"
                if decision == "KEEP"
                else "gain is not robustly above the static ensemble"
            )
        )
        feature_set = feature_map.get(row["method_id"], "B6 only")
        payload = {
            "iteration": iteration,
            "model_id": row["method_id"],
            "diagnosis": diagnosis,
            "hypothesis": "test whether this information/architecture family converts frozen expert diversity into unseen-subject gain",
            "architecture": row["method_id"],
            "features": feature_set,
            "loss": "binary log-loss; hyperparameters and threshold selected on inner calibration subjects",
            "grouped_validation": "five outer subject folds with disjoint model-fit/calibration/heldout subjects",
            "BA": row["mean_subject_BA"],
            "Delta_BA_vs_B_STRONG": row["Delta_BA_vs_B_STRONG"],
            "Macro_F1": row["macro_f1"],
            "NLL": row["NLL"],
            "Brier": row["Brier"],
            "switch_rate": row["switch_rate"],
            "rescue_count": row["rescue_count"],
            "harm_count": row["harm_count"],
            "worst_subject_effect": row["worst_subject_delta"],
            "positive_subject_fraction": row["positive_subject_fraction"],
            "result": decision,
            "OUTER_TEST_USED": False,
        }
        text = f"""# Iteration {iteration:03d}: {row['method_id']}

- Diagnosis: {diagnosis}
- Hypothesis: {payload['hypothesis']}
- Architecture: `{row['method_id']}`
- Features: `{feature_set}`
- Validation: {payload['grouped_validation']}
- Mean subject BA: `{row['mean_subject_BA']:.6f}`
- Delta vs B_STRONG: `{100 * row['Delta_BA_vs_B_STRONG']:+.3f} pp`
- Subject bootstrap CI95: `[{100 * row['CI95_L']:+.3f}, {100 * row['CI95_U']:+.3f}] pp`
- Macro-F1 / NLL / Brier: `{row['macro_f1']:.6f}` / `{row['NLL']:.6f}` / `{row['Brier']:.6f}`
- Switch rate: `{100 * row['switch_rate']:.3f}%`
- Rescue / harm: `{row['rescue_count']}` / `{row['harm_count']}`
- Worst-subject delta: `{100 * row['worst_subject_delta']:+.3f} pp`
- Positive-subject fraction: `{row['positive_subject_fraction']:.3f}`
- Result: `{decision}`
- `OUTER_TEST_USED=false`
"""
        (RESEARCH_LOG / f"ITERATION_{iteration:03d}.md").write_text(text, encoding="utf-8")
        rows.append(payload)
    write_csv(RESEARCH_LOG / "ITERATION_SUMMARY.csv", pd.DataFrame(rows))


def run_initial(cache_root: Path, wbcic_repo: Path) -> pd.DataFrame:
    ensure_directories()
    reconstruct_openbmi(cache_root)
    audit_legality(cache_root, wbcic_repo)
    data = load_openbmi(cache_root)
    _write_protocol(data, cache_root)
    _write_design_review()
    bundle = build_openbmi_features(data)
    jobs = [
        ("M1_DYNAMIC_KEEP_LINEAR", "KEEP", "linear", linear.configurations(), linear.build),
        ("M1_DYNAMIC_KEEP_HGB", "KEEP", "tree", trees.configurations(), trees.build),
        ("M2_KEEP_ACTION_LINEAR", "KEEP_ACTION", "linear", linear.configurations(), linear.build),
        ("M2_KEEP_ACTION_HGB", "KEEP_ACTION", "tree", trees.configurations(), trees.build),
        ("M3_KEEP_ACTION_PERSIST_LINEAR", "KEEP_ACTION_PERSIST", "linear", linear.configurations(), linear.build),
        ("M3_KEEP_ACTION_PERSIST_HGB", "KEEP_ACTION_PERSIST", "tree", trees.configurations(), trees.build),
        ("M2_BOUNDED_RESIDUAL", "KEEP_ACTION", "residual", residual.configurations(), residual.build),
        ("M3_BOUNDED_RESIDUAL_PERSIST", "KEEP_ACTION_PERSIST", "residual", residual.configurations(), residual.build),
    ]
    results: list[OOFResult] = [baseline_result(data)]
    for method_id, feature_set, family, configurations, builder in jobs:
        print(f"[V4 initial] {method_id}", flush=True)
        results.append(
            run_nested_oof(
                data,
                method_id,
                bundle.matrices[feature_set],
                family,
                configurations,
                builder,
            )
        )

    leaderboard_rows, subject_tables, fold_tables, selection_tables, prediction_tables = [], [], [], [], []
    for result in results:
        row, subjects, folds = summarize_method(
            data,
            result.method_id,
            result.prediction,
            result.probability,
            result.outer_fold,
        )
        leaderboard_rows.append(row)
        subject_tables.append(subjects)
        fold_tables.append(folds)
        if not result.selections.empty:
            selection_tables.append(result.selections)
        prediction_tables.append(
            pd.DataFrame(
                {
                    "dataset": data.dataset_id,
                    "trial_uid": data.trial_uid,
                    "subject_id": data.subjects,
                    "session_id": data.sessions,
                    "method_id": result.method_id,
                    "outer_fold": result.outer_fold,
                    "label": data.labels,
                    "B_STRONG_prediction": data.base_prediction,
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "OUTER_TEST_USED": False,
                }
            )
        )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["Delta_BA_vs_B_STRONG", "NLL"], ascending=[False, True]
    ).reset_index(drop=True)
    write_csv(LEADERBOARD / "OPENBMI_MODEL_LEADERBOARD.csv", leaderboard)
    write_csv(DIAGNOSTICS / "SUBJECT_RESULTS.csv", pd.concat(subject_tables, ignore_index=True))
    write_csv(DIAGNOSTICS / "FOLD_RESULTS.csv", pd.concat(fold_tables, ignore_index=True))
    write_csv(DIAGNOSTICS / "OPENBMI_OOF_PREDICTIONS.csv", pd.concat(prediction_tables, ignore_index=True))
    if selection_tables:
        write_csv(DIAGNOSTICS / "CALIBRATION_SELECTION.csv", pd.concat(selection_tables, ignore_index=True))
    feature_map = {method_id: feature_set for method_id, feature_set, *_ in jobs}
    _write_iteration_logs(leaderboard, feature_map)
    runtime = {
        "status": "V4_INITIAL_SEARCH_COMPLETE",
        "command": "python experiments/persist_eeg_final_model_v4/code/run_search.py --stage initial",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "artifacts": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
            for path in sorted(Path(__file__).resolve().parents[1].joinpath("outputs").rglob("*"))
            if path.is_file()
        },
        "OUTER_TEST_USED": False,
    }
    write_json(Path(__file__).resolve().parents[1] / "outputs" / "REPRODUCIBILITY.json", runtime)
    print(markdown_table(leaderboard[["method_id", "mean_subject_BA", "Delta_BA_vs_B_STRONG", "CI95_L", "CI95_U"]]))
    return leaderboard


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("initial",), default="initial")
    parser.add_argument("--openbmi-cache", type=Path, default=default_openbmi_cache())
    parser.add_argument("--wbcic-repo", type=Path, default=default_wbcic_repo())
    args = parser.parse_args()
    run_initial(args.openbmi_cache, args.wbcic_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
